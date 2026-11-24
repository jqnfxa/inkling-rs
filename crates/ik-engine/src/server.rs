//! The serving layer, after mini-SGLang: an HTTP front end (OpenAI API) that renders and tokenizes requests,
//! a scheduler thread that owns the engine and runs the requests, and an incremental detokenizer and reply parser
//! that stream the result back (server-sent events).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, LazyLock, Mutex};

use serde_json::{Value, json};

use crate::chat::{self, Delta, Parser};
use crate::compact;
use crate::engine::Engine;
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::Tokenizer;

pub struct Job {
    pub prompt: Vec<u32>,
    pub params: SamplingParams,
    /// at most this many tokens, or until the context is full
    pub max_tokens: Option<usize>,
    /// thinking tokens at most, then the thinking is closed so that the model answers
    pub thinking_budget: Option<usize>,
    pub stop: Vec<String>,
    /// raw text out (completions) rather than parsed chat messages
    pub raw: bool,
    /// set by POST /v1/chat/completions/control
    pub control: Arc<Control>,
    /// compaction of the thinking when it fills the context
    pub compaction: compact::Settings,
    /// stream the prompt's progress (return_progress) and the timings of each token (timings_per_token)
    pub progress: bool,
    pub timings: bool,
    pub tx: Sender<Event>,
}

/// What a running job can be told by POST /v1/chat/completions/control.
#[derive(Default)]
pub struct Control {
    /// reasoning_end: close the thinking now
    pub end_reasoning: AtomicBool,
    /// compact: compact the context now, if the model is thinking
    pub compact: AtomicBool,
}

/// The running jobs, by completion id.
static RUNNING: LazyLock<Mutex<HashMap<String, Arc<Control>>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub enum Event {
    /// the prompt so far: tokens in the cache of total (reused ones included), and milliseconds
    PromptProgress { processed: usize, total: usize, cache: usize, ms: f64 },
    /// before the deltas of each token: the prompt's numbers, and tokens and seconds of generation so far
    Tick { prompt_n: usize, cache_n: usize, prompt_s: f64, n: usize, decode_s: f64 },
    Delta(Delta),
    /// a compaction under way: the summary's tokens so far ("summary"), or the tokens of the rebuilt context ("rebuild")
    Compaction { stage: &'static str, done: usize, total: usize },
    /// decode_s leaves out the compactions, which took compact_s
    Done { finish: &'static str, prompt_tokens: usize, completion_tokens: usize, reused: usize, prefill_s: f64, decode_s: f64, compactions: usize, compact_s: f64 },
    Error(String),
}

/// Bytes to text, holding back an unfinished UTF-8 sequence.
#[derive(Default)]
struct Utf8Buf(Vec<u8>);

impl Utf8Buf {
    fn push(&mut self, b: &[u8]) -> String {
        self.0.extend_from_slice(b);
        match std::str::from_utf8(&self.0) {
            Ok(s) => {
                let r = s.to_string();
                self.0.clear();
                r
            }
            Err(e) => {
                let ok = e.valid_up_to();
                if e.error_len().is_some() {
                    // invalid bytes, not a cut: replace them
                    let r = String::from_utf8_lossy(&self.0).into_owned();
                    self.0.clear();
                    return r;
                }
                let r = String::from_utf8_lossy(&self.0[..ok]).into_owned();
                self.0.drain(..ok);
                r
            }
        }
    }
}

/// The compactions of one answer.
struct Compactor {
    cfg: compact::Settings,
    /// ask for the summary in a system message rather than a user one (IK_COMPACT_ROLE=system)
    system: bool,
    /// the end of the model's thinking tag in the sequence, once found
    open_end: Option<usize>,
    /// where the thinking kept verbatim begins, after the last summary
    body: usize,
    count: usize,
    /// a compaction did not fit: the thinking is closed at the reserve instead
    failed: bool,
    secs: f64,
    /// tokens waited so far for the end of a line, to compact there
    wait: usize,
}

/// Tokens to wait at most for the end of a line before a compaction (and a quarter of the reserve at most).
const LINE_WAIT: usize = 256;

enum Compacted {
    /// the rebuilt sequence's logits
    Done(Vec<f32>),
    /// no room for a summary that helps: the sequence is as it was
    NoRoom,
    /// the client left
    Gone,
    Error(String),
}

/// Compacts the thinking that fills the context: asks the model for a summary on the cache it has, then rebuilds the
/// sequence as the prompt, the model's message up to its thinking tag, the summary, and the end of the thinking.
/// The prompt ends at `prompt_end`, where the engine's snapshot is.
fn compact(eng: &mut Engine, tok: &Tokenizer, c: &mut Compactor, prompt_end: usize, params: &SamplingParams, tx: &Sender<Event>) -> Compacted {
    let think = tok.token_id(chat::CONTENT_THINKING);
    let open_end = match c.open_end.or_else(|| eng.tokens[prompt_end..].iter().position(|&t| Some(t) == think).map(|i| prompt_end + i + 1)) {
        Some(e) => e,
        None => return Compacted::NoRoom,
    };
    c.open_end = Some(open_end);
    c.body = c.body.max(open_end);
    let len = eng.tokens.len();
    // the summary's length: a third of what it replaces, within 768..summary_tokens; the model is told it in words
    let budget = ((len - open_end.min(len)) / 3).clamp(768, c.cfg.summary_tokens.max(768));
    let req = tok.encode(&compact::request(c.count > 0, c.system, budget * 3 / 5), true);
    let (head, tail) = (tok.encode(compact::SUMMARY_HEAD, false), tok.encode(compact::SUMMARY_TAIL, false));
    // the rebuilt sequence takes half the context before the reserve at most, so that the next compaction is far
    let target = eng.n_ctx.saturating_sub(c.cfg.reserve) / 2;
    let fixed = open_end + head.len() + tail.len();
    let cap = budget.min(c.cfg.summary_tokens).min(eng.n_ctx.saturating_sub(len + req.len() + 1)).min(target.saturating_sub(fixed));
    if cap < 256 || len <= c.body {
        return Compacted::NoRoom;
    }
    let body: Vec<u32> = eng.tokens[c.body..len].to_vec();
    let open: Vec<u32> = eng.tokens[prompt_end..open_end].to_vec();
    let gone = |ev: Event| tx.send(ev).is_err();

    // the summary, on the cache: the thinking ends, the request follows, and the model answers it in plain text
    if gone(Event::Delta(Delta::Reasoning(format!("\n\n{}", compact::SUMMARY_HEAD)))) {
        return Compacted::Gone;
    }
    let mut logits = match eng.extend(&req, &mut |_| {}) {
        Ok(l) => l,
        Err(e) => return Compacted::Error(e),
    };
    // sampled as the answer is but no hotter than 0.6 (at 1.0 it went on solving, at 0.3 it looped); a line that
    // repeats an earlier one is left out of it
    let mut sampler = Sampler::new(SamplingParams { temperature: params.temperature.min(0.6), ..params.clone() });
    let mut summary = vec![];
    let mut utf8 = Utf8Buf::default();
    let mut lines = compact::Lines::default();
    while summary.len() < cap && lines.repeats < 4 {
        let t = sampler.sample(&logits, &eng.tokens, &[]);
        if tok.is_control(t) || tok.eos.contains(&t) {
            break;
        }
        summary.push(t);
        let piece = utf8.push(&tok.piece_bytes(t, false));
        if let Some(at) = lines.push(summary.len() - 1, &piece) {
            summary.truncate(at);
        }
        if !piece.is_empty() && gone(Event::Delta(Delta::Reasoning(piece))) {
            return Compacted::Gone;
        }
        if summary.len() % 32 == 0 && gone(Event::Compaction { stage: "summary", done: summary.len(), total: cap }) {
            return Compacted::Gone;
        }
        logits = match eng.decode(t) {
            Ok(l) => l,
            Err(e) => return Compacted::Error(e),
        };
    }

    // the rebuilt sequence; a summary of almost nothing is left out, and as much of the thinking kept as fits
    let pieces: Vec<Vec<u8>> = body.iter().map(|&t| tok.piece_bytes(t, true)).collect();
    let failed = summary.len() < 32;
    let (seq, kept) = if failed {
        compact::rebuild(&open, &[], &[], &[], &body, &pieces, target.saturating_sub(open_end))
    } else {
        compact::rebuild(&open, &head, &summary, &tail, &body, &pieces, c.cfg.keep.min(target.saturating_sub(fixed + summary.len())))
    };
    let p = match eng.rollback(seq.len()) {
        Ok(p) => p,
        Err(e) => return Compacted::Error(e),
    };
    let total = seq.len();
    let mut left = false;
    let rebuilt = eng.extend(&seq, &mut |pos| left |= tx.send(Event::Compaction { stage: "rebuild", done: pos - p, total }).is_err());
    let logits = match rebuilt {
        Ok(l) => l,
        Err(e) => return Compacted::Error(e),
    };
    c.body = p + seq.len() - kept;
    c.count += 1;
    let note = if failed {
        format!("\n[context compacted: no summary came, {} tokens of reasoning dropped, the last {kept} kept]\n\n", len - open_end - kept)
    } else {
        format!("\n[context compacted: {} tokens of reasoning summarized in {}, the last {kept} kept]\n\n", len - open_end, summary.len())
    };
    if left || gone(Event::Delta(Delta::Reasoning(note))) {
        return Compacted::Gone;
    }
    eprintln!("compaction {}: {} -> {} tokens ({} summary, {kept} kept)", c.count, len, eng.tokens.len(), summary.len());
    Compacted::Done(logits)
}

/// The scheduler: owns the engine, runs the jobs in order, and keeps the cached sequence for the next prompt.
pub fn scheduler(mut eng: Engine, tok: Arc<Tokenizer>, rx: Receiver<Job>) {
    let end_sampling = tok.token_id(chat::END_SAMPLING);
    let eot = tok.token_id("<|endoftext|>");
    let system_role = std::env::var("IK_COMPACT_ROLE").is_ok_and(|v| v == "system");
    // what the model writes to close its thinking and begin the answer
    let close: Vec<(u32, &str)> = [chat::END_MESSAGE, chat::MESSAGE_MODEL, chat::CONTENT_TEXT].iter().filter_map(|&s| tok.token_id(s).map(|t| (t, s))).collect();
    while let Ok(job) = rx.recv() {
        let t0 = std::time::Instant::now();
        let total = job.prompt.len();
        let tx = job.tx.clone();
        let progress = job.progress;
        let fed = eng.feed_progress(&job.prompt, &mut |processed, cache| {
            if progress {
                let _ = tx.send(Event::PromptProgress { processed, total, cache, ms: t0.elapsed().as_secs_f64() * 1e3 });
            }
        });
        let (mut logits, reused) = match fed {
            Ok(r) => r,
            Err(e) => {
                let _ = job.tx.send(Event::Error(e));
                let _ = eng.reset();
                continue;
            }
        };
        let prefill_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        // no more than the context holds; a finite max_tokens keeps a quarter (up to 1024) for the answer after thinking
        let limit = job.max_tokens.unwrap_or(usize::MAX);
        let mut budget = job.thinking_budget.unwrap_or(usize::MAX);
        if let Some(m) = job.max_tokens {
            budget = budget.min(m - (m / 4).min(1024));
        }
        let mut sampler = Sampler::new(job.params.clone());
        let mut parser = Parser::default();
        let mut utf8 = Utf8Buf::default();
        let mut text = String::new();
        let (mut n, mut n_think, mut closed, mut line_end) = (0, 0, false, true);
        let mut finish = "length";
        let banned: Vec<u32> = vec![];
        let mut comp = Compactor { cfg: job.compaction.clone(), system: system_role, open_end: None, body: 0, count: 0, failed: false, secs: 0.0, wait: 0 };
        let send = |d: Delta, text: &mut String, stop_hit: &mut bool| -> bool {
            if let Delta::Content(c) = &d {
                *text += c;
                if job.stop.iter().any(|s| !s.is_empty() && text.contains(s.as_str())) {
                    *stop_hit = true;
                }
            }
            job.tx.send(Event::Delta(d)).is_ok()
        };
        'tokens: loop {
            let room = eng.n_ctx.saturating_sub(eng.tokens.len() + 1);
            if n >= limit || room == 0 {
                break;
            }
            let thinking = !job.raw && !closed && parser.thinking();
            // the thinking fills the context (or the user asked): compact it at the end of a line (up to LINE_WAIT tokens
            // later), or if that cannot be, close it at the reserve. Compaction switched off: only when asked, and the
            // thinking runs to the end of the context as before.
            let mut full = false;
            if thinking {
                full = comp.cfg.enabled && eng.tokens.len() + comp.cfg.reserve >= eng.n_ctx;
                let asked = job.control.compact.load(Ordering::Relaxed);
                let can = !comp.failed && comp.count < comp.cfg.max;
                if (full || asked) && can && (line_end || comp.wait >= LINE_WAIT.min(comp.cfg.reserve / 4)) {
                    comp.wait = 0;
                    job.control.compact.store(false, Ordering::Relaxed);
                    let tc = std::time::Instant::now();
                    let r = compact(&mut eng, &tok, &mut comp, total, &job.params, &job.tx);
                    comp.secs += tc.elapsed().as_secs_f64();
                    match r {
                        Compacted::Done(l) => {
                            logits = l;
                            line_end = true;
                            continue;
                        }
                        Compacted::NoRoom => comp.failed = true,
                        Compacted::Gone => {
                            finish = "abort";
                            break;
                        }
                        Compacted::Error(e) => {
                            let _ = job.tx.send(Event::Error(e));
                            let _ = eng.reset();
                            finish = "error";
                            break;
                        }
                    }
                } else if (full || asked) && can {
                    comp.wait += 1;
                }
                full = full && (comp.failed || comp.count >= comp.cfg.max);
            }
            // the thinking is over its budget, or the user ended it: close it as the model would, and let it answer
            if thinking && (full || n_think >= budget || job.control.end_reasoning.load(Ordering::Relaxed)) {
                closed = true;
                if n + close.len() >= limit || close.len() >= room {
                    break;
                }
                for &(t, s) in &close {
                    for d in parser.push(s) {
                        let mut stop_hit = false;
                        if !send(d, &mut text, &mut stop_hit) {
                            finish = "abort";
                            break 'tokens;
                        }
                    }
                    n += 1;
                    logits = match eng.decode(t) {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = job.tx.send(Event::Error(e));
                            let _ = eng.reset();
                            finish = "error";
                            break 'tokens;
                        }
                    };
                }
                continue;
            }
            let t = sampler.sample(&logits, &eng.tokens, &banned);
            n += 1;
            if parser.thinking() {
                n_think += 1;
            }
            if Some(t) == end_sampling || Some(t) == eot || tok.eos.contains(&t) {
                // the end of the reply is part of the sequence, so the next turn continues it
                let _ = eng.decode(t);
                finish = "stop";
                break;
            }
            if job.timings {
                let decode_s = t1.elapsed().as_secs_f64() - comp.secs;
                let _ = job.tx.send(Event::Tick { prompt_n: total - reused, cache_n: reused, prompt_s: prefill_s, n, decode_s });
            }
            let piece = utf8.push(&tok.piece_bytes(t, true));
            if !piece.is_empty() {
                line_end = piece.ends_with('\n');
            }
            let deltas = if job.raw { vec![Delta::Content(piece.clone())] } else { parser.push(&piece) };
            let mut stop_hit = false;
            let mut gone = false;
            for d in deltas {
                if !send(d, &mut text, &mut stop_hit) {
                    gone = true;
                }
            }
            if gone {
                finish = "abort";
                break;
            }
            if stop_hit {
                finish = "stop";
                break;
            }
            logits = match eng.decode(t) {
                Ok(l) => l,
                Err(e) => {
                    let _ = job.tx.send(Event::Error(e));
                    let _ = eng.reset();
                    finish = "error";
                    break;
                }
            };
        }
        let _ = job.tx.send(Event::Done {
            finish,
            prompt_tokens: job.prompt.len(),
            completion_tokens: n,
            reused,
            prefill_s,
            decode_s: t1.elapsed().as_secs_f64() - comp.secs,
            compactions: comp.count,
            compact_s: comp.secs,
        });
    }
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

fn read_request(s: &mut BufReader<TcpStream>) -> Option<Request> {
    let mut line = String::new();
    s.read_line(&mut line).ok()?;
    let mut it = line.split_whitespace();
    let method = it.next()?.to_string();
    let path = it.next()?.to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        if s.read_line(&mut h).ok()? == 0 {
            return None;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                len = v.trim().parse().ok()?;
            }
        }
    }
    if len > 64 << 20 {
        return None;
    }
    let mut body = vec![0u8; len];
    s.read_exact(&mut body).ok()?;
    Some(Request { method, path, body })
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

fn respond(s: &mut TcpStream, code: u16, body: &Value) {
    let b = body.to_string();
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Internal Server Error",
    };
    let _ = write!(s, "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{CORS}Connection: close\r\n\r\n{b}", b.len());
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn params_of(r: &Value) -> SamplingParams {
    let d = SamplingParams::default();
    let f = |k: &str, dv: f32| r.get(k).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(dv);
    SamplingParams {
        temperature: f("temperature", d.temperature),
        top_k: r.get("top_k").and_then(Value::as_u64).map(|v| v as usize).unwrap_or(d.top_k),
        top_p: f("top_p", d.top_p),
        min_p: f("min_p", d.min_p),
        repeat_penalty: f("repeat_penalty", f("repetition_penalty", d.repeat_penalty)),
        presence_penalty: f("presence_penalty", d.presence_penalty),
        frequency_penalty: f("frequency_penalty", d.frequency_penalty),
        repeat_last_n: d.repeat_last_n,
        seed: r.get("seed").and_then(Value::as_u64).unwrap_or(0),
    }
}

/// What the server says about itself, and the browser UI it serves.
pub struct Site {
    /// GET /props, as llama-server answers it (the llama.cpp chat UI reads it)
    pub props: Value,
    /// GET /v1/models
    pub models: Value,
    /// the files of a browser UI, served at / (llama.cpp's built chat UI), if any
    pub ui: Option<std::path::PathBuf>,
    /// compaction unless a request says otherwise
    pub compaction: compact::Settings,
}

/// The request's `compaction` (`false`, or {enabled, reserve_tokens, keep_recent_tokens, max_compactions,
/// summary_tokens}) over the server's settings.
fn compaction_of(r: &Value, d: &compact::Settings) -> compact::Settings {
    let mut c = d.clone();
    match r.get("compaction") {
        Some(Value::Bool(b)) => c.enabled = *b,
        Some(o @ Value::Object(_)) => {
            let n = |k: &str| o.get(k).and_then(Value::as_u64).map(|v| v as usize);
            c.enabled = o.get("enabled").and_then(Value::as_bool).unwrap_or(c.enabled);
            c.reserve = n("reserve_tokens").unwrap_or(c.reserve);
            c.keep = n("keep_recent_tokens").unwrap_or(c.keep);
            c.max = n("max_compactions").unwrap_or(c.max);
            c.summary_tokens = n("summary_tokens").unwrap_or(c.summary_tokens);
        }
        _ => {}
    }
    c
}

/// Serves file `path` of the UI directory `root` (index.html for /); false if there is no such file.
fn serve_file(s: &mut TcpStream, root: &std::path::Path, path: &str) -> bool {
    let rel = if path == "/" { "index.html" } else { path.trim_start_matches('/') };
    if rel.split('/').any(|c| c == ".." || c.starts_with('.')) {
        return false;
    }
    let Ok(file) = root.join(rel).canonicalize() else { return false };
    let Ok(base) = root.canonicalize() else { return false };
    if !file.starts_with(&base) || !file.is_file() {
        return false;
    }
    let Ok(body) = std::fs::read(&file) else { return false };
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    let ty = match ext {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "webmanifest" => "application/manifest+json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "wasm" => "application/wasm",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    };
    // hashed assets never change under their name; index.html names them, so it is always checked again
    let cache = if rel.contains("/immutable/") { "public, max-age=31536000, immutable" } else { "no-cache" };
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: {ty}\r\nContent-Length: {}\r\nCache-Control: {cache}\r\n{CORS}Connection: close\r\n\r\n", body.len());
    let _ = stream_write(s, head.as_bytes()).and_then(|_| stream_write(s, &body));
    true
}

fn stream_write(s: &mut TcpStream, b: &[u8]) -> std::io::Result<()> {
    s.write_all(b)
}

fn handle(mut stream: TcpStream, jobs: Sender<Job>, tok: Arc<Tokenizer>, model: Arc<String>, site: Arc<Site>) {
    let mut rd = BufReader::new(stream.try_clone().unwrap());
    let Some(req) = read_request(&mut rd) else { return };
    let path = req.path.split('?').next().unwrap_or("");
    match (req.method.as_str(), path) {
        ("OPTIONS", _) => {
            let _ = write!(stream, "HTTP/1.1 204 No Content\r\n{CORS}Content-Length: 0\r\n\r\n");
        }
        ("GET", "/health") | ("GET", "/v1/health") => respond(&mut stream, 200, &json!({"status": "ok"})),
        ("GET", "/props") => respond(&mut stream, 200, &site.props),
        // llama-server's own tools (--tools), and its resumable streams: none here
        ("GET", "/tools") => respond(&mut stream, 200, &json!([])),
        ("POST", "/v1/streams/lookup") => respond(&mut stream, 200, &json!([])),
        ("GET", "/v1/models") | ("GET", "/models") => respond(&mut stream, 200, &site.models),
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") | ("POST", "/v1/completions") | ("POST", "/completions") => {
            let chat = path.contains("chat");
            let r: Value = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return respond(&mut stream, 400, &json!({"error": {"message": format!("bad JSON: {e}")}})),
            };
            let prompt = if chat {
                let msgs = r.get("messages").and_then(Value::as_array).cloned().unwrap_or_default();
                // thinking switched off (the llama.cpp UI sends enable_thinking: false): the lowest effort
                let off = Value::String("none".into());
                let effort = r.get("reasoning_effort").or_else(|| r.pointer("/chat_template_kwargs/reasoning_effort")).or_else(|| {
                    (r.pointer("/chat_template_kwargs/enable_thinking") == Some(&Value::Bool(false))).then_some(&off)
                });
                match chat::render(&msgs, r.get("tools"), effort) {
                    Ok(p) => p,
                    Err(e) => return respond(&mut stream, 400, &json!({"error": {"message": e}})),
                }
            } else {
                r.get("prompt").and_then(Value::as_str).unwrap_or("").to_string()
            };
            let ids = tok.encode(&prompt, true);
            let stop: Vec<String> = match r.get("stop") {
                Some(Value::String(s)) => vec![s.clone()],
                Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
                _ => vec![],
            };
            // missing, 0 or -1: until the context is full, as llama-server
            let max_tokens = r.get("max_completion_tokens").or_else(|| r.get("max_tokens")).and_then(Value::as_i64).filter(|&m| m > 0).map(|m| m as usize);
            let thinking_budget = r.get("thinking_budget_tokens").and_then(Value::as_i64).filter(|&b| b >= 0).map(|b| b as usize);
            let streaming = r.get("stream").and_then(Value::as_bool).unwrap_or(false);
            let flag = |k: &str| streaming && r.get(k).and_then(Value::as_bool).unwrap_or(false);
            let id = format!("{}-{}{:03}", if chat { "chatcmpl" } else { "cmpl" }, now(), NEXT_ID.fetch_add(1, Ordering::Relaxed) % 1000);
            let control = Arc::new(Control::default());
            if let Ok(mut m) = RUNNING.lock() {
                m.insert(id.clone(), control.clone());
            }
            let (tx, rx) = channel();
            let job = Job {
                prompt: ids,
                params: params_of(&r),
                max_tokens,
                thinking_budget,
                stop,
                raw: !chat,
                control,
                compaction: compaction_of(&r, &site.compaction),
                progress: flag("return_progress"),
                timings: flag("timings_per_token"),
                tx,
            };
            if jobs.send(job).is_err() {
                return respond(&mut stream, 500, &json!({"error": {"message": "the scheduler is gone"}}));
            }
            if streaming {
                stream_reply(&mut stream, rx, &id, &model, chat);
            } else {
                whole_reply(&mut stream, rx, &id, &model, chat);
            }
            if let Ok(mut m) = RUNNING.lock() {
                m.remove(&id);
            }
        }
        // {"action": "reasoning_end" | "compact", "id": <completion id>}; reasoning_end is the llama.cpp UI's "end thinking"
        ("POST", "/v1/chat/completions/control") | ("POST", "/chat/completions/control") => {
            let r: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            let (action, id) = (r.get("action").and_then(Value::as_str).unwrap_or(""), r.get("id").and_then(Value::as_str).unwrap_or(""));
            let job = RUNNING.lock().ok().and_then(|m| m.get(id).cloned());
            match (action, job) {
                ("reasoning_end" | "compact", Some(c)) => {
                    let flag = if action == "compact" { &c.compact } else { &c.end_reasoning };
                    flag.store(true, Ordering::Relaxed);
                    respond(&mut stream, 200, &json!({"success": true}))
                }
                ("reasoning_end" | "compact", None) => respond(&mut stream, 404, &json!({"success": false, "error": {"message": format!("no running completion {id}")}})),
                _ => respond(&mut stream, 400, &json!({"success": false, "error": {"message": format!("unknown action {action}")}})),
            }
        }
        ("GET", p) if site.ui.as_deref().is_some_and(|root| serve_file(&mut stream, root, p)) => {}
        _ => respond(&mut stream, 404, &json!({"error": {"message": format!("no route {} {}", req.method, path)}})),
    }
}

fn usage(p: usize, c: usize, reused: usize, prefill_s: f64, decode_s: f64, compactions: usize, compact_s: f64) -> Value {
    let mut u = json!({
        "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
        "prompt_tokens_details": {"cached_tokens": reused},
        "timings": {"prefill_s": prefill_s, "decode_s": decode_s, "decode_tokens_per_s": if decode_s > 0.0 { c as f64 / decode_s } else { 0.0 }},
    });
    if compactions > 0 {
        u["timings"]["compactions"] = json!(compactions);
        u["timings"]["compact_s"] = json!(compact_s);
    }
    u
}

fn chunk(id: &str, model: &str, chat: bool, delta: Value, finish: Option<&str>) -> Value {
    if chat {
        json!({"id": id, "object": "chat.completion.chunk", "created": now(), "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    } else {
        json!({"id": id, "object": "text_completion", "created": now(), "model": model,
               "choices": [{"index": 0, "text": delta.get("content").cloned().unwrap_or(Value::String(String::new())), "finish_reason": finish}]})
    }
}

fn stream_reply(s: &mut TcpStream, rx: Receiver<Event>, id: &str, model: &str, chat: bool) {
    if write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n").is_err() {
        return;
    }
    let mut send = |v: &Value| write!(s, "data: {v}\n\n").and_then(|_| s.flush()).is_ok();
    if chat && !send(&chunk(id, model, chat, json!({"role": "assistant", "content": ""}), None)) {
        return;
    }
    let mut ntool = 0;
    // the timings of the last token (when asked for), on each chunk as llama-server puts them
    let mut tick: Option<Value> = None;
    let with_tick = |mut c: Value, tick: &Option<Value>| {
        if let Some(t) = tick {
            c["timings"] = t.clone();
        }
        c
    };
    for ev in rx {
        let ok = match ev {
            Event::PromptProgress { processed, total, cache, ms } => {
                let mut c = chunk(id, model, chat, json!({}), None);
                c["prompt_progress"] = json!({"total": total, "cache": cache, "processed": processed, "time_ms": ms});
                send(&c)
            }
            Event::Tick { prompt_n, cache_n, prompt_s, n, decode_s } => {
                tick = Some(timings(prompt_n + cache_n, n, cache_n, prompt_s, decode_s));
                true
            }
            Event::Delta(Delta::Content(c)) => send(&with_tick(chunk(id, model, chat, json!({"content": c}), None), &tick)),
            Event::Delta(Delta::Reasoning(r)) => send(&with_tick(chunk(id, model, chat, json!({"reasoning_content": r}), None), &tick)),
            Event::Delta(Delta::ToolCall(name, args)) => {
                ntool += 1;
                let c = chunk(id, model, chat, json!({"tool_calls": [{"index": ntool - 1, "id": format!("call_{ntool}"), "type": "function", "function": {"name": name, "arguments": args}}]}), None);
                send(&with_tick(c, &tick))
            }
            Event::Compaction { stage, done, total } => {
                let mut c = chunk(id, model, chat, json!({}), None);
                c["compaction"] = json!({"stage": stage, "done": done, "total": total});
                send(&c)
            }
            Event::Done { finish, prompt_tokens, completion_tokens, reused, prefill_s, decode_s, compactions, compact_s } => {
                let finish = if ntool > 0 && finish == "stop" { "tool_calls" } else { finish };
                let mut last = chunk(id, model, chat, json!({}), Some(finish));
                last["usage"] = usage(prompt_tokens, completion_tokens, reused, prefill_s, decode_s, compactions, compact_s);
                last["timings"] = timings(prompt_tokens, completion_tokens, reused, prefill_s, decode_s);
                if compactions > 0 {
                    last["timings"]["compactions"] = json!(compactions);
                    last["timings"]["compact_ms"] = json!(compact_s * 1e3);
                }
                send(&last);
                let _ = write!(s, "data: [DONE]\n\n");
                return;
            }
            Event::Error(e) => {
                send(&json!({"error": {"message": e}}));
                return;
            }
        };
        if !ok {
            return; // the client left: dropping rx makes the scheduler stop the job
        }
    }
}

/// The speeds as llama-server reports them (the llama.cpp chat UI shows them).
fn timings(prompt_tokens: usize, completion_tokens: usize, reused: usize, prefill_s: f64, decode_s: f64) -> Value {
    let pn = prompt_tokens.saturating_sub(reused);
    json!({
        "cache_n": reused,
        "prompt_n": pn,
        "prompt_ms": prefill_s * 1e3,
        "prompt_per_token_ms": if pn > 0 { prefill_s * 1e3 / pn as f64 } else { 0.0 },
        "prompt_per_second": if prefill_s > 0.0 { pn as f64 / prefill_s } else { 0.0 },
        "predicted_n": completion_tokens,
        "predicted_ms": decode_s * 1e3,
        "predicted_per_token_ms": if completion_tokens > 0 { decode_s * 1e3 / completion_tokens as f64 } else { 0.0 },
        "predicted_per_second": if decode_s > 0.0 { completion_tokens as f64 / decode_s } else { 0.0 },
    })
}

fn whole_reply(s: &mut TcpStream, rx: Receiver<Event>, id: &str, model: &str, chat: bool) {
    let (mut content, mut reasoning) = (String::new(), String::new());
    let mut tools = vec![];
    for ev in rx {
        match ev {
            Event::PromptProgress { .. } | Event::Tick { .. } | Event::Compaction { .. } => {}
            Event::Delta(Delta::Content(c)) => content += &c,
            Event::Delta(Delta::Reasoning(r)) => reasoning += &r,
            Event::Delta(Delta::ToolCall(name, args)) => {
                tools.push(json!({"id": format!("call_{}", tools.len() + 1), "type": "function", "function": {"name": name, "arguments": args}}))
            }
            Event::Done { finish, prompt_tokens, completion_tokens, reused, prefill_s, decode_s, compactions, compact_s } => {
                let finish = if !tools.is_empty() && finish == "stop" { "tool_calls" } else { finish };
                let u = usage(prompt_tokens, completion_tokens, reused, prefill_s, decode_s, compactions, compact_s);
                let body = if chat {
                    let mut msg = json!({"role": "assistant", "content": content});
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = Value::String(reasoning.clone());
                    }
                    if !tools.is_empty() {
                        msg["tool_calls"] = Value::Array(tools.clone());
                    }
                    json!({"id": id, "object": "chat.completion", "created": now(), "model": model,
                           "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": u,
                           "timings": timings(prompt_tokens, completion_tokens, reused, prefill_s, decode_s)})
                } else {
                    json!({"id": id, "object": "text_completion", "created": now(), "model": model,
                           "choices": [{"index": 0, "text": content, "finish_reason": finish}], "usage": u})
                };
                return respond(s, 200, &body);
            }
            Event::Error(e) => return respond(s, 500, &json!({"error": {"message": e}})),
        }
    }
}

/// Serves the OpenAI API on `addr` until the process ends.
pub fn serve(addr: &str, jobs: Sender<Job>, tok: Arc<Tokenizer>, model: String, site: Site) -> std::io::Result<()> {
    let l = TcpListener::bind(addr)?;
    let model = Arc::new(model);
    let site = Arc::new(site);
    for conn in l.incoming() {
        let Ok(conn) = conn else { continue };
        let (jobs, tok, model, site) = (jobs.clone(), tok.clone(), model.clone(), site.clone());
        std::thread::spawn(move || handle(conn, jobs, tok, model, site));
    }
    Ok(())
}
