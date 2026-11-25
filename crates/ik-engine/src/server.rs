//! The serving layer, after mini-SGLang: an HTTP front end (OpenAI API) that renders and tokenizes requests,
//! a scheduler thread that owns the engine and runs the requests, and an incremental detokenizer and reply parser
//! that stream the result back (server-sent events).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Condvar, LazyLock, Mutex};

use serde_json::{Value, json};

use crate::chat::{self, Delta, Parser};
use crate::compact;
use crate::engine::{Engine, T_MAX};
use crate::sampler::{Sampler, SamplingParams};
use crate::tokenizer::Tokenizer;

/// A chat request as it came, to render again with its older messages summarized (compaction between turns).
pub struct ChatIn {
    pub messages: Vec<Value>,
    pub tools: Option<Value>,
    pub effort: Option<Value>,
}

pub struct Job {
    /// the completion id (the ids of its tool calls derive from it)
    pub id: String,
    pub prompt: Vec<u32>,
    /// the chat the prompt was rendered from (None for a raw completion)
    pub chat: Option<ChatIn>,
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
    /// progress notes on the thinking (reasoning_summary)
    pub notes: compact::Notes,
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
    /// the client left (or DELETE /v1/stream): stop the job at its next token
    pub stop: AtomicBool,
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
    /// a progress note on the thinking
    Note(String),
    /// decode_s leaves out the compactions and the notes, which took aside.1 and aside.3
    Done { finish: &'static str, prompt_tokens: usize, completion_tokens: usize, reused: usize, prefill_s: f64, decode_s: f64, aside: Aside },
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

/// Work beside the answer's decoding: compactions and their seconds, notes and their seconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Aside(pub usize, pub f64, pub usize, pub f64);

/// The progress notes of one answer.
struct Noter {
    cfg: compact::Notes,
    system: bool,
    /// tokens of thinking since the last note
    since: usize,
    /// tokens waited for the end of a line
    wait: usize,
    count: usize,
    secs: f64,
}

impl Noter {
    /// A note is due: the first after a quarter of the interval, then one each interval.
    fn due(&self) -> bool {
        self.cfg.mode != compact::NoteMode::Off && self.since >= if self.count == 0 { (self.cfg.every / 4).max(64) } else { self.cfg.every.max(64) }
    }

    /// When the thinking ends: a last note on what came after the one before, if notes are all the client sees.
    fn last_due(&self) -> bool {
        self.cfg.mode == compact::NoteMode::Only && self.since >= 128
    }
}

/// Generates text after `logits` until a control token or `cap` tokens; the last is not decoded, so the sequence
/// grows by `cap` tokens at most. Sampled as the request asks but no hotter than 0.6 (at 1.0 a summary went on solving,
/// at 0.3 it looped); a line that repeats an earlier one is dropped, and after 4 such the text ends. `on(piece, tokens
/// kept so far)` sees each piece; false from it (the client left) stops it: Ok(None).
fn generate(eng: &mut Engine, tok: &Tokenizer, mut logits: Vec<f32>, cap: usize, params: &SamplingParams, on: &mut dyn FnMut(&str, usize) -> bool) -> Result<Option<Vec<u32>>, String> {
    let mut sampler = Sampler::new(SamplingParams { temperature: params.temperature.min(0.6), ..params.clone() });
    let (mut out, mut utf8, mut lines) = (vec![], Utf8Buf::default(), compact::Lines::default());
    for step in 1..=cap {
        let t = sampler.sample(&logits, &eng.tokens, &[]);
        if tok.is_control(t) || tok.eos.contains(&t) {
            break;
        }
        out.push(t);
        let piece = utf8.push(&tok.piece_bytes(t, false));
        if let Some(at) = lines.push(out.len() - 1, &piece) {
            out.truncate(at);
        }
        if !on(&piece, out.len()) {
            return Ok(None);
        }
        if step == cap || lines.repeats >= 4 {
            break;
        }
        logits = eng.decode(t)?;
    }
    Ok(Some(out))
}

/// Takes a progress note on a branch of the cache: the thinking ends, the request follows, the model writes the note,
/// and the sequence goes back to the checkpoint, so the logits the caller holds stay right. Streams it; returns
/// whether the client is still there.
fn take_note(eng: &mut Engine, tok: &Tokenizer, nt: &mut Noter, params: &SamplingParams, tx: &Sender<Event>) -> Result<bool, String> {
    let t0 = std::time::Instant::now();
    nt.since = 0;
    nt.wait = 0;
    let req = tok.encode(&compact::note_request(nt.system), true);
    // the branch stays within T_MAX tokens, which the checkpoint can come back from
    if req.len() + 24 > T_MAX || eng.tokens.len() + T_MAX >= eng.n_ctx {
        return Ok(true);
    }
    eng.checkpoint()?;
    let logits = eng.extend(&req, &mut |_| {})?;
    let out = generate(eng, tok, logits, T_MAX - req.len(), params, &mut |_, _| true)?.unwrap_or_default();
    eng.back_to_checkpoint()?;
    nt.secs += t0.elapsed().as_secs_f64();
    let text = compact::clean_note(&format!("{}{}", compact::NOTE_START, tok.decode(&out, false)));
    if text.is_empty() {
        return Ok(true);
    }
    nt.count += 1;
    let shown = if nt.cfg.mode == compact::NoteMode::Only { format!("{text}\n\n") } else { format!("\n\n{}\n\n", compact::quote(&text)) };
    Ok(tx.send(Event::Note(text)).is_ok() && tx.send(Event::Delta(Delta::Reasoning(shown))).is_ok())
}

/// Summaries of the older messages of chats: (hash of the messages summarized, how many, the summary). The latest
/// few, so that a chat UI that sends the whole history at each turn finds the summary made at an earlier turn.
type Summaries = Vec<(u64, usize, String)>;

/// Compaction between turns, after Pi: when a chat's prompt and the reserve do not fit, the older messages are
/// replaced by a summary (made now, in chunks if they are longer than the context, or made at an earlier turn of the
/// same chat), and the last ones kept. Returns the new prompt and how many messages a new summary covers (0: a
/// summary from before); None to keep the prompt there is.
fn compact_history(eng: &mut Engine, tok: &Tokenizer, job: &Job, cache: &mut Summaries) -> Result<Option<(Vec<u32>, usize)>, String> {
    let (Some(ch), cfg) = (&job.chat, &job.compaction) else { return Ok(None) };
    let fits = |len: usize| len + cfg.reserve <= eng.n_ctx;
    if !cfg.enabled || fits(job.prompt.len()) {
        return Ok(None);
    }
    let msgs = &ch.messages;
    let role = |m: &Value| m.get("role").and_then(Value::as_str).unwrap_or("").to_string();
    let lead = msgs.iter().take_while(|m| role(m) == "system").count();
    let render = |ms: &[Value], tools: bool| -> Result<Vec<u32>, String> { Ok(tok.encode(&chat::render(ms, if tools { ch.tools.as_ref() } else { None }, ch.effort.as_ref())?, true)) };
    let with = |n: usize, summary: &str| -> Vec<Value> {
        msgs[..lead].iter().cloned().chain(std::iter::once(compact::history_message(summary))).chain(msgs[n..].iter().cloned()).collect()
    };
    let h = compact::prefix_hashes(compact::render_seed(ch.tools.as_ref(), ch.effort.as_ref()), msgs);
    // a summary made at an earlier turn of this chat: the same prompt as then, so the KV cache is reused too
    let cached = cache.iter().filter(|(hh, n, _)| *n > lead && *n < msgs.len() && h[*n] == *hh).max_by_key(|e| e.1).cloned();
    if let Some((_, n, sum)) = &cached {
        let p = render(&with(*n, sum), true)?;
        if fits(p.len()) {
            return Ok(Some((p, 0)));
        }
    }
    // a new one, of the messages after the cached summary's (updating it) up to the ones kept
    let (start, mut prev) = cached.map_or((lead, None), |(_, n, s)| (n, Some(s)));
    let budget = cfg.summary_tokens.max(768);
    let empty = render(&[], false)?.len();
    let sizes: Vec<usize> = msgs.iter().map(|m| render(std::slice::from_ref(m), false).map(|p| p.len().saturating_sub(empty))).collect::<Result<_, _>>()?;
    let roles: Vec<String> = msgs.iter().map(role).collect();
    let roles: Vec<&str> = roles.iter().map(String::as_str).collect();
    let head = render(&msgs[..lead], true)?.len();
    let keep = cfg.keep.min((eng.n_ctx.saturating_sub(cfg.reserve) / 2).saturating_sub(head + budget));
    let Some(cut) = compact::cut_point(&sizes, &roles, start, keep) else { return Ok(None) };
    // the summarized messages in chunks that fit with the request, the previous summary and the new one
    let room = eng.n_ctx.saturating_sub(3 * budget + 1024).max(1024);
    let mut chunks: Vec<Vec<String>> = vec![vec![]];
    let mut used = 0;
    for m in &msgs[start..cut] {
        let mut text = compact::serialize(m);
        let mut n = tok.encode(&text, false).len();
        if n > room {
            // one message larger than a chunk: its start and its end
            let c: Vec<char> = text.chars().collect();
            let k = c.len() * room / n / 2;
            text = format!("{} [...] {}", c[..k].iter().collect::<String>(), c[c.len() - k..].iter().collect::<String>());
            n = room;
        }
        if used + n > room && !chunks.last().unwrap().is_empty() {
            chunks.push(vec![]);
            used = 0;
        }
        chunks.last_mut().unwrap().push(text);
        used += n;
    }
    let none = Value::String("none".into());
    let (nchunks, total) = (chunks.len(), cut - start);
    for c in chunks {
        let req = compact::history_request(&c.join("\n\n"), prev.as_deref(), budget * 3 / 5);
        let ids = tok.encode(&(chat::render(&req, None, Some(&none))? + chat::MESSAGE_MODEL + chat::CONTENT_TEXT), true);
        let n_ids = ids.len();
        let (logits, _) = eng.feed_progress(&ids, &mut |done, _| {
            let _ = job.tx.send(Event::Compaction { stage: "history", done, total: n_ids });
        })?;
        let cap = budget.min(eng.n_ctx.saturating_sub(n_ids + 1));
        let out = generate(eng, tok, logits, cap, &job.params, &mut |_, _| true)?.unwrap_or_default();
        let text = tok.decode(&out, false).trim().to_string();
        if text.is_empty() {
            return Err("the summary of the earlier conversation came out empty".into());
        }
        prev = Some(text);
    }
    let summary = prev.unwrap_or_default();
    eprintln!("history: {} messages summarized in {} chunks into {} chars, {} kept", total, nchunks, summary.len(), msgs.len() - cut);
    cache.push((h[cut], cut, summary.clone()));
    if cache.len() > 16 {
        cache.remove(0);
    }
    Ok(Some((render(&with(cut, &summary), true)?, total)))
}

/// The compactions of one answer.
struct Compactor {
    cfg: compact::Settings,
    /// ask for the summary in a system message rather than a user one (unless IK_COMPACT_ROLE=user)
    system: bool,
    /// where the thinking kept verbatim begins, after the last summary (0 until a compaction of this answer)
    body: usize,
    count: usize,
    /// of which while the prompt was fed (they do not count against cfg.max)
    at_prompt: usize,
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

/// Compacts the turn that fills the context: everything after position `t`, where the turn starts (after the last
/// user message, or the prompt's end): the model's earlier messages of the turn with their thinking, its tool calls
/// and their results, and the thinking under way. Asks the model for a summary on the cache it has, then rebuilds the
/// sequence as the tokens before `t`, and one model message whose thinking is the summary and the end of the turn
/// kept verbatim (tool calls and results in it as they are). `close`: a section is open at the end of the sequence
/// (false when a prompt fed in parts was cut between messages).
fn compact(eng: &mut Engine, tok: &Tokenizer, c: &mut Compactor, t: usize, params: &SamplingParams, show: bool, tx: &Sender<Event>, close: bool) -> Compacted {
    let open = tok.encode(&format!("{}{}", chat::MESSAGE_MODEL, chat::CONTENT_THINKING), true);
    let (head, tail) = (tok.encode(compact::SUMMARY_HEAD, false), tok.encode(compact::SUMMARY_TAIL, false));
    let len = eng.tokens.len();
    if t + open.len() >= len {
        return Compacted::NoRoom;
    }
    let turn = &eng.tokens[t..len];
    // a turn that begins with an earlier summary: it is updated
    let again = turn.starts_with(&[open.as_slice(), head.as_slice()].concat());
    let b = if c.body > t { c.body } else { t + compact::body_start(turn, &open, &head, &tail) };
    let tools = tok.token_id(chat::CONTENT_TOOL_JSON).is_some_and(|j| turn.contains(&j));
    let start = t + open.len();
    // the summary's length: a third of what it replaces, within 768..summary_tokens; the model is told it in words
    let budget = ((len - start) / 3).clamp(768, c.cfg.summary_tokens.max(768));
    let req = tok.encode(&compact::request(again, c.system, budget * 3 / 5, close, tools), true);
    // the rebuilt sequence takes half the context before the reserve at most, so that the next compaction is far
    let target = eng.n_ctx.saturating_sub(c.cfg.reserve) / 2;
    let fixed = start + head.len() + tail.len();
    let cap = budget.min(c.cfg.summary_tokens).min(eng.n_ctx.saturating_sub(len + req.len() + 1)).min(target.saturating_sub(fixed));
    if cap < 256 || len <= b {
        return Compacted::NoRoom;
    }
    let body: Vec<u32> = eng.tokens[b..len].to_vec();
    let base: Vec<u32> = eng.tokens[..t].to_vec();
    let gone = |ev: Event| tx.send(ev).is_err();

    // the summary, on the cache: the section ends, the request follows, and the model answers it in plain text
    if show && gone(Event::Delta(Delta::Reasoning(format!("\n\n{}", compact::SUMMARY_HEAD)))) {
        return Compacted::Gone;
    }
    let logits = match eng.extend(&req, &mut |_| {}) {
        Ok(l) => l,
        Err(e) => return Compacted::Error(e),
    };
    let r = generate(eng, tok, logits, cap, params, &mut |piece, n| {
        (!show || piece.is_empty() || tx.send(Event::Delta(Delta::Reasoning(piece.to_string()))).is_ok())
            && (n % 32 != 0 || tx.send(Event::Compaction { stage: "summary", done: n, total: cap }).is_ok())
    });
    let summary = match r {
        Ok(Some(s)) => s,
        Ok(None) => return Compacted::Gone,
        Err(e) => return Compacted::Error(e),
    };

    // the rebuilt sequence; a summary of almost nothing is left out, and as much of the turn kept as fits
    let pieces: Vec<Vec<u8>> = body.iter().map(|&t| tok.piece_bytes(t, true)).collect();
    let failed = summary.len() < 32;
    let (seq, kept) = if failed {
        compact::rebuild(&open, &[], &[], &[], &body, &pieces, target.saturating_sub(start))
    } else {
        compact::rebuild(&open, &head, &summary, &tail, &body, &pieces, c.cfg.keep.min(target.saturating_sub(fixed + summary.len())))
    };
    let mut left = false;
    // the state at the turn's start: the snapshot is there (the prompt ended there), or the tokens before it are fed
    // again, which puts the snapshot there for the next compaction
    let rebase = if eng.snapshot_pos() == Some(t) { 0 } else { base.len() };
    let total = rebase + seq.len();
    if rebase > 0 {
        if let Err(e) = eng.feed_progress(&base, &mut |pos, _| left |= tx.send(Event::Compaction { stage: "rebuild", done: pos, total }).is_err()) {
            return Compacted::Error(e);
        }
    }
    let p = match eng.rollback(seq.len()) {
        Ok(p) => p,
        Err(e) => return Compacted::Error(e),
    };
    let rebuilt = eng.extend(&seq, &mut |pos| left |= tx.send(Event::Compaction { stage: "rebuild", done: rebase + pos - p, total }).is_err());
    let logits = match rebuilt {
        Ok(l) => l,
        Err(e) => return Compacted::Error(e),
    };
    c.body = p + seq.len() - kept;
    c.count += 1;
    let note = if failed {
        format!("\n[context compacted: no summary came, {} tokens of reasoning dropped, the last {kept} kept]\n\n", len - start - kept)
    } else {
        format!("\n[context compacted: {} tokens of reasoning summarized in {}, the last {kept} kept]\n\n", len - start, summary.len())
    };
    if left || gone(Event::Delta(Delta::Reasoning(note))) {
        return Compacted::Gone;
    }
    eprintln!("compaction {}: {} -> {} tokens ({} summary, {kept} kept, turn at {t}{})", c.count, len, eng.tokens.len(), summary.len(), if rebase > 0 { ", fed again before it" } else { "" });
    Compacted::Done(logits)
}

/// The structure of tokens, for feed_cut.
fn tags(tok: &Tokenizer, ids: &[u32]) -> Vec<compact::Tag> {
    use compact::Tag;
    let id = |s: &str| tok.token_id(s);
    let opens: Vec<u32> = [chat::CONTENT_TEXT, chat::CONTENT_THINKING, chat::CONTENT_TOOL_JSON, chat::CONTENT_XML].iter().filter_map(|s| id(s)).collect();
    let ends: Vec<u32> = [chat::END_MESSAGE, chat::END_SAMPLING].iter().filter_map(|s| id(s)).collect();
    let heads: Vec<u32> = [chat::MESSAGE_MODEL, chat::MESSAGE_USER, chat::MESSAGE_SYSTEM, chat::MESSAGE_TOOL].iter().filter_map(|s| id(s)).collect();
    ids.iter().map(|t| if opens.contains(t) { Tag::Open } else if ends.contains(t) { Tag::End } else if heads.contains(t) { Tag::Header } else { Tag::Text }).collect()
}

/// Feeds a prompt too long for the context whose overflow is in the current turn (it starts at `t`): up to the
/// compaction threshold, cut at a line end or between messages; then the turn is compacted as during generation and
/// the rest of the prompt follows, as many times as it takes. Returns the logits and the tokens reused.
fn feed_turn(eng: &mut Engine, tok: &Tokenizer, c: &mut Compactor, prompt: &[u32], t: usize, params: &SamplingParams, show: bool, tx: &Sender<Event>, progress: &mut dyn FnMut(usize, usize)) -> Result<Option<(Vec<f32>, usize)>, String> {
    let limit = eng.n_ctx.saturating_sub(c.cfg.reserve);
    let tags = tags(tok, prompt);
    let ends_line = |i: usize| tok.piece_bytes(prompt[i], true).last() == Some(&b'\n');
    // on top of the whole cache when the prompt extends it (compacted at its end when the cut is not past it: an agentic
    // client's tool result crossing the threshold), or of the snapshot; else the part before the turn first, so that
    // the snapshot is where the compactions rebuild from
    let plan = compact::feed_plan(&eng.tokens, eng.snapshot_pos(), prompt, &tags, &ends_line, t, limit);
    let (cut, mut close) = (plan.cut, plan.close);
    let (mut logits, reused) = match plan.start {
        compact::Start::Cache if cut > plan.base => eng.feed_progress(&prompt[..cut], progress)?,
        compact::Start::Cache => (vec![], plan.base),
        compact::Start::Snapshot => {
            let p = eng.rollback(prompt.len() - plan.base)?;
            (eng.extend(&prompt[p..cut], &mut |pos| progress(pos, p))?, p)
        }
        compact::Start::Turn => {
            let (l, r) = eng.feed_progress(&prompt[..t], progress)?;
            (if cut > t { eng.extend(&prompt[t..cut], &mut |pos| progress(pos, r))? } else { l }, r)
        }
    };
    eprintln!("prompt over the compaction threshold: {reused} tokens reused (from the {}), {} fed before the compaction at {cut}", match plan.start {
        compact::Start::Cache => "cache",
        compact::Start::Snapshot => "snapshot",
        compact::Start::Turn => "turn's start",
    }, cut - reused.min(cut));
    let mut fed = cut;
    while fed < prompt.len() {
        if c.at_prompt >= 64 {
            return Err("the prompt does not fit in the context even compacted".into());
        }
        let tc = std::time::Instant::now();
        let r = compact(eng, tok, c, t, params, show, tx, close);
        c.secs += tc.elapsed().as_secs_f64();
        c.at_prompt += 1;
        match r {
            Compacted::Done(_) => {}
            Compacted::NoRoom => return Err(format!("the prompt ({} tokens) does not fit in the context, and its turn could not be compacted", prompt.len())),
            Compacted::Gone => return Ok(None),
            Compacted::Error(e) => return Err(e),
        }
        let room = limit.saturating_sub(eng.tokens.len());
        if room < 1024 {
            return Err("no room left after a compaction of the prompt".into());
        }
        // the rest when it fits, else up to the threshold again
        let (next, cl) = if fed + room >= prompt.len() { (prompt.len(), true) } else { compact::feed_cut(&tags, &ends_line, fed, fed + room, 2048) };
        // progress in the prompt's own positions: the compacted sequence is shorter than what was fed
        let before = eng.tokens.len();
        logits = eng.extend(&prompt[fed..next], &mut |pos| progress(fed + pos - before, reused))?;
        (fed, close) = (next, cl);
    }
    Ok(Some((logits, reused)))
}

/// The prompt of a chat request: the sequence a stored reply ended with (an agentic client sending the history
/// again, with the tool results after it) and the new messages rendered after it, or the request as rendered. Also
/// where its turn starts (the end of the last user message), and the prefix hashes of its messages.
fn prepare(tok: &Tokenizer, ch: &ChatIn, rendered: &[u32], seqs: &compact::Sequences) -> Result<(Vec<u32>, Option<usize>, Vec<u64>, bool), String> {
    let (tools, effort) = (ch.tools.as_ref(), ch.effort.as_ref());
    let h = compact::prefix_hashes(compact::render_seed(tools, effort), &ch.messages);
    let roles: Vec<&str> = ch.messages.iter().map(|m| m.get("role").and_then(Value::as_str).unwrap_or("")).collect();
    if let Some((k, s)) = seqs.find(&h, &roles) {
        let mut p = s.tokens.clone();
        p.extend(tok.encode(&chat::render_from(&ch.messages, k, tools, effort)?, true));
        return Ok((p, Some(s.turn), h, true));
    }
    let turn = match roles.iter().rposition(|r| *r == "user") {
        Some(u) => Some(tok.encode(&chat::render(&ch.messages[..=u], tools, effort)?, true).len()),
        None => None,
    };
    Ok((rendered.to_vec(), turn, h, false))
}

/// The scheduler: owns the engine, runs the jobs in order, and keeps the cached sequence for the next prompt.
pub fn scheduler(mut eng: Engine, tok: Arc<Tokenizer>, rx: Receiver<Job>) {
    let end_sampling = tok.token_id(chat::END_SAMPLING);
    let end_message = tok.token_id(chat::END_MESSAGE);
    let eot = tok.token_id("<|endoftext|>");
    // the summary and note requests come as system messages (IK_COMPACT_ROLE=user: as user messages); on two traces
    // of 6k tokens the system role's summaries listed more of the approaches tried, and no contradictions
    let system_role = std::env::var("IK_COMPACT_ROLE").map_or(true, |v| v != "user");
    // what the model writes to close its thinking and begin the answer
    let close: Vec<(u32, &str)> = [chat::END_MESSAGE, chat::MESSAGE_MODEL, chat::CONTENT_TEXT].iter().filter_map(|&s| tok.token_id(s).map(|t| (t, s))).collect();
    let mut summaries: Summaries = vec![];
    let mut seqs = compact::Sequences::default();
    while let Ok(job) = rx.recv() {
        if job.control.stop.load(Ordering::Relaxed) {
            continue; // stopped while it waited
        }
        let mut comp = Compactor { cfg: job.compaction.clone(), system: system_role, body: 0, count: 0, at_prompt: 0, failed: false, secs: 0.0, wait: 0 };
        // notes in place of the thinking: the raw thinking is not sent
        let hide = job.notes.mode == compact::NoteMode::Only;
        // the prompt: the continuation of a stored sequence (a tool result after a reply of this server), or as
        // rendered; and where the current turn starts
        let (prompt, mut turn, hashes, hit) = match &job.chat {
            Some(ch) => match prepare(&tok, ch, &job.prompt, &seqs) {
                Ok((p, t, h, hit)) => (p, t, Some(h), hit),
                Err(e) => {
                    let _ = job.tx.send(Event::Error(e));
                    continue;
                }
            },
            None => (job.prompt.clone(), None, None, true),
        };
        let total = prompt.len();
        let over = comp.cfg.enabled && total + comp.cfg.reserve > eng.n_ctx;
        // the overflow is in the current turn when what comes before it leaves room for a compacted turn
        let in_turn = turn.is_some_and(|t| t + 4096 <= eng.n_ctx.saturating_sub(comp.cfg.reserve) / 2);
        let t0 = std::time::Instant::now();
        let tx = job.tx.clone();
        let progress = job.progress;
        let mut on_progress = |processed: usize, cache: usize| {
            if progress {
                let _ = tx.send(Event::PromptProgress { processed, total, cache, ms: t0.elapsed().as_secs_f64() * 1e3 });
            }
        };
        let (fed, prompt, summarized, hist_s) = if over && in_turn {
            match feed_turn(&mut eng, &tok, &mut comp, &prompt, turn.unwrap(), &job.params, !hide, &job.tx, &mut on_progress) {
                // the client left during a compaction: the cache stays as it is for the next request
                Ok(None) => continue,
                r => (r.map(Option::unwrap), prompt, 0, 0.0),
            }
        } else {
            // a chat longer than the context, before its current turn: its older messages summarized
            let th = std::time::Instant::now();
            let (prompt, summarized) = if over {
                match compact_history(&mut eng, &tok, &job, &mut summaries) {
                    Ok(Some((p, n))) => {
                        turn = None;
                        (p, n)
                    }
                    Ok(None) => (prompt, 0),
                    Err(e) => {
                        let _ = job.tx.send(Event::Error(e));
                        let _ = eng.reset();
                        continue;
                    }
                }
            } else {
                (prompt, 0)
            };
            let hist_s = if summarized > 0 { th.elapsed().as_secs_f64() } else { 0.0 };
            (eng.feed_progress(&prompt, &mut on_progress), prompt, summarized, hist_s)
        };
        let (mut logits, reused) = match fed {
            Ok(r) => r,
            Err(e) => {
                let _ = job.tx.send(Event::Error(e));
                let _ = eng.reset();
                continue;
            }
        };
        eprintln!("prompt: {total} tokens, {reused} from the cache{}{}", if over { ", over the compaction threshold" } else { "" }, if hit { "" } else { " (no stored reply matched)" });
        // where a compaction rebuilds from: the turn's start, or the prompt's end
        let turn_start = turn.unwrap_or(prompt.len()).min(prompt.len());
        // compactions while the prompt was fed count as compaction time
        let prefill_s = t0.elapsed().as_secs_f64() - comp.secs;
        let comp0 = comp.secs;
        if summarized > 0 {
            let _ = job.tx.send(Event::Delta(Delta::Reasoning(format!("[earlier conversation compacted: {summarized} messages summarized]\n\n"))));
        }
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
        let mut notes = Noter { cfg: job.notes.clone(), system: system_role, since: 0, wait: 0, count: 0, secs: 0.0 };
        // the reply's tool calls (name, arguments), to store the sequence by the message a client sends back
        let calls = std::cell::RefCell::new(Vec::<(String, String)>::new());
        let send = |d: Delta, text: &mut String, stop_hit: &mut bool| -> bool {
            if let Delta::ToolCall(name, args) = &d {
                calls.borrow_mut().push((name.clone(), args.clone()));
            }
            if hide && matches!(d, Delta::Reasoning(_)) {
                return true;
            }
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
            // the client is gone or asked to stop (with the thinking hidden, no send would have told)
            if job.control.stop.load(Ordering::Relaxed) {
                finish = "abort";
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
                let can = !comp.failed && comp.count - comp.at_prompt < comp.cfg.max;
                if (full || asked) && can && (line_end || comp.wait >= LINE_WAIT.min(comp.cfg.reserve / 4)) {
                    comp.wait = 0;
                    job.control.compact.store(false, Ordering::Relaxed);
                    let tc = std::time::Instant::now();
                    let r = compact(&mut eng, &tok, &mut comp, turn_start, &job.params, !hide, &job.tx, true);
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
                full = full && (comp.failed || comp.count - comp.at_prompt >= comp.cfg.max);
                // a progress note, at the end of a line (up to 64 tokens later)
                if notes.due() {
                    if line_end || notes.wait >= 64 {
                        match take_note(&mut eng, &tok, &mut notes, &job.params, &job.tx) {
                            Ok(true) => {}
                            Ok(false) => {
                                finish = "abort";
                                break;
                            }
                            Err(e) => {
                                let _ = job.tx.send(Event::Error(e));
                                let _ = eng.reset();
                                finish = "error";
                                break;
                            }
                        }
                    } else {
                        notes.wait += 1;
                    }
                }
            }
            // the thinking is over its budget, or the user ended it: close it as the model would, and let it answer
            let closing = thinking && (full || n_think >= budget || job.control.end_reasoning.load(Ordering::Relaxed));
            if closing && full {
                // the thinking is closed at the reserve, uncompacted (no room for a summary, or max_compactions reached)
                let _ = job.tx.send(Event::Compaction { stage: "closed", done: comp.count, total: comp.cfg.max });
            }
            let t = if closing { None } else { Some(sampler.sample(&logits, &eng.tokens, &banned)) };
            // the thinking ends (closed, or the model ends it): the last note, before the end goes into the cache
            if thinking && notes.last_due() && (closing || t == end_message) {
                match take_note(&mut eng, &tok, &mut notes, &job.params, &job.tx) {
                    Ok(true) => {}
                    Ok(false) => {
                        finish = "abort";
                        break;
                    }
                    Err(e) => {
                        let _ = job.tx.send(Event::Error(e));
                        let _ = eng.reset();
                        finish = "error";
                        break;
                    }
                }
            }
            let Some(t) = t else {
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
            };
            n += 1;
            if parser.thinking() {
                n_think += 1;
                notes.since += 1;
            }
            if Some(t) == end_sampling || Some(t) == eot || tok.eos.contains(&t) {
                // the end of the reply is part of the sequence, so the next turn continues it
                let _ = eng.decode(t);
                finish = "stop";
                break;
            }
            if job.timings {
                let decode_s = t1.elapsed().as_secs_f64() - (comp.secs - comp0) - notes.secs;
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
        // a reply that ended as the model ends one: its sequence, for the client's next request in the same turn
        if let (Some(h), Some(_)) = (&hashes, turn) {
            if finish == "stop" && end_sampling.is_some() && eng.tokens.last() == end_sampling.as_ref() {
                let reply = compact::reply_message(&job.id, &text, &calls.borrow());
                let hash = compact::prefix_hashes(*h.last().unwrap(), std::slice::from_ref(&reply))[1];
                seqs.put(compact::Sequence { hash, tokens: eng.tokens.clone(), turn: turn_start });
            }
        }
        let _ = job.tx.send(Event::Done {
            finish,
            prompt_tokens: total,
            completion_tokens: n,
            reused,
            prefill_s,
            decode_s: t1.elapsed().as_secs_f64() - (comp.secs - comp0) - notes.secs,
            aside: Aside(comp.count + (summarized > 0) as usize, comp.secs + hist_s, notes.count, notes.secs),
        });
    }
}

struct Request {
    method: String,
    path: String,
    /// X-Conversation-Id (the llama.cpp UI's, "<conversation>::<model>"): the stream is a resumable session
    conv: Option<String>,
    body: Vec<u8>,
}

fn read_request(s: &mut BufReader<TcpStream>) -> Option<Request> {
    let mut line = String::new();
    s.read_line(&mut line).ok()?;
    let mut it = line.split_whitespace();
    let method = it.next()?.to_string();
    let path = it.next()?.to_string();
    let mut len = 0usize;
    let mut conv = None;
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
            } else if k.eq_ignore_ascii_case("x-conversation-id") && !v.trim().is_empty() {
                conv = Some(v.trim().to_string());
            }
        }
    }
    if len > 64 << 20 {
        return None;
    }
    let mut body = vec![0u8; len];
    s.read_exact(&mut body).ok()?;
    Some(Request { method, path, conv, body })
}

/// A query parameter, percent-decoded.
fn query(path: &str, key: &str) -> Option<String> {
    let q = path.split_once('?')?.1;
    let v = q.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))?;
    let b = v.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = if b[i] == b'%' { b.get(i + 1..i + 3).and_then(|h| std::str::from_utf8(h).ok()).and_then(|h| u8::from_str_radix(h, 16).ok()) } else { None };
        match hex {
            Some(c) => {
                out.push(c);
                i += 3;
            }
            None => {
                out.push(b[i]);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: GET, POST, DELETE, OPTIONS\r\n";

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

/// The request's sampling settings over the server's (`d`).
fn params_of(r: &Value, d: &SamplingParams) -> SamplingParams {
    let f = |k: &str, dv: f32| r.get(k).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(dv);
    SamplingParams {
        temperature: f("temperature", d.temperature),
        top_k: r.get("top_k").and_then(Value::as_u64).map(|v| v as usize).unwrap_or(d.top_k),
        top_p: f("top_p", d.top_p),
        min_p: f("min_p", d.min_p),
        repeat_penalty: f("repeat_penalty", f("repetition_penalty", d.repeat_penalty)),
        presence_penalty: f("presence_penalty", d.presence_penalty),
        frequency_penalty: f("frequency_penalty", d.frequency_penalty),
        repeat_last_n: r.get("repeat_last_n").and_then(Value::as_u64).map_or(d.repeat_last_n, |v| v as usize),
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
    /// progress notes unless a request says otherwise
    pub notes: compact::Notes,
    /// sampling settings unless a request says otherwise
    pub sampling: SamplingParams,
}

/// The request's `reasoning_summary` (or `reasoning.summary`) and `reasoning_summary_interval` over the server's.
fn notes_of(r: &Value, d: &compact::Notes) -> Result<compact::Notes, String> {
    let mut n = d.clone();
    if let Some(v) = r.get("reasoning_summary").or_else(|| r.pointer("/reasoning/summary")) {
        n.mode = compact::NoteMode::parse(v).ok_or_else(|| format!("reasoning_summary {v}: off, inline or only"))?;
    }
    if let Some(e) = r.get("reasoning_summary_interval").and_then(Value::as_u64) {
        n.every = e as usize;
    }
    Ok(n)
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

/// A stream the llama.cpp UI can come back to (its requests carry X-Conversation-Id), after llama-server: the job runs
/// on when the connection drops or the page reloads, its bytes are kept, `GET /v1/stream?conv_id=K&from=N` replays them
/// from byte N and follows, `DELETE /v1/stream?conv_id=K` stops the job, `POST /v1/streams/lookup` lists them.
struct Session {
    key: String,
    started: u64,
    control: Arc<Control>,
    /// the response body so far, and whether it ended
    log: Mutex<(Vec<u8>, bool)>,
    more: Condvar,
}

/// The sessions, oldest first: the running ones and the last few that ended.
static SESSIONS: LazyLock<Mutex<Vec<Arc<Session>>>> = LazyLock::new(|| Mutex::new(vec![]));
const KEEP_ENDED: usize = 4;

impl Session {
    fn ended(&self) -> bool {
        self.log.lock().map_or(true, |g| g.1)
    }

    fn end(&self) {
        if let Ok(mut g) = self.log.lock() {
            g.1 = true;
        }
        self.more.notify_all();
    }

    fn find(key: &str) -> Option<Arc<Session>> {
        SESSIONS.lock().ok()?.iter().rev().find(|s| s.key == key).cloned()
    }

    /// Adds a session: a job of the same conversation still running is stopped (a new message, or the answer asked
    /// again), and only the last KEEP_ENDED ended sessions stay.
    fn open(s: Arc<Session>) {
        let Ok(mut all) = SESSIONS.lock() else { return };
        for o in all.iter().filter(|o| o.key == s.key) {
            o.control.stop.store(true, Ordering::Relaxed);
        }
        all.retain(|o| o.key != s.key);
        let ended = all.iter().filter(|o| o.ended()).count();
        let mut drop = ended.saturating_sub(KEEP_ENDED);
        all.retain(|o| {
            let gone = drop > 0 && o.ended();
            drop -= gone as usize;
            !gone
        });
        all.push(s);
    }
}

/// The body a session's job writes: kept, and the followers woken.
struct SessionLog<'a>(&'a Session);

impl Write for SessionLog<'_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.log.lock().map_err(|_| std::io::Error::other("poisoned"))?.0.extend_from_slice(b);
        self.0.more.notify_all();
        Ok(b.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Sends a session's body from byte `from`, then what comes, to its end or until the client leaves.
fn follow(s: &mut TcpStream, ss: &Session, mut at: usize) {
    if write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n").is_err() {
        return;
    }
    loop {
        let (bytes, ended) = {
            let Ok(mut g) = ss.log.lock() else { return };
            while g.0.len() <= at && !g.1 {
                g = match ss.more.wait(g) {
                    Ok(g) => g,
                    Err(_) => return,
                };
            }
            at = at.min(g.0.len());
            (g.0[at..].to_vec(), g.1)
        };
        if !bytes.is_empty() && s.write_all(&bytes).and_then(|_| s.flush()).is_err() {
            return;
        }
        at += bytes.len();
        if ended && bytes.is_empty() {
            return;
        }
    }
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
        // llama-server's own tools (--tools): none here
        ("GET", "/tools") => respond(&mut stream, 200, &json!([])),
        ("POST", "/v1/streams/lookup") => {
            let r: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            let keys: Vec<&str> = r.get("conversation_ids").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
            let list: Vec<Value> = SESSIONS.lock().map(|all| {
                all.iter().filter(|s| keys.contains(&s.key.as_str())).map(|s| json!({"conversation_id": s.key, "is_done": s.ended(), "started_at": s.started})).collect()
            }).unwrap_or_default();
            respond(&mut stream, 200, &Value::Array(list))
        }
        ("GET", "/v1/stream") => {
            let from = query(&req.path, "from").and_then(|v| v.parse().ok()).unwrap_or(0);
            match query(&req.path, "conv_id").and_then(|k| Session::find(&k)) {
                Some(ss) => follow(&mut stream, &ss, from),
                None => respond(&mut stream, 404, &json!({"error": {"message": "no such stream"}})),
            }
        }
        ("DELETE", "/v1/stream") => match query(&req.path, "conv_id").and_then(|k| Session::find(&k)) {
            Some(ss) => {
                ss.control.stop.store(true, Ordering::Relaxed);
                respond(&mut stream, 200, &json!({"success": true}))
            }
            None => respond(&mut stream, 404, &json!({"success": false, "error": {"message": "no such stream"}})),
        },
        ("GET", "/v1/models") | ("GET", "/models") => respond(&mut stream, 200, &site.models),
        ("POST", "/v1/chat/completions") | ("POST", "/chat/completions") | ("POST", "/v1/completions") | ("POST", "/completions") => {
            let chat = path.contains("chat");
            let r: Value = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return respond(&mut stream, 400, &json!({"error": {"message": format!("bad JSON: {e}")}})),
            };
            let mut chat_in = None;
            let prompt = if chat {
                let msgs = r.get("messages").and_then(Value::as_array).cloned().unwrap_or_default();
                // thinking switched off (the llama.cpp UI sends enable_thinking: false): the lowest effort
                let off = Value::String("none".into());
                let effort = r.get("reasoning_effort").or_else(|| r.pointer("/chat_template_kwargs/reasoning_effort")).or_else(|| {
                    (r.pointer("/chat_template_kwargs/enable_thinking") == Some(&Value::Bool(false))).then_some(&off)
                });
                let p = match chat::render(&msgs, r.get("tools"), effort) {
                    Ok(p) => p,
                    Err(e) => return respond(&mut stream, 400, &json!({"error": {"message": e}})),
                };
                chat_in = Some(ChatIn { messages: msgs.clone(), tools: r.get("tools").cloned(), effort: effort.cloned() });
                p
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
            let notes = match notes_of(&r, &site.notes) {
                Ok(n) => n,
                Err(e) => return respond(&mut stream, 400, &json!({"error": {"message": e}})),
            };
            let id = format!("{}-{}{:03}", if chat { "chatcmpl" } else { "cmpl" }, now(), NEXT_ID.fetch_add(1, Ordering::Relaxed) % 1000);
            let control = Arc::new(Control::default());
            if let Ok(mut m) = RUNNING.lock() {
                m.insert(id.clone(), control.clone());
            }
            let (tx, rx) = channel();
            let job = Job {
                id: id.clone(),
                prompt: ids,
                chat: chat_in,
                params: params_of(&r, &site.sampling),
                max_tokens,
                thinking_budget,
                stop,
                raw: !chat,
                control: control.clone(),
                compaction: compaction_of(&r, &site.compaction),
                notes,
                progress: flag("return_progress"),
                timings: flag("timings_per_token"),
                tx,
            };
            if jobs.send(job).is_err() {
                return respond(&mut stream, 500, &json!({"error": {"message": "the scheduler is gone"}}));
            }
            if streaming && let Some(key) = req.conv {
                // a session: the job writes to its log on a thread of its own, and this connection follows it
                let ss = Arc::new(Session { key, started: now(), control, log: Mutex::new((vec![], false)), more: Condvar::new() });
                Session::open(ss.clone());
                let (job_ss, model) = (ss.clone(), model.clone());
                std::thread::spawn(move || {
                    let ss = job_ss;
                    if !stream_body(&mut SessionLog(&ss), &rx, &id, &model, chat, &ss.control.stop) {
                        ss.control.stop.store(true, Ordering::Relaxed);
                    }
                    drop(rx); // the scheduler's next send fails: a compaction or a note under way ends too
                    ss.end();
                    if let Ok(mut m) = RUNNING.lock() {
                        m.remove(&id);
                    }
                });
                return follow(&mut stream, &ss, 0);
            }
            if streaming {
                stream_reply(&mut stream, rx, &id, &model, chat, &control);
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

fn usage(p: usize, c: usize, reused: usize, prefill_s: f64, decode_s: f64, aside: Aside) -> Value {
    let Aside(compactions, compact_s, notes, notes_s) = aside;
    let mut u = json!({
        "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
        "prompt_tokens_details": {"cached_tokens": reused},
        "timings": {"prefill_s": prefill_s, "decode_s": decode_s, "decode_tokens_per_s": if decode_s > 0.0 { c as f64 / decode_s } else { 0.0 }},
    });
    if compactions > 0 {
        u["timings"]["compactions"] = json!(compactions);
        u["timings"]["compact_s"] = json!(compact_s);
    }
    if notes > 0 {
        u["timings"]["notes"] = json!(notes);
        u["timings"]["notes_s"] = json!(notes_s);
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

fn stream_reply(s: &mut TcpStream, rx: Receiver<Event>, id: &str, model: &str, chat: bool, control: &Control) {
    let head = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n");
    if head.is_err() || !stream_body(s, &rx, id, model, chat, &control.stop) {
        // the client left: the scheduler stops the job at its next token
        control.stop.store(true, Ordering::Relaxed);
    }
}

/// An SSE comment when nothing else went out for this long (thinking hidden behind notes, a compaction, a long
/// prefill): the llama.cpp UI, when its tab comes back, takes 3 s without a byte for a lost connection.
const KEEPALIVE: std::time::Duration = std::time::Duration::from_secs(1);

/// A writer that knows when it last wrote.
struct Timed<'a> {
    w: &'a mut dyn Write,
    last: std::time::Instant,
}

impl Write for Timed<'_> {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.last = std::time::Instant::now();
        self.w.write(b)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

/// A job's events as server-sent events: true at the end of the reply, false if a write failed, the job was stopped or
/// the scheduler dropped it.
fn stream_body(s: &mut dyn Write, rx: &Receiver<Event>, id: &str, model: &str, chat: bool, stop: &AtomicBool) -> bool {
    let s = &mut Timed { w: s, last: std::time::Instant::now() };
    let put = |s: &mut dyn Write, b: &[u8]| s.write_all(b).and_then(|_| s.flush()).is_ok();
    let send = |s: &mut dyn Write, v: &Value| put(s, format!("data: {v}\n\n").as_bytes());
    if chat && !send(s, &chunk(id, model, chat, json!({"role": "assistant", "content": ""}), None)) {
        return false;
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
    loop {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        // at most KEEPALIVE after the last write, whatever came between
        let wait = KEEPALIVE.saturating_sub(s.last.elapsed()).max(std::time::Duration::from_millis(10));
        let ev = match rx.recv_timeout(wait) {
            Ok(ev) => ev,
            Err(RecvTimeoutError::Timeout) => {
                if s.last.elapsed() >= KEEPALIVE && !put(s, b": keepalive\n\n") {
                    return false;
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => return false,
        };
        // events that write nothing (the timings of hidden thinking tokens) come too often for the timeout
        if s.last.elapsed() >= KEEPALIVE && !put(s, b": keepalive\n\n") {
            return false;
        }
        let ok = match ev {
            Event::PromptProgress { processed, total, cache, ms } => {
                let mut c = chunk(id, model, chat, json!({}), None);
                c["prompt_progress"] = json!({"total": total, "cache": cache, "processed": processed, "time_ms": ms});
                send(s, &c)
            }
            Event::Tick { prompt_n, cache_n, prompt_s, n, decode_s } => {
                tick = Some(timings(prompt_n + cache_n, n, cache_n, prompt_s, decode_s));
                true
            }
            Event::Delta(Delta::Content(c)) => send(s, &with_tick(chunk(id, model, chat, json!({"content": c}), None), &tick)),
            Event::Delta(Delta::Reasoning(r)) => send(s, &with_tick(chunk(id, model, chat, json!({"reasoning_content": r}), None), &tick)),
            Event::Delta(Delta::ToolCall(name, args)) => {
                ntool += 1;
                let c = chunk(id, model, chat, json!({"tool_calls": [{"index": ntool - 1, "id": compact::tool_call_id(id, ntool - 1), "type": "function", "function": {"name": name, "arguments": args}}]}), None);
                send(s, &with_tick(c, &tick))
            }
            Event::Compaction { stage, done, total } => {
                let mut c = chunk(id, model, chat, json!({}), None);
                c["compaction"] = json!({"stage": stage, "done": done, "total": total});
                send(s, &c)
            }
            Event::Note(t) => send(s, &with_tick(chunk(id, model, chat, json!({"reasoning_summary": t}), None), &tick)),
            Event::Done { finish, prompt_tokens, completion_tokens, reused, prefill_s, decode_s, aside } => {
                let finish = if ntool > 0 && finish == "stop" { "tool_calls" } else { finish };
                let mut last = chunk(id, model, chat, json!({}), Some(finish));
                last["usage"] = usage(prompt_tokens, completion_tokens, reused, prefill_s, decode_s, aside);
                last["timings"] = timings(prompt_tokens, completion_tokens, reused, prefill_s, decode_s);
                let Aside(compactions, compact_s, notes, notes_s) = aside;
                if compactions > 0 {
                    last["timings"]["compactions"] = json!(compactions);
                    last["timings"]["compact_ms"] = json!(compact_s * 1e3);
                }
                if notes > 0 {
                    last["timings"]["notes"] = json!(notes);
                    last["timings"]["notes_ms"] = json!(notes_s * 1e3);
                }
                return send(s, &last) && put(s, b"data: [DONE]\n\n");
            }
            Event::Error(e) => {
                return send(s, &json!({"error": {"message": e}}));
            }
        };
        if !ok {
            return false;
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
    let (mut content, mut reasoning, mut summary) = (String::new(), String::new(), vec![]);
    let mut tools = vec![];
    for ev in rx {
        match ev {
            Event::PromptProgress { .. } | Event::Tick { .. } | Event::Compaction { .. } => {}
            Event::Delta(Delta::Content(c)) => content += &c,
            Event::Delta(Delta::Reasoning(r)) => reasoning += &r,
            Event::Note(t) => summary.push(t),
            Event::Delta(Delta::ToolCall(name, args)) => {
                tools.push(json!({"id": compact::tool_call_id(id, tools.len()), "type": "function", "function": {"name": name, "arguments": args}}))
            }
            Event::Done { finish, prompt_tokens, completion_tokens, reused, prefill_s, decode_s, aside } => {
                let finish = if !tools.is_empty() && finish == "stop" { "tool_calls" } else { finish };
                let u = usage(prompt_tokens, completion_tokens, reused, prefill_s, decode_s, aside);
                let body = if chat {
                    let mut msg = json!({"role": "assistant", "content": content});
                    if !reasoning.is_empty() {
                        msg["reasoning_content"] = Value::String(reasoning.clone());
                    }
                    if !summary.is_empty() {
                        msg["reasoning_summary"] = Value::String(summary.join("\n\n"));
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
