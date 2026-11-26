//! The Inkling-2-Mini server: `inkling-server <checkpoint dir> [--host 127.0.0.1] [--port 8080] [--ctx 16384] [--slots N] [--ui DIR]
//! [--compact-reserve min(16384, ctx/8)] [--compact-keep 20000] [--compact-max 8] [--no-compact]
//! [--reasoning-summary off|inline|only] [--summary-every 1024]
//! [--temp 1.0] [--top-k 40] [--top-p 0.95] [--min-p 0.05] [--repeat-penalty 1.0] [--presence-penalty 0.0]
//! [--frequency-penalty 0.0] [--repeat-last-n 64]`
//! The sampling flags set the defaults; a request's own temperature, top_k, ... take precedence.
//! OpenAI-compatible: /v1/chat/completions, /v1/completions, /v1/models, /health; /props as llama-server answers it, and
//! with --ui the files of a browser UI at / (the llama.cpp chat UI: tools/fetch-ui.sh).
#![recursion_limit = "256"]

use std::sync::Arc;

use ik_engine::engine::{Engine, Options, threads_from_env};
use ik_engine::server;

fn arg(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let model = a.get(1).ok_or("usage: inkling-server <checkpoint dir> [--host H] [--port P] [--ctx N] [--slots N]")?.clone();
    let host = arg(&a, "--host").unwrap_or("127.0.0.1".into());
    let port = arg(&a, "--port").unwrap_or("8080".into());
    let n_ctx: usize = arg(&a, "--ctx").and_then(|v| v.parse().ok()).unwrap_or(16384);
    let slots: usize = arg(&a, "--slots").and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let ui = arg(&a, "--ui").map(std::path::PathBuf::from);
    let num = |k: &str, d: usize| -> Result<usize, String> { arg(&a, k).map_or(Ok(d), |v| v.parse().map_err(|_| format!("{k} {v}: not a number"))) };
    let d = ik_engine::compact::Settings::default();
    let compaction = ik_engine::compact::Settings {
        enabled: !a.iter().any(|x| x == "--no-compact"),
        reserve: num("--compact-reserve", d.reserve.min(n_ctx / 8))?,
        keep: num("--compact-keep", d.keep)?,
        max: num("--compact-max", d.max)?,
        summary_tokens: d.summary_tokens,
    };
    let mut notes = ik_engine::compact::Notes { every: num("--summary-every", 1024)?, ..Default::default() };
    if let Some(m) = arg(&a, "--reasoning-summary") {
        notes.mode = ik_engine::compact::NoteMode::parse(&serde_json::Value::String(m.clone())).ok_or(format!("--reasoning-summary {m}: off, inline or only"))?;
    }
    let s = ik_engine::sampler::SamplingParams::default();
    let real = |k: &str, d: f32| -> Result<f32, String> { arg(&a, k).map_or(Ok(d), |v| v.parse().map_err(|_| format!("{k} {v}: not a number"))) };
    let sampling = ik_engine::sampler::SamplingParams {
        temperature: real("--temp", s.temperature)?,
        top_k: num("--top-k", s.top_k)?,
        top_p: real("--top-p", s.top_p)?,
        min_p: real("--min-p", s.min_p)?,
        repeat_penalty: real("--repeat-penalty", s.repeat_penalty)?,
        presence_penalty: real("--presence-penalty", s.presence_penalty)?,
        frequency_penalty: real("--frequency-penalty", s.frequency_penalty)?,
        repeat_last_n: num("--repeat-last-n", s.repeat_last_n)?,
        seed: 0,
    };
    if compaction.enabled && compaction.reserve >= n_ctx / 2 {
        return Err(format!("--compact-reserve {} must be under half of --ctx {n_ctx}", compaction.reserve));
    }
    if let Some(d) = &ui
        && !d.join("index.html").is_file()
    {
        return Err(format!("--ui {}: no index.html there", d.display()));
    }
    let t0 = std::time::Instant::now();
    let eng = Engine::new(std::path::Path::new(&model), Options { n_ctx, cache_slots: slots, threads: threads_from_env(), graph: true, upload_budget: 4 })?;
    let tok = Arc::new(eng.ck.tokenizer()?);
    let name = eng.ck.name();
    eprintln!("{name}: loaded in {:.1}s on {}, context {n_ctx}, {} expert cache slots (fewer as the context fills)", t0.elapsed().as_secs_f64(), eng.ctx.name, eng.cache.n_slots());
    eprintln!("sampling unless a request says otherwise: temp {} top-k {} top-p {} min-p {} repeat-penalty {} presence-penalty {} frequency-penalty {} (last {})", sampling.temperature, sampling.top_k, sampling.top_p, sampling.min_p, sampling.repeat_penalty, sampling.presence_penalty, sampling.frequency_penalty, sampling.repeat_last_n);
    let site = site(&eng, &model, &name, n_ctx, ui.clone(), compaction, notes, sampling);
    let (tx, rx) = std::sync::mpsc::channel();
    let addr = format!("{host}:{port}");
    let tok2 = tok.clone();
    std::thread::Builder::new()
        .name("ik-http".into())
        .spawn(move || {
            if let Err(e) = server::serve(&addr, tx, tok2, name, site) {
                eprintln!("server: {e}");
                std::process::exit(1);
            }
        })
        .map_err(|e| e.to_string())?;
    eprintln!("listening on http://{host}:{port}{}", if ui.is_some() { " (open it in a browser to chat)" } else { "" });
    // the scheduler runs on this thread, where the CUDA context is current
    server::scheduler(eng, tok, rx);
    Ok(())
}

/// /props and /v1/models in the form llama-server gives them, for the llama.cpp chat UI.
fn site(eng: &Engine, path: &str, name: &str, n_ctx: usize, ui: Option<std::path::PathBuf>, compaction: ik_engine::compact::Settings, notes: ik_engine::compact::Notes, sampling: ik_engine::sampler::SamplingParams) -> server::Site {
    use serde_json::json;
    let d = &sampling;
    let template = eng.ck.text("chat_template.jinja").unwrap_or_default();
    let props = json!({
        "default_generation_settings": {
            "n_ctx": n_ctx,
            "params": {
                "seed": 4294967295u32, "temperature": d.temperature, "dynatemp_range": 0.0, "dynatemp_exponent": 1.0,
                "top_k": d.top_k, "top_p": d.top_p, "min_p": d.min_p, "top_n_sigma": -1.0, "xtc_probability": 0.0,
                "xtc_threshold": 0.1, "typical_p": 1.0, "repeat_last_n": d.repeat_last_n, "repeat_penalty": d.repeat_penalty,
                "presence_penalty": d.presence_penalty, "frequency_penalty": d.frequency_penalty, "dry_multiplier": 0.0,
                "dry_base": 1.75, "dry_allowed_length": 2, "dry_penalty_last_n": 64, "mirostat": 0, "mirostat_tau": 5.0,
                "mirostat_eta": 0.1, "max_tokens": -1, "n_predict": -1, "n_keep": 0, "n_discard": 0, "ignore_eos": false,
                "stream": false, "n_probs": 0, "min_keep": 0, "chat_format": "Content-only", "reasoning_format": "auto",
                "reasoning_in_content": false, "generation_prompt": "",
                "samplers": ["penalties", "top_k", "top_p", "min_p", "temperature"], "timings_per_token": false,
                "post_sampling_probs": false, "backend_sampling": false, "lora": []
            }
        },
        "total_slots": 1,
        "model_alias": name,
        "model_path": path,
        "modalities": {"vision": false, "video": false, "audio": false},
        "endpoint_slots": false,
        "endpoint_props": false,
        "endpoint_metrics": false,
        "ui": ui.is_some(),
        "ui_settings": {},
        "chat_template": template,
        "chat_template_caps": {
            "supports_object_arguments": true, "supports_parallel_tool_calls": true, "supports_preserve_reasoning": true,
            "supports_reasoning_effort": true, "supports_string_content": true, "supports_system_role": true,
            "supports_tool_calls": true, "supports_tools": true, "supports_typed_content": true,
            "reasoning_effort_levels": ["none", "minimal", "low", "medium", "high", "max"], "reasoning_effort_rejected": []
        },
        "bos_token": "",
        "eos_token": "<|content_model_end_sampling|>",
        "build_info": format!("inkling-rs {}", env!("CARGO_PKG_VERSION")),
        "is_sleeping": false,
        "cors_proxy_enabled": false
    });
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_secs()).unwrap_or(0);
    let models = json!({
        "object": "list",
        "models": [{"name": name, "model": name, "type": "model", "capabilities": ["completion"], "details": {"format": "safetensors"}}],
        "data": [{
            "id": name, "aliases": [name], "tags": [], "object": "model", "owned_by": "inkling-rs", "created": created,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "meta": {"n_ctx": n_ctx, "n_ctx_train": 1048576, "n_vocab": eng.hp.n_vocab, "n_embd": eng.hp.n_embd}
        }]
    });
    server::Site { props, models, ui, compaction, notes, sampling }
}
