//! Decode speed: `ik-bench <model.gguf> [n_gen] [cache_slots] [prompt]`; IK_THREADS sets the expert threads, IK_NO_GRAPH
//! launches each kernel instead of one CUDA graph per token, IK_PREFILL prefills the prompt (else it is decoded).
use ik_engine::engine::{Engine, Options, threads_from_env};
use ik_engine::tokenizer::Tokenizer;

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let n_gen: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(128);
    let slots: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let prompt = a.get(4).cloned().unwrap_or("<|message_user|><|content_text|>Write a detailed technical explanation of how a CPU cache hierarchy works, including L1, L2 and L3 caches, cache lines, associativity, and replacement policies.<|end_message|><|message_model|><|content_text|>".into());
    let t0 = std::time::Instant::now();
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: 8192, cache_slots: slots, threads: threads_from_env(), graph: std::env::var("IK_NO_GRAPH").is_err(), upload_budget: std::env::var("IK_UPLOADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4) })?;
    let tok = Tokenizer::from_gguf(&e.g)?;
    eprintln!("loaded in {:?}: {} MiB weights, {} cache slots, free VRAM {} MiB", t0.elapsed(), e.w.bytes >> 20, e.cache.n_slots(), e.ctx.mem_info()?.0 >> 20);
    let ids = tok.encode(&prompt, true);
    let t1 = std::time::Instant::now();
    let mut logits = vec![];
    let prefill = std::env::var("IK_PREFILL").is_ok();
    if prefill {
        logits = e.feed(&ids)?.0;
    } else {
        for &t in &ids {
            logits = e.decode(t)?;
        }
    }
    eprintln!("prompt: {} tokens in {:?} ({:.1} t/s, {})", ids.len(), t1.elapsed(), ids.len() as f64 / t1.elapsed().as_secs_f64(), if prefill { "prefilled" } else { "one at a time" });
    let mut out = vec![];
    let t2 = std::time::Instant::now();
    let mut lap = std::time::Instant::now();
    let (h0, m0) = { let u = e.pool.usage.lock().unwrap(); (u.hits, u.miss_count) };
    for i in 0..n_gen {
        let next = argmax(&logits);
        out.push(next);
        if tok.eos.contains(&next) {
            break;
        }
        logits = e.decode(next)?;
        if (i + 1) % 32 == 0 {
            let (h, m) = { let u = e.pool.usage.lock().unwrap(); (u.hits, u.miss_count) };
            eprintln!("  {:4} tokens: {:.2} t/s (last 32), hit rate {:.1}%", i + 1, 32.0 / lap.elapsed().as_secs_f64(), 100.0 * (h - h0) as f64 / ((h - h0 + m - m0).max(1)) as f64);
            lap = std::time::Instant::now();
        }
    }
    let dt = t2.elapsed().as_secs_f64();
    let busy = e.pool.busy_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;
    eprintln!("CPU expert time {:.1} ms/token over the run", 1e3 * busy / (ids.len() + out.len()) as f64);
    println!("{}", tok.decode(&out, false));
    {
        // the share of recent uses (decayed counts) the most used experts take: a static cache's best hit rate
        let u = e.pool.usage.lock().unwrap();
        let mut f: Vec<f32> = u.freq.iter().flatten().copied().collect();
        f.sort_by(|a, b| b.total_cmp(a));
        let tot: f32 = f.iter().sum();
        let share = |k: usize| 100.0 * f.iter().take(k).sum::<f32>() / tot.max(1e-9);
        eprintln!("top {} experts take {:.1}% of recent uses (top 1000: {:.1}%, top 1500: {:.1}%, top 2000: {:.1}%)", e.cache.n_slots(), share(e.cache.n_slots()), share(1000), share(1500), share(2000));
    }
    {
        use ik_engine::cpu_pool::DEC_NS;
        let v: Vec<f64> = DEC_NS.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed) as f64).collect();
        let j = v[0].max(1.0);
        eprintln!(
            "CPU decode jobs {:.0} ({:.2} misses each): to epoch {:.1} us, to gate_up done {:.1} us, to down done {:.1} us, to ack {:.1} us; {:.1} us a miss",
            v[0], v[1] / j, v[2] / j / 1e3, v[3] / j / 1e3, v[4] / j / 1e3, v[5] / j / 1e3, v[5] / v[1].max(1.0) / 1e3
        );
    }
    eprintln!("misses {} of which {} repeat an expert missed earlier in the same token ({:.1}%)", e.stats.misses_all, e.stats.misses_repeat, 100.0 * e.stats.misses_repeat as f64 / e.stats.misses_all.max(1) as f64);
    if let Some((v, total)) = e.profile()? {
        eprintln!("profile of the last token (IK_PROF): {total:.0} us between the first and the last stamp");
        for (name, n, us) in v.iter().take(30) {
            eprintln!("  {name:24} {n:5} x {:7.1} us = {us:8.0} us ({:4.1}%)", us / *n as f64, 100.0 * us / total);
        }
    }
    eprintln!("decode: {} tokens in {:.2}s = {:.2} t/s; cache uploads {}; free VRAM at the end {} MiB", out.len(), dt, out.len() as f64 / dt, e.cache.uploads, e.ctx.mem_info()?.0 >> 20);
    Ok(())
}
