//! Speed against context depth: `ik-ctx <checkpoint dir> <text file> <n_ctx> [n_gen] [max depth]`.
//! Feeds the text in growing prefixes (1k, 2k, 4k, ... tokens, up to n_ctx) through the batched prefill and, at
//! each depth, decodes n_gen tokens greedily; reports prefill speed of each new segment, decode speed and the
//! expert-cache hit rate. IK_THREADS sets the expert threads. IK_PROF adds, at each depth, the GPU time of each kind
//! of kernel in the last token, and the CPU expert time and misses per token.
use ik_engine::engine::{Engine, Options, threads_from_env};
use std::time::Instant;

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 {
        return Err("usage: ik-ctx <checkpoint dir> <text file> <n_ctx> [n_gen]".into());
    }
    let n_ctx: usize = a[3].parse().map_err(|_| "bad n_ctx")?;
    let n_gen: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let max_depth: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);
    let t0 = Instant::now();
    let opt = Options { n_ctx, cache_slots: usize::MAX, threads: threads_from_env(), graph: true, upload_budget: 4 };
    let mut e = Engine::new(std::path::Path::new(&a[1]), opt)?;
    let tok = e.ck.tokenizer()?;
    let (free, total) = e.ctx.mem_info()?;
    eprintln!(
        "n_ctx {n_ctx}: loaded in {:.1?}, {} expert cache slots ({} MiB), {} MiB VRAM free of {}",
        t0.elapsed(),
        e.cache.n_slots(),
        e.cache.bytes_for([e.cache.banks[0].n_slots, e.cache.banks[1].n_slots]) >> 20,
        free >> 20,
        total >> 20
    );
    let text = std::fs::read_to_string(&a[2]).map_err(|x| x.to_string())?;
    let mut ids = tok.encode("<|message_user|><|content_text|>", true);
    ids.extend(tok.encode(&text, false));
    let mut depths = vec![];
    let mut d = 1024;
    while d + n_gen < n_ctx {
        depths.push(d);
        d *= 2;
    }
    depths.push(n_ctx - n_gen - 1);
    depths.retain(|&d| d <= max_depth);
    if ids.len() < *depths.last().unwrap() {
        return Err(format!("text has only {} tokens", ids.len()));
    }
    println!("{:>7} {:>10} {:>12} {:>11} {:>9}", "depth", "prefill/s", "prefill tot", "decode t/s", "hit rate");
    let mut prev = 0;
    let mut total_pf = 0.0;
    for &d in &depths {
        let t = Instant::now();
        let (mut logits, reused) = e.feed(&ids[..d])?;
        let dt = t.elapsed().as_secs_f64();
        total_pf += dt;
        assert_eq!(reused, prev, "prefix was not reused");
        let (h0, m0) = {
            let u = e.pool.usage.lock().unwrap();
            (u.hits, u.miss_count)
        };
        // the first decoded token warms up; time the rest
        logits = e.decode(argmax(&logits))?;
        let busy0 = e.pool.busy_ns.load(std::sync::atomic::Ordering::Relaxed);
        let (h1, m1) = {
            let u = e.pool.usage.lock().unwrap();
            (u.hits, u.miss_count)
        };
        let t = Instant::now();
        for _ in 1..n_gen {
            logits = e.decode(argmax(&logits))?;
        }
        let dec = (n_gen - 1) as f64 / t.elapsed().as_secs_f64();
        let (h, m) = {
            let u = e.pool.usage.lock().unwrap();
            (u.hits, u.miss_count)
        };
        let hit = 100.0 * (h - h0) as f64 / ((h - h0 + m - m0).max(1)) as f64;
        println!("{d:>7} {:>10.1} {:>11.1}s {dec:>11.2} {hit:>8.1}%", (d - prev) as f64 / dt, total_pf);
        if let Some((v, total)) = e.profile()? {
            let busy = (e.pool.busy_ns.load(std::sync::atomic::Ordering::Relaxed) - busy0) as f64 / 1e6 / (n_gen - 1) as f64;
            let misses = (m - m1) as f64 / (n_gen - 1) as f64;
            let uses = (h - h1 + m - m1) as f64 / (n_gen - 1) as f64;
            println!("        last token {total:.0} us on the GPU timeline; CPU experts {busy:.2} ms/token (all threads), {misses:.1} misses of {uses:.0} uses a token; {} slots", e.cache.n_slots());
            for (name, n, us) in v.iter().take(10) {
                println!("          {name:22} {n:5} x {:7.1} us = {us:7.0} us ({:4.1}%)", us / *n as f64, 100.0 * us / total);
            }
        }
        prev = d;
    }
    Ok(())
}
