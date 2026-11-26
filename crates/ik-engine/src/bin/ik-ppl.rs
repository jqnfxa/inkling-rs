//! Perplexity of a text, decoded a token at a time: `ik-ppl <checkpoint dir> <text file> [n_tokens] [skip]`.
//! Scores tokens skip.. of the first n_tokens (the first ones have too little context to say much). The engine's switches
//! (IK_* variables) apply, to compare.
use ik_engine::engine::{Engine, Options, threads_from_env};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let skip: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(64);
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: n + 8, cache_slots: usize::MAX, threads: threads_from_env(), graph: true, upload_budget: 4 })?;
    let tok = e.ck.tokenizer()?;
    let text = std::fs::read_to_string(&a[2]).map_err(|x| x.to_string())?;
    let ids = tok.encode(&text, false);
    let n = n.min(ids.len());
    let t0 = std::time::Instant::now();
    let (mut nll, mut cnt) = (0f64, 0usize);
    for i in 0..n - 1 {
        let l = e.decode(ids[i])?;
        if i + 1 >= skip {
            let m = l.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let lse = m as f64 + l.iter().map(|&v| ((v - m) as f64).exp()).sum::<f64>().ln();
            nll += lse - l[ids[i + 1] as usize] as f64;
            cnt += 1;
        }
    }
    println!("{} tokens scored of {n}: ppl {:.4}, mean nll {:.5} ({:.1}s)", cnt, (nll / cnt as f64).exp(), nll / cnt as f64, t0.elapsed().as_secs_f64());
    Ok(())
}
