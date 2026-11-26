//! Rollback check: prompt A, generate, then prompt A + B through the snapshot, vs A + B from scratch:
//! `ik-rollback <checkpoint dir> [repeats of A's sentence, 30]` (100 makes A longer than the sliding window).
use ik_engine::engine::{Engine, Options, threads_from_env};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: 8192, cache_slots: 0, threads: threads_from_env(), graph: true, upload_budget: 0 })?;
    let tok = e.ck.tokenizer()?;
    let reps: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let pa = tok.encode(&"The history of the bicycle begins in the early nineteenth century. ".repeat(reps), false);
    let pb = tok.encode(" Later designs added pedals, chains and pneumatic tires, which made", false);
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0;
    // A, then 300 generated tokens (past the slack of the ring), then A + B
    let (mut l, _) = e.feed(&pa)?;
    for _ in 0..300 {
        let t = am(&l) as u32;
        l = e.decode(t)?;
    }
    let ab: Vec<u32> = pa.iter().chain(&pb).copied().collect();
    let t0 = std::time::Instant::now();
    let (lr, reused) = e.feed(&ab)?;
    let tr = t0.elapsed();
    e.reset()?;
    let (lf, _) = e.feed(&ab)?;
    let num: f32 = lr.iter().zip(&lf).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let den: f32 = lf.iter().map(|y| y * y).sum::<f32>().sqrt();
    println!("A {} tokens, B {} tokens: rolled back and reused {reused} in {tr:?}; rel diff vs fresh {:.5}, argmax {} vs {}", pa.len(), pb.len(), num / den, am(&lr), am(&lf));
    Ok(())
}
