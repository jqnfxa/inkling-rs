//! Branch check: a sequence that goes on straight, against the same sequence with a branch of `len` tokens (at most
//! T_MAX) taken at a checkpoint and dropped: `ik-branch <checkpoint dir> [branch tokens, 256]`. The logits after it
//! should be the ones without it.
use ik_engine::engine::{Engine, Options, T_MAX, threads_from_env};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: 8192, cache_slots: 0, threads: threads_from_env(), graph: true, upload_budget: 0 })?;
    let tok = e.ck.tokenizer()?;
    let len: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(T_MAX).min(T_MAX);
    // a prompt longer than the sliding window, so that the rings have wrapped
    let pa = tok.encode(&"The history of the bicycle begins in the early nineteenth century. ".repeat(100), false);
    let branch = tok.encode(&"Pause, and write a note about something else entirely: the weather, the sea, the mountains. ".repeat(20), false);
    let am = |v: &[f32]| v.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).unwrap().0 as u32;
    let run = |e: &mut Engine, forced: Option<&[u32]>, with_branch: bool| -> Result<(Vec<u32>, Vec<Vec<f32>>), String> {
        e.reset()?;
        let (mut l, _) = e.feed(&pa)?;
        let mut toks = vec![];
        let mut logits = vec![];
        for i in 0..114 {
            if i == 50 && with_branch {
                e.checkpoint()?;
                // a prefill of most of the branch, then decoded tokens to its length
                let pre = len.saturating_sub(20).max(1);
                let mut bl = e.extend(&branch[..pre], &mut |_| {})?;
                for _ in pre..len {
                    bl = e.decode(am(&bl))?;
                }
                e.back_to_checkpoint()?;
            }
            let t = forced.map_or_else(|| am(&l), |f| f[i]);
            toks.push(t);
            l = e.decode(t)?;
            if i >= 50 {
                logits.push(l.clone());
            }
        }
        Ok((toks, logits))
    };
    let (toks, straight) = run(&mut e, None, false)?;
    let (_, branched) = run(&mut e, Some(&toks), true)?;
    let mut worst = 0f32;
    let mut same = 0;
    for (x, y) in branched.iter().zip(&straight) {
        let num: f32 = x.iter().zip(y).map(|(p, q)| (p - q) * (p - q)).sum::<f32>().sqrt();
        let den: f32 = y.iter().map(|q| q * q).sum::<f32>().sqrt();
        worst = worst.max(num / den);
        same += (am(x) == am(y)) as usize;
    }
    println!("prompt {} tokens, branch of {len} at position {}: over the next {} tokens, worst rel diff {worst:.6}, argmax same {same}/{}", pa.len(), pa.len() + 50, straight.len(), straight.len());
    Ok(())
}
