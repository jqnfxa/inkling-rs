//! Prefill vs one-token-at-a-time decode on the same text: `ik-pfcheck <checkpoint dir> <text file> [n_tokens]`
use ik_engine::engine::{Engine, Options, threads_from_env};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&a[2]).map_err(|e| e.to_string())?;
    let n: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(300);
    let slots: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: 8192, cache_slots: slots, threads: threads_from_env(), graph: true, upload_budget: 4 })?;
    let tok = e.ck.tokenizer()?;
    let ids: Vec<u32> = tok.encode(&text, false).into_iter().take(n).collect();
    let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    // decode, one at a time
    let t0 = std::time::Instant::now();
    let mut ld = vec![];
    for &t in &ids {
        ld = e.decode(t)?;
    }
    let td = t0.elapsed().as_secs_f64();
    // prefill, in ubatches
    e.reset()?;
    let t1 = std::time::Instant::now();
    let (lp, _) = e.feed(&ids)?;
    let tp = t1.elapsed().as_secs_f64();
    let num: f32 = lp.iter().zip(&ld).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
    let den: f32 = ld.iter().map(|b| b * b).sum::<f32>().sqrt();
    println!("{} tokens: decode {:.2}s ({:.1} t/s), prefill {:.2}s ({:.1} t/s)", ids.len(), td, ids.len() as f64 / td, tp, ids.len() as f64 / tp);
    println!("last logits: rel diff {:.4}, argmax decode {} prefill {}", num / den, am(&ld), am(&lp));
    // a second prefill pass, with a warmer cache
    e.reset()?;
    let t2 = std::time::Instant::now();
    let _ = e.feed(&ids)?;
    let tp2 = t2.elapsed().as_secs_f64();
    println!("prefill again: {:.2}s ({:.1} t/s)", tp2, ids.len() as f64 / tp2);
    let st = &e.stats;
    let ph: Vec<f64> = ik_engine::cpu_pool::PHASE_NS.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9).collect();
    println!("CPU batch phases: gate_up {:.2}s, activations {:.2}s, down {:.2}s", ph[0], ph[1], ph[2]);
    println!(
        "prefill totals (both runs): GPU wait {:.2}s, host {:.2}s, CPU experts {:.2}s; pairs GPU {} ({} grouped) CPU {} streamed {} (of {} experts)",
        st.pf_gpu_s, st.pf_host_s, st.pf_cpu_s, st.pf_gpu_pairs, st.pf_grouped_pairs, st.pf_cpu_pairs - st.pf_stream_pairs, st.pf_stream_pairs, st.pf_stream_experts
    );
    println!(
        "expert stream: {:.2} GB/s over {:.2}s; still running after the CPU team at {} layers, for {:.2}s",
        st.pf_stream_bytes as f64 / st.pf_stream_s.max(1e-9) / 1e9,
        st.pf_stream_s,
        st.pf_stream_late,
        st.pf_stream_late_s
    );
    println!(
        "bounce slots: the host copied {:.2} GB/s over {:.2}s, and waited {:.2}s for slots",
        st.pf_stream_bytes as f64 / st.pf_stream_copy_s.max(1e-9) / 1e9,
        st.pf_stream_copy_s,
        st.pf_stream_wait_s
    );
    Ok(())
}
