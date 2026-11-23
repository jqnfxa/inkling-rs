//! Kernel timings on real weights: `ik-kbench <model.gguf>`; IK_CTX (default 8192) bounds the attention lengths timed
use ik_engine::engine::{Engine, Options};
fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: std::env::var("IK_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(8192), cache_slots: 16, threads: vec![None], graph: false, upload_budget: 0 })?;
    for (name, us, bytes) in e.kernel_bench()? {
        println!("{name:32} {us:9.1} us  {:7.1} GB/s", bytes / us / 1e3);
    }
    Ok(())
}
