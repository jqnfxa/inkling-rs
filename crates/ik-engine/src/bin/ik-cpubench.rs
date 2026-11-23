//! CPU expert kernels on real rows: single-token dots vs the batched kernel, one thread: `ik-cpubench <model.gguf>`
use ik_engine::cpu::{Q8_0, Q8K, dot_q4k, dot_q50, quantize_q8_0, quantize_q8k};
use ik_engine::cpu_batch::{TokQ8K, TokQ8_0, q4k_dot_tokens_packed, q50_dot_tokens_packed};
use ik_engine::experts::Experts;
use ik_engine::gguf::Gguf;
use ik_engine::hp::Hp;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    unsafe { std::env::set_var("IK_NO_HUGE", "1") };
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let hp = Hp::from_gguf(&g)?;
    let ex = Experts::new(&g, &hp)?;
    let x: Vec<f32> = (0..2304).map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0).collect();
    let mut xq = Q8K { d: vec![], qs: vec![], bsums: vec![] };
    quantize_q8k(&x, &mut xq);
    let mut aq = Q8_0 { d: vec![], qs: vec![], sums: vec![] };
    quantize_q8_0(&x[..1152], &mut aq);
    for nt in [1usize, 4, 5, 8] {
        let xs: Vec<TokQ8K> = (0..nt).map(|_| TokQ8K::new(&xq)).collect();
        let as_: Vec<TokQ8_0> = (0..nt).map(|_| TokQ8_0::new(&aq)).collect();
        let mut out = vec![0f32; nt];
        // 8 experts of layer 3, read once each per repetition
        let t0 = std::time::Instant::now();
        let mut s = 0f32;
        let mut bytes = 0usize;
        for e in 0..8 {
            let gu = ex.gate_up(&g, 3, e);
            for r in 0..2304 {
                let row = &gu[r * ex.gu_row..(r + 1) * ex.gu_row];
                if nt == 1 { s += dot_q4k(row, &xq) } else { q4k_dot_tokens_packed(row, &xs, &mut out); s += out[0] }
            }
            bytes += gu.len();
        }
        let dt = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let mut bytes2 = 0usize;
        for e in 8..16 {
            let dn = ex.down(&g, 3, e);
            for r in 0..2304 {
                let row = &dn[r * ex.down_row..(r + 1) * ex.down_row];
                if nt == 1 { s += dot_q50(row, &aq) } else { q50_dot_tokens_packed(row, &as_, &mut out); s += out[0] }
            }
            bytes2 += dn.len();
        }
        let dt2 = t1.elapsed().as_secs_f64();
        println!("{nt} tokens: gate_up {:.2} GB/s ({:.1} us/row), down {:.2} GB/s ({:.1} us/row)   [{s:.1}]", bytes as f64 / dt / 1e9, dt * 1e6 / (8.0 * 2304.0), bytes2 as f64 / dt2 / 1e9, dt2 * 1e6 / (8.0 * 2304.0));
    }
    Ok(())
}
