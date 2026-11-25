//! CPU expert kernels on real rows: single-token dots vs the batched kernel, one thread: `ik-cpubench <model.gguf>`.
//! Layer 3 (NVFP4) and the first Q8_0 layer, 8 experts each, read once per pass.
use ik_engine::cpu::{Q8_0, dot_row, quantize_q8_0};
use ik_engine::cpu_batch::dot_tokens;
use ik_engine::experts::{Experts, Fmt};
use ik_engine::gguf::Gguf;
use ik_engine::hp::Hp;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let hp = Hp::from_gguf(&g)?;
    let ex = Experts::new(&g, &hp)?;
    let x: Vec<f32> = (0..2304).map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0).collect();
    let (mut xq, mut aq) = (Q8_0::default(), Q8_0::default());
    quantize_q8_0(&x, &mut xq);
    quantize_q8_0(&x[..1152], &mut aq);
    let layers = [Some(3), ex.fmt.iter().position(|&f| f == Fmt::Q8)];
    for layer in layers.into_iter().flatten() {
        for nt in [1usize, 4, 8] {
            let xs: Vec<&Q8_0> = (0..nt).map(|_| &xq).collect();
            let as_: Vec<&Q8_0> = (0..nt).map(|_| &aq).collect();
            let mut out = vec![0f32; nt];
            let mut s = 0f32;
            let (t0, mut bytes) = (std::time::Instant::now(), 0);
            for e in 0..8 {
                let gu = ex.gate_up(layer, e);
                for r in 0..gu.rows {
                    if nt == 1 { s += dot_row(&gu, r, &xq) } else { dot_tokens(&gu, r, &xs, &mut out); s += out[0] }
                }
                bytes += gu.b.len();
            }
            let dt = t0.elapsed().as_secs_f64();
            let (t1, mut bytes2) = (std::time::Instant::now(), 0);
            for e in 8..16 {
                let dn = ex.down(layer, e);
                for r in 0..dn.rows {
                    if nt == 1 { s += dot_row(&dn, r, &aq) } else { dot_tokens(&dn, r, &as_, &mut out); s += out[0] }
                }
                bytes2 += dn.b.len();
            }
            let dt2 = t1.elapsed().as_secs_f64();
            println!("layer {layer} ({:?}), {nt} tokens: gate_up {:.2} GB/s, down {:.2} GB/s   [{s:.1}]", ex.fmt[layer], bytes as f64 / dt / 1e9, bytes2 as f64 / dt2 / 1e9);
        }
        // compute speed alone: one expert's gate_up (in the caches) again and again for about a second
        {
            let gu = ex.gate_up(layer, 40);
            let mut s = 0f32;
            for r in 0..gu.rows {
                s += dot_row(&gu, r, &xq);
            }
            let (t0, mut bytes) = (std::time::Instant::now(), 0usize);
            while t0.elapsed().as_secs_f64() < 1.0 {
                for r in 0..gu.rows {
                    s += dot_row(&gu, r, &xq);
                }
                bytes += gu.b.len();
            }
            println!("layer {layer} ({:?}): gate_up from the caches, 1 s: {:.2} GB/s   [{s:.1}]", ex.fmt[layer], bytes as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
        // one token, the gate_up of an expert read from RAM (first time) and again from the caches (L3 holds it)
        let mut s = 0f32;
        for e in 32..36 {
            let gu = ex.gate_up(layer, e);
            let mut v = [0f64; 2];
            for pass in v.iter_mut() {
                let t0 = std::time::Instant::now();
                for r in 0..gu.rows {
                    s += dot_row(&gu, r, &xq);
                }
                *pass = gu.b.len() as f64 / t0.elapsed().as_secs_f64() / 1e9;
            }
            println!("layer {layer} expert {e}: gate_up ({} KiB) from RAM {:.2} GB/s, again (cached) {:.2} GB/s   [{s:.1}]", gu.b.len() >> 10, v[0], v[1]);
        }
    }
    // the pool's ceiling: n threads, each through its own NVFP4 experts for a second, either the same 2 again and again
    // (from the caches) or a new one each time (from RAM)
    let layer = 3;
    for warm in [true, false] {
        for nt in [1usize, 4, 8, 12] {
            let ex = &ex;
            let xq = &xq;
            let stop = std::sync::atomic::AtomicBool::new(false);
            let bytes = std::sync::atomic::AtomicU64::new(0);
            let t0 = std::time::Instant::now();
            std::thread::scope(|sc| {
                for t in 0..nt {
                    let (stop, bytes) = (&stop, &bytes);
                    sc.spawn(move || {
                        let mut s = 0f32;
                        let mut e = t * 8;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            let gu = ex.gate_up(layer, if warm { t % 64 } else { e % 128 });
                            for r in 0..gu.rows {
                                s += dot_row(&gu, r, xq);
                            }
                            bytes.fetch_add(gu.b.len() as u64, std::sync::atomic::Ordering::Relaxed);
                            e += 1;
                        }
                        std::hint::black_box(s);
                    });
                }
                std::thread::sleep(std::time::Duration::from_secs(1));
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
            });
            let gbs = bytes.load(std::sync::atomic::Ordering::Relaxed) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            println!("{nt:2} threads, {}: {gbs:.1} GB/s in all", if warm { "each its own expert again (caches)" } else { "a new expert each time (RAM)   " });
        }
    }
    Ok(())
}
