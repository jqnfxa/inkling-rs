//! CPU expert kernels on real rows: single-token dots vs the batched kernel, one thread: `ik-cpubench <checkpoint dir>`.
//! Layer 3 (NVFP4) and the first MXFP8 layer, 8 experts each, read once per pass.
use ik_engine::checkpoint::Checkpoint;
use ik_engine::cpu::{dot_row, order_x};
use ik_engine::cpu_batch::dot_rows;
use ik_engine::experts::{Experts, Fmt, Mat};
use ik_engine::hp::Hp;

/// Bytes of a matrix's codes and scales.
fn bytes(m: &Mat) -> usize {
    m.q.len() + m.s.len()
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ck = std::sync::Arc::new(Checkpoint::open(std::path::Path::new(&a[1]))?);
    let hp = Hp::from_config(&ck.config)?;
    let ex = Experts::new(ck.clone(), &hp)?;
    let x: Vec<f32> = (0..hp.n_embd).map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0).collect();
    let layers = [Some(3), ex.fmt.iter().position(|&f| f == Fmt::MxFp8)];
    for layer in layers.into_iter().flatten() {
        // dot_row takes the activations in its order, the batched kernel as they are
        let mut xo = vec![0f32; hp.n_embd];
        order_x(ex.fmt[layer], &x, &mut xo);
        let mut ao = vec![0f32; hp.n_ff_exp];
        order_x(ex.fmt[layer], &x[..hp.n_ff_exp], &mut ao);
        let (xq, aq) = (&x[..], &x[..hp.n_ff_exp]);
        for nt in [1usize, 4, 8] {
            let xs: Vec<&[f32]> = (0..nt).map(|_| xq).collect();
            let as_: Vec<&[f32]> = (0..nt).map(|_| aq).collect();
            let mut out = vec![0f32; 32 * nt];
            let mut s = 0f32;
            let (t0, mut bytes) = (std::time::Instant::now(), 0);
            for e in 0..8 {
                let gu = ex.gate_up(layer, e);
                for r in (0..gu.rows).step_by(if nt == 1 { 1 } else { 32 }) {
                    if nt == 1 { s += dot_row(&gu, r, &xo) } else { dot_rows(&gu, r, 32.min(gu.rows - r), &xs, &mut out); s += out[0] }
                }
                bytes += self::bytes(&gu);
            }
            let dt = t0.elapsed().as_secs_f64();
            let (t1, mut bytes2) = (std::time::Instant::now(), 0);
            for e in 8..16 {
                let dn = ex.down(layer, e);
                for r in (0..dn.rows).step_by(if nt == 1 { 1 } else { 32 }) {
                    if nt == 1 { s += dot_row(&dn, r, &ao) } else { dot_rows(&dn, r, 32.min(dn.rows - r), &as_, &mut out); s += out[0] }
                }
                bytes2 += self::bytes(&dn);
            }
            let dt2 = t1.elapsed().as_secs_f64();
            println!("layer {layer} ({:?}), {nt} tokens: gate_up {:.2} GB/s, down {:.2} GB/s   [{s:.1}]", ex.fmt[layer], bytes as f64 / dt / 1e9, bytes2 as f64 / dt2 / 1e9);
        }
        // compute speed alone: one expert's gate_up (in the caches) again and again for about a second
        {
            let gu = ex.gate_up(layer, 40);
            let mut s = 0f32;
            for r in 0..gu.rows {
                s += dot_row(&gu, r, &xo);
            }
            let (t0, mut bytes) = (std::time::Instant::now(), 0usize);
            while t0.elapsed().as_secs_f64() < 1.0 {
                for r in 0..gu.rows {
                    s += dot_row(&gu, r, &xo);
                }
                bytes += self::bytes(&gu);
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
                    s += dot_row(&gu, r, &xo);
                }
                *pass = bytes(&gu) as f64 / t0.elapsed().as_secs_f64() / 1e9;
            }
            println!("layer {layer} expert {e}: gate_up ({} KiB) from RAM {:.2} GB/s, again (cached) {:.2} GB/s   [{s:.1}]", bytes(&gu) >> 10, v[0], v[1]);
        }
    }
    // prefill's batched kernel, one thread, on distinct tokens as a ubatch has them: an expert's rows in chunks of 32 as
    // the pool takes them; GFLOP/s of the FMAs
    for layer in layers.into_iter().flatten() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % 2000) as f32 / 1000.0 - 1.0
        };
        let xs_all: Vec<Vec<f32>> = (0..32).map(|_| (0..hp.n_embd).map(|_| rnd()).collect()).collect();
        for nt in [4usize, 12, 32] {
            let xs: Vec<&[f32]> = xs_all[..nt].iter().map(|x| &x[..]).collect();
            let axs: Vec<&[f32]> = xs_all[..nt].iter().map(|x| &x[..hp.n_ff_exp]).collect();
            let mut out = vec![0f32; 32 * nt];
            let mut gf = [0f64; 2];
            for (wi, g) in gf.iter_mut().enumerate() {
                let (t0, mut flops, mut s) = (std::time::Instant::now(), 0f64, 0f32);
                for e in 40..48 {
                    let (m, x) = if wi == 0 { (ex.gate_up(layer, e), &xs) } else { (ex.down(layer, e), &axs) };
                    for c in (0..m.rows).step_by(32) {
                        dot_rows(&m, c, (m.rows - c).min(32), x, &mut out);
                        s += out[0];
                    }
                    flops += 2.0 * (m.rows * m.k * nt) as f64;
                }
                *g = flops / t0.elapsed().as_secs_f64() / 1e9;
                std::hint::black_box(s);
            }
            println!("layer {layer} ({:?}), {nt:2} distinct tokens, one thread: gate_up {:.1} GFLOP/s, down {:.1}", ex.fmt[layer], gf[0], gf[1]);
        }
    }
    // the pool's ceiling: n threads, each through its own NVFP4 experts for a second, either the same 2 again and again
    // (from the caches) or a new one each time (from RAM)
    let layer = 3;
    let mut xo = vec![0f32; hp.n_embd];
    order_x(ex.fmt[layer], &x, &mut xo);
    for warm in [true, false] {
        for nt in [1usize, 4, 8, 12] {
            let ex = &ex;
            let xq = &xo[..];
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
                            bytes.fetch_add(self::bytes(&gu) as u64, std::sync::atomic::Ordering::Relaxed);
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
