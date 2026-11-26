//! The CPU dot kernel alone, one thread, on random rows that the caches hold: `ik-cpudot [seconds]`. The speed of its
//! arithmetic, without the checkpoint; GB/s of codes and scales.
use ik_engine::cpu::{dot_row, order_x};
use ik_engine::experts::{Fmt, Mat};

fn main() {
    let secs: f64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for (fmt, rows, k) in [(Fmt::NvFp4, 64, 2304), (Fmt::NvFp4, 128, 1152), (Fmt::MxFp8, 32, 2304), (Fmt::MxFp8, 64, 1152)] {
        let (qb, sb) = fmt.parts(rows, k);
        let q: Vec<u8> = (0..qb).map(|_| if fmt == Fmt::MxFp8 { (rnd() % 0x7E) as u8 | (rnd() as u8 & 0x80) } else { rnd() as u8 }).collect();
        let s: Vec<u8> = (0..sb).map(|_| if fmt == Fmt::MxFp8 { 110 + (rnd() % 12) as u8 } else { 0x20 + (rnd() % 24) as u8 }).collect();
        let m = Mat { fmt, q: &q, s: &s, factor: 1.0, rows, k };
        let x: Vec<f32> = (0..k).map(|_| (rnd() % 2001) as f32 / 1000.0 - 1.0).collect();
        let mut xo = vec![0f32; k];
        order_x(fmt, &x, &mut xo);
        let mut acc = 0f32;
        let (t0, mut passes) = (std::time::Instant::now(), 0usize);
        while t0.elapsed().as_secs_f64() < secs {
            for r in 0..rows {
                acc += dot_row(&m, r, &xo);
            }
            passes += 1;
        }
        let dt = t0.elapsed().as_secs_f64();
        println!("{fmt:?} {rows}x{k}: {:.2} GB/s, {:.1} ns a row   [{acc:.1}]", (passes * (qb + sb)) as f64 / dt / 1e9, 1e9 * dt / (passes * rows) as f64);
    }
}
