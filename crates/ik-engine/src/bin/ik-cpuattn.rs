//! The CPU decode attention alone (cpu_attn::rows) on random f16 rows, without the checkpoint:
//! `ik-cpuattn [rows] [threads] [seconds]`. The threads split the distances 0..rows into ranges, as the CPU team's
//! chunks; GB/s of K and V read, and ns a row on one thread. Few rows (64 a thread) stay in the caches: the speed of the
//! arithmetic. The rows are on huge pages; IK_NO_HUGE=1 puts them on 4 KiB ones.
use ik_engine::cpu_attn::{self, DIM, HEADS, N_REL, Partial, ROW};
use ik_engine::experts::HugeMem;
use std::time::Instant;

fn main() -> Result<(), String> {
    let arg = |i: usize, d: f64| std::env::args().nth(i).and_then(|s| s.parse::<f64>().ok()).unwrap_or(d);
    let (rows, threads, secs) = (arg(1, 65536.0) as usize, arg(2, 8.0) as usize, arg(3, 2.0));
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    // normal f16 of magnitude 2^-4 .. 2^2, either sign
    let mut f16 = || {
        let x = rnd();
        ((x as u16) & 0x8000) | ((11 + (x >> 16) % 6) as u16) << 10 | ((x >> 32) as u16 & 1023)
    };
    let (k, v) = (HugeMem::new(rows * ROW * 2)?, HugeMem::new(rows * ROW * 2)?);
    for m in [&k, &v] {
        // SAFETY: the mapping holds rows * ROW u16, aligned
        let s = unsafe { std::slice::from_raw_parts_mut(m.ptr as *mut u16, rows * ROW) };
        s.iter_mut().for_each(|x| *x = f16());
    }
    let mut f = |n: usize, a: f32| -> Vec<f32> { (0..n).map(|_| ((rnd() % 2001) as f32 / 1000.0 - 1.0) * a).collect() };
    let (q, r, sink) = (f(HEADS * DIM, 0.3), f(HEADS * N_REL, 1.0), f(HEADS, 2.0));
    let extent = 1024;
    let proj: Vec<u16> = f(N_REL * extent, 1.0).iter().map(|x| (x.to_bits() >> 16) as u16).collect();
    let qy = cpu_attn::Query { q: &q, r: &r, sink: &sink, proj: &proj, extent, scale: 1.0 / (DIM as f32).sqrt(), pos: rows - 1 };
    let (kp, vp) = (k.ptr as usize, v.ptr as usize);
    let per = rows.div_ceil(threads);
    let t0 = Instant::now();
    // each thread repeats its range until the time is up: (rows done, a sum to keep the work)
    let (done, l) = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|t| {
                let qy = &qy;
                s.spawn(move || {
                    let (lo, hi) = (t * per, ((t + 1) * per).min(rows));
                    let (mut n, mut l) = (0usize, 0f32);
                    while t0.elapsed().as_secs_f64() < secs {
                        let mut p = [Partial::EMPTY; HEADS];
                        // SAFETY: rows 0..rows of k and v exist
                        unsafe { cpu_attn::rows(qy, kp as *const u16, vp as *const u16, lo, hi, &mut p) };
                        (n, l) = (n + hi - lo, l + p[0].l);
                    }
                    (n, l)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).fold((0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
    });
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{rows} rows, {threads} threads: {:.2} GB/s, {:.0} ns a row on one thread   [{l:.1}]",
        (done * 2 * ROW * 2) as f64 / dt / 1e9,
        1e9 * dt * threads as f64 / done as f64
    );
    Ok(())
}
