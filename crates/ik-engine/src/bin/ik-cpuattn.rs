//! The CPU decode attention alone (cpu_attn::rows) on random f16 rows, without the checkpoint:
//! `ik-cpuattn [rows] [threads] [seconds] [jobs]`. The threads split the distances 0..rows into ranges, as the CPU team's
//! chunks; GB/s of K and V read, and ns a row on one thread. Few rows (64 a thread) stay in the caches: the speed of the
//! arithmetic. The rows are on huge pages; IK_NO_HUGE=1 puts them on 4 KiB ones, IK_VMM=1 on CUDA's host pages as the
//! engine's host KV rows are.
//!
//! With `jobs`, the threads run jobs as the engine's attention jobs: each over all the rows of one of 8 layers in turn
//! (each layer its own rows, as in the engine, so that they come from DRAM), in chunks the threads take as they come
//! (IK_CHUNKS of them, default 4 a thread, at least 128 rows each), the partials merged in order once all are done; the
//! time a job is from its start to the merge's end.
use ik_engine::cpu_attn::{self, DIM, HEADS, N_REL, Partial, ROW};
use ik_engine::cuda::{Ctx, Stream, VmBuf};
use ik_engine::experts::HugeMem;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::time::Instant;

/// The layers of jobs, each with its own rows.
const LAYERS: usize = 8;

/// K or V: on huge pages, or on CUDA's host pages past one page of VRAM, as the engine maps them.
enum Mem {
    Huge(HugeMem),
    Vmm(VmBuf, usize),
}

impl Mem {
    fn ptr(&self) -> *mut u16 {
        match self {
            Mem::Huge(m) => m.ptr as *mut u16,
            Mem::Vmm(m, off) => (m.ptr as usize + off) as *mut u16,
        }
    }
}

fn main() -> Result<(), String> {
    let arg = |i: usize, d: f64| std::env::args().nth(i).and_then(|s| s.parse::<f64>().ok()).unwrap_or(d);
    let (rows, threads, secs) = (arg(1, 65536.0) as usize, arg(2, 8.0) as usize, arg(3, 2.0));
    let jobs = std::env::args().nth(4).is_some_and(|s| s == "jobs");
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
    let vmm = std::env::var("IK_VMM").is_ok_and(|v| v == "1");
    let ctx = if vmm { Some(Ctx::new(0)?) } else { None };
    let all = if jobs { LAYERS * rows } else { rows };
    let alloc = || -> Result<Mem, String> {
        match &ctx {
            Some(c) => {
                let s = Stream::new()?;
                let page = VmBuf::granularity(c.device)?;
                let mut m = VmBuf::reserve(page + all * ROW * 2, c.device)?;
                m.set_host_from(page);
                m.resize(page + all * ROW * 2, &s)?;
                s.sync()?;
                Ok(Mem::Vmm(m, page))
            }
            None => Ok(Mem::Huge(HugeMem::new(all * ROW * 2)?)),
        }
    };
    let (k, v) = (alloc()?, alloc()?);
    for m in [&k, &v] {
        // SAFETY: the mapping holds `all` rows of ROW u16, aligned, and the host can read and write it
        let s = unsafe { std::slice::from_raw_parts_mut(m.ptr(), all * ROW) };
        s.iter_mut().for_each(|x| *x = f16());
    }
    let mut f = |n: usize, a: f32| -> Vec<f32> { (0..n).map(|_| ((rnd() % 2001) as f32 / 1000.0 - 1.0) * a).collect() };
    let (q, r, sink) = (f(HEADS * DIM, 0.3), f(HEADS * N_REL, 1.0), f(HEADS, 2.0));
    let extent = 1024;
    let proj: Vec<u16> = f(N_REL * extent, 1.0).iter().map(|x| (x.to_bits() >> 16) as u16).collect();
    let qy = cpu_attn::Query { q: &q, r: &r, sink: &sink, proj: &proj, extent, scale: 1.0 / (DIM as f32).sqrt(), pos: rows - 1 };
    let (kp, vp) = (k.ptr() as usize, v.ptr() as usize);
    if jobs {
        return run_jobs(&qy, kp, vp, rows, threads, secs);
    }
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

/// Jobs as the engine's: the main thread starts each and works on it, helpers spin between jobs.
fn run_jobs(qy: &cpu_attn::Query, kp: usize, vp: usize, rows: usize, threads: usize, secs: f64) -> Result<(), String> {
    let want = std::env::var("IK_CHUNKS").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(4 * threads);
    let chunk = rows.div_ceil(rows.div_ceil(128).min(want).max(1));
    let n_chunks = rows.div_ceil(chunk);
    let parts: Vec<Mutex<[Partial; HEADS]>> = (0..n_chunks).map(|_| Mutex::new([Partial::EMPTY; HEADS])).collect();
    let (epoch, next, done, stop) = (AtomicU32::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicBool::new(false));
    let work = |g: u32| {
        // the layer of job g: its rows from row `base` on
        let base = g as usize % LAYERS * rows * ROW;
        loop {
            let c = next.fetch_add(1, Ordering::AcqRel);
            if c >= n_chunks {
                break;
            }
            let (lo, hi) = (c * chunk, ((c + 1) * chunk).min(rows));
            let mut p = [Partial::EMPTY; HEADS];
            // SAFETY: rows 0..LAYERS rows of k and v exist
            unsafe { cpu_attn::rows(qy, (kp as *const u16).add(base), (vp as *const u16).add(base), lo, hi, &mut p) };
            *parts[c].lock().unwrap() = p;
            done.fetch_add(1, Ordering::AcqRel);
        }
    };
    let (mut n_jobs, mut merge_s, mut l) = (0usize, 0f64, 0f32);
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 1..threads {
            s.spawn(|| {
                let mut seen = 0;
                while !stop.load(Ordering::Acquire) {
                    let g = epoch.load(Ordering::Acquire);
                    if g == seen {
                        std::hint::spin_loop();
                        continue;
                    }
                    seen = g;
                    work(g);
                }
            });
        }
        while t0.elapsed().as_secs_f64() < secs {
            next.store(0, Ordering::Release);
            done.store(0, Ordering::Release);
            let g = epoch.fetch_add(1, Ordering::AcqRel) + 1;
            work(g);
            while done.load(Ordering::Acquire) < n_chunks {
                std::hint::spin_loop();
            }
            let tm = Instant::now();
            let mut all = [Partial::EMPTY; HEADS];
            for p in &parts {
                for (a, b) in all.iter_mut().zip(p.lock().unwrap().iter()) {
                    a.merge(b);
                }
            }
            merge_s += tm.elapsed().as_secs_f64();
            (n_jobs, l) = (n_jobs + 1, l + all[0].l);
        }
        stop.store(true, Ordering::Release);
    });
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{rows} rows, {threads} threads, jobs of {n_chunks} chunks of {chunk} rows: {:.2} GB/s, {:.1} us a job ({:.1} us of it merging)   [{l:.1}]",
        (n_jobs * rows * 2 * ROW * 2) as f64 / dt / 1e9,
        1e6 * dt / n_jobs as f64,
        1e6 * merge_s / n_jobs as f64
    );
    Ok(())
}
