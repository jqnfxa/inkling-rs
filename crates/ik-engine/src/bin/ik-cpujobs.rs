//! The CPU experts on decode jobs, without the GPU: `ik-cpujobs <checkpoint dir> [tokens] [threads...]`.
//! Posts the jobs of `tokens` tokens (default 8) to the expert team through host memory, as moe_dispatch does: the
//! layers of the decode schedule, 8 random experts each, misses at the rate of the decode runs (5 or 6 of 8). Each
//! `threads` argument is a team, as IK_THREADS takes it ("0,2,4" pins, "8" is 8 unpinned threads); by default a few.
//! Reports the bytes of the missed experts over the time from each job to its ack, as GB/s, and the phases of the jobs.
use std::sync::atomic::{AtomicU32, Ordering};

use ik_engine::checkpoint::Checkpoint;
use ik_engine::cpu_pool::{DEC_NS, Pool};
use ik_engine::cuda::{Ctx, HostBuf};
use ik_engine::experts::{Experts, Fmt};
use ik_engine::hp::Hp;

fn team(s: &str) -> Vec<Option<usize>> {
    if s.contains(',') { s.split(',').filter_map(|c| c.trim().parse().ok()).map(Some).collect() } else { vec![None; s.parse().unwrap_or(8)] }
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let tokens: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let teams: Vec<String> = if a.len() > 3 { a[3..].to_vec() } else { ["8", "0,2,4", "0,2,4,6", "0,2,4,6,8,10", "0,2,4,6,8,10,1,3,5,7,9,11"].map(String::from).to_vec() };
    let _ctx = Ctx::new(0)?;
    let ck = std::sync::Arc::new(Checkpoint::open(std::path::Path::new(&a[1]))?);
    let hp = Hp::from_config(&ck.config)?;
    let ex = std::sync::Arc::new(Experts::new(ck.clone(), &hp)?);
    let (n, ne) = (hp.n_embd, hp.n_expert);
    // the physical layer of each step of a token: the prelude, the loop's passes, the coda
    let mut sched = vec![];
    sched.extend(0..hp.loop_start);
    for _ in 0..hp.n_passes {
        sched.extend(hp.loop_start..hp.loop_end);
    }
    sched.extend(hp.loop_end..hp.n_phys);
    let job = HostBuf::new(4 * (32 + n))?;
    let ack = HostBuf::new(64)?;
    let out = HostBuf::new(4 * 8 * n)?;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for t in &teams {
        let pool = Pool::new(ex.clone(), &job, &ack, &out, hp.n_phys, ne, team(t));
        let p = job.ptr as *mut u32;
        // SAFETY: host-mapped memory the pool reads after seq, as the GPU writes it
        let (seq, ackv) = unsafe { (&*(p as *const AtomicU32), &*(ack.ptr as *const AtomicU32)) };
        let base: Vec<u64> = DEC_NS.iter().map(|v| v.load(Ordering::Relaxed)).collect();
        let (mut bytes, mut misses, mut busy) = (0usize, 0usize, 0f64);
        let mut s = seq.load(Ordering::Relaxed);
        for (j, &layer) in sched.iter().cycle().take(tokens * sched.len()).enumerate() {
            let mut ids = [0u32; 8];
            for k in 0..8 {
                ids[k] = loop {
                    let e = (rnd() % ne as u64) as u32;
                    if !ids[..k].contains(&e) {
                        break e;
                    }
                };
            }
            let n_miss = 5 + (j % 7 < 3) as usize; // 5.43 a job
            let (gu, dn) = match ex.fmt[layer] {
                Fmt::NvFp4 => (ex.gu_bytes[0], ex.down_bytes[0]),
                Fmt::MxFp8 => (ex.gu_bytes[1], ex.down_bytes[1]),
            };
            // SAFETY: the job's words, written before seq
            unsafe {
                *p.add(1) = layer as u32;
                for k in 0..8 {
                    *p.add(2 + k) = ids[k];
                    *(p.add(10 + k) as *mut i32) = if k < n_miss { -1 } else { k as i32 };
                }
                let x = p.add(32) as *mut f32;
                for i in 0..n {
                    *x.add(i) = (rnd() % 2001) as f32 / 1000.0 - 1.0;
                }
            }
            s = s.wrapping_add(1).max(1);
            let t0 = std::time::Instant::now();
            seq.store(s, Ordering::Release);
            while ackv.load(Ordering::Acquire) != s {
                std::hint::spin_loop();
            }
            busy += t0.elapsed().as_secs_f64();
            bytes += n_miss * (gu + dn);
            misses += n_miss;
        }
        drop(pool);
        let d: Vec<f64> = DEC_NS.iter().zip(&base).map(|(v, b)| (v.load(Ordering::Relaxed) - b) as f64).collect();
        let jobs = d[0].max(1.0);
        println!(
            "{t:>26}: {:5.1} GB/s, {:6.1} us a miss; a job: to gate_up done {:6.1} us, to down done {:6.1} us",
            bytes as f64 / busy / 1e9,
            1e6 * busy / misses as f64,
            d[3] / jobs / 1e3,
            d[4] / jobs / 1e3
        );
    }
    Ok(())
}
