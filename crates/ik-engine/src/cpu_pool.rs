//! The CPU half of the MoE: a team of threads that serves the jobs the GPU posts in host-mapped memory
//! (see the kernels moe_dispatch and moe_wait), computes the experts that miss the GPU cache, and acknowledges.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;

use crate::cpu::{dot_row, order_x, x_pos};
use crate::cpu_attn::{self, DIM, HEADS, N_REL, Partial};
use crate::cpu_batch::dot_rows;
use crate::cuda::HostBuf;
use crate::experts::Experts;

const JOB_IDS: usize = 2; // u32 index of ids
const JOB_SLOTS: usize = 10;
const JOB_X: usize = 32;
/// The steps of a token's jobs (seq = token * 64 + step + 1): its MoE layers from 0, the attention of its
/// full-attention layers from ATTN_STEP (8 of them).
pub const ATTN_STEP: u32 = 48;
/// floats of an attention job: q [24][128], r [24][16], sink [24]
pub const ATTN_JOB: usize = HEADS * DIM + HEADS * N_REL + HEADS;
/// floats of the CPU's attention partials: [24] of m, l, acc[128]
pub const ATTN_OUT: usize = HEADS * (2 + DIM);

/// What the attention jobs need besides the job: the full-attention caches of each layer slot (host addresses of K
/// and V row 0), the rows in VRAM (the CPU takes those past them), and the bias profiles of each physical layer.
pub struct AttnCtx {
    pub kv: Vec<Option<(usize, usize)>>,
    pub phys: Vec<usize>,
    pub proj: Vec<Vec<u16>>,
    /// the rows in VRAM (the GPU's), which the engine moves between tokens
    pub vram: AtomicUsize,
    pub extent: usize,
    pub scale: f32,
    /// host address of the partials, [ATTN_OUT] f32
    pub out: usize,
}
const ROWS: usize = 32; // rows per chunk of the interleaved split (prefill)
const ACT: usize = 32; // decode: activations per gate_up chunk (its gate and up rows, and one block of the activations)
const DN_ROWS: usize = 32; // decode: down rows per chunk

/// Usage of the experts, decayed, as the CPU sees the jobs: [layer][expert].
pub struct Usage {
    pub freq: Vec<Vec<f32>>,
    pub misses: Vec<(usize, usize)>, // recent misses (layer, expert)
    pub hits: u64,
    pub miss_count: u64,
}

struct Shared {
    stop: AtomicBool,
    epoch: AtomicU32,     // the leader bumps it to start a job on the helpers
    arrive: AtomicU32,  // spin barrier
    phase: AtomicU32,
    next: [AtomicU32; 3], // prefill: the next chunk of the gate_up, activation and down phases
    // decode: the next chunk of each phase, epoch << 32 | total << 16 | taken, and how many chunks are done. A thread
    // reads the job only while it holds a chunk of its epoch, which the epoch cannot end without.
    gu_next: AtomicU64,
    dn_next: AtomicU64,
    gu_done: AtomicU32,
    dn_done: AtomicU32,
    kind: AtomicU64,      // epoch << 2 | KIND_*
    batch_req: AtomicU32,  // the host posts a batch job (prefill): 1 pending, 0 done
    n: usize,
    work: UnsafeCell<Work>,
    attn: std::sync::OnceLock<AttnCtx>,
}

const KIND_DECODE: u64 = 0;
const KIND_BATCH: u64 = 1;
const KIND_ATTN: u64 = 2;

// SAFETY: `work` is written by the leader before it bumps `epoch`, and read by helpers after: a batch job's phases are
// separated by barriers, a decode job's by the done counters (see `take`).
unsafe impl Sync for Shared {}

/// A prefill job: the experts of one layer that missed the GPU cache, each with its (token, pair) list.
#[derive(Default)]
pub struct BatchJob {
    pub layer: usize,
    pub x: Vec<f32>,                             // [token][n]
    pub misses: Vec<(usize, Vec<(usize, usize)>)>, // (expert, [(token, pair)])
    pub out: usize,                              // host address of [pair][n] outputs
}

#[derive(Default)]
struct Work {
    batch: Option<BatchJob>,
    bh: Vec<Vec<f32>>,  // batch: gate_up per CPU pair
    bact: Vec<Vec<f32>>, // batch: activations per CPU pair
    layer: usize,
    misses: Vec<(usize, usize)>, // (k, expert)
    x: Vec<f32>,
    acts: Vec<Vec<f32>>, // activations per miss, written a block per gate_up chunk
    out: usize,    // host address of cpu_out [8][n]
    // an attention job: its query (q, r, sink), slot, position, the distances 0..a_d0 in chunks of a_chunk, and the
    // chunks' partials
    aq: Vec<f32>,
    a_il: usize,
    a_pos: usize,
    a_d0: usize,
    a_chunk: usize,
    a_parts: Vec<[Partial; HEADS]>,
}

/// nanoseconds of the batch phases: gate_up, activations, down
pub static PHASE_NS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

static GU_AT: AtomicU64 = AtomicU64::new(0);
thread_local! {
    static LEADER_T0: std::cell::Cell<std::time::Instant> = std::cell::Cell::new(std::time::Instant::now());
}

/// decode jobs with misses: [jobs, misses, ns from the job to its epoch, to gate_up done, to down done, to the ack]
pub static DEC_NS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

pub struct Pool {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
    pub usage: Arc<Mutex<Usage>>,
    pub busy_ns: Arc<AtomicU64>,
}

struct Ctx {
    ex: Arc<Experts>,
    job: usize, // host addresses
    ack: usize,
    out: usize,
}

/// Spins a while, then yields the CPU: when the machine is oversubscribed, the thread that everyone waits for may be
/// descheduled, and pure spinning would hold its CPU.
#[inline]
fn pause(spins: &mut u32) {
    *spins += 1;
    if *spins < 2000 {
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

fn barrier(sh: &Shared) {
    let phase = sh.phase.load(Ordering::Acquire);
    if sh.arrive.fetch_add(1, Ordering::AcqRel) + 1 == sh.n as u32 {
        sh.arrive.store(0, Ordering::Relaxed);
        sh.phase.store(phase.wrapping_add(1), Ordering::Release);
    } else {
        let mut spins = 0;
        while sh.phase.load(Ordering::Acquire) == phase {
            pause(&mut spins);
        }
    }
}

fn pin(cpu: Option<usize>) {
    if let Some(c) = cpu {
        // SAFETY: plain libc call on the current thread.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(c, &mut set);
            libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

/// The work of thread `t` on a batch job.
fn run_batch(sh: &Shared, cx: &Ctx, t: usize) {
    // SAFETY: see Shared.
    let w = unsafe { &mut *sh.work.get() };
    let job = w.batch.as_ref().unwrap();
    let ex = &cx.ex;
    let n = ex.n_embd;
    let nff = ex.n_ff;
    // CPU pair index of each (miss, token): consecutive per miss
    let mut base = Vec::with_capacity(job.misses.len());
    let mut acc = 0;
    for (_, toks) in &job.misses {
        base.push(acc);
        acc += toks.len();
    }
    let chunks_gu = (2 * nff).div_ceil(ROWS);
    let tp = std::time::Instant::now();
    loop {
        let c = sh.next[0].fetch_add(1, Ordering::Relaxed) as usize;
        if c >= chunks_gu * job.misses.len() {
            break;
        }
        let (m, c) = (c / chunks_gu, c % chunks_gu);
        let (e, toks) = &job.misses[m];
        let gu = ex.gate_up(job.layer, *e);
        let sg = gu.scale();
        let xs: Vec<&[f32]> = toks.iter().map(|&(tk, _)| &job.x[tk * n..(tk + 1) * n]).collect();
        let (r0, nr) = (c * ROWS, ROWS.min(2 * nff - c * ROWS));
        let mut out = vec![0f32; nr * toks.len()];
        dot_rows(&gu, r0, nr, &xs, &mut out);
        for (j, oj) in out.chunks(toks.len()).enumerate() {
            for (i, &v) in oj.iter().enumerate() {
                // SAFETY: row r0 + j of pair base[m] + i is written by the thread that took the chunk
                unsafe { *(w.bh[base[m] + i].as_ptr() as *mut f32).add(r0 + j) = v * sg };
            }
        }
    }
    barrier(sh);
    let tp1 = std::time::Instant::now();
    loop {
        let p = sh.next[1].fetch_add(1, Ordering::Relaxed) as usize;
        if p >= acc {
            break;
        }
        // gate i in row 2i, up i in row 2i + 1
        let h = &w.bh[p];
        // SAFETY: pair p is written by the thread that took it
        let act = unsafe { std::slice::from_raw_parts_mut(w.bact[p].as_ptr() as *mut f32, nff) };
        for (i, a) in act.iter_mut().enumerate() {
            let (gv, uv) = (h[2 * i], h[2 * i + 1]);
            *a = ex.beta * (gv / ex.beta).tanh() / (1.0 + (-gv).exp()) * ex.lb * (uv / ex.lb).tanh();
        }
    }
    barrier(sh);
    let tp2 = std::time::Instant::now();
    let chunks_dn = n.div_ceil(ROWS);
    loop {
        let c = sh.next[2].fetch_add(1, Ordering::Relaxed) as usize;
        if c >= chunks_dn * job.misses.len() {
            break;
        }
        let (m, c) = (c / chunks_dn, c % chunks_dn);
        let (e, toks) = &job.misses[m];
        let dn = ex.down(job.layer, *e);
        let sd = dn.scale();
        let xs: Vec<&[f32]> = (0..toks.len()).map(|i| &w.bact[base[m] + i][..]).collect();
        let (r0, nr) = (c * ROWS, ROWS.min(n - c * ROWS));
        let mut out = vec![0f32; nr * toks.len()];
        dot_rows(&dn, r0, nr, &xs, &mut out);
        for (j, oj) in out.chunks(toks.len()).enumerate() {
            for (&v, &(_, pair)) in oj.iter().zip(toks) {
                // SAFETY: output row r0 + j of `pair` is written by the thread that took the chunk
                unsafe { *(job.out as *mut f32).add(pair * n + r0 + j) = v * sd };
            }
        }
    }
    barrier(sh);
    if t == 0 {
        let tp3 = std::time::Instant::now();
        PHASE_NS[0].fetch_add((tp1 - tp).as_nanos() as u64, Ordering::Relaxed);
        PHASE_NS[1].fetch_add((tp2 - tp1).as_nanos() as u64, Ordering::Relaxed);
        PHASE_NS[2].fetch_add((tp3 - tp2).as_nanos() as u64, Ordering::Relaxed);
    }
}

/// A chunk of a decode phase of epoch `epoch`, from its counter.
enum Take {
    Chunk(usize),
    /// all taken: the total
    Done(u32),
    /// the epoch is over
    Stale,
}

fn take(next: &AtomicU64, epoch: u32) -> Take {
    let mut v = next.load(Ordering::Acquire);
    loop {
        if (v >> 32) as u32 != epoch {
            return Take::Stale;
        }
        let (total, i) = ((v >> 16) & 0xffff, v & 0xffff);
        if i >= total {
            return Take::Done(total as u32);
        }
        match next.compare_exchange_weak(v, v + 1, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Take::Chunk(i as usize),
            Err(x) => v = x,
        }
    }
}

/// Waits for `total` chunks of a phase of `epoch` to be done; false if the epoch ended first (a late helper).
fn wait_done(sh: &Shared, done: &AtomicU32, total: u32, epoch: u32) -> bool {
    let mut spins = 0;
    while done.load(Ordering::Acquire) < total {
        if sh.epoch.load(Ordering::Acquire) != epoch {
            return false;
        }
        pause(&mut spins);
    }
    true
}

/// The work of thread `t` on the job of epoch `epoch`.
fn run(sh: &Shared, cx: &Ctx, t: usize, epoch: u32) {
    let kind = sh.kind.load(Ordering::Acquire);
    if (kind >> 2) as u32 != epoch {
        return; // a later job has started: this thread takes part in it next
    }
    match kind & 3 {
        KIND_BATCH => run_batch(sh, cx, t),
        KIND_ATTN => run_attn(sh, epoch),
        _ => run_decode(sh, cx, t, epoch),
    }
}

/// Chunks of an attention job: the partials of their distances.
fn run_attn(sh: &Shared, epoch: u32) {
    let w = sh.work.get() as *const Work;
    let Some(ac) = sh.attn.get() else { return };
    while let Take::Chunk(c) = take(&sh.gu_next, epoch) {
        // SAFETY: the epoch holds while this chunk is not done; the leader writes the job only between epochs, and
        // chunk c's partials are this thread's alone
        unsafe {
            let w = &*w;
            let (lo, hi) = (c * w.a_chunk, ((c + 1) * w.a_chunk).min(w.a_d0));
            let (k, v) = ac.kv[w.a_il].unwrap();
            let aq = &w.aq;
            let qy = cpu_attn::Query {
                q: &aq[..HEADS * DIM],
                r: &aq[HEADS * DIM..HEADS * (DIM + N_REL)],
                sink: &aq[HEADS * (DIM + N_REL)..ATTN_JOB],
                proj: &ac.proj[ac.phys[w.a_il]],
                extent: ac.extent,
                scale: ac.scale,
                pos: w.a_pos,
            };
            let part = &mut *(w.a_parts.as_ptr().add(c) as *mut [Partial; HEADS]);
            *part = [Partial::EMPTY; HEADS];
            cpu_attn::rows(&qy, k as *const u16, v as *const u16, lo, hi, part);
        }
        sh.gu_done.fetch_add(1, Ordering::Release);
    }
}

/// A decode job: the misses of one token at one layer. Chunks of gate_up rows with their activations (a block of
/// them), then, once all are done, chunks of down rows. No barrier: a thread that the OS put aside only delays the
/// chunk it holds, and one that wakes late finds the chunks taken.
fn run_decode(sh: &Shared, cx: &Ctx, t: usize, epoch: u32) {
    let ex = &cx.ex;
    let (n, nff) = (ex.n_embd, ex.n_ff);
    let w = sh.work.get() as *const Work;
    let gu_total = loop {
        match take(&sh.gu_next, epoch) {
            Take::Stale => return,
            Take::Done(total) => break total,
            Take::Chunk(c) => {
                // SAFETY: the epoch holds while this chunk is not done; the leader writes the job only between epochs
                let w = unsafe { &*w };
                let (m, blk) = (c / (nff / ACT), c % (nff / ACT));
                let gu = ex.gate_up(w.layer, w.misses[m].1);
                let sg = gu.scale();
                let acts = w.acts[m].as_ptr() as *mut f32;
                for i in blk * ACT..(blk + 1) * ACT {
                    // gate i in row 2i, up i in row 2i + 1
                    let gv = dot_row(&gu, 2 * i, &w.x) * sg;
                    let uv = dot_row(&gu, 2 * i + 1, &w.x) * sg;
                    let s = ex.beta * (gv / ex.beta).tanh() / (1.0 + (-gv).exp());
                    // SAFETY: the activations of block `blk` of miss m (ACT a multiple of 16, so their places in the
                    // order of the down rows too) are written by this chunk alone, and read only after all are done
                    unsafe { *acts.add(x_pos(gu.fmt, i)) = s * ex.lb * (uv / ex.lb).tanh() };
                }
                sh.gu_done.fetch_add(1, Ordering::Release);
            }
        }
    };
    if !wait_done(sh, &sh.gu_done, gu_total, epoch) {
        return;
    }
    if t == 0 {
        GU_AT.store(LEADER_T0.with(|c| c.get()).elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
    let dn_total = loop {
        match take(&sh.dn_next, epoch) {
            Take::Stale => return,
            Take::Done(total) => break total,
            Take::Chunk(c) => {
                // SAFETY: as above
                let w = unsafe { &*w };
                let (m, blk) = (c / (n / DN_ROWS), c % (n / DN_ROWS));
                let (k, e) = w.misses[m];
                let dn = ex.down(w.layer, e);
                let sd = dn.scale();
                // SAFETY: rows blk * DN_ROWS.. of miss k's output are written by this chunk alone
                let out = unsafe { std::slice::from_raw_parts_mut((w.out as *mut f32).add(k * n), n) };
                for r in blk * DN_ROWS..(blk + 1) * DN_ROWS {
                    out[r] = dot_row(&dn, r, &w.acts[m]) * sd;
                }
                sh.dn_done.fetch_add(1, Ordering::Release);
            }
        }
    };
    if t == 0 {
        wait_done(sh, &sh.dn_done, dn_total, epoch);
    }
}

impl Pool {
    /// `cpus`: the CPU of each thread (the first is the leader), or None to leave them unpinned.
    pub fn new(ex: Arc<Experts>, job: &HostBuf, ack: &HostBuf, out: &HostBuf, n_layers: usize, n_expert: usize, cpus: Vec<Option<usize>>) -> Pool {
        let n = cpus.len().max(1);
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            epoch: AtomicU32::new(0),
            arrive: AtomicU32::new(0),
            phase: AtomicU32::new(0),
            next: [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)],
            gu_next: AtomicU64::new(0),
            dn_next: AtomicU64::new(0),
            gu_done: AtomicU32::new(0),
            dn_done: AtomicU32::new(0),
            kind: AtomicU64::new(0),
            batch_req: AtomicU32::new(0),
            n,
            work: UnsafeCell::new(Work::default()),
            attn: std::sync::OnceLock::new(),
        });
        let usage = Arc::new(Mutex::new(Usage { freq: vec![vec![0.0; n_expert]; n_layers], misses: vec![], hits: 0, miss_count: 0 }));
        let busy_ns = Arc::new(AtomicU64::new(0));
        let mut threads = vec![];
        for (t, cpu) in cpus.iter().copied().enumerate() {
            let sh = shared.clone();
            let cx = Ctx { ex: ex.clone(), job: job.ptr as usize, ack: ack.ptr as usize, out: out.ptr as usize };
            let usage = usage.clone();
            let busy = busy_ns.clone();
            threads.push(
                std::thread::Builder::new()
                    .name(format!("ik-moe-{t}"))
                    .spawn(move || {
                        pin(cpu);
                        if t == 0 {
                            leader(&sh, &cx, &usage, &busy);
                        } else {
                            helper(&sh, &cx, t);
                        }
                    })
                    .unwrap(),
            );
        }
        Pool { shared, threads, usage, busy_ns }
    }
}

fn helper(sh: &Shared, cx: &Ctx, t: usize) {
    let mut seen = 0u32;
    let mut idle = 0u64;
    loop {
        let g = sh.epoch.load(Ordering::Acquire);
        if g != seen {
            seen = g;
            idle = 0;
            run(sh, cx, t, g);
            continue;
        }
        if sh.stop.load(Ordering::Relaxed) {
            return;
        }
        idle += 1;
        if idle > 400_000 {
            std::thread::sleep(std::time::Duration::from_micros(50));
        } else if idle > 4_000 {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
}

fn leader(sh: &Shared, cx: &Ctx, usage: &Mutex<Usage>, busy: &AtomicU64) {
    let job = cx.job as *const u32;
    // SAFETY: host-mapped page-locked memory that lives as long as the pool; the GPU writes the job, the CPU the ack.
    let seq_p = unsafe { &*(job as *const AtomicU32) };
    let ack = unsafe { &*(cx.ack as *const AtomicU32) };
    let mut last = 0u32;
    let mut idle = 0u64;
    let fake = std::env::var("IK_FAKE_CPU").is_ok(); // diagnostics: acknowledge without computing
    let n = cx.ex.n_embd;
    let nff = cx.ex.n_ff;
    loop {
        if sh.batch_req.load(Ordering::Acquire) == 1 {
            // SAFETY: the host wrote the batch before batch_req; helpers are idle until epoch changes
            let w = unsafe { &mut *sh.work.get() };
            let npairs: usize = w.batch.as_ref().map_or(0, |b| b.misses.iter().map(|m| m.1.len()).sum());
            w.bh.resize_with(npairs.max(w.bh.len()), Vec::new);
            for h in w.bh.iter_mut().take(npairs) {
                h.resize(2 * nff, 0.0);
            }
            w.bact.resize_with(npairs.max(w.bact.len()), Vec::new);
            for a in w.bact.iter_mut().take(npairs) {
                a.resize(nff, 0.0);
            }
            for c in &sh.next {
                c.store(0, Ordering::Relaxed);
            }
            let t0 = std::time::Instant::now();
            let g = sh.epoch.load(Ordering::Relaxed).wrapping_add(1);
            sh.kind.store((g as u64) << 2 | KIND_BATCH, Ordering::Release);
            sh.epoch.store(g, Ordering::Release);
            run_batch(sh, cx, 0);
            busy.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            if let (Ok(mut u), Some(b)) = (usage.lock(), w.batch.as_ref()) {
                for (e, toks) in &b.misses {
                    u.freq[b.layer][*e] += toks.len() as f32;
                    u.misses.push((b.layer, *e));
                }
            }
            w.batch = None;
            sh.batch_req.store(0, Ordering::Release);
            continue;
        }
        let seq = seq_p.load(Ordering::Acquire);
        if seq == 0 || seq == last {
            if sh.stop.load(Ordering::Relaxed) {
                sh.stop.store(true, Ordering::Relaxed);
                return;
            }
            idle += 1;
            if idle > 4_000_000 {
                std::thread::sleep(std::time::Duration::from_micros(20));
            } else if idle > 20_000 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
            continue;
        }
        idle = 0;
        let t0 = std::time::Instant::now();
        LEADER_T0.with(|c| c.set(t0));
        last = seq;
        if (seq - 1) % 64 >= ATTN_STEP {
            attn_job(sh, job);
            ack.store(seq, Ordering::Release);
            ATTN_NS[2].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            continue;
        }
        // SAFETY: the GPU wrote the job before seq, behind a system fence
        let (layer, ids, slots) = unsafe {
            let layer = *job.add(1) as usize;
            let ids: [usize; 8] = std::array::from_fn(|k| std::ptr::read_volatile(job.add(JOB_IDS + k)) as usize);
            let slots: [i32; 8] = std::array::from_fn(|k| std::ptr::read_volatile(job.add(JOB_SLOTS + k)) as i32);
            (layer, ids, slots)
        };
        let n_miss = slots.iter().filter(|&&s| s < 0).count();
        if n_miss > 0 && !fake {
            // SAFETY: no thread reads the job between epochs
            let w = unsafe { &mut *sh.work.get() };
            w.layer = layer;
            w.misses.clear();
            w.misses.extend((0..8).filter(|&k| slots[k] < 0).map(|k| (k, ids[k])));
            // SAFETY: host-mapped memory the GPU wrote before seq
            let x = unsafe { std::slice::from_raw_parts(job.add(JOB_X) as *const f32, n) };
            w.x.resize(n, 0.0);
            order_x(cx.ex.fmt[layer], x, &mut w.x);
            if w.acts.len() < n_miss {
                w.acts.resize_with(n_miss, || vec![0.0; nff]);
            }
            w.out = cx.out;
            let g = sh.epoch.load(Ordering::Relaxed).wrapping_add(1);
            let (gu_total, dn_total) = ((n_miss * nff / ACT) as u64, (n_miss * n / DN_ROWS) as u64);
            sh.gu_done.store(0, Ordering::Relaxed);
            sh.dn_done.store(0, Ordering::Relaxed);
            sh.gu_next.store((g as u64) << 32 | gu_total << 16, Ordering::Release);
            sh.dn_next.store((g as u64) << 32 | dn_total << 16, Ordering::Release);
            sh.kind.store((g as u64) << 2 | KIND_DECODE, Ordering::Release);
            sh.epoch.store(g, Ordering::Release);
            let t1 = t0.elapsed().as_nanos() as u64;
            run_decode(sh, cx, 0, g);
            let t3 = t0.elapsed().as_nanos() as u64;
            DEC_NS[0].fetch_add(1, Ordering::Relaxed);
            DEC_NS[1].fetch_add(n_miss as u64, Ordering::Relaxed);
            DEC_NS[2].fetch_add(t1, Ordering::Relaxed);
            DEC_NS[3].fetch_add(GU_AT.load(Ordering::Relaxed).saturating_sub(0), Ordering::Relaxed);
            DEC_NS[4].fetch_add(t3, Ordering::Relaxed);
        }
        let misses = (0..8).filter(|&k| slots[k] < 0).map(|k| (k, ids[k]));
        ack.store(seq, Ordering::Release);
        if n_miss > 0 && !fake {
            DEC_NS[5].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        busy.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if let Ok(mut u) = usage.lock() {
            for &e in &ids {
                u.freq[layer][e] += 1.0;
            }
            u.hits += (8 - n_miss) as u64;
            u.miss_count += n_miss as u64;
            for (_, e) in misses {
                u.misses.push((layer, e));
            }
        }
    }
}

/// attention jobs: [jobs, of them with rows for the CPU, ns from the job to its ack, rows]
pub static ATTN_NS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The attention job in `job`: the partials of the rows past the VRAM ones, in chunks on the team, merged in order
/// into the output; nothing when the token is not past them (attn_reduce then reads nothing).
fn attn_job(sh: &Shared, job: *const u32) {
    ATTN_NS[0].fetch_add(1, Ordering::Relaxed);
    let Some(ac) = sh.attn.get() else { return };
    // SAFETY: the GPU wrote the job before seq, behind a system fence
    let (il, pos) = unsafe { (std::ptr::read_volatile(job.add(1)) as usize, std::ptr::read_volatile(job.add(2)) as usize) };
    let d0 = (pos + 1).saturating_sub(ac.vram.load(Ordering::Acquire));
    if d0 == 0 || ac.kv.get(il).is_none_or(|kv| kv.is_none()) {
        return;
    }
    ATTN_NS[1].fetch_add(1, Ordering::Relaxed);
    ATTN_NS[3].fetch_add(d0 as u64, Ordering::Relaxed);
    let n_chunks = {
        // SAFETY: no thread reads the job between epochs
        let w = unsafe { &mut *sh.work.get() };
        w.aq.clear();
        // SAFETY: host-mapped memory the GPU wrote before seq
        w.aq.extend_from_slice(unsafe { std::slice::from_raw_parts(job.add(JOB_X) as *const f32, ATTN_JOB) });
        let n_chunks = d0.div_ceil(128).min(4 * sh.n).max(1);
        (w.a_il, w.a_pos, w.a_d0, w.a_chunk) = (il, pos, d0, d0.div_ceil(n_chunks));
        let n_chunks = d0.div_ceil(w.a_chunk);
        if w.a_parts.len() < n_chunks {
            w.a_parts.resize(n_chunks, [Partial::EMPTY; HEADS]);
        }
        n_chunks
    };
    let g = sh.epoch.load(Ordering::Relaxed).wrapping_add(1);
    sh.gu_done.store(0, Ordering::Relaxed);
    sh.gu_next.store((g as u64) << 32 | (n_chunks as u64) << 16, Ordering::Release);
    sh.kind.store((g as u64) << 2 | KIND_ATTN, Ordering::Release);
    sh.epoch.store(g, Ordering::Release);
    run_attn(sh, g);
    wait_done(sh, &sh.gu_done, n_chunks as u32, g);
    // SAFETY: all chunks are done, so no thread writes the work now
    let w = unsafe { &*sh.work.get() };
    let mut all = [Partial::EMPTY; HEADS];
    for p in &w.a_parts[..n_chunks] {
        for (a, b) in all.iter_mut().zip(p) {
            a.merge(b);
        }
    }
    // SAFETY: host-mapped memory of ATTN_OUT floats that the GPU reads after the ack
    let out = unsafe { std::slice::from_raw_parts_mut(ac.out as *mut f32, ATTN_OUT) };
    for (h, p) in all.iter().enumerate() {
        out[h * (2 + DIM)] = p.m;
        out[h * (2 + DIM) + 1] = p.l;
        out[h * (2 + DIM) + 2..(h + 1) * (2 + DIM)].copy_from_slice(&p.acc);
    }
}

impl Pool {
    /// Sets what the attention jobs need; once, before the first.
    pub fn set_attn(&self, ac: AttnCtx) {
        let _ = self.shared.attn.set(ac);
    }

    /// Moves the split of the attention jobs: the rows from `rows` on are the CPU's. No job may be in flight.
    pub fn set_attn_vram(&self, rows: usize) {
        if let Some(ac) = self.shared.attn.get() {
            ac.vram.store(rows, Ordering::Release);
        }
    }

    /// Starts a prefill job on the team; wait_batch returns when it is done.
    pub fn start_batch(&self, job: BatchJob) {
        assert_eq!(self.shared.batch_req.load(Ordering::Acquire), 0, "a batch job is running");
        // SAFETY: the leader reads the job only after batch_req is set, and the previous one is done
        unsafe { (*self.shared.work.get()).batch = Some(job) };
        self.shared.batch_req.store(1, Ordering::Release);
    }

    /// Returns when the job of start_batch is done.
    pub fn wait_batch(&self) {
        let mut spins = 0;
        while self.shared.batch_req.load(Ordering::Acquire) != 0 {
            pause(&mut spins);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
