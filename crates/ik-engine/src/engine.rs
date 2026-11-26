//! One sequence of Inkling-2-Mini from its safetensors checkpoint: weights, caches and the decode step.
//! The GPU posts each MoE layer to the CPU team (cpu_pool) through host-mapped memory, so a token runs as one CUDA graph.

use std::collections::HashMap;
use std::sync::Arc;

use crate::checkpoint::{Checkpoint, Dtype};
use crate::cpu_pool::{ATTN_JOB, ATTN_OUT, ATTN_STEP, AttnCtx, Pool};
use crate::cuda::{self, Ctx, DevBuf, Graph, HostBuf, Result, Stream, VmBuf};
use crate::experts::{ExpertCache, Experts, Fmt};
use crate::hp::{Hp, Step};
use crate::kernels::{Kernels, P};
use crate::weights::{Mat, Weights};

/// The expert threads from IK_THREADS ("0,2,4" pins, "8" is 8 unpinned threads); default: 8 unpinned.
pub fn threads_from_env() -> Vec<Option<usize>> {
    match std::env::var("IK_THREADS") {
        Ok(v) if v.contains(',') => v.split(',').filter_map(|c| c.trim().parse().ok()).map(Some).collect(),
        Ok(v) => vec![None; v.trim().parse().unwrap_or(8)],
        Err(_) => vec![None; 8],
    }
}

/// The GPU computes the cached experts (else all experts run on the CPU, and the cache has no slots).
const GPU_EXPERTS: bool = true;

/// The row of `Scratch::msgs` that holds the message of controller c (q, k, v, the MoE input): the loop's cross-pass
/// queries give the q and MoE messages (rows 0, 1), its kv queries the k and v ones (rows 2, 3).
const MSG: [usize; 4] = [0, 2, 3, 1];

/// Tokens of a prefill ubatch, at most.
pub const T_MAX: usize = 256;

/// The expert usage counts decay by this much at each token: an expert that every token uses counts about 100.
const DECAY: f32 = 0.99;

/// Bytes of a row (position) of a K or a V cache: 8 kv heads of 128 f16, the f32 that K and V come to rounded once.
const KV_ROW: usize = 2048;

/// The full-attention KV caches get memory in steps of this many rows (a 2 MiB page).
const KV_STEP: usize = 1024;
/// host memory left free when the KV cache spills there
const HOST_MARGIN: usize = 1 << 30;
/// Prefill reads the host rows of the KV cache from copies in VRAM, this many rows at a time.
const STAGE_ROWS: usize = 4096;
/// floats of attn_prefill's state between passes, per token and kv head: m[3], l[3], acc[3][128]
const PF_STATE: usize = 390;
/// Prefill runs the pairs of cached experts on the GPU this many tokens at a time.
const PF_GROUP: usize = 32;
/// A cached NVFP4 expert with at least this many of a ubatch's tokens runs them grouped (moe_nvfp4_g, a tile of 32
/// tokens decoding each weight once), the others pair by pair (moe_nvfp4_d, the weights read a pair at a time).
const G_MIN: usize = 12;
/// Chunks of grouped pairs in a ubatch, at most: they fill the pf buffers' 8 PF_GROUP rows.
const G_CHUNKS: usize = T_MAX / PF_GROUP;

/// Bytes of BatchBufs::grp: for nt tokens, the tokens at 0, the pair by pair slots at 32 nt, chunk c's rows at
/// 32 nt (2 + c), and the entries after the last chunk's rows.
fn grp_bytes(hp: &Hp) -> usize {
    32 * T_MAX * (2 + G_CHUNKS) + 64 * (hp.n_expert + G_CHUNKS)
}
/// Staging slots of prefill's expert stream: the copy stream fills one while the GPU runs the pairs of another.
const STREAM_SLOTS: usize = 4;
/// From this many tokens on, prefill's dense matrices run as gemm_bf16 (W read once for 128 tokens) and not as
/// gemv_bf16 (W read once a token).
const GEMM_MIN: usize = 8;

/// Decode attention: keys per chunk of attn_part, log2, of the sliding-window layers (16 chunks of the window) and of
/// the full ones.
const CS_SWA: u32 = 6;
const CS_FULL: u32 = 8;

pub struct Options {
    pub n_ctx: usize,
    pub cache_slots: usize,
    /// CPU of each expert thread (the first leads), None = unpinned
    pub threads: Vec<Option<usize>>,
    /// capture the decode step as a CUDA graph
    pub graph: bool,
    /// experts uploaded to the GPU cache per token at most, once it is full
    pub upload_budget: usize,
}

/// Device scratch of one token.
struct Scratch {
    x: DevBuf,       // the residual stream, cur
    inp_sa: DevBuf,  // layer input, when it is not a block
    mixed: DevBuf,   // [3][n]
    inputs: DevBuf,  // [3][n] attention inputs after control and norm
    ctl_tmp: DevBuf, // [3][n]
    ctl_g: DevBuf,   // [2n]
    ctl_gate: DevBuf,
    ctl_g3: DevBuf,    // the inputs of 3 controllers [3][2n]
    ctl_gate3: DevBuf, // [3][64]
    msgs: DevBuf,      // [4][n] (MSG)
    g0: DevBuf,      // q | gate | rel | sink
    gk: DevBuf,      // k | k taps
    gv: DevBuf,      // v | v taps
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    r: DevBuf,
    att: DevBuf,
    att2: DevBuf,
    a: DevBuf,
    a2: DevBuf,
    ffm: DevBuf,
    ffm2: DevBuf,
    xn: DevBuf,
    ids: DevBuf,
    wts: DevBuf,
    slots: DevBuf,
    gu: DevBuf,  // [8][2 n_ff]
    dn: DevBuf,  // [8][n]
    f: DevBuf,
    logits_raw: DevBuf,
    af: DevBuf, // the expert activations [8][n_ff]
    rlog: DevBuf, // router logits [n_expert]
    pm: DevBuf,   // attention partials
    pl: DevBuf,
    pacc: DevBuf,
}

/// Device buffers of a prefill ubatch of up to T_MAX tokens, [t][..] each.
struct BatchBufs {
    x: DevBuf,
    inp_sa: DevBuf,
    mix: Vec<DevBuf>,
    msg: Vec<DevBuf>,
    ctl: Vec<DevBuf>,
    ctl_g: DevBuf,
    ctl_gate: DevBuf,
    inn: Vec<DevBuf>,
    g0: DevBuf,
    gk: DevBuf,
    gv: DevBuf,
    kc: DevBuf,
    vc: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    r: DevBuf,
    att: DevBuf,
    att2: DevBuf,
    a: DevBuf,
    a2: DevBuf,
    ffm: DevBuf,
    ffm2: DevBuf,
    xn: DevBuf,
    rlog: DevBuf,
    ids: DevBuf,
    wts: DevBuf,
    slots: DevBuf,
    moe: DevBuf,
    f: DevBuf,
    tau: DevBuf,
    one: DevBuf,
    bl: Blocks,
    host_ids: HostBuf,
    host_xn: HostBuf,
    host_slots: HostBuf,
    cpu_out: HostBuf,
    /// the GPU's pairs of PF_GROUP tokens: gate_up [8 PF_GROUP][2 n_ff], activations [8 PF_GROUP][n_ff], down
    /// [8 PF_GROUP][n]
    pf_gu: DevBuf,
    pf_af: DevBuf,
    pf_dn: DevBuf,
    /// the lists of the grouped cached experts (experts_g): the tokens of the grouped pairs [8 T_MAX] u32, the slots
    /// left to the pair by pair path [8 T_MAX] i32, each chunk's rows of the pairs [G_CHUNKS][8 T_MAX] i32, and the
    /// moe_nvfp4_g entries of gate_up, then of down, [n_expert + G_CHUNKS][4] u64 each
    grp: DevBuf,
    grp_h: HostBuf,
}

/// Prefill's stream of uncached experts: the copy stream uploads each to a staging slot and the GPU runs its pairs,
/// beside the CPU team, which takes the other uncached experts.
struct PfStream {
    /// the staging slots of gate_up and of down, sized for the larger format
    gu: DevBuf,
    dn: DevBuf,
    /// an expert's pairs: x [T_MAX][n] (gathered, then the down outputs), gate_up [T_MAX][2 n_ff], activations
    /// [T_MAX][n_ff]
    x: DevBuf,
    gu_out: DevBuf,
    af: DevBuf,
    /// the streamed pairs in expert order, [3][8 T_MAX] u32: staging slot, token, pair (8 t + k); and the host copy
    /// they are written to
    lists: DevBuf,
    lists_h: HostBuf,
    /// moe_nvfp4_g's entries of each streamed expert, [2 n_expert][4] u64: its gate_up, then its down; and the host
    /// copy they are written to
    tab: DevBuf,
    tab_h: HostBuf,
    /// the expert stream, at the highest priority: the cached experts' kernels on the main stream, thousands of blocks
    /// each, would hold its small ones back, and its uploads with them
    s: Stream,
    /// per staging slot: its upload is done; its last reader is done
    up: Vec<cuda::Event>,
    free: Vec<cuda::Event>,
    /// the copy stream at a layer's first upload, and the stream past its last pair, for the throughput
    t0: cuda::Event,
    t1: cuda::Event,
    /// bytes a second through the stream, and the CPU team's seconds a pair, as measured (moving averages)
    bw: f64,
    cpu_pair_s: f64,
    /// bytes of the layer in flight, measured once its events are done
    last_bytes: usize,
    /// page-locked slots that the host copies each expert to for its upload ([STREAM_SLOTS] of gate_up then down),
    /// and per slot the copy stream past its last upload from it
    bounce: HostBuf,
    bounced: Vec<cuda::Event>,
}

impl PfStream {
    fn new(ex: &Experts, hp: &Hp) -> Result<PfStream> {
        let (n, nff, cap) = (hp.n_embd, hp.n_ff_exp, 8 * T_MAX);
        let events = || (0..STREAM_SLOTS).map(|_| cuda::Event::new(false)).collect::<Result<Vec<_>>>();
        let slot = ex.gu_bytes.iter().max().unwrap() + ex.down_bytes.iter().max().unwrap();
        Ok(PfStream {
            gu: DevBuf::new(STREAM_SLOTS * ex.gu_bytes.iter().max().unwrap())?,
            dn: DevBuf::new(STREAM_SLOTS * ex.down_bytes.iter().max().unwrap())?,
            x: DevBuf::new(T_MAX * n * 4)?,
            gu_out: DevBuf::new(T_MAX * 2 * nff * 4)?,
            af: DevBuf::new(T_MAX * nff * 4)?,
            lists: DevBuf::new(3 * cap * 4)?,
            lists_h: HostBuf::new(3 * cap * 4)?,
            tab: DevBuf::new(2 * hp.n_expert * 32)?,
            tab_h: HostBuf::new(2 * hp.n_expert * 32)?,
            s: Stream::new_nonblocking_high()?,
            up: events()?,
            free: events()?,
            t0: cuda::Event::new(true)?,
            t1: cuda::Event::new(true)?,
            bw: 6.0e9,
            cpu_pair_s: 55e-6,
            last_bytes: 0,
            bounce: HostBuf::new(STREAM_SLOTS * slot)?,
            bounced: events()?,
        })
    }
}

/// Storage of the depth blocks and loop banks.
struct Blocks {
    b0: DevBuf,
    b6: Vec<DevBuf>,
    b12: Vec<DevBuf>,
    b18: DevBuf,
    prelude: DevBuf,
    pass_out: Vec<DevBuf>,
}

pub struct Engine {
    /// first, so that it stops before the host buffers it reads are freed
    pub pool: Pool,
    pub ck: Arc<Checkpoint>,
    pub hp: Hp,
    pub ctx: Ctx,
    pub k: Kernels,
    pub w: Weights,
    pub ex: Arc<Experts>,
    pub cache: ExpertCache,
    pub s: Stream,
    pub n_ctx: usize,
    sched: Vec<Step>,
    /// the sliding-window rings of K and of V: f16 rows [rows][8][128]
    kcache: Vec<Option<DevBuf>>,
    vcache: Vec<Option<DevBuf>>,
    /// the full-attention caches, K and V in f16 rows, at addresses for n_ctx rows; memory for kv_mapped
    kv_vm: Vec<Option<[VmBuf; 2]>>,
    kv_mapped: usize,
    /// the rows of kv_vm in VRAM; the rows past them are host memory
    kv_vram_rows: usize,
    /// whether the caches have rows past kv_vram_rows: then decode posts attention jobs to the CPU team for them
    kv_cpu: bool,
    /// the order of each full-attention slot among them: its attention job's step is ATTN_STEP + this
    attn_idx: Vec<u32>,
    /// the CPU team's partials of the host rows, [24] of m, l, acc[128]
    attn_out: HostBuf,
    /// when there are host rows: the VRAM copies of a chunk of them (K, V) that prefill reads, and its state between
    /// passes [T_MAX][8][PF_STATE]
    stage: Option<[DevBuf; 3]>,
    /// VRAM for the expert cache and the mapped rows of kv_vm together
    vram_budget: usize,
    ring: Vec<u32>,
    conv: Vec<DevBuf>, // per slot: k [kv][3], v [kv][3], post [n][3], ffn [n][3]
    sc: Scratch,
    bl: Blocks,
    params: DevBuf,  // pos u32, tau f32, one f32
    ptrs: DevBuf,    // pointer lists of the mix calls
    ptr_lists: HashMap<Vec<u64>, usize>,
    ptr_used: usize,
    pass_bank: (Vec<P>, P),
    /// debug: called with (name, slot, data) for the named intermediates (Qcur, kqv_out, ffn_inp, l_out, ...), if set
    pub probe: Option<Box<dyn FnMut(&str, usize, &[f32])>>,
    job: HostBuf,      // the MoE job the GPU posts: seq, layer, ids, slots, x
    ack: HostBuf,      // the seq the CPU acknowledged
    err: HostBuf,      // the seq of a wait that timed out
    cpu_out: HostBuf,  // [8][n] the experts computed on the CPU
    logits_h: HostBuf, // the logits, written by the GPU
    map: DevBuf,       // [n_phys][n_expert] slot of each expert, -1 if not cached
    graph: Option<Graph>,
    graph_on: bool,
    eager_done: bool,
    tok: u32,
    upload_budget: usize,
    pub pos: usize,
    pub stats: Stats,
    maxs: usize,
    /// the tokens of the sequence in the caches
    pub tokens: Vec<u32>,
    b: BatchBufs,
    pass_bank_t: (Vec<P>, P),
    /// the state at the end of the last prompt, to roll back to: conv states, and the window rows of the SWA caches
    snap_pos: Option<usize>,
    snap_conv: DevBuf,
    snap_swa: HostBuf,
    /// a checkpoint to come back to after a short branch: its position and conv states
    ckpt_pos: Option<usize>,
    ckpt_conv: DevBuf,
    copy_s: Stream,       // async cache uploads
    /// prefill's expert stream (IK_PF_STREAM=0: none)
    pf_stream: Option<PfStream>,
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub tokens: u64,
    /// decode: misses of experts, and those of an expert that had missed earlier in the same token
    pub misses_all: usize,
    pub misses_repeat: usize,
    /// prefill: seconds waiting for the GPU up to the router, grouping on the host, in the CPU experts
    pub pf_gpu_s: f64,
    pub pf_host_s: f64,
    pub pf_cpu_s: f64,
    pub pf_cpu_pairs: u64,
    pub pf_gpu_pairs: u64,
    /// prefill: the GPU's pairs that ran grouped (experts_g)
    pub pf_grouped_pairs: u64,
    /// prefill: pairs and experts of the expert stream (counted in pf_cpu_pairs too: they are not cached), its bytes
    /// and seconds, and the layers where it was still running when the CPU team was done, and the seconds it ran on
    pub pf_stream_pairs: u64,
    pub pf_stream_experts: u64,
    pub pf_stream_bytes: u64,
    pub pf_stream_s: f64,
    pub pf_stream_late: u64,
    pub pf_stream_late_s: f64,
    /// prefill: the host's seconds copying the streamed experts to the bounce slots, and waiting for a slot's last
    /// upload to be done
    pub pf_stream_copy_s: f64,
    pub pf_stream_wait_s: f64,
}

/// Host memory that can still be taken: MemAvailable, and within the memory limit of this process's cgroup if it has
/// one (memory.max less what the cgroup holds other than file pages).
fn host_mem_free() -> usize {
    let num = |v: &str| v.trim().trim_end_matches("kB").trim().parse::<usize>().ok();
    let mut free = std::fs::read_to_string("/proc/meminfo").ok().and_then(|m| m.lines().find_map(|l| l.strip_prefix("MemAvailable:")).and_then(num)).map_or(usize::MAX, |k| k << 10);
    if let Some(path) = std::fs::read_to_string("/proc/self/cgroup").ok().and_then(|c| c.lines().find_map(|l| l.strip_prefix("0::").map(|p| p.trim().to_string()))) {
        let rd = |f: &str| std::fs::read_to_string(format!("/sys/fs/cgroup{path}/{f}")).ok();
        if let (Some(max), Some(cur)) = (rd("memory.max").as_deref().and_then(num), rd("memory.current").as_deref().and_then(num)) {
            let file = rd("memory.stat").and_then(|s| s.lines().find_map(|l| l.strip_prefix("file ")).and_then(num)).unwrap_or(0);
            free = free.min(max.saturating_sub(cur.saturating_sub(file)));
        }
    }
    free
}

impl Engine {
    pub fn new(path: &std::path::Path, opt: Options) -> Result<Engine> {
        let ctx = Ctx::new(0)?;
        let ck = Arc::new(Checkpoint::open(path)?);
        let hp = Hp::from_config(&ck.config)?;
        let k = Kernels::load()?;
        let w = Weights::load(&ck, &hp)?;
        let ex = Arc::new(Experts::new(ck.clone(), &hp)?);
        let n_swa = hp.n_swa;
        let s = Stream::new()?;
        let n = hp.n_embd;
        let kv = hp.kv_dim();
        let f = |len: usize| DevBuf::zeroed(len * 4, &s);

        let n_slots = hp.n_slots();
        let mut kcache = vec![];
        let mut vcache = vec![];
        let mut kv_vm = vec![];
        let mut ring = vec![];
        let mut conv = vec![];
        for il in 0..n_slots {
            let ip = Self::phys(&hp, il);
            let rows = if hp.is_swa[ip] { hp.n_swa + T_MAX } else { opt.n_ctx };
            if hp.is_swa[ip] {
                kcache.push(Some(DevBuf::zeroed(rows * KV_ROW, &s)?));
                vcache.push(Some(DevBuf::zeroed(rows * KV_ROW, &s)?));
                kv_vm.push(None);
            } else {
                // memory comes as the context grows (plan_kv)
                let r = || VmBuf::reserve(rows.div_ceil(KV_STEP) * KV_STEP * KV_ROW, ctx.device);
                kcache.push(None);
                vcache.push(None);
                kv_vm.push(Some([r()?, r()?]));
            }
            ring.push(if hp.is_swa[ip] { (hp.n_swa + T_MAX) as u32 } else { 0 });
            conv.push(f((2 * kv + 2 * n) * 3)?);
        }
        // the GPU's chunks of full attention cover the rows in VRAM, which are fewer than VRAM holds
        let full = (0..n_slots).filter(|&il| !hp.is_swa[Self::phys(&hp, il)]).count().max(1);
        let vram_rows_max = ctx.mem_info()?.1 / (full * 2 * KV_ROW);
        let maxs = opt.n_ctx.min(vram_rows_max).div_ceil(1 << CS_FULL).max(hp.n_swa.div_ceil(1 << CS_SWA));
        let sc = Scratch {
            x: f(n)?,
            inp_sa: f(n)?,
            mixed: f(3 * n)?,
            inputs: f(3 * n)?,
            ctl_tmp: f(3 * n)?,
            ctl_g: f(2 * n)?,
            ctl_gate: f(64)?,
            ctl_g3: f(3 * 2 * n)?,
            ctl_gate3: f(3 * hp.gate_rank)?,
            msgs: f(4 * n)?,
            g0: f(w.layers[0].g0_rows())?,
            gk: f(kv + 64)?,
            gv: f(kv + 64)?,
            q: f(hp.n_head * hp.head_dim)?,
            k: f(kv)?,
            v: f(kv)?,
            r: f(hp.n_head * hp.n_rel)?,
            att: f(hp.n_head * hp.head_dim)?,
            att2: f(hp.n_head * hp.head_dim)?,
            a: f(n)?,
            a2: f(n)?,
            ffm: f(n)?,
            ffm2: f(n)?,
            xn: f(n)?,
            ids: f(8)?,
            wts: f(8)?,
            slots: f(8)?,
            gu: f(8 * 2 * hp.n_ff_exp)?,
            dn: f(8 * n)?,
            f: f(n)?,
            logits_raw: f(hp.n_vocab_padded)?,
            af: f(8 * hp.n_ff_exp)?,
            rlog: f(hp.n_expert)?,
            pm: f(8 * maxs * 3)?,
            pl: f(8 * maxs * 3)?,
            pacc: f(8 * maxs * 3 * 128)?,
        };
        let bl = Blocks {
            b0: f(n)?,
            b6: (0..hp.n_passes).map(|_| f(n)).collect::<Result<_>>()?,
            b12: (0..hp.n_passes).map(|_| f(n)).collect::<Result<_>>()?,
            b18: f(n)?,
            prelude: f(n)?,
            pass_out: (0..hp.n_passes).map(|_| f(n)).collect::<Result<_>>()?,
        };
        let sched = hp.schedule();
        let tm = T_MAX;
        let fv = |m: usize| (0..m).map(|_| f(tm * n)).collect::<Result<Vec<_>>>();
        let blocks_t = Blocks {
            b0: f(tm * n)?,
            b6: fv(hp.n_passes)?,
            b12: fv(hp.n_passes)?,
            b18: f(tm * n)?,
            prelude: f(tm * n)?,
            pass_out: fv(hp.n_passes)?,
        };
        let b = BatchBufs {
            x: f(tm * n)?,
            inp_sa: f(tm * n)?,
            mix: fv(3)?,
            msg: fv(4)?,
            ctl: fv(4)?,
            ctl_g: f(tm * 2 * n)?,
            ctl_gate: f(tm * 64)?,
            inn: fv(3)?,
            g0: f(tm * w.layers[0].g0_rows())?,
            gk: f(tm * (kv + 32))?,
            gv: f(tm * (kv + 32))?,
            kc: f(tm * kv)?,
            vc: f(tm * kv)?,
            q: f(tm * hp.n_head * hp.head_dim)?,
            k: f(tm * kv)?,
            v: f(tm * kv)?,
            r: f(tm * hp.n_head * hp.n_rel)?,
            att: f(tm * hp.n_head * hp.head_dim)?,
            att2: f(tm * hp.n_head * hp.head_dim)?,
            a: f(tm * n)?,
            a2: f(tm * n)?,
            ffm: f(tm * n)?,
            ffm2: f(tm * n)?,
            xn: f(tm * n)?,
            rlog: f(tm * hp.n_expert)?,
            ids: f(tm * 8)?,
            wts: f(tm * 8)?,
            slots: f(tm * 8)?,
            moe: f(tm * n)?,
            f: f(tm * n)?,
            tau: f(tm)?,
            one: DevBuf::from_slice(&vec![1f32; tm])?,
            bl: blocks_t,
            host_ids: HostBuf::new(4 * tm * 8)?,
            host_xn: HostBuf::new(4 * tm * n)?,
            host_slots: HostBuf::new(4 * tm * 8)?,
            cpu_out: HostBuf::new(4 * tm * 8 * n)?,
            pf_gu: f(8 * PF_GROUP * 2 * hp.n_ff_exp)?,
            pf_af: f(8 * PF_GROUP * hp.n_ff_exp)?,
            pf_dn: f(8 * PF_GROUP * n)?,
            grp: DevBuf::new(grp_bytes(&hp))?,
            grp_h: HostBuf::new(grp_bytes(&hp))?,
        };
        let stream_on = std::env::var("IK_PF_STREAM").map_or(true, |v| v != "0");
        let pf_stream = if GPU_EXPERTS && stream_on { Some(PfStream::new(&ex, &hp)?) } else { None };
        // the expert cache and the full-attention KV rows share what is free once the rest exists, less a margin for
        // the CUDA graph and the driver (IK_VRAM_MARGIN_MIB, default 160); the KV rows take memory as the context grows
        // and give it back when a new sequence is shorter (plan_kv)
        let (free, _) = ctx.mem_info()?;
        let margin = std::env::var("IK_VRAM_MARGIN_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(160) << 20;
        let vram_budget = free.saturating_sub(margin);
        let max_slots = if GPU_EXPERTS { (vram_budget / (ex.gu_bytes[0] + ex.down_bytes[0])).min(opt.cache_slots) } else { 0 };
        let cache = ExpertCache::new(&ex, &hp, max_slots, ctx.device)?;
        s.sync()?;
        let job = HostBuf::new(4 * (32 + n.max(ATTN_JOB)))?;
        let ack = HostBuf::new(64)?;
        let err = HostBuf::new(64)?;
        let cpu_out = HostBuf::new(4 * 8 * n)?;
        let threads = if opt.threads.is_empty() { vec![None] } else { opt.threads.clone() };
        let pool = Pool::new(ex.clone(), &job, &ack, &cpu_out, hp.n_phys, hp.n_expert, threads);
        let mut e = Engine {
            params: DevBuf::zeroed(16, &s)?,
            ptrs: DevBuf::zeroed(8 * 16384, &s)?,
            ptr_lists: HashMap::new(),
            ptr_used: 0,
            pass_bank: (vec![], 0),
            probe: None,
            job,
            ack,
            err,
            cpu_out,
            logits_h: HostBuf::new(4 * hp.n_vocab)?,
            map: DevBuf::from_slice(&vec![-1i32; hp.n_phys * hp.n_expert])?,
            pool,
            graph: None,
            graph_on: opt.graph,
            eager_done: false,
            tok: 0,
            upload_budget: opt.upload_budget,
            ck,
            hp,
            ctx,
            k,
            w,
            ex,
            cache,
            s,
            n_ctx: opt.n_ctx,
            sched,
            kcache,
            vcache,
            kv_vm,
            kv_mapped: 0,
            kv_vram_rows: 0,
            kv_cpu: false,
            attn_idx: vec![],
            attn_out: HostBuf::new(4 * ATTN_OUT)?,
            stage: None,
            vram_budget,
            ring,
            conv,
            sc,
            bl,
            pos: 0,
            stats: Stats::default(),
            maxs,
            tokens: vec![],
            b,
            pass_bank_t: (vec![], 0),
            snap_pos: None,
            snap_conv: DevBuf::new(n_slots * (2 * kv + 2 * n) * 3 * 4)?,
            ckpt_pos: None,
            ckpt_conv: DevBuf::new(n_slots * (2 * kv + 2 * n) * 3 * 4)?,
            snap_swa: HostBuf::new(n_slots * 2 * n_swa * KV_ROW)?,
            copy_s: Stream::new_nonblocking()?,
            pf_stream,
        };
        // the full-attention KV rows: in VRAM as far as the budget goes (IK_KV_VRAM_ROWS caps them), the rest in host
        // memory where the device can map it there
        let all = e.n_ctx.div_ceil(KV_STEP) * KV_STEP;
        let per_step = e.kv_bytes(KV_STEP).max(1);
        let cap = std::env::var("IK_KV_VRAM_ROWS").ok().and_then(|v| v.parse::<usize>().ok());
        let rows_for = |budget: usize| {
            let r = (budget / per_step * KV_STEP).min(all);
            cap.map_or(r, |c| r.min(c.div_ceil(KV_STEP) * KV_STEP))
        };
        let host_ok = VmBuf::host_pages_supported(e.ctx.device);
        let mut vram_rows = if host_ok { rows_for(e.vram_budget) } else { all };
        if vram_rows < all {
            let stage = [DevBuf::new(STAGE_ROWS * KV_ROW)?, DevBuf::new(STAGE_ROWS * KV_ROW)?, DevBuf::new(T_MAX * e.hp.n_head_kv * PF_STATE * 4)?];
            e.vram_budget = e.vram_budget.saturating_sub(stage.iter().map(|b| b.len).sum());
            vram_rows = rows_for(e.vram_budget);
            e.stage = Some(stage);
        }
        let (need_vram, need_host) = (e.kv_bytes(vram_rows), e.kv_bytes(all) - e.kv_bytes(vram_rows));
        let host_free = host_mem_free().saturating_sub(HOST_MARGIN);
        if need_vram > e.vram_budget || need_host > host_free {
            let per = per_step / KV_STEP;
            let most = if host_ok { (e.vram_budget / per_step * KV_STEP) + host_free / per } else { e.vram_budget / per };
            return Err(format!(
                "a context of {} tokens needs {} MiB of KV cache; there are {} MiB of VRAM for it and the experts{}: {} tokens at most",
                e.n_ctx,
                e.kv_bytes(all) >> 20,
                e.vram_budget >> 20,
                if host_ok { format!(", and {} MiB of host memory past that", host_free >> 20) } else { String::new() },
                most
            ));
        }
        for v in e.kv_vm.iter_mut().flatten().flatten() {
            v.set_host_from(vram_rows * KV_ROW);
        }
        e.kv_vram_rows = vram_rows;
        e.kv_cpu = vram_rows < all;
        let mut g = 0;
        for b in &e.kv_vm {
            e.attn_idx.push(g);
            g += b.is_some() as u32;
        }
        if g > 64 - ATTN_STEP {
            return Err(format!("{g} full-attention layers: more attention jobs than a token has steps for"));
        }
        if e.kv_cpu {
            let hp = &e.hp;
            let proj = (0..hp.n_phys)
                .map(|ip| if hp.is_swa[ip] { Ok(vec![]) } else { e.w.layers[ip].rel_proj.download::<u16>(hp.n_rel * hp.rel_extent) })
                .collect::<Result<Vec<_>>>()?;
            e.pool.set_attn(AttnCtx {
                kv: e.kv_vm.iter().map(|b| b.as_ref().map(|b| (b[0].ptr as usize, b[1].ptr as usize))).collect(),
                phys: (0..e.kv_vm.len()).map(|il| Self::phys(hp, il)).collect(),
                proj,
                vram: vram_rows,
                extent: hp.rel_extent,
                scale: 1.0 / (hp.head_dim as f32).sqrt(),
                out: e.attn_out.ptr as usize,
            });
        }
        e.plan_kv(T_MAX)?;
        Ok(e)
    }

    /// Bytes of the full-attention KV caches in host memory now.
    pub fn kv_host_bytes(&self) -> usize {
        self.kv_vm.iter().flatten().flatten().map(|v| v.host_bytes()).sum()
    }

    /// Bytes of the full-attention KV caches with `rows` rows mapped.
    fn kv_bytes(&self, rows: usize) -> usize {
        self.kv_vm.iter().flatten().flatten().map(|v| v.pages_for(rows * KV_ROW)).sum()
    }

    /// Gives the full-attention KV caches memory for `rows` positions (rounded up to KV_STEP) and the expert cache the
    /// rest of the budget. As the context grows the last slots go, with their experts; when a new sequence is shorter
    /// they come back empty and fill as experts are used.
    fn plan_kv(&mut self, rows: usize) -> Result<()> {
        let rows = (rows.div_ceil(KV_STEP) * KV_STEP).min(self.n_ctx.div_ceil(KV_STEP) * KV_STEP);
        // the VRAM rows; new ensured they fit
        let kv = self.kv_bytes(rows.min(self.kv_vram_rows));
        let slots = self.cache.fit(self.vram_budget.saturating_sub(kv));
        if rows == self.kv_mapped && slots == [self.cache.banks[0].n_slots, self.cache.banks[1].n_slots] {
            return Ok(());
        }
        // nothing in flight may touch the pages that go: finish the token's work and the uploads
        self.s.sync()?;
        self.copy_s.sync()?;
        let resize = |e: &mut Engine, rows: usize| -> Result<()> {
            for v in e.kv_vm.iter_mut().flatten().flatten() {
                v.resize(rows * KV_ROW, &e.s)?;
            }
            Ok(())
        };
        // host pages are page-locked, which the cgroup's memory.max does not reliably count: new checked the whole
        // context against what was free then, and each step past it is checked against what is free now
        if rows > self.kv_mapped.max(self.kv_vram_rows) {
            let more = self.kv_bytes(rows) - self.kv_bytes(self.kv_mapped.max(self.kv_vram_rows));
            let free = host_mem_free().saturating_sub(HOST_MARGIN);
            if more > free {
                return Err(format!("the KV cache of {rows} tokens needs {} MiB more of host memory; {} MiB are free", more >> 20, free >> 20));
            }
        }
        // release before taking: the KV rows first when they shrink, the slots first when they do
        if rows < self.kv_mapped {
            resize(self, rows)?;
        }
        self.cache.set_slots(slots, &self.s)?;
        if rows > self.kv_mapped {
            resize(self, rows)?;
        }
        self.kv_mapped = rows;
        self.s.sync()?;
        let flat: Vec<i32> = self.cache.slot_of.iter().flatten().copied().collect();
        self.map.upload(&flat)
    }

    pub fn phys(hp: &Hp, il: usize) -> usize {
        if il < hp.n_phys { il } else { hp.loop_start + (il - hp.n_phys) % (hp.loop_end - hp.loop_start) }
    }

    /// The device address of a pointer list, uploaded the first time it is seen.
    fn ptr_list(&mut self, list: &[P]) -> Result<P> {
        if let Some(&o) = self.ptr_lists.get(list) {
            return Ok(self.ptrs.at(o));
        }
        let o = self.ptr_used;
        if o + list.len() * 8 > self.ptrs.len {
            return Err("out of pointer list space".into());
        }
        self.ptrs.upload_at(o, list)?;
        self.ptr_lists.insert(list.to_vec(), o);
        self.ptr_used += list.len() * 8;
        Ok(self.ptrs.at(o))
    }

    /// y = W x: W a BF16 matrix of the checkpoint, x f32
    fn gemv(&self, m: &Mat, x: P, y: P) -> Result<()> {
        self.k.gemv_bf16(&self.s, m.w.ptr, x, y, m.rows as u32, m.k as u32, 1, m.rows as u32)
    }

    /// y[t] = W x[t] for nt tokens: x f32 [nt][k], y[t] at t y_stride (floats)
    fn gemm(&self, m: &Mat, x: P, y: P, nt: usize, y_stride: usize) -> Result<()> {
        let (rows, k, ntu, ys) = (m.rows as u32, m.k as u32, nt as u32, y_stride as u32);
        if nt >= GEMM_MIN { self.k.gemm_bf16(&self.s, m.w.ptr, x, y, rows, k, ntu, ys) } else { self.k.gemv_bf16(&self.s, m.w.ptr, x, y, rows, k, ntu, ys) }
    }

    /// The embedding of `token` as f32: its BF16 row of the checkpoint, read from the mapped file.
    fn embed(&self, token: u32, out: &mut [f32]) -> Result<()> {
        let n = self.hp.n_embd;
        if token as usize >= self.hp.n_vocab {
            return Err(format!("token {token} is past the vocabulary of {}", self.hp.n_vocab));
        }
        let e = self.ck.get("model.llm.embed.weight", Dtype::BF16, &[self.hp.n_vocab_padded, n])?;
        let row = &e[token as usize * n * 2..(token as usize + 1) * n * 2];
        for (o, b) in out.iter_mut().zip(row.chunks_exact(2)) {
            *o = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
        }
        Ok(())
    }

    fn mix(&mut self, states: &[P], cur: P, queries: P, out: P, nq: u32) -> Result<()> {
        let mut list: Vec<P> = states.to_vec();
        list.push(cur);
        let p = self.ptr_list(&list)?;
        self.k.mix(&self.s, p, list.len() as u32, queries, out, nq, self.hp.n_embd as u32)
    }

    /// out = control(cur, msg) of controller c of physical layer ip
    fn control(&mut self, ip: usize, c: usize, cur: P, msg: P, out: P) -> Result<()> {
        let n = self.hp.n_embd as u32;
        self.k.ctrl_in(&self.s, cur, msg, self.sc.ctl_g.ptr, n)?;
        let lp = self.w.layers[ip].lp.as_ref().unwrap();
        self.gemv(&lp.down[c], self.sc.ctl_g.ptr, self.sc.ctl_gate.ptr)?;
        self.k.ctrl_out(&self.s, cur, msg, self.sc.ctl_gate.ptr, lp.gain[c].ptr, lp.forget[c].ptr, out, n)
    }

    /// Starts a new sequence: the conv states are zeroed; the KV rows past the position are never read.
    pub fn reset(&mut self) -> Result<()> {
        for c in &self.conv {
            c.memset(0, &self.s)?;
        }
        self.s.sync()?;
        self.pos = 0;
        self.tokens.clear();
        self.ckpt_pos = None;
        Ok(())
    }

    /// Feeds a prompt: keeps the cached sequence if the prompt extends it; else rolls back to the end of the previous
    /// prompt if the new one starts with it (a chat UI resends the history, often without the reasoning of the reply);
    /// else starts over. Returns the logits after the last token, and how many prompt tokens were reused.
    pub fn feed(&mut self, prompt: &[u32]) -> Result<(Vec<f32>, usize)> {
        self.feed_progress(prompt, &mut |_, _| {})
    }

    /// feed, telling `progress(processed, reused)` after each ubatch: the prompt tokens in the cache so far, reused ones
    /// included, as llama-server counts them.
    pub fn feed_progress(&mut self, prompt: &[u32], progress: &mut dyn FnMut(usize, usize)) -> Result<(Vec<f32>, usize)> {
        let mut reuse = 0;
        if prompt.len() > self.tokens.len() && prompt.starts_with(&self.tokens) {
            reuse = self.tokens.len();
        } else if let Some(p) = self.snap_pos.filter(|&p| p < prompt.len() && p <= self.tokens.len() && prompt[..p] == self.tokens[..p]) {
            self.restore(p)?;
            reuse = p;
        } else {
            self.reset()?;
        }
        // a shorter sequence than the last gives its KV memory back to the expert cache
        if self.kv_mapped > prompt.len() + 2 * KV_STEP {
            self.plan_kv(prompt.len() + 1)?;
        }
        let mut logits = vec![];
        for chunk in prompt[reuse..].chunks(T_MAX) {
            logits = if chunk.len() == 1 { self.decode(chunk[0])? } else { self.prefill(chunk)? };
            progress(self.pos, reuse);
        }
        self.snapshot()?;
        Ok((logits, reuse))
    }

    /// Appends tokens to the sequence in ubatches, as a prompt is fed, but keeps the snapshot where it is (the end of
    /// the last prompt). `progress(pos)` after each ubatch. Returns the logits after the last token.
    pub fn extend(&mut self, toks: &[u32], progress: &mut dyn FnMut(usize)) -> Result<Vec<f32>> {
        let mut logits = vec![];
        for chunk in toks.chunks(T_MAX) {
            logits = if chunk.len() == 1 { self.decode(chunk[0])? } else { self.prefill(chunk)? };
            progress(self.pos);
        }
        Ok(logits)
    }

    /// Where the snapshot that rollback goes back to is: the end of the last prompt fed.
    pub fn snapshot_pos(&self) -> Option<usize> {
        self.snap_pos
    }

    /// Rolls the sequence back to the end of the last prompt, and gives the KV memory past it and `upcoming` more tokens
    /// back to the expert cache. Returns that position.
    pub fn rollback(&mut self, upcoming: usize) -> Result<usize> {
        let p = self.snap_pos.filter(|&p| p <= self.tokens.len()).ok_or_else(|| "no prompt to roll back to".to_string())?;
        self.restore(p)?;
        if self.kv_mapped > p + upcoming + 2 * KV_STEP {
            self.plan_kv(p + upcoming + 1)?;
        }
        Ok(p)
    }

    /// Marks the current position to come back to after a branch of at most T_MAX tokens (back_to_checkpoint). Only
    /// the conv states are saved: the SWA rings hold T_MAX rows past the window, so such a branch does not overwrite
    /// the rows the checkpoint's window needs, and the full-attention rows before it stay as they are.
    pub fn checkpoint(&mut self) -> Result<()> {
        let per = self.conv[0].len;
        for (il, c) in self.conv.iter().enumerate() {
            cuda::copy_d2d(self.ckpt_conv.at(il * per), c.ptr, per, &self.s)?;
        }
        self.s.sync()?;
        self.ckpt_pos = Some(self.pos);
        Ok(())
    }

    /// Goes back to the checkpoint, dropping the branch after it.
    pub fn back_to_checkpoint(&mut self) -> Result<()> {
        let c = self.ckpt_pos.take().ok_or_else(|| "no checkpoint".to_string())?;
        if c > self.pos || self.pos - c > T_MAX {
            return Err(format!("a branch of {} tokens is too long to come back from", self.pos.saturating_sub(c)));
        }
        let per = self.conv[0].len;
        for (il, c) in self.conv.iter().enumerate() {
            cuda::copy_d2d(c.ptr, self.ckpt_conv.at(il * per), per, &self.s)?;
        }
        self.s.sync()?;
        self.pos = c;
        self.tokens.truncate(c);
        Ok(())
    }

    /// The rows of the SWA ring that the positions max(0, pos - n_swa)..pos use: (first row, count) ranges.
    fn window_rows(&self, ring: usize) -> Vec<(usize, usize, usize)> {
        let pos = self.pos;
        let q0 = pos.saturating_sub(self.hp.n_swa);
        let (start, count) = (q0 % ring, pos - q0);
        // (ring row, offset in the window, rows)
        if start + count <= ring { vec![(start, 0, count)] } else { vec![(start, 0, ring - start), (0, ring - start, count - (ring - start))] }
    }

    /// Slot il's K and V caches.
    fn kv(&self, il: usize) -> [P; 2] {
        if let Some(b) = &self.kv_vm[il] {
            return [b[0].ptr, b[1].ptr];
        }
        [self.kcache[il].as_ref().unwrap().ptr, self.vcache[il].as_ref().unwrap().ptr]
    }

    /// Saves the state at the current position, the end of a prompt.
    fn snapshot(&mut self) -> Result<()> {
        let per = self.conv[0].len;
        for (il, c) in self.conv.iter().enumerate() {
            cuda::copy_d2d(self.snap_conv.at(il * per), c.ptr, per, &self.s)?;
        }
        let w = self.hp.n_swa;
        for il in 0..self.kcache.len() {
            let ring = self.ring[il] as usize;
            if ring == 0 {
                continue;
            }
            for (ci, cache) in [&self.kcache[il], &self.vcache[il]].into_iter().flatten().enumerate() {
                // the window's rows
                let base = (il * 2 + ci) * w * KV_ROW;
                for (row, off, cnt) in self.window_rows(ring) {
                    // SAFETY: inside the page-locked snapshot buffer
                    let dst = unsafe { self.snap_swa.ptr.add(base + off * KV_ROW) };
                    cuda::copy_d2h_async(dst, cache.at(row * KV_ROW), cnt * KV_ROW, &self.s)?;
                }
            }
        }
        self.s.sync()?;
        self.snap_pos = Some(self.pos);
        Ok(())
    }

    /// Rolls the sequence back to the snapshot at position p (the KV rows of the full layers up to p are intact).
    fn restore(&mut self, p: usize) -> Result<()> {
        let per = self.conv[0].len;
        for (il, c) in self.conv.iter().enumerate() {
            cuda::copy_d2d(c.ptr, self.snap_conv.at(il * per), per, &self.s)?;
        }
        self.pos = p;
        self.tokens.truncate(p);
        self.ckpt_pos = None;
        let w = self.hp.n_swa;
        for il in 0..self.kcache.len() {
            let ring = self.ring[il] as usize;
            if ring == 0 {
                continue;
            }
            for (ci, cache) in [&self.kcache[il], &self.vcache[il]].into_iter().flatten().enumerate() {
                let base = (il * 2 + ci) * w * KV_ROW;
                for (row, off, cnt) in self.window_rows(ring) {
                    // SAFETY: inside the page-locked snapshot buffer
                    let src = unsafe { self.snap_swa.ptr.add(base + off * KV_ROW) };
                    cuda::copy_h2d_async(cache.at(row * KV_ROW), src, cnt * KV_ROW, &self.s)?;
                }
            }
        }
        self.s.sync()
    }

    /// Runs one token at position `self.pos`, returns the logits of the next token.
    pub fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        let hp = self.hp.clone();
        let n = hp.n_embd;
        if self.pos >= self.n_ctx {
            return Err(format!("context of {} tokens is full", self.n_ctx));
        }
        if self.pos + 1 > self.kv_mapped {
            self.plan_kv(self.pos + 1)?;
        }
        let mut emb = vec![0f32; n];
        self.embed(token, &mut emb)?;
        self.sc.x.upload(&emb)?;
        self.tok += 1;
        let tau = hp.tau(self.pos);
        let params: [u32; 4] = [self.pos as u32, tau.to_bits(), 1f32.to_bits(), self.tok];
        self.params.upload(&params)?;

        let use_graph = self.graph_on && self.probe.is_none() && self.eager_done;
        if use_graph {
            if self.graph.is_none() {
                self.s.begin_capture()?;
                let r = self.token_body();
                let g = self.s.end_capture();
                r?;
                self.graph = Some(g?);
            }
            self.graph.as_ref().unwrap().launch(&self.s)?;
        } else {
            self.token_body()?;
            self.eager_done = true;
        }
        self.s.sync()?;
        let e = self.err.as_slice::<u32>()[0];
        if e != 0 {
            return Err(format!("the CPU experts did not answer job {e}"));
        }
        let logits = self.logits_h.as_slice::<f32>()[..hp.n_vocab].to_vec();
        self.tokens.push(token);
        self.pos += 1;
        self.stats.tokens += 1;
        self.admit()?;
        Ok(logits)
    }

    /// All the launches of one token, from the embedding in `x` to the logits in host memory.
    /// IK_PROF: the GPU time of each kind of kernel in the last token, from the stamps after each launch: (name, launches,
    /// microseconds), most first, and the token's total. A kernel's time includes the stamp before it (~1-2 us).
    pub fn profile(&self) -> Result<Option<(Vec<(String, usize, f64)>, f64)>> {
        let Some((buf, names, _)) = &self.k.prof else { return Ok(None) };
        let names = names.borrow();
        let st: Vec<u64> = buf.download(names.len())?;
        let mut by: HashMap<&str, (usize, f64)> = HashMap::new();
        for i in 1..names.len() {
            let e = by.entry(names[i]).or_default();
            e.0 += 1;
            e.1 += st[i].saturating_sub(st[i - 1]) as f64 / 1e3;
        }
        let mut v: Vec<(String, usize, f64)> = by.into_iter().map(|(k, (n, us))| (k.to_string(), n, us)).collect();
        v.sort_by(|a, b| b.2.total_cmp(&a.2));
        let total = st.last().copied().unwrap_or(0).saturating_sub(st.first().copied().unwrap_or(0)) as f64 / 1e3;
        Ok(Some((v, total)))
    }

    fn token_body(&mut self) -> Result<()> {
        self.k.prof_reset();
        let hp = self.hp.clone();
        let n = hp.n_embd;
        let nu = n as u32;
        let p_pos = self.params.ptr;
        let p_tau = self.params.at(4);
        let p_one = self.params.at(8);
        self.k.rms_norm(&self.s, self.sc.x.ptr, Some(self.w.tok_norm.ptr), self.sc.x.ptr, nu, 1, hp.eps)?;

        let mut blocks: Vec<P> = vec![];
        let mut prelude_blocks: Vec<P> = vec![];
        let sched = self.sched.clone();
        for (si, st) in sched.iter().enumerate() {
            if st.ip == hp.loop_start {
                if st.pass == 0 {
                    cuda::copy_d2d(self.bl.prelude.ptr, self.sc.x.ptr, n * 4, &self.s)?;
                    prelude_blocks = blocks.clone();
                } else {
                    // the bank of the previous pass: its blocks after the first core block, and its output
                    let bank: Vec<P> = blocks[prelude_blocks.len() + 1..].to_vec();
                    let out = self.bl.pass_out[st.pass - 1].ptr;
                    cuda::copy_d2d(out, self.sc.x.ptr, n * 4, &self.s)?;
                    cuda::copy_d2d(self.sc.x.ptr, self.bl.prelude.ptr, n * 4, &self.s)?;
                    blocks = prelude_blocks.clone();
                    self.pass_bank = (bank, out);
                }
            }
            self.layer(*st, si as u32, &mut blocks, p_pos, p_tau, p_one)?;
        }

        let x = self.sc.x.ptr;
        let orq = self.w.output_res_query.ptr;
        self.mix(&blocks.clone(), x, orq, self.sc.a.ptr, 1)?;
        self.k.rms_norm(&self.s, self.sc.a.ptr, Some(self.w.output_norm.ptr), self.sc.a.ptr, nu, 1, hp.eps)?;
        self.gemv(&self.w.output, self.sc.a.ptr, self.sc.logits_raw.ptr)?;
        self.k.softcap(&self.s, self.sc.logits_raw.ptr, self.logits_h.dev, hp.n_vocab as u32, hp.softcap)?;
        self.k.prof_stop();
        Ok(())
    }

    /// Between tokens: uploads experts that miss often to the GPU cache, in place of experts used less.
    fn admit(&mut self) -> Result<()> {
        self.admit_d(true)
    }

    /// Multiplies the expert usage counts by k.
    fn scale_usage(&self, k: f32) {
        let mut u = self.pool.usage.lock().unwrap();
        for f in u.freq.iter_mut().flatten() {
            *f *= k;
        }
    }

    /// admit, with the decay of the usage counts once per token (`decay`), not at each layer of a prefill
    fn admit_d(&mut self, decay: bool) -> Result<()> {
        if self.cache.n_slots() == 0 {
            return Ok(());
        }
        let mut changed = false;
        let mut cands = vec![];
        {
            let mut u = self.pool.usage.lock().unwrap();
            if decay {
                for l in u.freq.iter_mut() {
                    for f in l.iter_mut() {
                        *f *= DECAY;
                    }
                }
            }
            let mut ms = std::mem::take(&mut u.misses);
            ms.sort();
            let all = ms.len();
            ms.dedup();
            // misses of an expert already missed in an earlier pass of the same token (decode only)
            if decay {
                self.stats.misses_all += all;
                self.stats.misses_repeat += all - ms.len();
            }
            for (l, e) in ms {
                cands.push((u.freq[l][e], l, e));
            }
            self.cache.uses.clone_from(&u.freq);
        }
        cands.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut uploads = 0;
        for (f, l, e) in cands {
            if self.cache.slot_of[l][e] >= 0 {
                continue;
            }
            let bank = self.ex.fmt[l].index();
            let Some(v) = self.cache.victim(bank) else { continue };
            let (ol, oe) = self.cache.banks[bank].owner[v];
            let free = ol == usize::MAX;
            if !free {
                if uploads >= self.upload_budget || f <= 1.5 * self.cache.uses[ol][oe] + 0.5 {
                    continue;
                }
                uploads += 1;
            }
            self.cache.fill(&self.ex, v, l, e)?;
            changed = true;
        }
        if changed {
            let flat: Vec<i32> = self.cache.slot_of.iter().flatten().copied().collect();
            self.map.upload(&flat)?;
        }
        Ok(())
    }

    /// Runs up to T_MAX tokens at once (no CUDA graph; the host groups the experts at each MoE layer).
    /// Returns the logits after the last token.
    pub fn prefill(&mut self, toks: &[u32]) -> Result<Vec<f32>> {
        let hp = self.hp.clone();
        let n = hp.n_embd;
        let nt = toks.len();
        assert!(nt > 0 && nt <= T_MAX);
        if self.pos + nt > self.n_ctx {
            return Err(format!("context of {} tokens is full", self.n_ctx));
        }
        // the usage counts as decode leaves them: each token decays them once, and the ubatch's uses count at their
        // mean weight over it. With a decay per ubatch, a long prompt's experts counted thousands and held their slots
        // for hundreds of tokens of decode after it.
        let d = DECAY.powi(nt as i32);
        let wt = (1.0 - d) / (nt as f32 * (1.0 - DECAY));
        self.scale_usage(d / wt);
        if self.pos + nt > self.kv_mapped {
            self.plan_kv(self.pos + nt)?;
        }
        let mut emb = vec![0f32; nt * n];
        for (i, &t) in toks.iter().enumerate() {
            self.embed(t, &mut emb[i * n..(i + 1) * n])?;
        }
        self.b.x.upload(&emb)?;
        let taus: Vec<f32> = (0..nt).map(|i| hp.tau(self.pos + i)).collect();
        self.b.tau.upload(&taus)?;
        let params: [u32; 4] = [self.pos as u32, hp.tau(self.pos).to_bits(), 1f32.to_bits(), self.tok];
        self.params.upload(&params)?;
        let x = self.b.x.ptr;
        self.k.rms_norm(&self.s, x, Some(self.w.tok_norm.ptr), x, n as u32, nt as u32, hp.eps)?;

        let mut blocks: Vec<P> = vec![];
        let mut prelude_blocks: Vec<P> = vec![];
        let sched = self.sched.clone();
        for st in &sched {
            if st.ip == hp.loop_start {
                if st.pass == 0 {
                    cuda::copy_d2d(self.b.bl.prelude.ptr, x, nt * n * 4, &self.s)?;
                    prelude_blocks = blocks.clone();
                } else {
                    let bank: Vec<P> = blocks[prelude_blocks.len() + 1..].to_vec();
                    let out = self.b.bl.pass_out[st.pass - 1].ptr;
                    cuda::copy_d2d(out, x, nt * n * 4, &self.s)?;
                    cuda::copy_d2d(x, self.b.bl.prelude.ptr, nt * n * 4, &self.s)?;
                    blocks = prelude_blocks.clone();
                    self.pass_bank_t = (bank, out);
                }
            }
            self.layer_t(*st, &mut blocks, nt)?;
        }

        // the output of the last token
        let orq = self.w.output_res_query.ptr;
        let mut list = blocks.clone();
        list.push(x);
        let last: Vec<P> = list.iter().map(|&p| p + ((nt - 1) * n * 4) as u64).collect();
        let pl = self.ptr_list(&last)?;
        self.k.mix(&self.s, pl, last.len() as u32, orq, self.sc.a.ptr, 1, n as u32)?;
        self.k.rms_norm(&self.s, self.sc.a.ptr, Some(self.w.output_norm.ptr), self.sc.a.ptr, n as u32, 1, hp.eps)?;
        self.gemv(&self.w.output, self.sc.a.ptr, self.sc.logits_raw.ptr)?;
        self.k.softcap(&self.s, self.sc.logits_raw.ptr, self.logits_h.dev, hp.n_vocab as u32, hp.softcap)?;
        self.s.sync()?;
        let logits = self.logits_h.as_slice::<f32>()[..hp.n_vocab].to_vec();
        self.pos += nt;
        self.tokens.extend_from_slice(toks);
        self.stats.tokens += nt as u64;
        self.scale_usage(wt);
        self.admit_d(false)?;
        Ok(logits)
    }

    fn mix_t(&mut self, states: &[P], cur: P, queries: P, out: P, nt: usize) -> Result<()> {
        let mut list: Vec<P> = states.to_vec();
        list.push(cur);
        let p = self.ptr_list(&list)?;
        self.k.mix_t(&self.s, p, list.len() as u32, queries, out, 1, self.hp.n_embd as u32, nt as u32)
    }

    fn control_t(&mut self, ip: usize, c: usize, cur: P, msg: P, out: P, nt: usize) -> Result<()> {
        let n = self.hp.n_embd as u32;
        self.k.ctrl_in_t(&self.s, cur, msg, self.b.ctl_g.ptr, n, nt as u32)?;
        let lp = self.w.layers[ip].lp.as_ref().unwrap();
        let d = &lp.down[c];
        self.gemm(d, self.b.ctl_g.ptr, self.b.ctl_gate.ptr, nt, d.rows)?;
        self.k.ctrl_out_t(&self.s, cur, msg, self.b.ctl_gate.ptr, lp.gain[c].ptr, lp.forget[c].ptr, out, n, nt as u32)
    }

    fn layer_t(&mut self, st: Step, blocks: &mut Vec<P>, nt: usize) -> Result<()> {
        let hp = self.hp.clone();
        let (n, nu) = (hp.n_embd, hp.n_embd as u32);
        let (il, ip) = (st.il, st.ip);
        let kv = hp.kv_dim();
        let swa = self.w.layers[ip].swa;
        let x = self.b.x.ptr;
        let looped = st.pass > 0;
        let ntu = nt as u32;
        let nb = nt * n * 4;

        if looped {
            // b.msg[c]: the message of controller c (q, k, v, the MoE input)
            let (bank, out) = self.pass_bank_t.clone();
            for c in 0..4 {
                let (q, dst) = (self.w.layers[ip].lp.as_ref().unwrap().query(c, n), self.b.msg[c].ptr);
                self.mix_t(&bank, out, q, dst, nt)?;
            }
        }
        let block_start = ip % hp.block_size == 0;
        let inp_sa = if block_start {
            match ip {
                0 => self.b.bl.b0.ptr,
                6 => self.b.bl.b6[st.pass].ptr,
                12 => self.b.bl.b12[st.pass].ptr,
                18 => self.b.bl.b18.ptr,
                _ => return Err(format!("no block storage for layer {ip}")),
            }
        } else {
            self.b.inp_sa.ptr
        };
        cuda::copy_d2d(inp_sa, x, nb, &self.s)?;

        let mut ins: [P; 3] = [x, x, x];
        if let Some(q) = self.w.layers[ip].attn_res_query.as_ref().map(|b| b.ptr) {
            let nq = if swa { 1 } else { 3 };
            for j in 0..nq {
                let dst = self.b.mix[j].ptr;
                self.mix_t(&blocks.clone(), x, q + (j * n * 4) as u64, dst, nt)?;
            }
            for j in 0..3 {
                ins[j] = self.b.mix[if swa { 0 } else { j }].ptr;
            }
        }
        if looped {
            for j in 0..3 {
                let (out, msg) = (self.b.ctl[j].ptr, self.b.msg[j].ptr);
                self.control_t(ip, j, ins[j], msg, out, nt)?;
                ins[j] = out;
            }
        }
        let nin = if swa && !looped { 1 } else { 3 };
        for j in 0..nin {
            let w = self.w.layers[ip].in_norms[if swa { 0 } else { j }].ptr;
            let out = self.b.inn[j].ptr;
            self.k.rms_norm(&self.s, ins[j], Some(w), out, nu, ntu, hp.eps)?;
            ins[j] = out;
        }
        if nin == 1 {
            ins[1] = ins[0];
            ins[2] = ins[0];
        }
        // the projections of each input into rows [t][..]: g0 q | gate | rel | sink, gk k | k taps, gv v | v taps
        let l = &self.w.layers[ip];
        let g0r = l.g0_rows();
        let (gkr, gvr) = (l.gk.iter().map(|m| m.rows).sum::<usize>(), l.gv.iter().map(|m| m.rows).sum::<usize>());
        for (mats, x, y, stride) in [(&l.g0, ins[0], &self.b.g0, g0r), (&l.gk, ins[1], &self.b.gk, gkr), (&l.gv, ins[2], &self.b.gv, gvr)] {
            let mut off = 0;
            for m in mats {
                self.gemm(m, x, y.at(off * 4), nt, stride)?;
                off += m.rows;
            }
        }
        let (g0r, gkr, gvr) = (g0r as u32, gkr as u32, gvr as u32);
        let nh = hp.n_head * hp.head_dim;

        let conv = self.conv[il].ptr;
        let st_k = conv;
        let st_v = conv + (kv * 3 * 4) as u64;
        let st_post = conv + (2 * kv * 3 * 4) as u64;
        let st_ffn = st_post + (n * 3 * 4) as u64;
        let (k_taps, v_taps, cscale) = if swa {
            (None, None, 1.0 / ((hp.n_conv + 1) as f32).sqrt())
        } else {
            (Some(self.b.gk.at(kv * 4)), Some(self.b.gv.at(kv * 4)), 1.0 / ((2 * hp.n_conv + 1) as f32).sqrt())
        };
        let hd = hp.head_dim as u32;
        self.k.conv4_seq(&self.s, self.b.gk.ptr, l.k_conv.ptr, st_k, k_taps, self.b.kc.ptr, kv as u32, hd, cscale, ntu, gkr, gkr)?;
        self.k.conv4_seq(&self.s, self.b.gv.ptr, l.v_conv.ptr, st_v, v_taps, self.b.vc.ptr, kv as u32, hd, cscale, ntu, gvr, gvr)?;
        let post = if swa { self.b.one.ptr } else { self.b.tau.ptr };
        let (nhd, nkv) = (hp.n_head as u32, hp.n_head_kv as u32);
        self.k.clip_rows_t(&self.s, self.b.g0.ptr, Some(l.q_clip.ptr), self.b.q.ptr, hd, ntu * nhd, nhd, hp.qk_clip, post, nhd, g0r, nh as u32)?;
        self.k.clip_rows_t(&self.s, self.b.kc.ptr, None, self.b.k.ptr, hd, ntu * nkv, 1, hp.qk_clip, self.b.one.ptr, nkv, kv as u32, kv as u32)?;
        self.k.rms_norm(&self.s, self.b.vc.ptr, None, self.b.v.ptr, hd, ntu * nkv, hp.eps)?;
        let nr = hp.n_rel as u32;
        let rel = self.b.g0.at(2 * nh * 4);
        let sink = self.b.g0.at((2 * nh + hp.n_head * hp.n_rel) * 4);
        self.k.clip_rows_t(&self.s, rel, Some(l.rel_clip.ptr), self.b.r.ptr, nr, ntu * nhd, nhd, hp.rel_clip, post, nhd, g0r, nhd * nr)?;

        self.probe("Qcur", il, self.b.q.ptr, nt * nh)?;
        self.probe("Kcur", il, self.b.k.ptr, nt * kv)?;
        self.probe("Vcur", il, self.b.v.ptr, nt * kv)?;
        self.probe("Rcur", il, self.b.r.ptr, nt * hp.n_head * hp.n_rel)?;
        let l = &self.w.layers[ip];
        let ([kc, vc], ring) = (self.kv(il), self.ring[il]);
        self.k.kv_store_t(&self.s, self.b.k.ptr, self.b.v.ptr, kc, vc, self.params.ptr, ring, kv as u32, kv as u32, kv as u32, ntu)?;
        let extent = if swa { hp.n_swa } else { hp.rel_extent } as u32;
        let window = if swa { hp.n_swa as u32 } else { u32::MAX };
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        let host = !swa && self.stage.is_some() && self.pos + nt > self.kv_vram_rows;
        let pass = |e: &Engine, kc: P, vc: P, rows: (u32, u32), state: P, mode: u32| {
            e.k.attn_prefill(&e.s, e.b.q.ptr, e.b.r.ptr, sink, l.rel_proj.ptr, kc, vc, e.b.att.ptr, e.params.ptr, ring, window, extent, scale, nh as u32, nhd * nr, g0r, ntu, rows, state, mode)
        };
        if !host {
            pass(self, kc, vc, (0, u32::MAX), 0, 0)?;
        } else {
            // the rows in VRAM, then the host rows from a VRAM copy a chunk at a time (read over PCIe once, not once per
            // token), the softmax state carried between the passes
            let [sk, sv, state] = self.stage.as_ref().map(|b| [b[0].ptr, b[1].ptr, b[2].ptr]).unwrap();
            let (vram, end) = (self.kv_vram_rows, self.pos + nt);
            pass(self, kc, vc, (0, vram as u32), state, 2)?;
            let mut r0 = vram;
            while r0 < end {
                let r1 = (r0 + STAGE_ROWS).min(end);
                let off = (r0 * KV_ROW) as u64;
                cuda::copy_d2d(sk, kc + off, (r1 - r0) * KV_ROW, &self.s)?;
                cuda::copy_d2d(sv, vc + off, (r1 - r0) * KV_ROW, &self.s)?;
                pass(self, sk.wrapping_sub(off), sv.wrapping_sub(off), (r0 as u32, r1 as u32), state, if r1 == end { 1 } else { 3 })?;
                r0 = r1;
            }
        }
        self.probe("kqv_out", il, self.b.att.ptr, nt * nh)?;
        let gate = self.b.g0.at(nh * 4);
        self.k.mul_sigmoid_t(&self.s, self.b.att.ptr, gate, self.b.att2.ptr, ntu * nh as u32, nh as u32, g0r)?;
        let l = &self.w.layers[ip];
        self.gemm(&l.wo, self.b.att2.ptr, self.b.a.ptr, nt, n)?;
        let pscale = 1.0 / ((hp.n_conv + 1) as f32).sqrt();
        self.k.conv4_seq(&self.s, self.b.a.ptr, l.post_conv.ptr, st_post, None, self.b.a2.ptr, nu, 1, pscale, ntu, 0, nu)?;
        if block_start {
            blocks.push(inp_sa);
            cuda::copy_d2d(x, self.b.a2.ptr, nb, &self.s)?;
        } else {
            self.k.add(&self.s, inp_sa, self.b.a2.ptr, x, ntu * nu)?;
        }
        self.probe("ffn_inp", il, x, nt * n)?;

        let frq = self.w.layers[ip].ffn_res_query.ptr;
        let ffm = self.b.ffm.ptr;
        self.mix_t(&blocks.clone(), x, frq, ffm, nt)?;
        let mut m = ffm;
        if looped {
            let (msg, out) = (self.b.msg[3].ptr, self.b.ffm2.ptr);
            self.control_t(ip, 3, m, msg, out, nt)?;
            m = out;
        }
        let l = &self.w.layers[ip];
        self.k.rms_norm(&self.s, m, Some(l.ffn_norm.ptr), self.b.xn.ptr, nu, ntu, hp.eps)?;
        self.gemm(&l.router, self.b.xn.ptr, self.b.rlog.ptr, nt, hp.n_expert)?;
        self.k.router_topk_t(&self.s, self.b.rlog.ptr, l.router_bias.ptr, self.b.ids.ptr, self.b.wts.ptr, hp.w_scale, ntu)?;
        self.probe("ffn_norm_in", il, self.b.xn.ptr, nt * n)?;
        self.moe_t(ip, nt)?;
        self.probe("ffn_moe_out", il, self.b.moe.ptr, nt * n)?;
        let l = &self.w.layers[ip];
        self.k.conv4_seq(&self.s, self.b.moe.ptr, l.ffn_post_conv.ptr, st_ffn, None, self.b.f.ptr, nu, 1, pscale, ntu, 0, nu)?;
        self.k.add(&self.s, x, self.b.f.ptr, x, ntu * nu)?;
        self.probe("l_out", il, x, nt * n)?;
        Ok(())
    }

    /// The experts of a ubatch. The (token, expert) pairs of cached experts run on the GPU, PF_GROUP tokens at a time.
    /// Of the others, grouped by expert on the host, the expert stream takes those with the most tokens while its time
    /// stays under the CPU's for the rest, and the CPU team computes the rest meanwhile; the outputs of both are added
    /// to the GPU's once they are done.
    fn moe_t(&mut self, ip: usize, nt: usize) -> Result<()> {
        let n = self.hp.n_embd;
        let nu = n as u32;
        cuda::copy_d2h_async(self.b.host_ids.ptr, self.b.ids.ptr, nt * 8 * 4, &self.s)?;
        cuda::copy_d2h_async(self.b.host_xn.ptr, self.b.xn.ptr, nt * n * 4, &self.s)?;
        let t0 = std::time::Instant::now();
        self.s.sync()?;
        let t1 = std::time::Instant::now();
        self.stats.pf_gpu_s += (t1 - t0).as_secs_f64();
        // the last layer's stream is done (self.s waited for it): its throughput
        if let Some(ps) = self.pf_stream.as_mut()
            && ps.last_bytes > 0
        {
            let secs = ps.t1.ms_since(&ps.t0)? as f64 / 1e3;
            if secs > 0.0 {
                ps.bw = 0.8 * ps.bw + 0.2 * ps.last_bytes as f64 / secs;
            }
            self.stats.pf_stream_bytes += ps.last_bytes as u64;
            self.stats.pf_stream_s += secs;
            ps.last_bytes = 0;
        }
        let ids: Vec<u32> = self.b.host_ids.as_slice::<u32>()[..nt * 8].to_vec();
        let slots: Vec<i32> = ids.iter().map(|&e| self.cache.slot_of[ip][e as usize]).collect();
        let mut groups: Vec<(usize, Vec<(usize, usize)>)> = vec![];
        let mut gi: HashMap<usize, usize> = HashMap::new();
        for (p, &e) in ids.iter().enumerate().filter(|&(p, _)| slots[p] < 0) {
            let g = *gi.entry(e as usize).or_insert_with(|| {
                groups.push((e as usize, vec![]));
                groups.len() - 1
            });
            groups[g].1.push((p / 8, p));
        }
        // the stream's share: an expert costs it its bytes, and the CPU its pairs
        let mut streamed = vec![];
        if let Some(ps) = &self.pf_stream {
            let f = self.ex.fmt[ip].index();
            let bytes = (self.ex.gu_bytes[f] + self.ex.down_bytes[f]) as f64;
            groups.sort_by_key(|g| std::cmp::Reverse(g.1.len()));
            let mut rest: usize = groups.iter().map(|g| g.1.len()).sum();
            let mut k = 0;
            while k < groups.len() && (k + 1) as f64 * bytes / ps.bw <= (rest - groups[k].1.len()) as f64 * ps.cpu_pair_s {
                rest -= groups[k].1.len();
                k += 1;
            }
            streamed = groups.drain(..k).collect::<Vec<_>>();
        }
        self.b.host_slots.as_mut_slice::<i32>()[..nt * 8].copy_from_slice(&slots);
        cuda::copy_h2d_async(self.b.slots.ptr, self.b.host_slots.ptr, nt * 8 * 4, &self.s)?;
        let on_gpu = slots.iter().filter(|&&s| s >= 0).count();
        self.stats.pf_gpu_pairs += on_gpu as u64;
        self.stats.pf_cpu_pairs += (nt * 8 - on_gpu) as u64;
        // the CPU team counts the uses of its pairs; these are the GPU's
        if on_gpu > 0 || !streamed.is_empty() {
            let mut u = self.pool.usage.lock().unwrap();
            for (&e, _) in ids.iter().zip(&slots).filter(|&(_, &s)| s >= 0) {
                u.freq[ip][e as usize] += 1.0;
            }
            for (e, pairs) in &streamed {
                u.freq[ip][*e] += pairs.len() as f32;
                u.misses.push((ip, *e));
            }
        }
        // the CPU team starts, and the GPU's work is queued while it runs
        let cpu_pairs: usize = groups.iter().map(|g| g.1.len()).sum();
        let t3 = std::time::Instant::now();
        self.stats.pf_host_s += (t3 - t1).as_secs_f64();
        if !groups.is_empty() {
            let x = self.b.host_xn.as_slice::<f32>()[..nt * n].to_vec();
            self.pool.start_batch(crate::cpu_pool::BatchJob { layer: ip, x, misses: groups, out: self.b.cpu_out.ptr as usize });
        }
        // the cached experts with many tokens run grouped, and set b.moe; the others pair by pair, by the slots left
        let pp = self.experts_g(ip, nt, &ids, &slots)?;
        let grouped = pp.is_some();
        let (pp, pp_d) = match &pp {
            Some(v) => (&v[..], self.b.grp.at(32 * nt)),
            None => (&slots[..], self.b.slots.ptr),
        };
        let spans: Vec<(usize, usize, bool)> = (0..nt)
            .step_by(PF_GROUP)
            .map(|ta| {
                let tb = (ta + PF_GROUP).min(nt);
                (ta, tb, pp[8 * ta..8 * tb].iter().any(|&s| s >= 0))
            })
            .collect();
        for &(ta, tb, gpu) in &spans {
            if gpu {
                self.experts_pf(ip, ta, tb, pp_d + (4 * 8 * ta) as u64, grouped)?;
            }
        }
        if !streamed.is_empty() {
            self.stream_experts(ip, &streamed)?;
        }
        if cpu_pairs > 0 {
            let te = std::time::Instant::now();
            self.pool.wait_batch();
            let t4 = std::time::Instant::now();
            self.stats.pf_cpu_s += (t4 - t3).as_secs_f64();
            // the team's time is known when it was still running once the GPU's work was queued
            if let Some(ps) = self.pf_stream.as_mut()
                && (t4 - te).as_secs_f64() > 1e-3
            {
                ps.cpu_pair_s = 0.8 * ps.cpu_pair_s + 0.2 * (t4 - t3).as_secs_f64() / cpu_pairs as f64;
            }
        }
        if let Some(ps) = &self.pf_stream
            && !streamed.is_empty()
            && !ps.t1.done()
        {
            let t = std::time::Instant::now();
            ps.t1.sync()?;
            self.stats.pf_stream_late += 1;
            self.stats.pf_stream_late_s += t.elapsed().as_secs_f64();
        }
        // the CPU's and the stream's pairs (cpu_out), added to the GPU's where a group has some
        let cpu = self.b.cpu_out.dev;
        for &(ta, tb, gpu) in &spans {
            let (b, nn) = (&self.b, n * 4);
            self.k.moe_combine_t(&self.s, cpu + (8 * ta * nn) as u64, b.slots.at(4 * 8 * ta), b.wts.at(4 * 8 * ta), b.moe.at(ta * nn), nu, (tb - ta) as u32, false, gpu || grouped)?;
        }
        Ok(())
    }

    /// Queues the pairs of `streamed` (expert, its (token, pair)s) on the expert stream: the copy stream uploads each
    /// expert to a staging slot once the slot's last reader is done, and the stream gathers its tokens' x, runs gate_up,
    /// SITU and down (an NVFP4 expert's by moe_nvfp4_g, which decodes each weight once for 32 tokens), and puts the
    /// outputs in cpu_out's rows of their pairs, as the CPU team does. self.s waits for it.
    /// The experts go through the bounce slots: the host copies each to one once the slot's last upload is done, so
    /// this returns about when the last upload starts.
    fn stream_experts(&mut self, ip: usize, streamed: &[(usize, Vec<(usize, usize)>)]) -> Result<()> {
        let (n, ffu) = (self.hp.n_embd as u32, self.hp.n_ff_exp as u32);
        let f = self.ex.fmt[ip];
        let (gu_b, dn_b) = (self.ex.gu_bytes[f.index()], self.ex.down_bytes[f.index()]);
        let cap = 8 * T_MAX;
        let Some(ps) = self.pf_stream.as_mut() else { return Ok(()) };
        let l = ps.lists_h.as_mut_slice::<u32>();
        let mut i = 0;
        for (j, (_, pairs)) in streamed.iter().enumerate() {
            for &(t, p) in pairs {
                (l[i], l[cap + i], l[2 * cap + i]) = ((j % STREAM_SLOTS) as u32, t as u32, p as u32);
                i += 1;
            }
        }
        cuda::copy_h2d_async(ps.lists.ptr, ps.lists_h.ptr, 3 * cap * 4, &ps.s)?;
        // an NVFP4 expert's gate_up reads its staging slot and the gathered x, its down the activations
        if matches!(f, Fmt::NvFp4) {
            let tab = ps.tab_h.as_mut_slice::<u64>();
            for (j, (e, pairs)) in streamed.iter().enumerate() {
                let (slot, me) = (j % STREAM_SLOTS, pairs.len() as u64 | (*e as u64) << 32);
                tab[8 * j..8 * j + 8].copy_from_slice(&[ps.gu.at(slot * gu_b), ps.x.ptr, ps.gu_out.ptr, me, ps.dn.at(slot * dn_b), ps.af.ptr, ps.x.ptr, me]);
            }
            cuda::copy_h2d_async(ps.tab.ptr, ps.tab_h.ptr, 64 * streamed.len(), &ps.s)?;
        }
        ps.t0.record(&self.copy_s)?;
        let mut o = 0;
        for (j, (e, pairs)) in streamed.iter().enumerate() {
            let slot = j % STREAM_SLOTS;
            self.copy_s.wait(&ps.free[slot])?;
            let (gu, dn) = (self.ex.gate_up(ip, *e), self.ex.down(ip, *e));
            let tw = std::time::Instant::now();
            while !ps.bounced[slot].done() {
                std::thread::yield_now();
            }
            let tc = std::time::Instant::now();
            let sb = ps.bounce.len / STREAM_SLOTS;
            let dst = &mut ps.bounce.as_mut_slice::<u8>()[slot * sb..(slot + 1) * sb];
            let mut at = 0;
            for part in [gu.q, gu.s, dn.q, dn.s] {
                crate::cpu::copy_nt(&mut dst[at..at + part.len()], part);
                at += part.len();
            }
            crate::cpu::store_fence();
            assert_eq!(at, gu_b + dn_b, "an expert's parts fill its slots");
            self.stats.pf_stream_wait_s += (tc - tw).as_secs_f64();
            self.stats.pf_stream_copy_s += tc.elapsed().as_secs_f64();
            let src = dst.as_ptr();
            cuda::copy_h2d_async(ps.gu.at(slot * gu_b), src, gu_b, &self.copy_s)?;
            cuda::copy_h2d_async(ps.dn.at(slot * dn_b), src.wrapping_add(gu_b), dn_b, &self.copy_s)?;
            ps.bounced[slot].record(&self.copy_s)?;
            ps.up[slot].record(&self.copy_s)?;
            ps.s.wait(&ps.up[slot])?;
            let m = pairs.len() as u32;
            let at = |k: usize| ps.lists.at(4 * (k * cap + o));
            let (sl, tk, pr) = (at(0), at(1), at(2));
            self.k.copy_rows(&ps.s, self.b.xn.ptr, Some(tk), ps.x.ptr, None, n, m)?;
            match f {
                Fmt::NvFp4 => {
                    let ((gu_a, gu_d), (dn_a, dn_d)) = (self.cache.factor_ptrs(ip, 0), self.cache.factor_ptrs(ip, 1));
                    self.k.moe_nvfp4_g(&ps.s, ps.tab.at(64 * j), gu_a, gu_d, 2 * ffu, n, m, 1)?;
                    self.k.situ(&ps.s, ps.gu_out.ptr, sl, ps.af.ptr, ffu, self.ex.beta, self.ex.lb, m)?;
                    self.k.moe_nvfp4_g(&ps.s, ps.tab.at(64 * j + 32), dn_a, dn_d, n, ffu, m, 1)?;
                }
                Fmt::MxFp8 => {
                    self.k.moe_mxfp8_d(&ps.s, ps.gu.ptr, gu_b as u64, sl, ps.x.ptr, n, 0, ps.gu_out.ptr, 2 * ffu, n, m)?;
                    self.k.situ(&ps.s, ps.gu_out.ptr, sl, ps.af.ptr, ffu, self.ex.beta, self.ex.lb, m)?;
                    self.k.moe_mxfp8_d(&ps.s, ps.dn.ptr, dn_b as u64, sl, ps.af.ptr, ffu, 0, ps.x.ptr, n, ffu, m)?;
                }
            }
            ps.free[slot].record(&ps.s)?;
            self.k.copy_rows(&ps.s, ps.x.ptr, None, self.b.cpu_out.dev, Some(pr), n, m)?;
            o += pairs.len();
        }
        ps.t1.record(&ps.s)?;
        self.s.wait(&ps.t1)?;
        ps.last_bytes = streamed.len() * (gu_b + dn_b);
        self.stats.pf_stream_pairs += o as u64;
        self.stats.pf_stream_experts += streamed.len() as u64;
        Ok(())
    }

    fn probe(&mut self, name: &str, il: usize, buf: P, len: usize) -> Result<()> {
        if self.probe.is_none() {
            return Ok(());
        }
        self.s.sync()?;
        let mut v = vec![0f32; len];
        let tmp = DevBuf { ptr: buf, len: len * 4 };
        let r = tmp.download_at::<f32>(0, len);
        std::mem::forget(tmp);
        v.copy_from_slice(&r?);
        if let Some(p) = self.probe.as_mut() {
            p(name, il, &v);
        }
        Ok(())
    }

    fn layer(&mut self, st: Step, si: u32, blocks: &mut Vec<P>, p_pos: P, p_tau: P, p_one: P) -> Result<()> {
        let hp = self.hp.clone();
        let (n, nu) = (hp.n_embd, hp.n_embd as u32);
        let (il, ip) = (st.il, st.ip);
        let kv = hp.kv_dim();
        let swa = self.w.layers[ip].swa;
        let x = self.sc.x.ptr;
        let looped = st.pass > 0;

        // messages of a looped pass: from the bank of the previous pass
        if looped {
            let (bank, out) = self.pass_bank.clone();
            let lp = self.w.layers[ip].lp.as_ref().unwrap();
            let (q_mlp, k_v) = (lp.q_mlp, lp.k_v);
            self.mix(&bank, out, q_mlp, self.sc.msgs.ptr, 2)?;
            self.mix(&bank, out, k_v, self.sc.msgs.at(2 * n * 4), 2)?;
        }

        // the layer input: a new block, or kept for the residual
        let block_start = ip % hp.block_size == 0;
        let inp_sa = if block_start {
            match ip {
                0 => self.bl.b0.ptr,
                6 => self.bl.b6[st.pass].ptr,
                12 => self.bl.b12[st.pass].ptr,
                18 => self.bl.b18.ptr,
                _ => return Err(format!("no block storage for layer {ip}")),
            }
        } else {
            self.sc.inp_sa.ptr
        };
        cuda::copy_d2d(inp_sa, x, n * 4, &self.s)?;

        // attention inputs
        let mut ins: [P; 3] = [x, x, x];
        if let Some(q) = self.w.layers[ip].attn_res_query.as_ref().map(|b| b.ptr) {
            let nq = if swa { 1 } else { 3 };
            self.mix(&blocks.clone(), x, q, self.sc.mixed.ptr, nq)?;
            for j in 0..3 {
                ins[j] = self.sc.mixed.at(if swa { 0 } else { j * n * 4 });
            }
        }
        if looped {
            // the 3 input controllers at once: their inputs, down projections and outputs
            let lp = self.w.layers[ip].lp.as_ref().unwrap();
            let rows = lp.down[0].rows;
            let (mut t_in, mut t_down, mut t_out) = (vec![], vec![], vec![]);
            for j in 0..3 {
                let (msg, out) = (self.sc.msgs.at(MSG[j] * n * 4), self.sc.ctl_tmp.at(j * n * 4));
                t_in.extend([ins[j], msg]);
                t_down.extend([lp.down[j].w.ptr, self.sc.ctl_g3.at(j * 2 * n * 4), self.sc.ctl_gate3.at(j * rows * 4), rows as u64]);
                t_out.extend([ins[j], msg, lp.gain[j].ptr, lp.forget[j].ptr, out]);
            }
            let (t_in, t_down, t_out) = (self.ptr_list(&t_in)?, self.ptr_list(&t_down)?, self.ptr_list(&t_out)?);
            self.k.ctrl_in_z(&self.s, t_in, self.sc.ctl_g3.ptr, nu, 3)?;
            self.k.gemv_bf16t(&self.s, t_down, rows as u32, 2 * nu, 3)?;
            self.k.ctrl_outz(&self.s, t_out, self.sc.ctl_gate3.ptr, nu, 3)?;
            for (j, inp) in ins.iter_mut().enumerate() {
                *inp = self.sc.ctl_tmp.at(j * n * 4);
            }
        }
        let nin = if swa && !looped { 1 } else { 3 };
        for j in 0..nin {
            let w = self.w.layers[ip].in_norms[if swa { 0 } else { j }].ptr;
            let out = self.sc.inputs.at(j * n * 4);
            self.k.rms_norm(&self.s, ins[j], Some(w), out, nu, 1, hp.eps)?;
            ins[j] = out;
        }
        if nin == 1 {
            ins[1] = ins[0];
            ins[2] = ins[0];
        }

        // the projections of the q, k and v inputs in one launch, each checkpoint matrix into its part of g0 (q | gate |
        // rel | sink), gk (k | k taps) or gv (v | v taps)
        let l = &self.w.layers[ip];
        let mut tab = vec![];
        let mut max_rows = 0;
        for (mats, x, y) in [(&l.g0, ins[0], &self.sc.g0), (&l.gk, ins[1], &self.sc.gk), (&l.gv, ins[2], &self.sc.gv)] {
            let mut off = 0;
            for m in mats {
                tab.extend([m.w.ptr, x, y.at(off * 4), m.rows as u64]);
                off += m.rows;
                max_rows = max_rows.max(m.rows);
            }
        }
        let nz = (tab.len() / 4) as u32;
        let tab = self.ptr_list(&tab)?;
        self.k.gemv_bf16t(&self.s, tab, max_rows as u32, nu, nz)?;
        let l = &self.w.layers[ip];
        let nh = (hp.n_head * hp.head_dim) as u32;
        let q_raw = self.sc.g0.ptr;
        let gate = self.sc.g0.at(nh as usize * 4);
        let rel = self.sc.g0.at(2 * nh as usize * 4);
        let sink = self.sc.g0.at((2 * nh as usize + hp.n_head * hp.n_rel) * 4);

        // short convolutions of K and V (with input-dependent taps on full attention)
        let conv = self.conv[il].ptr;
        let st_k = conv;
        let st_v = conv + (kv * 3 * 4) as u64;
        let st_post = conv + (2 * kv * 3 * 4) as u64;
        let st_ffn = st_post + (n * 3 * 4) as u64;
        let cscale = if swa { 1.0 / ((hp.n_conv + 1) as f32).sqrt() } else { 1.0 / ((2 * hp.n_conv + 1) as f32).sqrt() };
        let _ = st_v;
        // K and V: convolutions, K clip, V norm, and the cache row, in one kernel
        let ([kc, vc], ring) = (self.kv(il), self.ring[il]);
        self.k.kv_prep(&self.s, !swa, self.sc.gk.ptr, self.sc.gv.ptr, l.k_conv.ptr, l.v_conv.ptr, st_k, self.sc.k.ptr, self.sc.v.ptr, kc, vc, p_pos, ring, cscale, hp.qk_clip, hp.eps)?;

        // Q and R clip, scaled by tau on full attention
        let post = if swa { p_one } else { p_tau };
        self.k.clip_qr(&self.s, q_raw, l.q_clip.ptr, self.sc.q.ptr, rel, l.rel_clip.ptr, self.sc.r.ptr, hp.n_head as u32, hp.qk_clip, hp.rel_clip, post)?;
        self.probe("Qcur", il, self.sc.q.ptr, nh as usize)?;
        self.probe("Kcur", il, self.sc.k.ptr, kv)?;
        self.probe("Vcur", il, self.sc.v.ptr, kv)?;
        self.probe("Rcur", il, self.sc.r.ptr, hp.n_head * hp.n_rel)?;

        // attention
        let l = &self.w.layers[ip];
        let extent = if swa { hp.n_swa } else { hp.rel_extent } as u32;
        let group = (hp.n_head / hp.n_head_kv) as u32;
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        let _ = group;
        let maxs = self.maxs as u32;
        let window = if swa { hp.n_swa as u32 } else { u32::MAX };
        let cs = if swa { CS_SWA } else { CS_FULL };
        // full attention with rows in host memory: the CPU team takes them (an attention job, posted first so the two
        // parts run together), the GPU the rows in VRAM
        let cpu = !swa && self.kv_cpu;
        let (vram, step) = if cpu { (self.kv_vram_rows as u32, ATTN_STEP + self.attn_idx[il]) } else { (u32::MAX, 0) };
        if cpu {
            let nr = (hp.n_head * hp.n_rel) as u32;
            self.k.job_post(&self.s, self.sc.q.ptr, nh, self.sc.r.ptr, nr, sink, hp.n_head as u32, self.job.dev, p_pos, step, il as u32)?;
        }
        let nsplit = if swa { hp.n_swa } else { self.n_ctx.min(self.kv_vram_rows) }.div_ceil(1 << cs).max(1) as u32;
        self.k.attn_part(&self.s, self.sc.q.ptr, self.sc.r.ptr, sink, l.rel_proj.ptr, kc, vc, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, p_pos, ring, extent, scale, maxs, nsplit, window, cs, vram)?;
        if !swa {
            // the per-chunk max and sum of each head: [kv head][chunk][3]
            self.probe("attn_pm", il, self.sc.pm.ptr, 8 * maxs as usize * 3)?;
            self.probe("attn_pl", il, self.sc.pl.ptr, 8 * maxs as usize * 3)?;
        }
        if cpu {
            self.k.moe_wait(&self.s, self.ack.dev, self.params.ptr, step, self.err.dev, 5_000_000_000)?;
        }
        self.k.attn_reduce(&self.s, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, self.sc.att.ptr, p_pos, window, maxs, cs, hp.n_head as u32, self.attn_out.dev, vram)?;
        self.probe("kqv_out", il, self.sc.att.ptr, nh as usize)?;
        self.k.mul_sigmoid(&self.s, self.sc.att.ptr, gate, self.sc.att2.ptr, nh)?;
        let l = &self.w.layers[ip];
        self.gemv(&l.wo, self.sc.att2.ptr, self.sc.a.ptr)?;
        let pscale = 1.0 / ((hp.n_conv + 1) as f32).sqrt();
        // residual streams: a block start begins a new one (the old one joins the blocks), else x = inp_sa + conv(a)
        if block_start {
            blocks.push(inp_sa);
            self.k.conv_res(&self.s, self.sc.a.ptr, l.post_conv.ptr, st_post, None, x, nu, pscale)?;
        } else {
            self.k.conv_res(&self.s, self.sc.a.ptr, l.post_conv.ptr, st_post, Some(inp_sa), x, nu, pscale)?;
        }
        self.probe("ffn_inp", il, x, n)?;

        // MoE input
        let frq = self.w.layers[ip].ffn_res_query.ptr;
        self.mix(&blocks.clone(), x, frq, self.sc.ffm.ptr, 1)?;
        let mut m = self.sc.ffm.ptr;
        if looped {
            self.control(ip, 3, m, self.sc.msgs.at(MSG[3] * n * 4), self.sc.ffm2.ptr)?;
            m = self.sc.ffm2.ptr;
        }
        let l = &self.w.layers[ip];
        self.k.rms_norm(&self.s, m, Some(l.ffn_norm.ptr), self.sc.xn.ptr, nu, 1, hp.eps)?;
        self.probe("ffn_norm_in", il, self.sc.xn.ptr, n)?;
        let l = &self.w.layers[ip];
        self.gemv(&l.router, self.sc.xn.ptr, self.sc.rlog.ptr)?;
        self.moe(ip, il, si, st_ffn, pscale)?;
        self.probe("l_out", il, x, n)?;
        Ok(())
    }

    /// The routed experts of one token: the cached ones on the GPU, the others on the CPU team meanwhile.
    fn moe(&mut self, ip: usize, il: usize, si: u32, st_ffn: P, pscale: f32) -> Result<()> {
        let n = self.hp.n_embd as u32;
        let map_row = self.map.at(ip * self.hp.n_expert * 4);
        let l = &self.w.layers[ip];
        self.k.router_dispatch(
            &self.s, self.sc.rlog.ptr, l.router_bias.ptr, self.sc.ids.ptr, self.sc.wts.ptr, self.hp.w_scale, map_row, self.sc.slots.ptr, self.sc.xn.ptr, self.job.dev,
            self.params.ptr, si, ip as u32, n,
        )?;
        if GPU_EXPERTS {
            self.experts_d(ip)?;
        }
        self.k.moe_wait(&self.s, self.ack.dev, self.params.ptr, si, self.err.dev, 5_000_000_000)?;
        let l = &self.w.layers[ip];
        self.k.moe_tail(&self.s, self.sc.dn.ptr, self.cpu_out.dev, self.sc.slots.ptr, self.sc.wts.ptr, l.ffn_post_conv.ptr, st_ffn, self.sc.x.ptr, n, pscale)?;
        if self.probe.is_some() {
            let ids: Vec<u32> = self.sc.ids.download(8)?;
            let wts: Vec<f32> = self.sc.wts.download(8)?;
            let idf: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
            if let Some(p) = self.probe.as_mut() {
                p("ffn_moe_topk", il, &idf);
                p("ffn_moe_weights_scaled", il, &wts);
            }
        }
        Ok(())
    }

    /// The cached experts of `sc.slots` (the 8 of physical layer ip) on the GPU: gate_up, the activation, and down into
    /// sc.dn. Both formats read x as f32: the norm's sc.xn, then the activations sc.af.
    fn experts_d(&self, ip: usize) -> Result<()> {
        let (n, ffu) = (self.hp.n_embd as u32, self.hp.n_ff_exp as u32);
        let (sc, f) = (&self.sc, self.ex.fmt[ip]);
        let bk = &self.cache.banks[f.index()];
        let (gu_b, dn_b) = (bk.gu_bytes as u64, bk.down_bytes as u64);
        let situ = || self.k.situ(&self.s, sc.gu.ptr, sc.slots.ptr, sc.af.ptr, ffu, self.ex.beta, self.ex.lb, 8);
        match f {
            Fmt::NvFp4 => {
                let ((gu_a, gu_d), (dn_a, dn_d)) = (self.cache.factor_ptrs(ip, 0), self.cache.factor_ptrs(ip, 1));
                self.k.moe_nvfp4_d(&self.s, bk.gu.ptr, gu_b, sc.slots.ptr, sc.ids.ptr, gu_a, gu_d, sc.xn.ptr, 0, 0, sc.gu.ptr, 2 * ffu, n, 8)?;
                situ()?;
                self.k.moe_nvfp4_d(&self.s, bk.down.ptr, dn_b, sc.slots.ptr, sc.ids.ptr, dn_a, dn_d, sc.af.ptr, ffu, 0, sc.dn.ptr, n, ffu, 8)
            }
            Fmt::MxFp8 => {
                self.k.moe_mxfp8_d(&self.s, bk.gu.ptr, gu_b, sc.slots.ptr, sc.xn.ptr, 0, 0, sc.gu.ptr, 2 * ffu, n, 8)?;
                situ()?;
                self.k.moe_mxfp8_d(&self.s, bk.down.ptr, dn_b, sc.slots.ptr, sc.af.ptr, ffu, 0, sc.dn.ptr, n, ffu, 8)
            }
        }
    }

    /// The pairs of tokens ta..tb on cached experts (`slots`, theirs of b.slots or of the pair by pair path's), on the
    /// GPU: gate_up, SITU and down of each, and their weighted sum into those tokens' rows of b.moe (added to them if
    /// `add`).
    fn experts_pf(&self, ip: usize, ta: usize, tb: usize, slots: P, add: bool) -> Result<()> {
        let (n, ffu) = (self.hp.n_embd as u32, self.hp.n_ff_exp as u32);
        let (b, f) = (&self.b, self.ex.fmt[ip]);
        let bk = &self.cache.banks[f.index()];
        let (gu_b, dn_b) = (bk.gu_bytes as u64, bk.down_bytes as u64);
        let pairs = 8 * (tb - ta) as u32;
        let (ids, x) = (b.ids.at(4 * 8 * ta), b.xn.at(4 * n as usize * ta));
        match f {
            Fmt::NvFp4 => {
                let ((gu_a, gu_d), (dn_a, dn_d)) = (self.cache.factor_ptrs(ip, 0), self.cache.factor_ptrs(ip, 1));
                self.k.moe_nvfp4_d(&self.s, bk.gu.ptr, gu_b, slots, ids, gu_a, gu_d, x, n, 3, b.pf_gu.ptr, 2 * ffu, n, pairs)?;
                self.k.situ(&self.s, b.pf_gu.ptr, slots, b.pf_af.ptr, ffu, self.ex.beta, self.ex.lb, pairs)?;
                self.k.moe_nvfp4_d(&self.s, bk.down.ptr, dn_b, slots, ids, dn_a, dn_d, b.pf_af.ptr, ffu, 0, b.pf_dn.ptr, n, ffu, pairs)?;
            }
            Fmt::MxFp8 => {
                self.k.moe_mxfp8_d(&self.s, bk.gu.ptr, gu_b, slots, x, n, 3, b.pf_gu.ptr, 2 * ffu, n, pairs)?;
                self.k.situ(&self.s, b.pf_gu.ptr, slots, b.pf_af.ptr, ffu, self.ex.beta, self.ex.lb, pairs)?;
                self.k.moe_mxfp8_d(&self.s, bk.down.ptr, dn_b, slots, b.pf_af.ptr, ffu, 0, b.pf_dn.ptr, n, ffu, pairs)?;
            }
        }
        self.k.moe_combine_t(&self.s, b.pf_dn.ptr, slots, b.wts.at(4 * 8 * ta), b.moe.at(4 * n as usize * ta), n, (tb - ta) as u32, true, add)
    }

    /// The cached pairs of the NVFP4 experts with at least G_MIN of the ubatch's nt tokens: each expert's tokens
    /// gathered and run by moe_nvfp4_g (which decodes a weight once for 32 tokens), the experts one after another in
    /// chunks of the pf buffers' 8 PF_GROUP rows (an expert split at a chunk's end), and the chunks' outputs added
    /// into b.moe (the first sets it). Returns the slots left to the pair by pair path (theirs, or -1), as uploaded to
    /// b.grp, if any expert ran grouped.
    fn experts_g(&mut self, ip: usize, nt: usize, ids: &[u32], slots: &[i32]) -> Result<Option<Vec<i32>>> {
        if !matches!(self.ex.fmt[ip], Fmt::NvFp4) {
            return Ok(None);
        }
        let ne = self.hp.n_expert;
        let mut pairs: Vec<Vec<usize>> = vec![vec![]; ne];
        for (p, &e) in ids.iter().enumerate().filter(|&(p, _)| slots[p] >= 0) {
            pairs[e as usize].push(p);
        }
        if pairs.iter().all(|v| v.len() < G_MIN) {
            return Ok(None);
        }
        let mut pp = slots.to_vec();
        let cap = 8 * PF_GROUP;
        let b = &self.b;
        let bk = &self.cache.banks[Fmt::NvFp4.index()];
        let (gu_b, dn_b) = (bk.gu_bytes as u64, bk.down_bytes as u64);
        let (n, ffu) = (self.hp.n_embd, self.hp.n_ff_exp);
        // the lists: row q of the tokens is the grouped pairs' q-th, expert after expert; chunk c holds rows c cap..
        let total: usize = pairs.iter().filter(|v| v.len() >= G_MIN).map(|v| v.len()).sum();
        let n_chunks = total.div_ceil(cap);
        let (rows_at, tab_at) = (64 * nt, 32 * nt * (2 + n_chunks));
        // per chunk: its first row, its rows, its first entry, its entries, and its most tokens of an expert
        let mut chunks: Vec<[usize; 5]> = vec![];
        let mut entries: Vec<[u64; 8]> = vec![];
        let l = b.grp_h.as_mut_slice::<u32>();
        l[rows_at / 4..tab_at / 4].fill(u32::MAX);
        let mut q = 0;
        for (e, ps) in pairs.iter().enumerate().filter(|(_, v)| v.len() >= G_MIN) {
            let slot = slots[ps[0]] as u64;
            let mut k = 0;
            while k < ps.len() {
                let (c, r) = (q / cap, q % cap);
                if chunks.len() == c {
                    chunks.push([q, 0, entries.len(), 0, 0]);
                }
                let m = (ps.len() - k).min(cap - r);
                for (j, &p) in ps[k..k + m].iter().enumerate() {
                    l[q + j] = (p / 8) as u32;
                    l[(rows_at + 32 * nt * c) / 4 + p] = (r + j) as u32;
                    pp[p] = -1;
                }
                let me = m as u64 | (e as u64) << 32;
                let (x, h, a) = (b.pf_dn.at(4 * n * r), b.pf_gu.at(8 * ffu * r), b.pf_af.at(4 * ffu * r));
                entries.push([bk.gu.ptr + slot * gu_b, x, h, me, bk.down.ptr + slot * dn_b, a, x, me]);
                let ch = &mut chunks[c];
                ch[1] += m;
                ch[3] += 1;
                ch[4] = ch[4].max(m);
                (q, k) = (q + m, k + m);
            }
        }
        for (p, &s) in pp.iter().enumerate() {
            l[8 * nt + p] = s as u32;
        }
        let t = b.grp_h.as_mut_slice::<u64>();
        let ne_ = entries.len();
        for (i, en) in entries.iter().enumerate() {
            t[tab_at / 8 + 4 * i..][..4].copy_from_slice(&en[..4]);
            t[tab_at / 8 + 4 * (ne_ + i)..][..4].copy_from_slice(&en[4..]);
        }
        cuda::copy_h2d_async(b.grp.ptr, b.grp_h.ptr, tab_at + 64 * ne_, &self.s)?;
        let ((gu_a, gu_d), (dn_a, dn_d)) = (self.cache.factor_ptrs(ip, 0), self.cache.factor_ptrs(ip, 1));
        let (nu, fu) = (n as u32, ffu as u32);
        for (c, &[q0, m, e0, nz, mm]) in chunks.iter().enumerate() {
            let tok = b.grp.at(4 * q0);
            let (m, nz, mm) = (m as u32, nz as u32, mm as u32);
            self.k.copy_rows(&self.s, b.xn.ptr, Some(tok), b.pf_dn.ptr, None, nu, m)?;
            self.k.moe_nvfp4_g(&self.s, b.grp.at(tab_at + 32 * e0), gu_a, gu_d, 2 * fu, nu, mm, nz)?;
            // the tokens stand for slots: SITU takes every row
            self.k.situ(&self.s, b.pf_gu.ptr, tok, b.pf_af.ptr, fu, self.ex.beta, self.ex.lb, m)?;
            self.k.moe_nvfp4_g(&self.s, b.grp.at(tab_at + 32 * (ne_ + e0)), dn_a, dn_d, nu, fu, mm, nz)?;
            self.k.moe_combine_r(&self.s, b.pf_dn.ptr, b.grp.at(rows_at + 32 * nt * c), b.wts.ptr, b.moe.ptr, nu, nt as u32, c > 0)?;
        }
        self.stats.pf_grouped_pairs += total as u64;
        Ok(Some(pp))
    }

    /// Times each kernel kind on layer 6's buffers: (name, microseconds per call, bytes read per call).
    pub fn kernel_bench(&mut self) -> Result<Vec<(String, f64, f64)>> {
        let hp = self.hp.clone();
        let n = hp.n_embd as u32;
        let reps = 50;
        // the attention cases read up to n_ctx keys
        self.plan_kv(self.n_ctx)?;
        let mut res = vec![];
        let ev0 = cuda::Event::new(true)?;
        let ev1 = cuda::Event::new(true)?;
        macro_rules! time {
            ($name:expr, $bytes:expr, $body:expr) => {{
                for _ in 0..3 {
                    $body;
                }
                ev0.record(&self.s)?;
                for _ in 0..reps {
                    $body;
                }
                ev1.record(&self.s)?;
                ev1.sync()?;
                let us = ev1.ms_since(&ev0)? as f64 * 1000.0 / reps as f64;
                res.push(($name.to_string(), us, $bytes as f64));
            }};
        }
        let x = self.sc.inputs.ptr;
        let l = &self.w.layers[6];
        let (wq, wo) = (&l.g0[0], &l.wo);
        time!("gemv_bf16 wq 3072x2304", wq.bytes(), self.gemv(wq, x, self.sc.g0.ptr)?);
        time!("gemv_bf16 wo 2304x3072", wo.bytes(), self.gemv(wo, x, self.sc.a.ptr)?);
        let o = &self.w.output;
        time!(format!("gemv_bf16 output {}x2304", o.rows), o.bytes(), self.gemv(o, x, self.sc.logits_raw.ptr)?);
        // prefill's, on T_MAX tokens: the weights' bytes count once a token, and as a weight is 2 bytes and 2 flops a
        // token, their GB/s are GFLOP/s
        let (b, nt) = (&self.b, T_MAX);
        for (name, m, x, y, ys) in [("wq", wq, b.xn.ptr, b.g0.ptr, l.g0_rows()), ("wk", &l.gk[0], b.xn.ptr, b.gk.ptr, l.gk.iter().map(|m| m.rows).sum::<usize>()), ("wo", wo, b.att2.ptr, b.a.ptr, wo.rows), ("router", &l.router, b.xn.ptr, b.rlog.ptr, l.router.rows)] {
            time!(format!("gemm_bf16 {name}, {nt} tokens"), nt * m.bytes(), self.gemm(m, x, y, nt, ys)?);
        }
        self.sc.slots.upload(&[0i32, 1, 2, 3, 4, 5, 6, 7])?;
        self.sc.ids.upload(&[0u32, 1, 2, 3, 4, 5, 6, 7])?;
        // an NVFP4 layer and the MXFP8 last one, on the first 8 slots of their banks
        for ip in [6, hp.n_phys - 1] {
            let f = self.ex.fmt[ip];
            let bk = &self.cache.banks[f.index()];
            if bk.n_slots >= 8 {
                time!(format!("experts {f:?} x8: gate_up, act, down"), 8 * (bk.gu_bytes + bk.down_bytes), self.experts_d(ip)?);
            }
        }
        // prefill's gate_up and down of one NVFP4 expert on m tokens, per pair (as the stream runs them) and grouped;
        // GB/s here are GFLOP/s (2 flops a weight and token)
        let bk = &self.cache.banks[Fmt::NvFp4.index()];
        if bk.n_slots >= 1 && matches!(self.ex.fmt[6], Fmt::NvFp4) {
            let (b, ffu, nn) = (&self.b, hp.n_ff_exp as u32, n as usize);
            let ((gu_a, gu_d), (dn_a, dn_d)) = (self.cache.factor_ptrs(6, 0), self.cache.factor_ptrs(6, 1));
            let (gu_b, dn_b) = (bk.gu_bytes as u64, bk.down_bytes as u64);
            b.slots.upload(&[0i32; T_MAX])?;
            b.ids.upload(&[0u32; T_MAX])?;
            for m in [24u32, 64, 256] {
                let flops = 2 * m as usize * 3 * nn * ffu as usize;
                time!(format!("NvFp4 expert per pair, {m} tokens"), flops, {
                    self.k.moe_nvfp4_d(&self.s, bk.gu.ptr, gu_b, b.slots.ptr, b.ids.ptr, gu_a, gu_d, b.xn.ptr, n, 0, b.pf_gu.ptr, 2 * ffu, n, m)?;
                    self.k.moe_nvfp4_d(&self.s, bk.down.ptr, dn_b, b.slots.ptr, b.ids.ptr, dn_a, dn_d, b.pf_af.ptr, ffu, 0, b.pf_dn.ptr, n, ffu, m)?
                });
                let tab = DevBuf::from_slice(&[bk.gu.ptr, b.xn.ptr, b.pf_gu.ptr, m as u64, bk.down.ptr, b.pf_af.ptr, b.pf_dn.ptr, m as u64])?;
                time!(format!("NvFp4 expert grouped, {m} tokens"), flops, {
                    self.k.moe_nvfp4_g(&self.s, tab.ptr, gu_a, gu_d, 2 * ffu, n, m, 1)?;
                    self.k.moe_nvfp4_g(&self.s, tab.at(32), dn_a, dn_d, n, ffu, m, 1)?
                });
            }
            // 8 cached experts on m tokens each, as a ubatch has them (pair 8t + k on expert k): per pair, and grouped
            // in one launch (experts_g); G_MIN is where grouped gets faster
            if bk.n_slots >= 8 {
                let sl: Vec<i32> = (0..T_MAX as i32).map(|p| p % 8).collect();
                let id: Vec<u32> = (0..T_MAX as u32).map(|p| p % 8).collect();
                b.slots.upload(&sl)?;
                b.ids.upload(&id)?;
                for m in [4u32, 8, 12, 16, 24, 32] {
                    let flops = 8 * 2 * m as usize * 3 * nn * ffu as usize;
                    time!(format!("NvFp4 8 experts per pair, {m} tokens"), flops, {
                        self.k.moe_nvfp4_d(&self.s, bk.gu.ptr, gu_b, b.slots.ptr, b.ids.ptr, gu_a, gu_d, b.xn.ptr, n, 3, b.pf_gu.ptr, 2 * ffu, n, 8 * m)?;
                        self.k.moe_nvfp4_d(&self.s, bk.down.ptr, dn_b, b.slots.ptr, b.ids.ptr, dn_a, dn_d, b.pf_af.ptr, ffu, 0, b.pf_dn.ptr, n, ffu, 8 * m)?
                    });
                    let mut t = vec![];
                    for k in 0..8u64 {
                        t.extend([bk.gu.ptr + k * gu_b, b.xn.ptr, b.pf_gu.at(8 * ffu as usize * m as usize * k as usize), m as u64 | k << 32]);
                    }
                    for k in 0..8u64 {
                        t.extend([bk.down.ptr + k * dn_b, b.pf_af.ptr, b.pf_dn.at(4 * nn * m as usize * k as usize), m as u64 | k << 32]);
                    }
                    let tab = DevBuf::from_slice(&t)?;
                    time!(format!("NvFp4 8 experts grouped, {m} tokens"), flops, {
                        self.k.moe_nvfp4_g(&self.s, tab.ptr, gu_a, gu_d, 2 * ffu, n, m, 8)?;
                        self.k.moe_nvfp4_g(&self.s, tab.at(256), dn_a, dn_d, n, ffu, m, 8)?
                    });
                }
            }
        }
        let maxs = self.maxs as u32;
        let mut cases = vec![(1024u32, (hp.n_swa + T_MAX) as u32, 6usize)];
        let mut keys = 4096;
        while keys <= self.n_ctx.min(self.kv_vram_rows) {
            cases.push((keys as u32, 0, 5));
            keys *= 2;
        }
        for &(keys, ring, kvi) in &cases {
            let p: [u32; 4] = [keys - 1, 1f32.to_bits(), 1f32.to_bits(), 0];
            self.params.upload(&p)?;
            let (window, cs) = if ring != 0 { (hp.n_swa as u32, CS_SWA) } else { (u32::MAX, CS_FULL) };
            let nsplit = if ring != 0 { hp.n_swa } else { self.n_ctx.min(self.kv_vram_rows) }.div_ceil(1 << cs) as u32;
            let [kc, vc] = self.kv(kvi);
            let lr = &self.w.layers[kvi];
            time!(format!("attn part+reduce {keys} keys"), keys as usize * 2 * KV_ROW, {
                self.k.attn_part(&self.s, self.sc.q.ptr, self.sc.r.ptr, self.sc.g0.ptr, lr.rel_proj.ptr, kc, vc, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, self.params.ptr, ring, 1024, 0.088, maxs, nsplit, window, cs, u32::MAX)?;
                self.k.attn_reduce(&self.s, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, self.sc.att.ptr, self.params.ptr, window, maxs, cs, hp.n_head as u32, 0, u32::MAX)?
            });
        }
        // prefill's attention on T_MAX tokens: a sliding layer's whole window, a full layer's keys up to each length
        // timed; GB/s here are GFLOP/s of q k and p v (24 heads of 128 dims: 12288 a key and token)
        let nh = hp.n_head * hp.head_dim;
        let sink = self.b.g0.at((2 * nh + hp.n_head * hp.n_rel) * 4);
        let (g0r, nr) = (self.w.layers[6].g0_rows() as u32, (hp.n_head * hp.n_rel) as u32);
        for &(keys, ring, kvi) in &cases {
            let (p0, window, extent) = if ring != 0 { (4096u32, hp.n_swa as u32, hp.n_swa as u32) } else { (keys - T_MAX as u32, u32::MAX, hp.rel_extent as u32) };
            self.params.upload(&[p0, 1f32.to_bits(), 1f32.to_bits(), 0])?;
            let pairs: u64 = (0..T_MAX as u64).map(|t| (p0 as u64 + t + 1).min(window as u64)).sum();
            let ([kc, vc], rb) = (self.kv(kvi), self.w.layers[kvi].rel_proj.ptr);
            let what = if ring != 0 { "sliding" } else { "full" };
            let (b, ntu, scale) = (&self.b, T_MAX as u32, 1.0 / (hp.head_dim as f32).sqrt());
            time!(format!("attn_prefill {what} to {}", p0 + ntu), pairs * 12288, {
                self.k.attn_prefill(&self.s, b.q.ptr, b.r.ptr, sink, rb, kc, vc, b.att.ptr, self.params.ptr, ring, window, extent, scale, nh as u32, nr, g0r, ntu, (0, u32::MAX), 0, 0)?
            });
        }
        let list = vec![self.bl.b0.ptr, self.bl.b6[0].ptr, self.bl.b12[0].ptr, self.sc.x.ptr];
        let pl = self.ptr_list(&list)?;
        let l = &self.w.layers[6];
        time!("mix 4 states 1 query", 4 * 4 * n as usize, self.k.mix(&self.s, pl, 4, l.ffn_res_query.ptr, self.sc.ffm.ptr, 1, n)?);
        time!("rms_norm 2304", 4 * n as usize, self.k.rms_norm(&self.s, self.sc.x.ptr, Some(l.ffn_norm.ptr), self.sc.xn.ptr, n, 1, hp.eps)?);
        time!("router logits+topk", l.router.bytes(), {
            self.gemv(&l.router, self.sc.xn.ptr, self.sc.rlog.ptr)?;
            self.k.router_topk(&self.s, self.sc.rlog.ptr, l.router_bias.ptr, self.sc.ids.ptr, self.sc.wts.ptr, 8.0)?
        });
        time!("conv4 2304", 4 * 8 * n as usize, self.k.conv4(&self.s, self.sc.a.ptr, l.post_conv.ptr, self.conv[6].ptr, None, self.sc.a2.ptr, n, 1, 0.4)?);
        time!("clip 24x128", 4 * 3072, self.k.clip_rows(&self.s, self.sc.g0.ptr, Some(l.q_clip.ptr), self.sc.q.ptr, 128, 24, 24, 22.6, self.params.at(8))?);
        time!("add 2304", 12 * n as usize, self.k.add(&self.s, self.sc.x.ptr, self.sc.f.ptr, self.sc.x.ptr, n)?);
        let lp = self.w.layers[6].lp.as_ref().unwrap();
        let d = &lp.down[0];
        time!("ctrl down 64x4608 (bf16)", d.bytes(), self.gemv(d, self.sc.ctl_g.ptr, self.sc.ctl_gate.ptr)?);
        let (gw, fw) = (lp.gain[0].ptr, lp.forget[0].ptr);
        time!("ctrl_out", 2 * 2 * n as usize * hp.gate_rank, self.k.ctrl_out(&self.s, self.sc.x.ptr, self.sc.msgs.ptr, self.sc.ctl_gate.ptr, gw, fw, self.sc.ffm2.ptr, n)?);
        time!("ctrl_in", 16 * n as usize, self.k.ctrl_in(&self.s, self.sc.x.ptr, self.sc.msgs.ptr, self.sc.ctl_g.ptr, n)?);
        Ok(res)
    }
}
