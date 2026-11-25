//! One sequence of Inkling-2-Mini: weights, caches and the decode step.
//! The GPU posts each MoE layer to the CPU team (cpu_pool) through host-mapped memory, so a token runs as one CUDA graph.

use std::collections::HashMap;
use std::sync::Arc;

use crate::cpu_pool::Pool;
use crate::cuda::{self, Ctx, DevBuf, Graph, HostBuf, Result, Stream, VmBuf};
use crate::experts::{ExpertCache, Experts};
use crate::gguf::Gguf;
use crate::hp::{Hp, Step};
use crate::kernels::{Kernels, P};
use crate::quant::dequant_q8_row;
use crate::weights::Weights;

/// The expert threads from IK_THREADS ("0,2,4" pins, "8" is 8 unpinned threads); default: 8 unpinned.
pub fn threads_from_env() -> Vec<Option<usize>> {
    match std::env::var("IK_THREADS") {
        Ok(v) if v.contains(',') => v.split(',').filter_map(|c| c.trim().parse().ok()).map(Some).collect(),
        Ok(v) => vec![None; v.trim().parse().unwrap_or(8)],
        Err(_) => vec![None; 8],
    }
}

/// Async cache uploads in flight, at most.
const STAGES: usize = 24;

/// The GPU computes the cached experts (else all experts run on the CPU, and the cache has no slots).
const GPU_EXPERTS: bool = true;
/// ... and in prefill too (else the prefill sends all experts to the CPU).
const GPU_PREFILL: bool = true;

/// Tokens of a prefill ubatch, at most.
pub const T_MAX: usize = 256;

/// The expert usage counts decay by this much at each token: an expert that every token uses counts about 100.
const DECAY: f32 = 0.99;

/// Bytes of a row (position) of the int8 KV cache: 8 kv heads of 128 int8, and a f16 scale per 32.
const KV_ROW: usize = 1024 + 64;

/// IK_KV_F16: the f16 KV cache of before (2048 bytes a row), to compare.
fn kv_f16() -> bool {
    std::env::var("IK_KV_F16").is_ok()
}

/// Whether the full-attention KV caches are int4 (K and V, the default) or int8 (IK_KV=8); the sliding-window rings
/// stay int8. Perplexity over 16k tokens of the model's reasoning: 1.6984 int8, 1.7058 int4 (+0.44%); the caches take
/// 576 bytes a row and layer pass instead of 1088, and full attention over 131k keys 519 us instead of 727.
fn kv_int4_env() -> bool {
    !kv_f16() && std::env::var("IK_KV").as_deref() != Ok("8")
}

/// Bytes a row of the 4 parts of a full-attention cache take: K values, K scales, V values, V scales.
fn kv_widths(int4: bool) -> [usize; 4] {
    let v = if int4 { 512 } else { 1024 };
    [v, 64, v, 64]
}

/// Bytes of a KV cache row.
fn kv_row() -> usize {
    if kv_f16() { 2048 } else { KV_ROW }
}
/// The full-attention KV caches get memory in steps of this many rows (a 2 MiB page of int8 values).
const KV_STEP: usize = 2048;

/// Decode attention: keys per chunk, log2, of the sliding window layers and of the full ones.
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
    cq: DevBuf, // the inputs of 3 controllers, q8_1
    cd: DevBuf,
    cs: DevBuf,
    ctl_gate3: DevBuf, // [64]
    msgs: DevBuf,    // [4][n]
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
    xq: DevBuf,  // q8_1 of a GEMV input: int8 [4608]
    xd: DevBuf,  // [144]
    xs: DevBuf,  // [144]
    xq2: DevBuf, // a second q8_1 input (k and v of full attention)
    xd2: DevBuf,
    xs2: DevBuf,
    xq3: DevBuf,
    xd3: DevBuf,
    xs3: DevBuf,
    aq: DevBuf, // q8_1 of the expert activations [8][n_ff]
    ad: DevBuf,
    asum: DevBuf,
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
    xq: Vec<DevBuf>,
    xd: Vec<DevBuf>,
    xs: Vec<DevBuf>,
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
    groups: DevBuf,
    gu: DevBuf,
    aq: DevBuf,
    ad: DevBuf,
    asum: DevBuf,
    dn: DevBuf,
    moe: DevBuf,
    f: DevBuf,
    tau: DevBuf,
    one: DevBuf,
    bl: Blocks,
    host_ids: HostBuf,
    host_xn: HostBuf,
    host_groups: HostBuf,
    host_slots: HostBuf,
    cpu_out: HostBuf,
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
    pub g: Arc<Gguf>,
    pub hp: Hp,
    pub ctx: Ctx,
    pub k: Kernels,
    pub w: Weights,
    pub ex: Arc<Experts>,
    pub cache: ExpertCache,
    pub s: Stream,
    pub n_ctx: usize,
    sched: Vec<Step>,
    kcache: Vec<Option<DevBuf>>, // int8 rows [rows][8][128], then their f16 scales [rows][8][4]; the SWA rings (all with IK_KV_F16)
    vcache: Vec<Option<DevBuf>>,
    /// the full-attention caches: int8 K, its scales, int8 V, its scales, at addresses for n_ctx rows; memory for kv_mapped
    kv_vm: Vec<Option<[VmBuf; 4]>>,
    kv_mapped: usize,
    /// the full-attention caches are int4 (kv_int4_env)
    kv_int4: bool,
    /// VRAM for the expert cache and the mapped rows of kv_vm together
    vram_budget: usize,
    kv_rows: Vec<usize>,
    ring: Vec<u32>,
    conv: Vec<DevBuf>, // per slot: k [kv][3], v [kv][3], post [n][3], ffn [n][3]
    sc: Scratch,
    bl: Blocks,
    params: DevBuf,  // pos u32, tau f32, one f32
    ptrs: DevBuf,    // pointer lists of the mix calls
    ptr_lists: HashMap<Vec<u64>, usize>,
    ptr_used: usize,
    pass_bank: (Vec<P>, P),
    /// debug: called with (name, slot, data) for the intermediates that llama.cpp names, if set
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
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub tokens: u64,
    /// decode: misses of experts, and those of an expert that had missed earlier in the same token
    pub misses_all: usize,
    pub misses_repeat: usize,
    pub gpu_experts: u64,
    pub cpu_experts: u64,
    /// prefill: seconds waiting for the GPU up to the router, grouping on the host, in the CPU experts, after them
    pub pf_gpu_s: f64,
    pub pf_host_s: f64,
    pub pf_cpu_s: f64,
    pub pf_tail_s: f64,
    pub pf_cpu_pairs: u64,
    pub pf_gpu_pairs: u64,
}

impl Engine {
    pub fn new(path: &std::path::Path, opt: Options) -> Result<Engine> {
        let ctx = Ctx::new(0)?;
        let g = Arc::new(Gguf::open(path)?);
        let hp = Hp::from_gguf(&g)?;
        let k = Kernels::load()?;
        let w = Weights::load(&g, &hp)?;
        let ex = Arc::new(Experts::new(&g, &hp)?);
        let n_swa = hp.n_swa;
        let s = Stream::new()?;
        let n = hp.n_embd;
        let kv = hp.kv_dim();
        let f = |len: usize| DevBuf::zeroed(len * 4, &s);

        let n_slots = hp.n_slots();
        let mut kcache = vec![];
        let mut vcache = vec![];
        let mut kv_vm = vec![];
        let kv_int4 = kv_int4_env();
        let mut kv_rows = vec![];
        let mut ring = vec![];
        let mut conv = vec![];
        for il in 0..n_slots {
            let ip = Self::phys(&hp, il);
            let rows = if hp.is_swa[ip] { hp.n_swa + T_MAX } else { opt.n_ctx };
            if hp.is_swa[ip] || kv_f16() {
                kcache.push(Some(DevBuf::zeroed(rows * kv_row(), &s)?));
                vcache.push(Some(DevBuf::zeroed(rows * kv_row(), &s)?));
                kv_vm.push(None);
            } else {
                // memory comes as the context grows (plan_kv)
                let r = |b: usize| VmBuf::reserve(rows.div_ceil(KV_STEP) * KV_STEP * b, ctx.device);
                kcache.push(None);
                vcache.push(None);
                let [kw, ksw, vw, vsw] = kv_widths(kv_int4);
                kv_vm.push(Some([r(kw)?, r(ksw)?, r(vw)?, r(vsw)?]));
            }
            kv_rows.push(rows);
            ring.push(if hp.is_swa[ip] { (hp.n_swa + T_MAX) as u32 } else { 0 });
            conv.push(f((2 * kv + 2 * n) * 3)?);
        }
        let maxs = (opt.n_ctx >> CS_FULL).max(hp.n_swa >> CS_SWA) + 1;
        let sc = Scratch {
            x: f(n)?,
            inp_sa: f(n)?,
            mixed: f(3 * n)?,
            inputs: f(3 * n)?,
            ctl_tmp: f(3 * n)?,
            ctl_g: f(2 * n)?,
            ctl_gate: f(64)?,
            cq: f(3 * 4608 / 4)?,
            cd: f(3 * 144)?,
            cs: f(3 * 144)?,
            ctl_gate3: f(3 * 64)?,
            msgs: f(4 * n)?,
            g0: f(w.layers[0].g0.rows)?,
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
            xq: f(4608 / 4)?,
            xd: f(144)?,
            xs: f(144)?,
            xq2: f(4608 / 4)?,
            xd2: f(144)?,
            xs2: f(144)?,
            xq3: f(4608 / 4)?,
            xd3: f(144)?,
            xs3: f(144)?,
            aq: f(8 * hp.n_ff_exp / 4)?,
            ad: f(8 * hp.n_ff_exp / 32)?,
            asum: f(8 * hp.n_ff_exp / 32)?,
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
            xq: (0..3).map(|_| f(tm * 4608 / 4)).collect::<Result<_>>()?,
            xd: (0..3).map(|_| f(tm * 144)).collect::<Result<_>>()?,
            xs: (0..3).map(|_| f(tm * 144)).collect::<Result<_>>()?,
            g0: f(tm * w.layers[0].g0.rows)?,
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
            groups: f(384 + 2 * tm * 8)?,
            gu: f(tm * 8 * 2 * hp.n_ff_exp)?,
            aq: f(tm * 8 * hp.n_ff_exp / 4)?,
            ad: f(tm * 8 * hp.n_ff_exp / 32)?,
            asum: f(tm * 8 * hp.n_ff_exp / 32)?,
            dn: f(tm * 8 * n)?,
            moe: f(tm * n)?,
            f: f(tm * n)?,
            tau: f(tm)?,
            one: DevBuf::from_slice(&vec![1f32; tm])?,
            bl: blocks_t,
            host_ids: HostBuf::new(4 * tm * 8)?,
            host_xn: HostBuf::new(4 * tm * n)?,
            host_groups: HostBuf::new(4 * (384 + 2 * tm * 8))?,
            host_slots: HostBuf::new(4 * tm * 8)?,
            cpu_out: HostBuf::new(4 * tm * 8 * n)?,
        };
        // the expert cache and the full-attention KV rows share what is free once the rest exists, less a margin for
        // the CUDA graph and the driver (IK_VRAM_MARGIN_MIB, default 160); the KV rows take memory as the context grows
        // and give it back when a new sequence is shorter (plan_kv)
        let (free, _) = ctx.mem_info()?;
        let margin = std::env::var("IK_VRAM_MARGIN_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(160) << 20;
        let vram_budget = free.saturating_sub(margin);
        let max_slots = if GPU_EXPERTS { (vram_budget / (ex.gu_bytes[0] + ex.down_bytes[0])).min(opt.cache_slots) } else { 0 };
        let cache = ExpertCache::new(&ex, &hp, max_slots, ctx.device)?;
        s.sync()?;
        let job = HostBuf::new(4 * (32 + n))?;
        let ack = HostBuf::new(64)?;
        let err = HostBuf::new(64)?;
        let cpu_out = HostBuf::new(4 * 8 * n)?;
        let threads = if opt.threads.is_empty() { vec![None] } else { opt.threads.clone() };
        let pool = Pool::new(ex.clone(), &job, &ack, &cpu_out, hp.n_phys, hp.n_expert, threads);
        let mut e = Engine {
            params: DevBuf::zeroed(16, &s)?,
            ptrs: DevBuf::zeroed(8 * 4096, &s)?,
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
            g,
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
            kv_int4,
            kv_mapped: 0,
            vram_budget,
            kv_rows,
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
            snap_swa: HostBuf::new(n_slots * 2 * n_swa * kv_row())?,
            copy_s: Stream::new_nonblocking()?,
        };
        let need = e.kv_bytes(e.n_ctx);
        if need > e.vram_budget {
            let per = e.kv_bytes(KV_STEP * 64) / (KV_STEP * 64);
            return Err(format!(
                "a context of {} tokens needs {} MiB of KV cache, more than the {} MiB of VRAM there are for it and the experts: {} tokens at most",
                e.n_ctx,
                need >> 20,
                e.vram_budget >> 20,
                e.vram_budget / per.max(1)
            ));
        }
        e.plan_kv(T_MAX)?;
        Ok(e)
    }

    /// Bytes of the full-attention KV caches with `rows` rows mapped.
    fn kv_bytes(&self, rows: usize) -> usize {
        self.kv_vm.iter().flatten().map(|b| b.iter().zip(kv_widths(self.kv_int4)).map(|(v, w)| v.pages_for(rows * w)).sum::<usize>()).sum()
    }

    /// Gives the full-attention KV caches memory for `rows` positions (rounded up to KV_STEP) and the expert cache the
    /// rest of the budget. As the context grows the last slots go, with their experts; when a new sequence is shorter
    /// they come back empty and fill as experts are used.
    fn plan_kv(&mut self, rows: usize) -> Result<()> {
        let rows = (rows.div_ceil(KV_STEP) * KV_STEP).min(self.n_ctx.div_ceil(KV_STEP) * KV_STEP);
        let kv = self.kv_bytes(rows);
        if kv > self.vram_budget {
            return Err(format!("not enough VRAM for the KV cache of {rows} tokens"));
        }
        let slots = self.cache.fit(self.vram_budget - kv);
        if rows == self.kv_mapped && slots == [self.cache.banks[0].n_slots, self.cache.banks[1].n_slots] {
            return Ok(());
        }
        // nothing in flight may touch the pages that go: finish the token's work and the uploads
        self.s.sync()?;
        self.copy_s.sync()?;
        self.join_uploads();
        let resize = |e: &mut Engine, rows: usize| -> Result<()> {
            let widths = kv_widths(e.kv_int4);
            for b in e.kv_vm.iter_mut().flatten() {
                for (v, w) in b.iter_mut().zip(widths) {
                    v.resize(rows * w, &e.s)?;
                }
            }
            Ok(())
        };
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

    /// Finished async uploads join the cache; returns whether any did.
    fn join_uploads(&mut self) -> bool {
        let mut changed = false;
        let mut k = 0;
        while k < self.cache.pending.len() {
            if self.cache.pending[k].ev.done() {
                let p = self.cache.pending.swap_remove(k);
                self.cache.slot_of[p.layer][p.e] = p.slot as i32;
                self.cache.banks[p.bank].owner[p.slot] = (p.layer, p.e);
                self.cache.banks[p.bank].busy[p.slot] = false;
                changed = true;
            } else {
                k += 1;
            }
        }
        changed
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

    /// y = W x with x quantized to q8_1 in the buffers `which` (0..3)
    fn qgemv(&self, m: &crate::weights::Q8Mat, x: P, y: P, which: usize, quantize: bool) -> Result<()> {
        let (q, d, s) = match which {
            0 => (&self.sc.xq, &self.sc.xd, &self.sc.xs),
            1 => (&self.sc.xq2, &self.sc.xd2, &self.sc.xs2),
            _ => (&self.sc.xq3, &self.sc.xd3, &self.sc.xs3),
        };
        if quantize {
            self.k.quant_q8_1(&self.s, x, q.ptr, d.ptr, s.ptr, m.k as u32, 1)?;
        }
        self.k.gemv_q8d(&self.s, m.qs.ptr, m.d.ptr, q.ptr, d.ptr, y, m.rows as u32, m.k as u32)
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
        let tab = self.ptr_list(&[cur, msg])?;
        self.k.ctrl_in_qz(&self.s, tab, self.sc.xq.ptr, self.sc.xd.ptr, self.sc.xs.ptr, n, 1)?;
        let lp = self.w.layers[ip].lp.as_ref().unwrap();
        let d = &lp.down[c];
        self.qgemv(d, self.sc.ctl_g.ptr, self.sc.ctl_gate.ptr, 0, false)?;
        let (gm, fm) = (&lp.gain[c], &lp.forget[c]);
        self.k.ctrl_out(&self.s, cur, msg, self.sc.ctl_gate.ptr, gm.qs.ptr, gm.d.ptr, fm.qs.ptr, fm.d.ptr, out, n)
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

    /// Whether slot il's KV cache is int4: the full-attention ones with kv_int4; the sliding-window rings are int8.
    fn int4_at(&self, il: usize) -> bool {
        self.kv_int4 && self.kv_vm[il].is_some()
    }

    /// The int8 values and the scales of slot il's K and V caches.
    fn kv8(&self, il: usize) -> [P; 4] {
        if let Some(b) = &self.kv_vm[il] {
            return [b[0].ptr, b[1].ptr, b[2].ptr, b[3].ptr];
        }
        let (k, v) = (self.kcache[il].as_ref().unwrap(), self.vcache[il].as_ref().unwrap());
        let q = self.kv_rows[il] * 1024;
        [k.ptr, k.ptr + q as u64, v.ptr, v.ptr + q as u64]
    }

    /// The (offset, bytes a row) in a cache of `rows` rows of row `row`'s int8 values and of its scales (or its f16 row).
    fn kv_parts(rows: usize, row: usize) -> Vec<(usize, usize)> {
        if kv_f16() { vec![(row * 2048, 2048)] } else { vec![(row * 1024, 1024), (rows * 1024 + row * 64, 64)] }
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
                // the window's int8 rows, then its scales
                let base = (il * 2 + ci) * w * kv_row();
                for (row, off, cnt) in self.window_rows(ring) {
                    let mut at = base;
                    for (o, b) in Self::kv_parts(ring, row) {
                        // SAFETY: inside the page-locked snapshot buffer
                        let dst = unsafe { self.snap_swa.ptr.add(at + off * b) };
                        cuda::copy_d2h_async(dst, cache.at(o), cnt * b, &self.s)?;
                        at += w * b;
                    }
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
                let base = (il * 2 + ci) * w * kv_row();
                for (row, off, cnt) in self.window_rows(ring) {
                    let mut at = base;
                    for (o, b) in Self::kv_parts(ring, row) {
                        // SAFETY: inside the page-locked snapshot buffer
                        let src = unsafe { self.snap_swa.ptr.add(at + off * b) };
                        cuda::copy_h2d_async(cache.at(o), src, cnt * b, &self.s)?;
                        at += w * b;
                    }
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
        // embedding row, dequantized on the host
        let te = self.g.tensor("token_embd.weight")?;
        let row = te.ty.row_bytes(n);
        let mut emb = vec![0f32; n];
        dequant_q8_row(&self.g.mmap[te.offset + token as usize * row..te.offset + (token as usize + 1) * row], &mut emb);
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
        let o = &self.w.output;
        self.qgemv(o, self.sc.a.ptr, self.sc.logits_raw.ptr, 0, true)?;
        self.k.softcap(&self.s, self.sc.logits_raw.ptr, self.logits_h.dev, hp.n_vocab as u32, hp.softcap)?;
        self.k.prof_stop();
        Ok(())
    }

    /// Between tokens: uploads experts that miss often to the GPU cache, in place of experts used less.
    /// With page-locked experts the uploads are async (copy stream); a slot is mapped once its upload is done.
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
        // finished uploads join the cache
        let mut changed = self.join_uploads();
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
            if self.cache.slot_of[l][e] >= 0 || self.cache.pending.iter().any(|p| p.layer == l && p.e == e) {
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
            if self.ex.pinned {
                if self.cache.pending.len() >= STAGES {
                    break;
                }
                // unmap the victim now; the next token's graph reads the new map
                if !free {
                    self.cache.slot_of[ol][oe] = -1;
                    changed = true;
                }
                let b = &mut self.cache.banks[bank];
                b.owner[v] = (usize::MAX, usize::MAX);
                b.busy[v] = true;
                // the host copy is in the slot layout: one copy each
                let (gu, dn) = (self.ex.gate_up(l, e).b, self.ex.down(l, e).b);
                cuda::copy_h2d_async(b.gu.at(v * b.gu_bytes), gu.as_ptr(), gu.len(), &self.copy_s)?;
                cuda::copy_h2d_async(b.down.at(v * b.down_bytes), dn.as_ptr(), dn.len(), &self.copy_s)?;
                let ev = cuda::Event::new(false)?;
                ev.record(&self.copy_s)?;
                self.cache.pending.push(crate::experts::Pending { bank, slot: v, layer: l, e, ev });
                self.cache.uploads += 1;
            } else {
                self.cache.fill(&self.ex, v, l, e)?;
                changed = true;
            }
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
        let te = self.g.tensor("token_embd.weight")?;
        let row = te.ty.row_bytes(n);
        let mut emb = vec![0f32; nt * n];
        for (i, &t) in toks.iter().enumerate() {
            dequant_q8_row(&self.g.mmap[te.offset + t as usize * row..te.offset + (t as usize + 1) * row], &mut emb[i * n..(i + 1) * n]);
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
        let o = &self.w.output;
        self.qgemv(o, self.sc.a.ptr, self.sc.logits_raw.ptr, 0, true)?;
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

    /// y[t] = W x[t]: x quantized into the batch's q8_1 buffers `which`
    fn qgemm(&self, m: &crate::weights::Q8Mat, x: P, y: P, which: usize, quantize: bool, nt: usize) -> Result<()> {
        let (q, d, sx) = (&self.b.xq[which], &self.b.xd[which], &self.b.xs[which]);
        if quantize {
            self.k.quant_q8_1(&self.s, x, q.ptr, d.ptr, sx.ptr, m.k as u32, nt as u32)?;
        }
        self.k.gemm_q8d_t(&self.s, m.qs.ptr, m.d.ptr, q.ptr, d.ptr, y, m.rows as u32, m.k as u32, nt as u32, m.rows as u32)
    }

    fn control_t(&mut self, ip: usize, c: usize, cur: P, msg: P, out: P, nt: usize) -> Result<()> {
        let n = self.hp.n_embd as u32;
        self.k.ctrl_in_t(&self.s, cur, msg, self.b.ctl_g.ptr, n, nt as u32)?;
        let lp = self.w.layers[ip].lp.as_ref().unwrap();
        self.qgemm(&lp.down[c], self.b.ctl_g.ptr, self.b.ctl_gate.ptr, 0, true, nt)?;
        let (gm, fm) = (&lp.gain[c], &lp.forget[c]);
        self.k.ctrl_out_t(&self.s, cur, msg, self.b.ctl_gate.ptr, gm.qs.ptr, gm.d.ptr, fm.qs.ptr, fm.d.ptr, out, n, nt as u32)
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
            let (bank, out) = self.pass_bank_t.clone();
            let mq = self.w.layers[ip].lp.as_ref().unwrap().msg_query.ptr;
            for j in 0..4 {
                let dst = self.b.msg[j].ptr;
                self.mix_t(&bank, out, mq + (j * n * 4) as u64, dst, nt)?;
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
        let l = &self.w.layers[ip];
        let (g0, gk, gv) = (&l.g0, &l.gk, &l.gv);
        let shared_in = nin == 1;
        self.qgemm(g0, ins[0], self.b.g0.ptr, 0, true, nt)?;
        self.qgemm(gk, ins[1], self.b.gk.ptr, if shared_in { 0 } else { 1 }, !shared_in, nt)?;
        self.qgemm(gv, ins[2], self.b.gv.ptr, if shared_in { 0 } else { 2 }, !shared_in, nt)?;
        let g0r = g0.rows as u32;
        let (gkr, gvr) = (gk.rows as u32, gv.rows as u32);
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
        let (kv8, ring) = (self.kv8(il), self.ring[il]);
        if kv_f16() {
            self.k.kv_store_t(&self.s, self.b.k.ptr, self.b.v.ptr, kv8[0], kv8[2], self.params.ptr, ring, kv as u32, kv as u32, kv as u32, ntu)?;
        } else {
            if self.int4_at(il) {
                self.k.kv_store4_t(&self.s, self.b.k.ptr, self.b.v.ptr, kv8, self.params.ptr, kv as u32, kv as u32, ntu)?;
            } else {
                self.k.kv_store8_t(&self.s, self.b.k.ptr, self.b.v.ptr, kv8, self.params.ptr, ring, kv as u32, kv as u32, ntu)?;
            }
        }
        let extent = if swa { hp.n_swa } else { hp.rel_extent } as u32;
        let window = if swa { hp.n_swa as u32 } else { u32::MAX };
        let scale = 1.0 / (hp.head_dim as f32).sqrt();
        if kv_f16() {
            self.k.attn_prefill(&self.s, self.b.q.ptr, self.b.r.ptr, sink, l.rel_bias.ptr, kv8[0], kv8[2], self.b.att.ptr, self.params.ptr, ring, window, extent, scale, nh as u32, nhd * nr, g0r, ntu)?;
        } else {
            let int4 = self.int4_at(il);
            let l = &self.w.layers[ip];
            self.k.attn_prefill8(&self.s, self.b.q.ptr, self.b.r.ptr, sink, l.rel_bias.ptr, kv8, self.b.att.ptr, self.params.ptr, ring, window, extent, scale, nh as u32, nhd * nr, g0r, ntu, int4)?;
        }
        self.probe("kqv_out", il, self.b.att.ptr, nt * nh)?;
        let gate = self.b.g0.at(nh * 4);
        self.k.mul_sigmoid_t(&self.s, self.b.att.ptr, gate, self.b.att2.ptr, ntu * nh as u32, nh as u32, g0r)?;
        let l = &self.w.layers[ip];
        self.qgemm(&l.wo, self.b.att2.ptr, self.b.a.ptr, 0, true, nt)?;
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
        self.k.router_logits_t(&self.s, l.router.ptr, self.b.xn.ptr, self.b.rlog.ptr, hp.n_expert as u32, nu, ntu)?;
        self.k.router_topk_t(&self.s, self.b.rlog.ptr, l.probs_b.ptr, self.b.ids.ptr, self.b.wts.ptr, hp.w_scale, ntu)?;
        self.probe("ffn_norm_in", il, self.b.xn.ptr, nt * n)?;
        self.moe_t(ip, nt)?;
        self.probe("ffn_moe_out", il, self.b.moe.ptr, nt * n)?;
        let l = &self.w.layers[ip];
        self.k.conv4_seq(&self.s, self.b.moe.ptr, l.ffn_post_conv.ptr, st_ffn, None, self.b.f.ptr, nu, 1, pscale, ntu, 0, nu)?;
        self.k.add(&self.s, x, self.b.f.ptr, x, ntu * nu)?;
        self.probe("l_out", il, x, nt * n)?;
        Ok(())
    }

    /// The experts of a ubatch: the host groups the (token, expert) pairs by expert; the cached experts run as
    /// grouped GEMMs on the GPU while the CPU team computes the others.
    fn moe_t(&mut self, ip: usize, nt: usize) -> Result<()> {
        let n = self.hp.n_embd;
        let nu = n as u32;
        let ffu = self.hp.n_ff_exp as u32;
        self.k.quant_q8_1(&self.s, self.b.xn.ptr, self.b.xq[0].ptr, self.b.xd[0].ptr, self.b.xs[0].ptr, nu, nt as u32)?;
        cuda::copy_d2h_async(self.b.host_ids.ptr, self.b.ids.ptr, nt * 8 * 4, &self.s)?;
        cuda::copy_d2h_async(self.b.host_xn.ptr, self.b.xn.ptr, nt * n * 4, &self.s)?;
        let t0 = std::time::Instant::now();
        self.s.sync()?;
        let t1 = std::time::Instant::now();
        self.stats.pf_gpu_s += (t1 - t0).as_secs_f64();
        let ids: Vec<u32> = self.b.host_ids.as_slice::<u32>()[..nt * 8].to_vec();
        // group the pairs: cached experts by slot, the others by expert
        let mut gpu_groups: Vec<(i32, Vec<(usize, usize)>)> = vec![];
        let mut cpu_groups: Vec<(usize, Vec<(usize, usize)>)> = vec![];
        let mut gi: HashMap<i32, usize> = HashMap::new();
        let mut ci: HashMap<usize, usize> = HashMap::new();
        let slots_h = self.b.host_slots.as_mut_slice::<i32>();
        let slot_of: Vec<i32> = if GPU_PREFILL { self.cache.slot_of[ip].clone() } else { vec![-1; self.hp.n_expert] };
        for (p, &e) in ids.iter().enumerate() {
            let e = e as usize;
            let slot = slot_of[e];
            slots_h[p] = slot;
            let t = p / 8;
            if slot >= 0 {
                let g = *gi.entry(slot).or_insert_with(|| {
                    gpu_groups.push((slot, vec![]));
                    gpu_groups.len() - 1
                });
                gpu_groups[g].1.push((t, p));
            } else {
                let c = *ci.entry(e).or_insert_with(|| {
                    cpu_groups.push((e, vec![]));
                    cpu_groups.len() - 1
                });
                cpu_groups[c].1.push((t, p));
            }
        }
        if let Ok(mut u) = self.pool.usage.lock() {
            for &e in &ids {
                if slot_of[e as usize] >= 0 {
                    u.freq[ip][e as usize] += 1.0;
                }
            }
            u.hits += ids.iter().filter(|&&e| slot_of[e as usize] >= 0).count() as u64;
            u.miss_count += ids.iter().filter(|&&e| slot_of[e as usize] < 0).count() as u64;
        }
        let gh = self.b.host_groups.as_mut_slice::<u32>();
        let mut off = 0;
        for (g, (slot, toks)) in gpu_groups.iter().enumerate() {
            gh[3 * g] = *slot as u32;
            gh[3 * g + 1] = toks.len() as u32;
            gh[3 * g + 2] = off as u32;
            for (i, &(t, p)) in toks.iter().enumerate() {
                gh[384 + 2 * (off + i)] = t as u32;
                gh[384 + 2 * (off + i) + 1] = p as u32;
            }
            off += toks.len();
        }
        let ng = gpu_groups.len();
        cuda::copy_h2d_async(self.b.slots.ptr, self.b.host_slots.ptr, nt * 8 * 4, &self.s)?;
        if ng > 0 && GPU_PREFILL {
            cuda::copy_h2d_async(self.b.groups.ptr, self.b.host_groups.ptr, (384 + 2 * off) * 4, &self.s)?;
            let tiles = gpu_groups.iter().map(|g| g.1.len()).max().unwrap_or(1).div_ceil(32) as u32;
            let f = self.ex.fmt[ip];
            let bk = &self.cache.banks[f.index()];
            self.k.moe_g(&self.s, f, false, bk.gu.ptr, bk.gu_bytes as u64, self.b.groups.ptr, self.b.xq[0].ptr, self.b.xd[0].ptr, self.b.gu.ptr, 2 * ffu, nu, ng as u32, tiles)?;
            self.k.situ_q8_n(&self.s, self.b.gu.ptr, self.b.aq.ptr, self.b.ad.ptr, self.b.asum.ptr, ffu, self.ex.beta, self.ex.lb, (nt * 8) as u32)?;
            self.k.moe_g(&self.s, f, true, bk.down.ptr, bk.down_bytes as u64, self.b.groups.ptr, self.b.aq.ptr, self.b.ad.ptr, self.b.dn.ptr, nu, ffu, ng as u32, tiles)?;
        }
        // the misses on the CPU, meanwhile
        self.stats.pf_gpu_pairs += gpu_groups.iter().map(|g| g.1.len() as u64).sum::<u64>();
        self.stats.pf_cpu_pairs += cpu_groups.iter().map(|g| g.1.len() as u64).sum::<u64>();
        let t2 = std::time::Instant::now();
        let mut t3 = t2;
        if !cpu_groups.is_empty() {
            let xn = self.b.host_xn.as_slice::<f32>();
            let mut need = vec![false; nt];
            for (_, toks) in &cpu_groups {
                for &(t, _) in toks {
                    need[t] = true;
                }
            }
            let xq: Vec<crate::cpu::Q8_0> = (0..nt)
                .map(|t| {
                    let mut q = crate::cpu::Q8_0::default();
                    if need[t] {
                        crate::cpu::quantize_q8_0(&xn[t * n..(t + 1) * n], &mut q);
                    }
                    q
                })
                .collect();
            t3 = std::time::Instant::now();
            self.pool.run_batch(crate::cpu_pool::BatchJob { layer: ip, xq, misses: cpu_groups, out: self.b.cpu_out.ptr as usize });
        }
        let t4 = std::time::Instant::now();
        self.stats.pf_host_s += (t3 - t1).as_secs_f64();
        self.stats.pf_cpu_s += (t4 - t3).as_secs_f64();
        self.k.moe_combine_t(&self.s, self.b.dn.ptr, self.b.cpu_out.dev, self.b.slots.ptr, self.b.wts.ptr, self.b.moe.ptr, nu, nt as u32)?;
        // the cache keeps up during long prompts: uploads run on the copy stream meanwhile
        if self.ex.pinned {
            self.admit_d(false)?;
        }
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
            let mq = self.w.layers[ip].lp.as_ref().unwrap().msg_query.ptr;
            self.mix(&bank, out, mq, self.sc.msgs.ptr, 4)?;
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
            // the 3 input controllers at once
            let lp = self.w.layers[ip].lp.as_ref().unwrap();
            let mut t_in = vec![];
            let mut t_w = vec![];
            let mut t_out = vec![];
            for j in 0..3 {
                let (msg, out) = (self.sc.msgs.at(j * n * 4), self.sc.ctl_tmp.at(j * n * 4));
                t_in.extend([ins[j], msg]);
                t_w.extend([lp.down[j].qs.ptr, lp.down[j].d.ptr]);
                t_out.extend([ins[j], msg, lp.gain[j].qs.ptr, lp.gain[j].d.ptr, lp.forget[j].qs.ptr, lp.forget[j].d.ptr, out]);
            }
            let rows = lp.down[0].rows as u32;
            let (t_in, t_w, t_out) = (self.ptr_list(&t_in)?, self.ptr_list(&t_w)?, self.ptr_list(&t_out)?);
            self.k.ctrl_in_qz(&self.s, t_in, self.sc.cq.ptr, self.sc.cd.ptr, self.sc.cs.ptr, nu, 3)?;
            self.k.gemv_q8dz(&self.s, t_w, self.sc.cq.ptr, self.sc.cd.ptr, self.sc.ctl_gate3.ptr, rows, 2 * nu, 3)?;
            self.k.ctrl_outz(&self.s, t_out, self.sc.ctl_gate3.ptr, nu, 3)?;
            for (j, inp) in ins.iter_mut().enumerate() {
                *inp = self.sc.ctl_tmp.at(j * n * 4);
            }
        }
        let nin = if swa && !looped { 1 } else { 3 };
        for j in 0..nin {
            let w = self.w.layers[ip].in_norms[if swa { 0 } else { j }].ptr;
            let out = self.sc.inputs.at(j * n * 4);
            let (q, d, sx) = match j {
                0 => (&self.sc.xq, &self.sc.xd, &self.sc.xs),
                1 => (&self.sc.xq2, &self.sc.xd2, &self.sc.xs2),
                _ => (&self.sc.xq3, &self.sc.xd3, &self.sc.xs3),
            };
            self.k.norm_q8(&self.s, ins[j], w, out, q.ptr, d.ptr, sx.ptr, nu, hp.eps, 1, nu)?;
        }

        // projections
        let l = &self.w.layers[ip];
        let (g0, gk, gv) = (&l.g0, &l.gk, &l.gv);
        let shared_in = nin == 1;
        // g0, gk and gv in one launch
        let xb = |w: usize| match w {
            0 => (self.sc.xq.ptr, self.sc.xd.ptr),
            1 => (self.sc.xq2.ptr, self.sc.xd2.ptr),
            _ => (self.sc.xq3.ptr, self.sc.xd3.ptr),
        };
        let mut tab = vec![];
        for (j, (m, y)) in [(g0, self.sc.g0.ptr), (gk, self.sc.gk.ptr), (gv, self.sc.gv.ptr)].into_iter().enumerate() {
            let (xq, xd) = xb(if shared_in { 0 } else { j });
            tab.extend([m.qs.ptr, m.d.ptr, xq, xd, y, m.rows as u64]);
        }
        let (max_rows, k) = (g0.rows.max(gk.rows).max(gv.rows) as u32, g0.k as u32);
        let tab = self.ptr_list(&tab)?;
        self.k.gemv_q8dt(&self.s, tab, max_rows, k, 3)?;
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
        let (kv8, ring) = (self.kv8(il), self.ring[il]);
        if kv_f16() {
            self.k.kv_prep(&self.s, !swa, self.sc.gk.ptr, self.sc.gv.ptr, l.k_conv.ptr, l.v_conv.ptr, st_k, self.sc.k.ptr, self.sc.v.ptr, kv8[0], kv8[2], p_pos, ring, cscale, hp.qk_clip, hp.eps)?;
        } else {
            match self.int4_at(il) {
                false => self.k.kv_prep8(&self.s, !swa, self.sc.gk.ptr, self.sc.gv.ptr, l.k_conv.ptr, l.v_conv.ptr, st_k, self.sc.k.ptr, self.sc.v.ptr, kv8, p_pos, ring, cscale, hp.qk_clip, hp.eps)?,
                true => self.k.kv_prep4(&self.s, self.sc.gk.ptr, self.sc.gv.ptr, l.k_conv.ptr, l.v_conv.ptr, st_k, self.sc.k.ptr, self.sc.v.ptr, kv8, p_pos, cscale, hp.qk_clip, hp.eps)?,
            }
        }

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
        let nsplit = if swa { (hp.n_swa >> CS_SWA) as u32 } else { self.n_ctx.div_ceil(1 << CS_FULL) as u32 };
        if kv_f16() {
            let nsplit = if swa { hp.n_swa.div_ceil(256) as u32 } else { self.n_ctx.div_ceil(256) as u32 };
            self.k.attn_part(&self.s, self.sc.q.ptr, self.sc.r.ptr, sink, l.rel_bias.ptr, kv8[0], kv8[2], self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, p_pos, ring, extent, scale, maxs, nsplit, window)?;
            self.k.attn_reduce(&self.s, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, self.sc.att.ptr, p_pos, window, maxs, hp.n_head as u32)?;
        } else {
            let int4 = self.int4_at(il);
            let l = &self.w.layers[ip];
            self.k.attn_part8(&self.s, self.sc.q.ptr, self.sc.r.ptr, sink, l.rel_bias.ptr, kv8, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, p_pos, ring, extent, scale, maxs, nsplit, window, cs, int4)?;
            if !swa {
                // the per-chunk max and sum of each head: [kv head][chunk][3]
                self.probe("attn_pm", il, self.sc.pm.ptr, 8 * maxs as usize * 3)?;
                self.probe("attn_pl", il, self.sc.pl.ptr, 8 * maxs as usize * 3)?;
            }
            self.k.attn_reduce8(&self.s, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, self.sc.att.ptr, p_pos, window, maxs, cs, hp.n_head as u32)?;
        }
        self.probe("kqv_out", il, self.sc.att.ptr, nh as usize)?;
        self.k.gate_q8(&self.s, self.sc.att.ptr, gate, self.sc.xq.ptr, self.sc.xd.ptr, self.sc.xs.ptr, nh)?;
        let l = &self.w.layers[ip];
        let wo = &l.wo;
        self.qgemv(wo, self.sc.att2.ptr, self.sc.a.ptr, 0, false)?;
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
            self.control(ip, 3, m, self.sc.msgs.at(3 * n * 4), self.sc.ffm2.ptr)?;
            m = self.sc.ffm2.ptr;
        }
        let l = &self.w.layers[ip];
        self.k.norm_q8(&self.s, m, l.ffn_norm.ptr, self.sc.xn.ptr, self.sc.xq.ptr, self.sc.xd.ptr, self.sc.xs.ptr, nu, hp.eps, 1, nu)?;
        self.probe("ffn_norm_in", il, self.sc.xn.ptr, n)?;
        let l = &self.w.layers[ip];
        self.k.router_logits(&self.s, l.router.ptr, self.sc.xn.ptr, self.sc.rlog.ptr, hp.n_expert as u32, nu)?;
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
            &self.s, self.sc.rlog.ptr, l.probs_b.ptr, self.sc.ids.ptr, self.sc.wts.ptr, self.hp.w_scale, map_row, self.sc.slots.ptr, self.sc.xn.ptr, self.job.dev,
            self.params.ptr, si, ip as u32, n,
        )?;
        let ffu = self.hp.n_ff_exp as u32;
        if GPU_EXPERTS {
            let f = self.ex.fmt[ip];
            let bk = &self.cache.banks[f.index()];
            self.k.moe_d(&self.s, f, bk.gu.ptr, bk.gu_bytes as u64, self.sc.slots.ptr, self.sc.xq.ptr, self.sc.xd.ptr, 0, self.sc.gu.ptr, 2 * ffu, n)?;
            self.k.situ_q8(&self.s, self.sc.gu.ptr, self.sc.slots.ptr, self.sc.aq.ptr, self.sc.ad.ptr, self.sc.asum.ptr, ffu, self.ex.beta, self.ex.lb)?;
            self.k.moe_d(&self.s, f, bk.down.ptr, bk.down_bytes as u64, self.sc.slots.ptr, self.sc.aq.ptr, self.sc.ad.ptr, ffu, self.sc.dn.ptr, n, ffu)?;
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
        self.k.quant_q8_1(&self.s, x, self.sc.xq.ptr, self.sc.xd.ptr, self.sc.xs.ptr, n, 1)?;
        time!("quant_q8_1 2304", 4 * n as usize, self.k.quant_q8_1(&self.s, x, self.sc.xq.ptr, self.sc.xd.ptr, self.sc.xs.ptr, n, 1)?);
        let l = &self.w.layers[6];
        let (g0, wo) = (&l.g0, &l.wo);
        time!("gemv_q8d g0 6552x2304", g0.bytes(), self.qgemv(g0, x, self.sc.g0.ptr, 0, false)?);
        time!("gemv_q8d wo 2304x3072", wo.bytes(), self.qgemv(wo, x, self.sc.a.ptr, 0, false)?);
        let o = &self.w.output;
        time!("gemv_q8d output 201024x2304", o.bytes(), self.qgemv(o, x, self.sc.logits_raw.ptr, 0, false)?);
        self.sc.slots.upload(&[0i32, 1, 2, 3, 4, 5, 6, 7])?;
        let ffu = hp.n_ff_exp as u32;
        let (nv, b0) = (crate::experts::Fmt::Nv4, &self.cache.banks[0]);
        time!("moe_nv4_d gate_up 8 experts", 8 * b0.gu_bytes, self.k.moe_d(&self.s, nv, b0.gu.ptr, b0.gu_bytes as u64, self.sc.slots.ptr, self.sc.xq.ptr, self.sc.xd.ptr, 0, self.sc.gu.ptr, 2 * ffu, n)?);
        time!("situ_q8 8 experts", 8 * 8 * ffu as usize, self.k.situ_q8(&self.s, self.sc.gu.ptr, self.sc.slots.ptr, self.sc.aq.ptr, self.sc.ad.ptr, self.sc.asum.ptr, ffu, 4.0, 25.0)?);
        time!("moe_nv4_d down 8 experts", 8 * b0.down_bytes, self.k.moe_d(&self.s, nv, b0.down.ptr, b0.down_bytes as u64, self.sc.slots.ptr, self.sc.aq.ptr, self.sc.ad.ptr, ffu, self.sc.dn.ptr, n, ffu)?);
        let maxs = self.maxs as u32;
        let mut cases = vec![(1024u32, (hp.n_swa + T_MAX) as u32, 6usize)];
        let mut keys = 4096;
        while keys <= self.n_ctx {
            cases.push((keys as u32, 0, 5));
            keys *= 2;
        }
        for &(keys, ring, kvi) in &cases {
            let p: [u32; 4] = [keys - 1, 1f32.to_bits(), 1f32.to_bits(), 0];
            self.params.upload(&p)?;
            let (window, cs) = if ring != 0 { (hp.n_swa as u32, CS_SWA) } else { (u32::MAX, CS_FULL) };
            let nsplit = if ring != 0 { (hp.n_swa >> CS_SWA) as u32 } else { self.n_ctx.div_ceil(1 << CS_FULL) as u32 };
            let kv8 = self.kv8(kvi);
            let lr = &self.w.layers[kvi];
            let int4 = self.int4_at(kvi);
            let row: usize = if ring != 0 { KV_ROW * 2 } else { kv_widths(int4).iter().sum() };
            time!(format!("attn part+reduce {keys} keys"), keys as usize * row, {
                self.k.attn_part8(&self.s, self.sc.q.ptr, self.sc.r.ptr, self.sc.g0.ptr, lr.rel_bias.ptr, kv8, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, self.params.ptr, ring, 1024, 0.088, maxs, nsplit, window, cs, int4)?;
                self.k.attn_reduce8(&self.s, self.sc.pm.ptr, self.sc.pl.ptr, self.sc.pacc.ptr, self.sc.att.ptr, self.params.ptr, window, maxs, cs, hp.n_head as u32)?
            });
        }
        let list = vec![self.bl.b0.ptr, self.bl.b6[0].ptr, self.bl.b12[0].ptr, self.sc.x.ptr];
        let pl = self.ptr_list(&list)?;
        let l = &self.w.layers[6];
        time!("mix 4 states 1 query", 4 * 4 * n as usize, self.k.mix(&self.s, pl, 4, l.ffn_res_query.ptr, self.sc.ffm.ptr, 1, n)?);
        time!("rms_norm 2304", 4 * n as usize, self.k.rms_norm(&self.s, self.sc.x.ptr, Some(l.ffn_norm.ptr), self.sc.xn.ptr, n, 1, hp.eps)?);
        time!("router logits+topk", 4 * n as usize * 128, {
            self.k.router_logits(&self.s, l.router.ptr, self.sc.xn.ptr, self.sc.rlog.ptr, hp.n_expert as u32, n)?;
            self.k.router_topk(&self.s, self.sc.rlog.ptr, l.probs_b.ptr, self.sc.ids.ptr, self.sc.wts.ptr, 8.0)?
        });
        time!("conv4 2304", 4 * 8 * n as usize, self.k.conv4(&self.s, self.sc.a.ptr, l.post_conv.ptr, self.conv[6].ptr, None, self.sc.a2.ptr, n, 1, 0.4)?);
        time!("clip 24x128", 4 * 3072, self.k.clip_rows(&self.s, self.sc.g0.ptr, Some(l.q_clip.ptr), self.sc.q.ptr, 128, 24, 24, 22.6, self.params.at(8))?);
        time!("add 2304", 12 * n as usize, self.k.add(&self.s, self.sc.x.ptr, self.sc.f.ptr, self.sc.x.ptr, n)?);
        let lp = self.w.layers[6].lp.as_ref().unwrap();
        let d = &lp.down[0];
        time!("ctrl down 64x4608 (q8d)", d.bytes(), self.qgemv(d, self.sc.ctl_g.ptr, self.sc.ctl_gate.ptr, 0, true)?);
        let (gm, fm) = (&lp.gain[0], &lp.forget[0]);
        time!("ctrl_out", gm.bytes() + fm.bytes(), self.k.ctrl_out(&self.s, self.sc.x.ptr, self.sc.msgs.ptr, self.sc.ctl_gate.ptr, gm.qs.ptr, gm.d.ptr, fm.qs.ptr, fm.d.ptr, self.sc.ffm2.ptr, n)?);
        time!("ctrl_in", 16 * n as usize, self.k.ctrl_in(&self.s, self.sc.x.ptr, self.sc.msgs.ptr, self.sc.ctl_g.ptr, n)?);
        Ok(res)
    }
}
