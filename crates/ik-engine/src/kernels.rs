//! The kernite PTX module, loaded once, and launchers of its kernels.

use std::cell::RefCell;
use std::collections::HashMap;

use crate::cuda::{Arg, CUdeviceptr, DevBuf, Func, Module, Result, Stream};

pub struct Kernels {
    _module: Module,
    f: HashMap<String, Func>,
    /// IK_PROF: a stamp of the GPU timer after each launch, and the names of the kernels launched since `prof_reset`
    pub prof: Option<(DevBuf, RefCell<Vec<&'static str>>, std::cell::Cell<bool>)>,
}

/// Stamps a profile holds, at most.
pub const PROF_MAX: usize = 8192;

/// `rms_norm_1` for `_ZN2ik10ik_kernels3gpu8rms_normILj1EEE`: the kernel's name, and its instance if any (the module
/// left out).
fn short(mangled: &str) -> String {
    let rest = &mangled[mangled.find("3gpu").map_or(0, |i| i + 4)..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    let len: usize = digits.parse().unwrap_or(0);
    let name = &rest[digits.len()..digits.len() + len];
    let inst: String = rest[digits.len() + len..].chars().filter(char::is_ascii_digit).collect();
    if inst.is_empty() { name.to_string() } else { format!("{name}_{inst}") }
}

pub type P = CUdeviceptr;

impl Kernels {
    pub fn load() -> Result<Kernels> {
        let ptx = ik_kernels::gpu::PTX;
        let module = Module::load(ptx)?;
        let mut f = HashMap::new();
        for line in ptx.lines() {
            if let Some(name) = line.strip_prefix(".visible .entry ").and_then(|l| l.strip_suffix('(')) {
                f.insert(short(name), module.func(name)?);
            }
        }
        let prof = if std::env::var("IK_PROF").is_ok() { Some((DevBuf::new(PROF_MAX * 8)?, RefCell::new(vec![]), std::cell::Cell::new(false))) } else { None };
        Ok(Kernels { _module: module, f, prof })
    }

    /// Profiling: forgets the kernels launched so far and stamps those that follow, until `prof_stop` (a token's body).
    pub fn prof_reset(&self) {
        if let Some((_, names, on)) = &self.prof {
            names.borrow_mut().clear();
            on.set(true);
        }
    }

    pub fn prof_stop(&self) {
        if let Some((_, _, on)) = &self.prof {
            on.set(false);
        }
    }

    fn get(&self, name: &str) -> Func {
        *self.f.get(name).unwrap_or_else(|| panic!("no kernel {name}"))
    }

    fn run(&self, s: &Stream, name: &'static str, grid: (u32, u32, u32), block: u32, args: &[&dyn Arg]) -> Result<()> {
        self.get(name).launch(s, grid, (block, 1, 1), 0, args)?;
        if let Some((buf, names, on)) = &self.prof
            && on.get()
        {
            let mut names = names.borrow_mut();
            let i = names.len() as u32;
            if (i as usize) < PROF_MAX {
                names.push(name);
                self.get("stamp").launch(s, (1, 1, 1), (32, 1, 1), 0, &[&buf.ptr, &i])?;
            }
        }
        Ok(())
    }

    /// y[t][r] = W x[t] for `ntok` tokens: W a BF16 [rows][k] (as the checkpoint has it), x f32 [ntok][k], y at
    /// t y_stride. k a multiple of 256.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_bf16(&self, s: &Stream, w: P, x: P, y: P, rows: u32, k: u32, ntok: u32, y_stride: u32) -> Result<()> {
        assert!(k % 256 == 0, "gemv_bf16: k {k}");
        self.run(s, "gemv_bf16", (rows.div_ceil(16), ntok, 1), 256, &[&w, &x, &y, &rows, &k, &y_stride])
    }

    /// What gemv_bf16 gives for `ntok` tokens, by tiles of 128 rows and 128 tokens, each reading W once for its tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_bf16(&self, s: &Stream, w: P, x: P, y: P, rows: u32, k: u32, ntok: u32, y_stride: u32) -> Result<()> {
        assert!(k.is_multiple_of(16) && ntok > 0, "gemm_bf16: k {k}, {ntok} tokens");
        self.run(s, "gemm_bf16", (rows.div_ceil(128), ntok.div_ceil(128), 1), 256, &[&w, &x, &y, &rows, &k, &ntok, &y_stride])
    }

    /// gemv_bf16 of `nz` matrices of the same k in one launch: tab[4 z ..] = w, x, y, rows of matrix z; `max_rows` the
    /// largest rows.
    pub fn gemv_bf16t(&self, s: &Stream, tab: P, max_rows: u32, k: u32, nz: u32) -> Result<()> {
        assert!(k % 256 == 0, "gemv_bf16t: k {k}");
        self.run(s, "gemv_bf16t", (max_rows.div_ceil(16), 1, nz), 256, &[&tab, &k])
    }

    /// RMS norm of `nvec` vectors of n floats, times w if given
    pub fn rms_norm(&self, s: &Stream, x: P, w: Option<P>, y: P, n: u32, nvec: u32, eps: f32) -> Result<()> {
        self.rms_norm_s(s, x, w, y, n, nvec, eps, n, n)
    }

    /// RMS norm of `nvec` vectors at strides xs, ys (floats)
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_s(&self, s: &Stream, x: P, w: Option<P>, y: P, n: u32, nvec: u32, eps: f32, xs: u32, ys: u32) -> Result<()> {
        let threads = n.min(1024).div_ceil(32) * 32;
        let (name, wp) = match w {
            Some(w) => ("rms_norm_1", w),
            None => ("rms_norm_0", 0),
        };
        self.run(s, name, (nvec, 1, 1), threads, &[&x, &wp, &y, &n, &eps, &xs, &ys])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn clip_rows(&self, s: &Stream, x: P, scale: Option<P>, y: P, d: u32, rows: u32, s_rows: u32, thr: f32, post: P) -> Result<()> {
        self.clip_rows_t(s, x, scale, y, d, rows, s_rows, thr, post, rows, 0, 0)
    }

    /// clip of `rows` rows of d floats, `rpt` rows per token; x, y advance by their token strides; post[token]
    #[allow(clippy::too_many_arguments)]
    pub fn clip_rows_t(
        &self, s: &Stream, x: P, scale: Option<P>, y: P, d: u32, rows: u32, s_rows: u32, thr: f32, post: P, rpt: u32, x_ts: u32, y_ts: u32,
    ) -> Result<()> {
        let (name, sp) = match scale {
            Some(p) => ("clip_rows_1", p),
            None => ("clip_rows_0", 0),
        };
        let eps = 1e-6f32;
        self.run(s, name, (rows.div_ceil(8), 1, 1), 256, &[&x, &sp, &y, &d, &rows, &s_rows, &thr, &eps, &post, &rpt, &x_ts, &y_ts])
    }

    pub fn mix(&self, s: &Stream, ptrs: P, ns: u32, qs: P, out: P, nq: u32, n: u32) -> Result<()> {
        self.mix_t(s, ptrs, ns, qs, out, nq, n, 1)
    }

    /// mix of `ntok` tokens: state j of token t at ptrs[j] + t n, out [t][nq][n]
    #[allow(clippy::too_many_arguments)]
    pub fn mix_t(&self, s: &Stream, ptrs: P, ns: u32, qs: P, out: P, nq: u32, n: u32, ntok: u32) -> Result<()> {
        assert!(ns <= 8, "mix8 takes 8 states at most");
        if n == 2304 && std::env::var("IK_OLD_MIX").is_err() {
            return self.run(s, "mix768", (nq, ntok, 1), 768, &[&ptrs, &ns, &qs, &out]);
        }
        self.run(s, "mix8", (nq, ntok, 1), 256, &[&ptrs, &ns, &qs, &out, &n])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv4(&self, s: &Stream, x: P, w: P, state: P, taps: Option<P>, y: P, c: u32, head_dim: u32, scale: f32) -> Result<()> {
        let (name, tp) = match taps {
            Some(t) => ("conv4_1", t),
            None => ("conv4_0", 0),
        };
        self.run(s, name, (c.div_ceil(256), 1, 1), 256, &[&x, &w, &state, &tp, &y, &c, &head_dim, &scale])
    }

    pub fn ctrl_in(&self, s: &Stream, cur: P, msg: P, g: P, n: u32) -> Result<()> {
        self.ctrl_in_t(s, cur, msg, g, n, 1)
    }

    pub fn ctrl_in_t(&self, s: &Stream, cur: P, msg: P, g: P, n: u32, ntok: u32) -> Result<()> {
        self.run(s, "ctrl_in", (1, ntok, 1), 1024, &[&cur, &msg, &g, &n])
    }

    /// ctrl_in of `nz` controllers of one token: cur and msg of controller z at tab[2 z], tab[2 z + 1]; its input at
    /// g + 2 n z.
    pub fn ctrl_in_z(&self, s: &Stream, tab: P, g: P, n: u32, nz: u32) -> Result<()> {
        self.run(s, "ctrl_in_z", (1, 1, nz), 1024, &[&tab, &g, &n])
    }

    /// y = (1 - tanh(F gate)) cur + tanh(G gate) msg: gw (G) and fw (F) the BF16 [n][64] gain and forget projections.
    #[allow(clippy::too_many_arguments)]
    pub fn ctrl_out(&self, s: &Stream, cur: P, msg: P, gate: P, gw: P, fw: P, y: P, n: u32) -> Result<()> {
        self.ctrl_out_t(s, cur, msg, gate, gw, fw, y, n, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ctrl_out_t(&self, s: &Stream, cur: P, msg: P, gate: P, gw: P, fw: P, y: P, n: u32, ntok: u32) -> Result<()> {
        self.run(s, "ctrl_out", (n.div_ceil(256), ntok, 1), 256, &[&cur, &msg, &gate, &gw, &fw, &y, &n])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn router(&self, s: &Stream, wt: P, x: P, bias: P, ids: P, wts: P, ids2: P, wts2: P, n: u32, scale: f32) -> Result<()> {
        self.run(s, "router", (1, 1, 1), 128, &[&wt, &x, &bias, &ids, &wts, &ids2, &wts2, &n, &scale])
    }

    pub fn add(&self, s: &Stream, a: P, b: P, y: P, n: u32) -> Result<()> {
        self.run(s, "add", (n.div_ceil(256), 1, 1), 256, &[&a, &b, &y, &n])
    }

    /// Rows of n floats (a multiple of 4): dst row i = src row sidx[i], for i < m.
    pub fn gather_rows(&self, s: &Stream, src: P, sidx: P, dst: P, n: u32, m: u32) -> Result<()> {
        assert!(n.is_multiple_of(4), "gather_rows: n {n}");
        self.run(s, "gather_rows", ((n / 4).div_ceil(128), m, 1), 128, &[&src, &sidx, &dst, &n])
    }

    /// Rows of n floats (a multiple of 4): dst row didx[i] += wts[widx[i]] * src row i, for i < m; the rows didx[i] are
    /// distinct.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rows_w(&self, s: &Stream, src: P, dst: P, didx: P, wts: P, widx: P, n: u32, m: u32) -> Result<()> {
        assert!(n.is_multiple_of(4), "add_rows_w: n {n}");
        self.run(s, "add_rows_w", ((n / 4).div_ceil(128), m, 1), 128, &[&src, &dst, &didx, &wts, &widx, &n])
    }

    pub fn mul_sigmoid(&self, s: &Stream, x: P, g: P, y: P, n: u32) -> Result<()> {
        self.mul_sigmoid_t(s, x, g, y, n, n, n)
    }

    /// y = x * sigmoid(g) over n floats in rows of n_row, the rows of g `g_stride` apart
    #[allow(clippy::too_many_arguments)]
    pub fn mul_sigmoid_t(&self, s: &Stream, x: P, g: P, y: P, n: u32, n_row: u32, g_stride: u32) -> Result<()> {
        self.run(s, "mul_sigmoid", (n.div_ceil(256), 1, 1), 256, &[&x, &g, &y, &n, &n_row, &g_stride])
    }

    pub fn softcap(&self, s: &Stream, x: P, y: P, n: u32, cap: f32) -> Result<()> {
        self.run(s, "softcap", (n.div_ceil(256), 1, 1), 256, &[&x, &y, &n, &cap])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_dispatch(&self, s: &Stream, ids: P, map: P, slots: P, x: P, job: P, params: P, step: u32, layer: u32, n: u32) -> Result<()> {
        self.run(s, "moe_dispatch", (1, 1, 1), 256, &[&ids, &map, &slots, &x, &job, &params, &step, &layer, &n])
    }

    pub fn moe_wait(&self, s: &Stream, ack: P, params: P, step: u32, err: P, timeout_ns: u64) -> Result<()> {
        self.run(s, "moe_wait", (1, 1, 1), 32, &[&ack, &params, &step, &err, &timeout_ns])
    }

    /// The SITU activation of the cached experts of the `pairs` slots: gate_up [pair][2 nff] (gate i row 2i, up i row
    /// 2i + 1) to af [pair][nff], f32.
    #[allow(clippy::too_many_arguments)]
    pub fn situ(&self, s: &Stream, gu: P, slots: P, af: P, nff: u32, beta: f32, lb: f32, pairs: u32) -> Result<()> {
        self.run(s, "situ", ((nff / 32).div_ceil(8), pairs, 1), 256, &[&gu, &slots, &af, &nff, &beta, &lb])
    }

    /// y = W x for the `pairs` slots (a token's 8, or 8 a token of several) on a bank of ModelOpt NVFP4 slots (codes,
    /// then scales): x f32 (pair j's at x + (j >> xshift) xstride), pair j's factor amax[ids[j]] / denom[0].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_nvfp4_d(
        &self, s: &Stream, bank: P, slot_bytes: u64, slots: P, ids: P, amax: P, denom: P, x: P, xstride: u32, xshift: u32, y: P, rows: u32, k: u32,
        pairs: u32,
    ) -> Result<()> {
        assert!(k % 32 == 0 && k <= 2304, "moe_nvfp4_d: k {k}");
        self.run(s, "moe_nvfp4_d", (pairs, rows.div_ceil(64), 1), 128, &[&bank, &slot_bytes, &slots, &ids, &amax, &denom, &x, &xstride, &xshift, &y, &rows, &k])
    }

    /// y = W x for the tokens of `nz` NVFP4 experts in one launch, each weight decoded once for 32 tokens. tab[4 z ..]
    /// holds the expert's slot (codes, then scales), x [m][k] f32 and y [m][rows], then m and the expert e as two u32s;
    /// the factor is amax[e] / denom[0]. `max_m` is the largest m.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_nvfp4_g(&self, s: &Stream, tab: P, amax: P, denom: P, rows: u32, k: u32, max_m: u32, nz: u32) -> Result<()> {
        assert!(k.is_multiple_of(32) && rows.is_multiple_of(4) && max_m > 0, "moe_nvfp4_g: rows {rows}, k {k}, m {max_m}");
        self.run(s, "moe_nvfp4_g", (rows.div_ceil(128), max_m.div_ceil(32), nz), 128, &[&tab, &amax, &denom, &rows, &k])
    }

    /// moe_nvfp4_g for MXFP8 experts: tab[4 z ..] holds the expert's slot (e4m3 values, then UE8M0 scales), x [m][k] f32
    /// and y [m][rows], then m and the expert as two u32s. `max_m` is the largest m.
    pub fn moe_mxfp8_g(&self, s: &Stream, tab: P, rows: u32, k: u32, max_m: u32, nz: u32) -> Result<()> {
        assert!(k.is_multiple_of(32) && rows.is_multiple_of(4) && max_m > 0, "moe_mxfp8_g: rows {rows}, k {k}, m {max_m}");
        self.run(s, "moe_mxfp8_g", (rows.div_ceil(128), max_m.div_ceil(32), nz), 128, &[&tab, &rows, &k])
    }

    /// y = W x for the `pairs` slots on a bank of MXFP8 slots (e4m3 values, then UE8M0 scales): x f32, as moe_nvfp4_d.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_mxfp8_d(&self, s: &Stream, bank: P, slot_bytes: u64, slots: P, x: P, xstride: u32, xshift: u32, y: P, rows: u32, k: u32, pairs: u32) -> Result<()> {
        assert!(k % 128 == 0, "moe_mxfp8_d: k {k}");
        self.run(s, "moe_mxfp8_d", (pairs, rows.div_ceil(64), 1), 128, &[&bank, &slot_bytes, &slots, &x, &xstride, &xshift, &y, &rows, &k])
    }

    pub fn router_topk(&self, s: &Stream, logits: P, bias: P, ids: P, wts: P, scale: f32) -> Result<()> {
        self.router_topk_t(s, logits, bias, ids, wts, scale, 1)
    }

    pub fn router_topk_t(&self, s: &Stream, logits: P, bias: P, ids: P, wts: P, scale: f32, ntok: u32) -> Result<()> {
        self.run(s, "router_topk", (ntok, 1, 1), 128, &[&logits, &bias, &ids, &wts, &scale])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_part(
        &self, s: &Stream, q: P, r: P, sink: P, rb: P, kc: P, vc: P, pm: P, pl: P, pacc: P, params: P, ring: u32, extent: u32, scale: f32, maxs: u32,
        nsplit: u32, window: u32, cs: u32, vram: u32,
    ) -> Result<()> {
        self.run(s, "attn_part", (8, nsplit, 1), 128, &[&q, &r, &sink, &rb, &kc, &vc, &pm, &pl, &pacc, &params, &ring, &extent, &scale, &maxs, &window, &cs, &vram])
    }

    /// `cpu`: the CPU's partials of the rows past the first `vram` (u32::MAX: none).
    #[allow(clippy::too_many_arguments)]
    pub fn attn_reduce(&self, s: &Stream, pm: P, pl: P, pacc: P, out: P, params: P, window: u32, maxs: u32, cs: u32, n_head: u32, cpu: P, vram: u32) -> Result<()> {
        self.run(s, "attn_reduce", (n_head, 1, 1), 128, &[&pm, &pl, &pacc, &out, &params, &window, &maxs, &cs, &cpu, &vram])
    }

    /// Posts a job of `step` to the CPU team: `word`, the position, and the floats of a, b and c (see the kernel).
    #[allow(clippy::too_many_arguments)]
    pub fn job_post(&self, s: &Stream, a: P, na: u32, b: P, nb: u32, c: P, nc: u32, job: P, params: P, step: u32, word: u32) -> Result<()> {
        self.run(s, "job_post", (1, 1, 1), 256, &[&a, &na, &b, &nb, &c, &nc, &job, &params, &step, &word])
    }

    // ---------------------------------------------------------------- controllers, several at once (grid.z)

    /// ctrl_out of `nz` controllers of one token: cur, msg, gw, fw, y of controller z at tab[5 z ..]; its gate at
    /// gate + 64 z.
    pub fn ctrl_outz(&self, s: &Stream, tab: P, gate: P, n: u32, nz: u32) -> Result<()> {
        self.run(s, "ctrl_outz", (n.div_ceil(256), 1, nz), 256, &[&tab, &gate, &n])
    }

    // ---------------------------------------------------------------- batched (prefill)

    #[allow(clippy::too_many_arguments)]
    pub fn conv4_seq(
        &self, s: &Stream, x: P, w: P, state: P, taps: Option<P>, y: P, c: u32, head_dim: u32, scale: f32, ntok: u32, taps_stride: u32, x_stride: u32,
    ) -> Result<()> {
        let (name, tp) = match taps {
            Some(t) => ("conv4_seq_1", t),
            None => ("conv4_seq_0", 0),
        };
        self.run(s, name, (c.div_ceil(256), 1, 1), 256, &[&x, &w, &state, &tp, &y, &c, &head_dim, &scale, &ntok, &taps_stride, &x_stride])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn kv_store_t(&self, s: &Stream, k: P, v: P, kc: P, vc: P, params: P, ring: u32, nkv: u32, kstride: u32, vstride: u32, ntok: u32) -> Result<()> {
        self.run(s, "kv_store_t", (nkv.div_ceil(256), ntok, 1), 256, &[&k, &v, &kc, &vc, &params, &ring, &nkv, &kstride, &vstride])
    }

    #[allow(clippy::too_many_arguments)]
    /// `rows` (lo, hi), `state` and `mode`: one pass of several (see the kernel); (0, u32::MAX), 0, 0 for all rows in one.
    pub fn attn_prefill(
        &self, s: &Stream, q: P, r: P, sink: P, rb: P, kc: P, vc: P, out: P, params: P, ring: u32, window: u32, extent: u32, scale: f32,
        q_stride: u32, r_stride: u32, s_stride: u32, ntok: u32, rows: (u32, u32), state: P, mode: u32,
    ) -> Result<()> {
        let (lo, hi) = rows;
        self.run(
            s,
            "attn_prefill",
            (8, ntok.div_ceil(16), 1),
            128,
            &[&q, &r, &sink, &rb, &kc, &vc, &out, &params, &ring, &window, &extent, &scale, &q_stride, &r_stride, &s_stride, &lo, &hi, &state, &mode, &ntok],
        )
    }

    /// out = (add ? out : 0) + the weighted sum of the GPU's pairs (gpu) or of the others (src [pair][n]).
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine_t(&self, s: &Stream, src: P, slots: P, wts: P, out: P, n: u32, ntok: u32, gpu: bool, add: bool) -> Result<()> {
        let (g, a) = (gpu as u32, add as u32);
        self.run(s, "moe_combine_t", (n.div_ceil(256), ntok, 1), 256, &[&src, &slots, &wts, &out, &n, &g, &a])
    }

    /// moe_combine_t over the pairs that `rows` ([token][8] i32) gives a row of src, -1 for the others.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine_r(&self, s: &Stream, src: P, rows: P, wts: P, out: P, n: u32, ntok: u32, add: bool) -> Result<()> {
        let a = add as u32;
        self.run(s, "moe_combine_r", (n.div_ceil(256), ntok, 1), 256, &[&src, &rows, &wts, &out, &n, &a])
    }

    // ---------------------------------------------------------------- fused decode

    #[allow(clippy::too_many_arguments)]
    pub fn kv_prep(
        &self, s: &Stream, dyn_taps: bool, gk: P, gv: P, kw: P, vw: P, state: P, k: P, v: P, kc: P, vc: P, params: P, ring: u32, cscale: f32, thr: f32, eps: f32,
    ) -> Result<()> {
        let name = if dyn_taps { "kv_prep_1" } else { "kv_prep_0" };
        self.run(s, name, (8, 1, 1), 128, &[&gk, &gv, &kw, &vw, &state, &k, &v, &kc, &vc, &params, &ring, &cscale, &thr, &eps])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn clip_qr(&self, s: &Stream, qin: P, qs: P, qout: P, rin: P, rs: P, rout: P, n_head: u32, thr_q: f32, thr_r: f32, post: P) -> Result<()> {
        self.run(s, "clip_qr", ((2 * n_head).div_ceil(8), 1, 1), 256, &[&qin, &qs, &qout, &rin, &rs, &rout, &n_head, &thr_q, &thr_r, &post])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv_res(&self, s: &Stream, x: P, w: P, state: P, res: Option<P>, y: P, n: u32, scale: f32) -> Result<()> {
        let (name, r) = match res {
            Some(r) => ("conv_res_1", r),
            None => ("conv_res_0", 0),
        };
        self.run(s, name, (n.div_ceil(256), 1, 1), 256, &[&x, &w, &state, &r, &y, &n, &scale])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_tail(&self, s: &Stream, gpu: P, cpu: P, slots: P, wts: P, w: P, state: P, x: P, n: u32, scale: f32) -> Result<()> {
        self.run(s, "moe_tail", (n.div_ceil(256), 1, 1), 256, &[&gpu, &cpu, &slots, &wts, &w, &state, &x, &n, &scale])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn router_dispatch(
        &self, s: &Stream, logits: P, bias: P, ids: P, wts: P, scale: f32, map: P, slots: P, x: P, job: P, params: P, step: u32, layer: u32, n: u32,
    ) -> Result<()> {
        self.run(s, if std::env::var("IK_OLD_ROUTER").is_ok() { "router_dispatch" } else { "router_dispatch2" }, (1, 1, 1), 128, &[&logits, &bias, &ids, &wts, &scale, &map, &slots, &x, &job, &params, &step, &layer, &n])
    }
}
