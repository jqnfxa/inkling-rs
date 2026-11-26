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

/// `rms_norm_1` for `_ZN2ik10ik_kernels3gpu8rms_normILj1EEE`: the kernel's name, and its instance if any (the module,
/// gpu or attn, left out).
fn short(mangled: &str) -> String {
    let rest = &mangled[["3gpu", "4attn"].iter().find_map(|m| mangled.find(m).map(|i| i + m.len())).unwrap_or(0)..];
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

    /// gemv_bf16 of `nz` matrices of the same k in one launch: tab[4 z ..] = w, x, y, rows of matrix z; `max_rows` the
    /// largest rows.
    pub fn gemv_bf16t(&self, s: &Stream, tab: P, max_rows: u32, k: u32, nz: u32) -> Result<()> {
        assert!(k % 256 == 0, "gemv_bf16t: k {k}");
        self.run(s, "gemv_bf16t", (max_rows.div_ceil(16), 1, nz), 256, &[&tab, &k])
    }

    /// y[t][r] = W x[t] for `ntok` tokens, W repacked Q8_0 [rows][k]
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8(&self, s: &Stream, qs: P, d: P, x: P, y: P, rows: u32, k: u32, ntok: u32, y_stride: u32) -> Result<()> {
        self.run(s, "gemv_q8", (rows.div_ceil(16), ntok, 1), 256, &[&qs, &d, &x, &y, &rows, &k, &y_stride])
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

    #[allow(clippy::too_many_arguments)]
    pub fn kv_store(&self, s: &Stream, k: P, v: P, kc: P, vc: P, params: P, ring: u32, nkv: u32) -> Result<()> {
        self.run(s, "kv_store", (nkv.div_ceil(256), 1, 1), 256, &[&k, &v, &kc, &vc, &params, &ring, &nkv])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_decode(
        &self, s: &Stream, q: P, r: P, sink: P, rb: P, kc: P, vc: P, out: P, params: P, ring: u32, extent: u32, n_head: u32,
        group: u32, scale: f32,
    ) -> Result<()> {
        self.run(s, "attn_decode", (n_head, 1, 1), 256, &[&q, &r, &sink, &rb, &kc, &vc, &out, &params, &ring, &extent, &group, &scale])
    }

    pub fn moe_combine(&self, s: &Stream, gpu: P, cpu: P, slots: P, wts: P, out: P, n: u32) -> Result<()> {
        self.run(s, "moe_combine", (n.div_ceil(256), 1, 1), 256, &[&gpu, &cpu, &slots, &wts, &out, &n])
    }

    pub fn add(&self, s: &Stream, a: P, b: P, y: P, n: u32) -> Result<()> {
        self.run(s, "add", (n.div_ceil(256), 1, 1), 256, &[&a, &b, &y, &n])
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

    /// q8_1 of `nvec` vectors of n floats (n a multiple of 32)
    pub fn quant_q8_1(&self, s: &Stream, x: P, q: P, d: P, sum: P, n: u32, nvec: u32) -> Result<()> {
        self.run(s, "quant_q8_1", ((n / 32).div_ceil(8), nvec, 1), 256, &[&x, &q, &d, &sum, &n])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8d(&self, s: &Stream, qs: P, dw: P, xq: P, xd: P, y: P, rows: u32, k: u32) -> Result<()> {
        self.run(s, "gemv_q8d", (rows.div_ceil(16), 1, 1), 256, &[&qs, &dw, &xq, &xd, &y, &rows, &k])
    }

    /// The SITU activation of the cached experts of `slots`: gate_up [8][2 nff] (gate i row 2i, up i row 2i + 1) to
    /// af [8][nff], f32.
    #[allow(clippy::too_many_arguments)]
    pub fn situ(&self, s: &Stream, gu: P, slots: P, af: P, nff: u32, beta: f32, lb: f32) -> Result<()> {
        self.run(s, "situ", ((nff / 32).div_ceil(8), 8, 1), 256, &[&gu, &slots, &af, &nff, &beta, &lb])
    }

    /// y = W x for the 8 experts of `slots` on a bank of ModelOpt NVFP4 slots (codes, then scales): x f32 (expert j's
    /// at x + j xstride), expert j's factor amax[ids[j]] / denom[0].
    #[allow(clippy::too_many_arguments)]
    pub fn moe_nvfp4_d(&self, s: &Stream, bank: P, slot_bytes: u64, slots: P, ids: P, amax: P, denom: P, x: P, xstride: u32, y: P, rows: u32, k: u32) -> Result<()> {
        assert!(k % 32 == 0 && k <= 2304, "moe_nvfp4_d: k {k}");
        self.run(s, "moe_nvfp4_d", (rows.div_ceil(64), 8, 1), 128, &[&bank, &slot_bytes, &slots, &ids, &amax, &denom, &x, &xstride, &y, &rows, &k])
    }

    /// y = W x for the 8 experts of `slots` on a bank of MXFP8 slots (e4m3 values, then UE8M0 scales): x f32.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_mxfp8_d(&self, s: &Stream, bank: P, slot_bytes: u64, slots: P, x: P, xstride: u32, y: P, rows: u32, k: u32) -> Result<()> {
        assert!(k % 128 == 0, "moe_mxfp8_d: k {k}");
        self.run(s, "moe_mxfp8_d", (rows.div_ceil(64), 8, 1), 128, &[&bank, &slot_bytes, &slots, &x, &xstride, &y, &rows, &k])
    }

    pub fn router_logits(&self, s: &Stream, w: P, x: P, logits: P, n_expert: u32, n: u32) -> Result<()> {
        self.router_logits_t(s, w, x, logits, n_expert, n, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn router_logits_t(&self, s: &Stream, w: P, x: P, logits: P, n_expert: u32, n: u32, ntok: u32) -> Result<()> {
        self.run(s, "router_logits", (n_expert.div_ceil(8), ntok, 1), 256, &[&w, &x, &logits, &n_expert, &n])
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
        nsplit: u32, window: u32, cs: u32,
    ) -> Result<()> {
        self.run(s, "attn_part", (8, nsplit, 1), 128, &[&q, &r, &sink, &rb, &kc, &vc, &pm, &pl, &pacc, &params, &ring, &extent, &scale, &maxs, &window, &cs])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attn_reduce(&self, s: &Stream, pm: P, pl: P, pacc: P, out: P, params: P, window: u32, maxs: u32, cs: u32, n_head: u32) -> Result<()> {
        self.run(s, "attn_reduce", (n_head, 1, 1), 128, &[&pm, &pl, &pacc, &out, &params, &window, &maxs, &cs])
    }

    // ---------------------------------------------------------------- controllers, several at once (grid.z)

    pub fn ctrl_in_qz(&self, s: &Stream, tab: P, q: P, d: P, sum: P, n: u32, nz: u32) -> Result<()> {
        if n == 2304 && std::env::var("IK_OLD_CTRL").is_err() {
            return self.run(s, "ctrl_in_z768", (1, 1, nz), 768, &[&tab, &q, &d, &sum]);
        }
        self.run(s, "ctrl_in_qz", (1, 1, nz), 256, &[&tab, &q, &d, &sum, &n])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q8dz(&self, s: &Stream, tab: P, xq: P, xd: P, y: P, rows: u32, k: u32, nz: u32) -> Result<()> {
        self.run(s, "gemv_q8dz", (rows.div_ceil(16), 1, nz), 256, &[&tab, &xq, &xd, &y, &rows, &k])
    }

    /// `nz` GEMVs in one launch, their arguments in `tab` (see the kernel); `max_rows` sizes the grid.
    pub fn gemv_q8dt(&self, s: &Stream, tab: P, max_rows: u32, k: u32, nz: u32) -> Result<()> {
        self.run(s, "gemv_q8dt", (max_rows.div_ceil(16), 1, nz), 256, &[&tab, &k])
    }

    /// ctrl_out of `nz` controllers of one token: cur, msg, gw, fw, y of controller z at tab[5 z ..]; its gate at
    /// gate + 64 z.
    pub fn ctrl_outz(&self, s: &Stream, tab: P, gate: P, n: u32, nz: u32) -> Result<()> {
        self.run(s, "ctrl_outz", (n.div_ceil(256), 1, nz), 256, &[&tab, &gate, &n])
    }

    // ---------------------------------------------------------------- batched (prefill)

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q8d_t(&self, s: &Stream, qs: P, dw: P, xq: P, xd: P, y: P, rows: u32, k: u32, ntok: u32, y_stride: u32) -> Result<()> {
        self.run(s, "gemm_q8d_t", (rows.div_ceil(32), ntok.div_ceil(32), 1), 128, &[&qs, &dw, &xq, &xd, &y, &rows, &k, &ntok, &y_stride])
    }

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
    pub fn attn_prefill(
        &self, s: &Stream, q: P, r: P, sink: P, rb: P, kc: P, vc: P, out: P, params: P, ring: u32, window: u32, extent: u32, scale: f32,
        q_stride: u32, r_stride: u32, s_stride: u32, ntok: u32,
    ) -> Result<()> {
        self.run(s, "attn_prefill", (8, ntok, 1), 128, &[&q, &r, &sink, &rb, &kc, &vc, &out, &params, &ring, &window, &extent, &scale, &q_stride, &r_stride, &s_stride])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn situ_q8_n(&self, s: &Stream, gu: P, aq: P, ad: P, asum: P, nff: u32, beta: f32, lb: f32, npairs: u32) -> Result<()> {
        self.run(s, "situ_q8_n", ((nff / 32).div_ceil(8), npairs, 1), 256, &[&gu, &aq, &ad, &asum, &nff, &beta, &lb])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_combine_t(&self, s: &Stream, gpu: P, cpu: P, slots: P, wts: P, out: P, n: u32, ntok: u32) -> Result<()> {
        self.run(s, "moe_combine_t", (n.div_ceil(256), ntok, 1), 256, &[&gpu, &cpu, &slots, &wts, &out, &n])
    }

    // ---------------------------------------------------------------- fused decode

    #[allow(clippy::too_many_arguments)]
    pub fn norm_q8(&self, s: &Stream, x: P, w: P, y: P, q: P, d: P, sum: P, n: u32, eps: f32, nvec: u32, xs: u32) -> Result<()> {
        if n == 2304 && std::env::var("IK_OLD_NORM").is_err() {
            return self.run(s, "norm_q8_768", (nvec, 1, 1), 768, &[&x, &w, &y, &q, &d, &sum, &eps, &xs]);
        }
        self.run(s, "norm_q8", (nvec, 1, 1), 256, &[&x, &w, &y, &q, &d, &sum, &n, &eps, &xs])
    }

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

    pub fn gate_q8(&self, s: &Stream, a: P, g: P, q: P, d: P, sum: P, n: u32) -> Result<()> {
        self.run(s, "gate_q8", ((n / 32).div_ceil(8), 1, 1), 256, &[&a, &g, &q, &d, &sum, &n])
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

    #[allow(clippy::too_many_arguments)]
    pub fn ctrl_in_q(&self, s: &Stream, cur: P, msg: P, q: P, d: P, sum: P, n: u32) -> Result<()> {
        self.run(s, "ctrl_in_q", (1, 1, 1), 256, &[&cur, &msg, &q, &d, &sum, &n])
    }
}
