//! GPU kernels of the Inkling-2-Mini engine, written in kernite. The engine loads `gpu::PTX` with the driver API.
//!
//! Layout conventions: vectors are f32, one token at a time (grid.y walks tokens where a kernel takes several).
//! The weights are read as the checkpoint stores them: the dense matrices in BF16, the experts in NVFP4 (E2M1 codes,
//! an E4M3 scale per 16 values) or MXFP8 (E4M3 values, a UE8M0 scale per 32).

#![allow(long_running_const_eval)]

#[kernite_macros::kernels]
pub mod gpu {
    // ---------------------------------------------------------------- helpers

    /// The lane of this thread, and the row of its warp: `ctaid.x * (ntid.x / 32) + tid.x / 32`.
    fn warp_row(lane: u32, row: u32) {
        let (t, w, n, b): (u32, u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        mov.u32(n, ntid.x);
        shr.u32(n, n, 5);
        mov.u32(b, ctaid.x);
        mad.lo.u32(row, b, n, w);
    }

    /// Sum of `v` over the warp, in every lane.
    fn warp_sum(v: f32) {
        for sh in [16, 8, 4, 2, 1] {
            let o: f32;
            shfl.sync.bfly.b32(o, v, sh, 31, 0xffff_ffff);
            add.rn.f32(v, v, o);
        }
    }

    /// Max of `v` over the warp, in every lane.
    fn warp_max(v: f32) {
        for sh in [16, 8, 4, 2, 1] {
            let o: f32;
            shfl.sync.bfly.b32(o, v, sh, 31, 0xffff_ffff);
            max.f32(v, v, o);
        }
    }

    /// Sum of `v` over the block, in every thread. `red` is the shared address of 32 floats.
    fn block_sum(v: f32, red: u32) {
        let (t, lane, w, nw, at): (u32, u32, u32, u32, u32);
        warp_sum(v);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        mov.u32(nw, ntid.x);
        add.u32(nw, nw, 31);
        shr.u32(nw, nw, 5);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        mad.lo.u32(at, w, 4, red);
        bar.sync(0);
        if first {
            st.shared.f32([at], v);
        }
        bar.sync(0);
        let inside: pred;
        setp.lt.u32(inside, lane, nw);
        mov.f32(v, 0.0);
        mad.lo.u32(at, lane, 4, red);
        if inside {
            ld.shared.f32(v, [at]);
        }
        warp_sum(v);
    }

    /// e^x
    fn exp(y: f32, x: f32) {
        mul.rn.f32(y, x, 1.442_695_04);
        ex2.approx.ftz.f32(y, y);
    }

    /// 1/(1 + e^-x)
    fn sigmoid(y: f32, x: f32) {
        let e: f32;
        mul.rn.f32(e, x, -1.442_695_04);
        ex2.approx.ftz.f32(e, e);
        add.rn.f32(e, e, 1.0);
        rcp.rn.f32(y, e);
    }

    /// tanh(x) = 1 - 2/(e^2x + 1), x clamped where tanh is +-1 in f32
    fn tanh(y: f32, x: f32) {
        let e: f32;
        min.f32(e, x, 15.0);
        max.f32(e, e, -15.0);
        mul.rn.f32(e, e, 2.885_390_08);
        ex2.approx.ftz.f32(e, e);
        add.rn.f32(e, e, 1.0);
        rcp.rn.f32(e, e);
        fma.rn.f32(y, e, -2.0, 1.0);
    }

    /// Copies `n` floats of global `src` to shared `dst`, by all threads of the block, then waits.
    fn stage(dst: u32, src: u64, n: u32) {
        let (i, nt): (u32, u32);
        mov.u32(i, tid.x);
        mov.u32(nt, ntid.x);
        label(top);
        let done: pred;
        setp.ge.u32(done, i, n);
        if done {
            bra(end);
        }
        let at: u64;
        let sat: u32;
        let v: f32;
        mad.wide.u32(at, i, 4, src);
        ld.global.f32(v, [at]);
        mad.lo.u32(sat, i, 4, dst);
        st.shared.f32([sat], v);
        add.u32(i, i, nt);
        bra(top);
        label(end);
        bar.sync(0);
    }

    // ---------------------------------------------------------------- BF16 (the checkpoint's dense weights, as stored)

    /// The bf16 at `a` as f32.
    fn ld_bf16(f: f32, a: u64) {
        let h: u16;
        let u: u32;
        ld.global.nc.u16(h, [a]);
        cvt.u32.u16(u, h);
        shl.b32(u, u, 16);
        mov.b32(f, u);
    }

    /// The two bf16 of `w` as f32: `lo` the one in its low half (the first in memory).
    fn bf16x2(lo: f32, hi: f32, w: u32) {
        let (a, b): (u32, u32);
        shl.b32(a, w, 16);
        and.b32(b, w, 0xFFFF_0000);
        mov.b32(lo, a);
        mov.b32(hi, b);
    }

    /// acc += the two bf16 of `w` times xa (the first) and xb.
    fn fma_bf16x2(acc: f32, w: u32, xa: f32, xb: f32) {
        let (lo, hi): (f32, f32);
        bf16x2(lo, hi, w);
        fma.rn.f32(acc, lo, xa, acc);
        fma.rn.f32(acc, hi, xb, acc);
    }

    /// The bf16 `h` as f32.
    fn bf16f(f: f32, h: u16) {
        let u: u32;
        cvt.u32.u16(u, h);
        shl.b32(u, u, 16);
        mov.b32(f, u);
    }

    /// The 4 static taps of channel c of a short convolution as f32, w0 the oldest token's and w3 the current one's,
    /// from the checkpoint's BF16 weights: a `[c][4]` conv1d weight (oldest first), or with `LAG` the `[lag][c_total]`
    /// static weight of a dynamic convolution (lag 0 the current token).
    fn conv_w<const LAG: u32>(w0: f32, w1: f32, w2: f32, w3: f32, w: u64, c: u32, c_total: u32) {
        if LAG == 0 {
            let (wa, p0, p1): (u64, u32, u32);
            let c4: u32;
            shl.b32(c4, c, 2);
            mad.wide.u32(wa, c4, 2, w);
            ld.global.nc.v2.u32((p0, p1), [wa]);
            bf16x2(w0, w1, p0);
            bf16x2(w2, w3, p1);
        }
        if LAG == 1 {
            let (a, st): (u64, u64);
            mad.wide.u32(a, c, 2, w);
            mul.wide.u32(st, c_total, 2);
            ld_bf16(w3, a);
            add.u64(a, a, st);
            ld_bf16(w2, a);
            add.u64(a, a, st);
            ld_bf16(w1, a);
            add.u64(a, a, st);
            ld_bf16(w0, a);
        }
    }

    /// y[r] = sum_c W[r][c] x[c] for a BF16 matrix W [rows][k] (row-major, as the checkpoint has it) and f32 x.
    /// A warp takes rows 2w and 2w + 1 of the block's 16, a lane 8 columns of each at a time (16-byte loads, 512
    /// contiguous bytes of a row per warp); x comes through L1. k a multiple of 256.
    fn gemv_bf16_body(w: u64, x: u64, y: u64, rows: u32, k: u32, b: u32) {
        let (t, lane, wp, r0, r1, last): (u32, u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(wp, t, 5);
        shl.b32(r0, b, 4);
        mad.lo.u32(r0, wp, 2, r0);
        let past: pred;
        setp.ge.u32(past, r0, rows);
        if past {
            ret();
        }
        // an odd last row: the warp's second row repeats it, and is not stored
        add.u32(r1, r0, 1);
        sub.u32(last, rows, 1);
        min.u32(r1, r1, last);
        let (c, o0, o1): (u32, u32, u32);
        shl.b32(c, lane, 3);
        mad.lo.u32(o0, r0, k, c);
        mad.lo.u32(o1, r1, k, c);
        let (a0, a1, xa): (u64, u64, u64);
        mad.wide.u32(a0, o0, 2, w);
        mad.wide.u32(a1, o1, 2, w);
        mad.wide.u32(xa, c, 4, x);
        let (acc0, acc1): (f32, f32);
        mov.f32(acc0, 0.0);
        mov.f32(acc1, 0.0);
        label(top);
        let done: pred;
        setp.ge.u32(done, c, k);
        if done {
            bra(end);
        }
        let p: [u32; 4];
        let q: [u32; 4];
        let xv: [f32; 8];
        ld.global.nc.v4.u32((p[0], p[1], p[2], p[3]), [a0]);
        ld.global.nc.v4.u32((q[0], q[1], q[2], q[3]), [a1]);
        ld.global.nc.v4.f32((xv[0], xv[1], xv[2], xv[3]), [xa]);
        ld.global.nc.v4.f32((xv[4], xv[5], xv[6], xv[7]), [xa + 16]);
        for j in 0..4 {
            fma_bf16x2(acc0, p[j], xv[2 * j], xv[2 * j + 1]);
            fma_bf16x2(acc1, q[j], xv[2 * j], xv[2 * j + 1]);
        }
        add.u64(a0, a0, 512);
        add.u64(a1, a1, 512);
        add.u64(xa, xa, 1024);
        add.u32(c, c, 256);
        bra(top);
        label(end);
        warp_sum(acc0);
        warp_sum(acc1);
        // lane 0 stores the first row, lane 1 the second (if there is one)
        let (st0, st1): (pred, pred);
        let ya: u64;
        mad.wide.u32(ya, r0, 4, y);
        setp.eq.u32(st0, lane, 0);
        if st0 {
            st.global.f32([ya], acc0);
        }
        setp.lt.u32(past, r0, last);
        setp.eq.and.u32(st1, lane, 1, past);
        if st1 {
            st.global.f32([ya + 4], acc1);
        }
    }

    /// gemv_bf16_body on token ctaid.y: x advances by k floats a token, y by `y_stride`.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn gemv_bf16(w: *const u16, x: *const f32, y: *mut f32, rows: u32, k: u32, y_stride: u32) {
        let (b, tk, o): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(tk, ctaid.y);
        let (xt, yt): (u64, u64);
        mul.lo.u32(o, tk, k);
        mad.wide.u32(xt, o, 4, x);
        mul.lo.u32(o, tk, y_stride);
        mad.wide.u32(yt, o, 4, y);
        gemv_bf16_body(w, xt, yt, rows, k, b);
    }

    /// gemv_bf16_body of matrix ctaid.z, from tab[4 z ..]: w, x, y, rows. Blocks past a matrix's rows return at once:
    /// the grid is sized for the largest.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn gemv_bf16t(tab: *const u64, k: u32) {
        let (z, b, zi, b16): (u32, u32, u32, u32);
        mov.u32(z, ctaid.z);
        mov.u32(b, ctaid.x);
        let (ta, wz, xz, yz): (u64, u64, u64, u64);
        let rows: u32;
        shl.b32(zi, z, 2);
        mad.wide.u32(ta, zi, 8, tab);
        ld.global.u32(rows, [ta + 24]);
        shl.b32(b16, b, 4);
        let idle: pred;
        setp.ge.u32(idle, b16, rows);
        if idle {
            ret();
        }
        ld.global.u64(wz, [ta]);
        ld.global.u64(xz, [ta + 8]);
        ld.global.u64(yz, [ta + 16]);
        gemv_bf16_body(wz, xz, yz, rows, k, b);
    }

    // ---------------------------------------------------------------- norms

    /// y = x / rms(x) * w (w BF16; without w where `HAS_W` is 0), over n floats; grid.x walks vectors.
    #[maxntid(1024)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn rms_norm<const HAS_W: u32>(x: *const f32, w: *const u16, y: *mut f32, n: u32, eps: f32, xs: u32, ys: u32) {
        let red: shared<[f32; 32], 16>;
        let (v, off, offy): (u32, u32, u32);
        mov.u32(v, ctaid.x);
        mul.lo.u32(off, v, xs);
        mul.lo.u32(offy, v, ys);
        let (xv, yv): (u64, u64);
        mad.wide.u32(xv, off, 4, x);
        mad.wide.u32(yv, offy, 4, y);
        let (i, nt): (u32, u32);
        mov.u32(nt, ntid.x);

        let ss: f32;
        mov.f32(ss, 0.0);
        mov.u32(i, tid.x);
        label(sum_top);
        let done: pred;
        setp.ge.u32(done, i, n);
        if done {
            bra(sum_end);
        }
        let at: u64;
        let f: f32;
        mad.wide.u32(at, i, 4, xv);
        ld.global.f32(f, [at]);
        fma.rn.f32(ss, f, f, ss);
        add.u32(i, i, nt);
        bra(sum_top);
        label(sum_end);

        let r: u32;
        mov.u32(r, red);
        block_sum(ss, r);
        let (nf, scale): (f32, f32);
        cvt.rn.f32.u32(nf, n);
        div.rn.f32(scale, ss, nf);
        add.rn.f32(scale, scale, eps);
        sqrt.rn.f32(scale, scale);
        rcp.rn.f32(scale, scale);

        mov.u32(i, tid.x);
        label(top);
        let done2: pred;
        setp.ge.u32(done2, i, n);
        if done2 {
            bra(end);
        }
        let (at2, yat): (u64, u64);
        let g: f32;
        mad.wide.u32(at2, i, 4, xv);
        ld.global.f32(g, [at2]);
        mul.rn.f32(g, g, scale);
        if HAS_W == 1 {
            let wat: u64;
            let wv: f32;
            mad.wide.u32(wat, i, 2, w);
            ld_bf16(wv, wat);
            mul.rn.f32(g, g, wv);
        }
        mad.wide.u32(yat, i, 4, yv);
        st.global.f32([yat], g);
        add.u32(i, i, nt);
        bra(top);
        label(end);
    }

    // ---------------------------------------------------------------- clip

    /// Row clip: `y = x*s * min(1, thr/(|x*s| + eps)) * post[0]` for each row of `d <= 128` floats.
    /// One warp per row. Row `r` uses row `r % s_rows` of the scale `s`, which `HAS_S = 0` leaves out.
    #[maxntid(256)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn clip_rows<const HAS_S: u32>(
        x: *const f32, s: *const u16, y: *mut f32, d: u32, rows: u32, s_rows: u32, thr: f32, eps: f32, post: *const f32,
        rpt: u32, x_tstride: u32, y_tstride: u32,
    ) {
        let (lane, row): (u32, u32);
        warp_row(lane, row);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        // row r is head r % rpt of token r / rpt; x and y advance by their token strides
        let (srow, xoff, soff, tk, hr, yoff): (u32, u32, u32, u32, u32, u32);
        rem.u32(srow, row, s_rows);
        div.u32(tk, row, rpt);
        rem.u32(hr, row, rpt);
        mul.lo.u32(xoff, hr, d);
        mad.lo.u32(yoff, tk, y_tstride, xoff);
        mad.lo.u32(xoff, tk, x_tstride, xoff);
        mul.lo.u32(soff, srow, d);

        let ss: f32;
        mov.f32(ss, 0.0);
        for k in 0..4 {
            let idx: u32;
            add.u32(idx, lane, k * 32);
            let out: pred;
            setp.ge.u32(out, idx, d);
            if !out {
                let v: f32;
                load_scaled::<HAS_S>(v, x, s, xoff, soff, idx);
                fma.rn.f32(ss, v, v, ss);
            }
        }
        warp_sum(ss);

        let (norm, f, p): (f32, f32, f32);
        sqrt.rn.f32(norm, ss);
        add.rn.f32(norm, norm, eps);
        div.rn.f32(f, thr, norm);
        max.f32(f, f, 0.0);
        min.f32(f, f, 1.0);
        let pa: u64;
        mad.wide.u32(pa, tk, 4, post);
        ld.global.f32(p, [pa]);
        for k in 0..4 {
            let idx: u32;
            add.u32(idx, lane, k * 32);
            let out: pred;
            setp.ge.u32(out, idx, d);
            if !out {
                let v: f32;
                load_scaled::<HAS_S>(v, x, s, xoff, soff, idx);
                mul.rn.f32(v, v, f);
                mul.rn.f32(v, v, p);
                let i: u32;
                let at: u64;
                add.u32(i, yoff, idx);
                mad.wide.u32(at, i, 4, y);
                st.global.f32([at], v);
            }
        }
    }

    /// `v = x[xoff + idx] * s[soff + idx]` (s BF16), or `x[xoff + idx]` alone where `HAS_S` is 0.
    fn load_scaled<const HAS_S: u32>(v: f32, x: u64, s: u64, xoff: u32, soff: u32, idx: u32) {
        let i: u32;
        let at: u64;
        add.u32(i, xoff, idx);
        mad.wide.u32(at, i, 4, x);
        ld.global.f32(v, [at]);
        if HAS_S == 1 {
            let j: u32;
            let sat: u64;
            let w: f32;
            add.u32(j, soff, idx);
            mad.wide.u32(sat, j, 2, s);
            ld_bf16(w, sat);
            mul.rn.f32(v, v, w);
        }
    }

    // ---------------------------------------------------------------- short convolution

    /// Causal short convolution of 4 taps with a residual, for one token: per channel c,
    /// `y = (x + sum_k w[c][k] * h[k] + dyn) * scale` where h = (state[c][0..3], x), then state[c] = (h[1], h[2], x);
    /// w the checkpoint's BF16 weight (see conv_w: `[c][4]`, or `[lag][c]` where DYN).
    /// With `DYN`, `dyn = sum_lag h[3 - lag] * taps[head(c)][lag]`, head(c) = c / (c_total / n_heads).
    #[maxntid(256)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn conv4<const DYN: u32>(
        x: *const f32, w: *const u16, state: *mut f32, taps: *const f32, y: *mut f32, c_total: u32, head_dim: u32, scale: f32,
    ) {
        let (b, nt, t, c): (u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, b, nt, t);
        let past: pred;
        setp.ge.u32(past, c, c_total);
        if past {
            ret();
        }
        let (xa, sa): (u64, u64);
        let c3: u32;
        mad.wide.u32(xa, c, 4, x);
        mul.lo.u32(c3, c, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (xv, h0, h1, h2): (f32, f32, f32, f32);
        ld.global.f32(xv, [xa]);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        let (w0, w1, w2, w3): (f32, f32, f32, f32);
        conv_w::<DYN>(w0, w1, w2, w3, w, c, c_total);
        let o: f32;
        mul.rn.f32(o, w0, h0);
        fma.rn.f32(o, w1, h1, o);
        fma.rn.f32(o, w2, h2, o);
        fma.rn.f32(o, w3, xv, o);
        if DYN == 1 {
            let hd: u32;
            let ta: u64;
            div.u32(hd, c, head_dim);
            shl.b32(hd, hd, 2);
            mad.wide.u32(ta, hd, 4, taps);
            let (t0, t1, t2, t3): (f32, f32, f32, f32);
            ld.global.v4.f32((t0, t1, t2, t3), [ta]);
            let dsum: f32;
            mul.rn.f32(dsum, xv, t0);
            fma.rn.f32(dsum, h2, t1, dsum);
            fma.rn.f32(dsum, h1, t2, dsum);
            fma.rn.f32(dsum, h0, t3, dsum);
            add.rn.f32(o, o, dsum);
        }
        add.rn.f32(o, o, xv);
        mul.rn.f32(o, o, scale);
        st.global.f32([sa], h1);
        st.global.f32([sa + 4], h2);
        st.global.f32([sa + 8], xv);
        let ya: u64;
        mad.wide.u32(ya, c, 4, y);
        st.global.f32([ya], o);
    }

    // ---------------------------------------------------------------- loop control

    /// The input of a controller's down projection: `g = (rms(cur), rms(msg))`, 2n floats; token ctaid.y.
    #[maxntid(1024)]
    pub unsafe extern "ptx-kernel" fn ctrl_in(cur: *const f32, msg: *const f32, g: *mut f32, n: u32) {
        let (tk, to, to2): (u32, u32, u32);
        let (curt, msgt, gt): (u64, u64, u64);
        mov.u32(tk, ctaid.y);
        mul.lo.u32(to, tk, n);
        shl.b32(to2, to, 1);
        mad.wide.u32(curt, to, 4, cur);
        mad.wide.u32(msgt, to, 4, msg);
        mad.wide.u32(gt, to2, 4, g);
        ctrl_in_f32(curt, msgt, gt, n);
    }

    /// ctrl_in of controller ctaid.z (one token): cur and msg at tab[2 z], tab[2 z + 1]; g at g + 2 n z.
    #[maxntid(1024)]
    pub unsafe extern "ptx-kernel" fn ctrl_in_z(tab: *const u64, g: *mut f32, n: u32) {
        let (z, zi, go): (u32, u32, u32);
        mov.u32(z, ctaid.z);
        let (ta, cz, mz, gz): (u64, u64, u64, u64);
        shl.b32(zi, z, 1);
        mad.wide.u32(ta, zi, 8, tab);
        ld.global.u64(cz, [ta]);
        ld.global.u64(mz, [ta + 8]);
        mul.lo.u32(go, zi, n);
        mad.wide.u32(gz, go, 4, g);
        ctrl_in_f32(cz, mz, gz, n);
    }

    /// g[0..n] = cur / rms(cur), g[n..2n] = msg / rms(msg).
    fn ctrl_in_f32(curt: u64, msgt: u64, gt: u64, n: u32) {
        let red: shared<[f32; 32], 16>;
        let r: u32;
        mov.u32(r, red);
        let (t, nt): (u32, u32);
        mov.u32(t, tid.x);
        mov.u32(nt, ntid.x);
        let nf: f32;
        cvt.rn.f32.u32(nf, n);
        for half in 0..2 {
            let src: u64;
            if half == 0 {
                mov.u64(src, curt);
            } else {
                mov.u64(src, msgt);
            }
            let ss: f32;
            mov.f32(ss, 0.0);
            let i: u32;
            mov.u32(i, t);
            label(a_top);
            let a_done: pred;
            setp.ge.u32(a_done, i, n);
            if a_done {
                bra(a_end);
            }
            let at: u64;
            let v: f32;
            mad.wide.u32(at, i, 4, src);
            ld.global.f32(v, [at]);
            fma.rn.f32(ss, v, v, ss);
            add.u32(i, i, nt);
            bra(a_top);
            label(a_end);
            block_sum(ss, r);
            div.rn.f32(ss, ss, nf);
            add.rn.f32(ss, ss, 0.000_001);
            sqrt.rn.f32(ss, ss);
            rcp.rn.f32(ss, ss);
            mov.u32(i, t);
            label(b_top);
            let b_done: pred;
            setp.ge.u32(b_done, i, n);
            if b_done {
                bra(b_end);
            }
            let (at2, gat): (u64, u64);
            let v2: f32;
            let gi: u32;
            mad.wide.u32(at2, i, 4, src);
            ld.global.f32(v2, [at2]);
            mul.rn.f32(v2, v2, ss);
            add.u32(gi, i, half * 0);
            if half == 1 {
                add.u32(gi, i, n);
            }
            mad.wide.u32(gat, gi, 4, gt);
            st.global.f32([gat], v2);
            add.u32(i, i, nt);
            bra(b_top);
            label(b_end);
        }
    }

    /// The controller's output: `y = cur - tanh(F gate) * cur + tanh(G gate) * msg`, per channel, where G and F are
    /// the checkpoint's BF16 gain and forget projections [n][64] and gate is 64 floats.
    /// The body of ctrl_out, for token tk.
    fn ctrl_out_body(cur: u64, msg: u64, gate: u64, gw: u64, fw: u64, y: u64, n: u32, tk: u32) {
        let gs: shared<[f32; 64], 16>;
        let base: u32;
        mov.u32(base, gs);
        let n64: u32;
        mov.u32(n64, 64);
        let (to, tg): (u32, u32);
        mul.lo.u32(to, tk, n);
        shl.b32(tg, tk, 6);
        let gatet: u64;
        mad.wide.u32(gatet, tg, 4, gate);
        stage(base, gatet, n64);
        let (b, nt, t, c): (u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, b, nt, t);
        let past: pred;
        setp.ge.u32(past, c, n);
        if past {
            ret();
        }
        let (gain, forget): (f32, f32);
        bf16_row64(gain, gw, c, base);
        bf16_row64(forget, fw, c, base);
        tanh(gain, gain);
        tanh(forget, forget);
        let (ca, ma, ya): (u64, u64, u64);
        let (cv, mv, o): (f32, f32, f32);
        let ct: u32;
        add.u32(ct, to, c);
        mad.wide.u32(ca, ct, 4, cur);
        mad.wide.u32(ma, ct, 4, msg);
        ld.global.f32(cv, [ca]);
        ld.global.f32(mv, [ma]);
        mul.rn.f32(o, forget, cv);
        sub.rn.f32(o, cv, o);
        fma.rn.f32(o, gain, mv, o);
        mad.wide.u32(ya, ct, 4, y);
        st.global.f32([ya], o);
    }

    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn ctrl_out(
        cur: *const f32, msg: *const f32, gate: *const f32, gw: *const u16, fw: *const u16, y: *mut f32, n: u32,
    ) {
        let tk: u32;
        mov.u32(tk, ctaid.y);
        ctrl_out_body(cur, msg, gate, gw, fw, y, n, tk);
    }

    /// ctrl_out of controller ctaid.z (one token): cur, msg, gw, fw, y at tab[5 z ..]; its gate at gate + 64 z.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn ctrl_outz(tab: *const u64, gate: *const f32, n: u32) {
        let (z, zi, tk): (u32, u32, u32);
        mov.u32(z, ctaid.z);
        mov.u32(tk, 0);
        let (ta, cz, mz, gwz, fwz, yz, gz): (u64, u64, u64, u64, u64, u64, u64);
        mul.lo.u32(zi, z, 5);
        mad.wide.u32(ta, zi, 8, tab);
        ld.global.u64(cz, [ta]);
        ld.global.u64(mz, [ta + 8]);
        ld.global.u64(gwz, [ta + 16]);
        ld.global.u64(fwz, [ta + 24]);
        ld.global.u64(yz, [ta + 32]);
        shl.b32(zi, z, 6);
        mad.wide.u32(gz, zi, 4, gate);
        ctrl_out_body(cz, mz, gz, gwz, fwz, yz, n, tk);
    }

    /// v = row `r` of a BF16 [*][64] matrix times the 64 floats at shared `xs`.
    fn bf16_row64(v: f32, w: u64, r: u32, xs: u32) {
        mov.f32(v, 0.0);
        let wa: u64;
        let r64: u32;
        shl.b32(r64, r, 6);
        mad.wide.u32(wa, r64, 2, w);
        for q in 0..8 {
            let (p0, p1, p2, p3): (u32, u32, u32, u32);
            ld.global.nc.v4.u32((p0, p1, p2, p3), [wa + (q * 16)]);
            let (x0, x1, x2, x3): (f32, f32, f32, f32);
            ld.shared.v4.f32((x0, x1, x2, x3), [xs + (q * 32)]);
            fma_bf16x2(v, p0, x0, x1);
            fma_bf16x2(v, p1, x2, x3);
            ld.shared.v4.f32((x0, x1, x2, x3), [xs + (q * 32 + 16)]);
            fma_bf16x2(v, p2, x0, x1);
            fma_bf16x2(v, p3, x2, x3);
        }
    }

    // ---------------------------------------------------------------- router

    /// MoE router of one token: logits = W x (W^T is [n][128] f32), probs = sigmoid, the top 8 of probs + bias are
    /// selected, their probs normalized to sum 1 (+eps) and scaled. Writes ids[8] and weights[8], to `ids`/`wts` and to
    /// `ids2`/`wts2` (host-mapped, for the CPU experts).
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn router(
        wt: *const f32, x: *const f32, bias: *const f32, ids: *mut u32, wts: *mut f32, ids2: *mut u32, wts2: *mut f32, n: u32, scale: f32,
    ) {
        let xs: shared<[f32; 4608], 16>;
        let sel: shared<[f32; 128], 16>;
        let win: shared<[u32; 8], 16>;
        let base: u32;
        mov.u32(base, xs);
        stage(base, x, n);
        let e: u32;
        mov.u32(e, tid.x);
        let acc: f32;
        mov.f32(acc, 0.0);
        let i: u32;
        mov.u32(i, 0);
        label(top);
        let done: pred;
        setp.ge.u32(done, i, n);
        if done {
            bra(end);
        }
        let (wi, xa): (u32, u32);
        let wa: u64;
        let (wv, xv): (f32, f32);
        mad.lo.u32(wi, i, 128, e);
        mad.wide.u32(wa, wi, 4, wt);
        ld.global.nc.f32(wv, [wa]);
        mad.lo.u32(xa, i, 4, base);
        ld.shared.f32(xv, [xa]);
        fma.rn.f32(acc, wv, xv, acc);
        add.u32(i, i, 1);
        bra(top);
        label(end);
        let (p, sv, bv): (f32, f32, f32);
        sigmoid(p, acc);
        let ba: u64;
        mad.wide.u32(ba, e, 4, bias);
        ld.global.nc.f32(bv, [ba]);
        add.rn.f32(sv, p, bv);
        let (sb, sat): (u32, u32);
        mov.u32(sb, sel);
        mad.lo.u32(sat, e, 4, sb);
        st.shared.f32([sat], sv);
        let wb: u32;
        mov.u32(wb, win);
        bar.sync(0);

        // 8 rounds: every warp finds the max of the 128 selection scores, ties to the lower expert; thread 0 records it
        let lane: u32;
        and.b32(lane, e, 31);
        let first: pred;
        setp.eq.u32(first, e, 0);
        for k in 0..8 {
            let (best, cand): (f32, f32);
            let (bi, ci): (u32, u32);
            mov.f32(best, -3.0e38);
            mov.u32(bi, 0);
            for q in 0..4 {
                let ca: u32;
                add.u32(ci, lane, q * 32);
                mad.lo.u32(ca, ci, 4, sb);
                ld.shared.f32(cand, [ca]);
                let better: pred;
                setp.gt.f32(better, cand, best);
                selp.f32(best, cand, best, better);
                selp.u32(bi, ci, bi, better);
            }
            for sh in [16, 8, 4, 2, 1] {
                let (ob, oi): (f32, u32);
                shfl.sync.bfly.b32(ob, best, sh, 31, 0xffff_ffff);
                shfl.sync.bfly.b32(oi, bi, sh, 31, 0xffff_ffff);
                let (gt, eq, lt, take): (pred, pred, pred, pred);
                setp.gt.f32(gt, ob, best);
                setp.eq.f32(eq, ob, best);
                setp.lt.u32(lt, oi, bi);
                and.pred(take, eq, lt);
                or.pred(take, take, gt);
                selp.f32(best, ob, best, take);
                selp.u32(bi, oi, bi, take);
            }
            bar.sync(0);
            if first {
                let kill: u32;
                st.shared.u32([wb + (k * 4)], bi);
                mad.lo.u32(kill, bi, 4, sb);
                st.shared.f32([kill], -3.0e38);
            }
            bar.sync(0);
        }

        // weights: the probs of the winners, normalized and scaled
        let my: pred;
        setp.lt.u32(my, e, 8);
        let hit: u32;
        mov.u32(hit, 0xffff_ffff);
        for k in 0..8 {
            let wk: u32;
            ld.shared.u32(wk, [wb + (k * 4)]);
            let is: pred;
            setp.eq.u32(is, wk, e);
            if is {
                mov.u32(hit, k);
            }
        }
        // each winner's thread holds its prob in p; gather the 8 probs with a warp-free shared pass
        let won: pred;
        setp.ne.u32(won, hit, 0xffff_ffff);
        let pb: u32;
        mov.u32(pb, sel);
        bar.sync(0);
        if won {
            let pa: u32;
            mad.lo.u32(pa, hit, 4, pb);
            st.shared.f32([pa], p);
        }
        bar.sync(0);
        if my {
            let (sum, pv, own): (f32, f32, f32);
            mov.f32(sum, 0.0);
            for k in 0..8 {
                ld.shared.f32(pv, [pb + (k * 4)]);
                add.rn.f32(sum, sum, pv);
            }
            add.rn.f32(sum, sum, 0.000_000_000_000_000_000_01);
            let oa: u32;
            mad.lo.u32(oa, e, 4, pb);
            ld.shared.f32(own, [oa]);
            div.rn.f32(own, own, sum);
            mul.rn.f32(own, own, scale);
            let id: u32;
            mad.lo.u32(oa, e, 4, wb);
            ld.shared.u32(id, [oa]);
            let (a1, a2, a3, a4): (u64, u64, u64, u64);
            mad.wide.u32(a1, e, 4, ids);
            mad.wide.u32(a2, e, 4, wts);
            mad.wide.u32(a3, e, 4, ids2);
            mad.wide.u32(a4, e, 4, wts2);
            st.global.u32([a1], id);
            st.global.f32([a2], own);
            st.global.u32([a3], id);
            st.global.f32([a4], own);
        }
    }

    // ---------------------------------------------------------------- elementwise

    /// y = a + b over n floats.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn add(a: *const f32, b: *const f32, y: *mut f32, n: u32) {
        let (bk, nt, t, c): (u32, u32, u32, u32);
        mov.u32(bk, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, bk, nt, t);
        let past: pred;
        setp.ge.u32(past, c, n);
        if past {
            ret();
        }
        let (aa, ba, ya): (u64, u64, u64);
        let (av, bv): (f32, f32);
        mad.wide.u32(aa, c, 4, a);
        mad.wide.u32(ba, c, 4, b);
        mad.wide.u32(ya, c, 4, y);
        ld.global.f32(av, [aa]);
        ld.global.f32(bv, [ba]);
        add.rn.f32(av, av, bv);
        st.global.f32([ya], av);
    }

    /// y = x * sigmoid(g) over n floats.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn mul_sigmoid(x: *const f32, g: *const f32, y: *mut f32, n: u32, n_row: u32, g_stride: u32) {
        let (bk, nt, t, c): (u32, u32, u32, u32);
        mov.u32(bk, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, bk, nt, t);
        let past: pred;
        setp.ge.u32(past, c, n);
        if past {
            ret();
        }
        let (xa, ga, ya): (u64, u64, u64);
        let (xv, gv, sg): (f32, f32, f32);
        let (gr, gc, gi): (u32, u32, u32);
        div.u32(gr, c, n_row);
        rem.u32(gc, c, n_row);
        mad.lo.u32(gi, gr, g_stride, gc);
        mad.wide.u32(xa, c, 4, x);
        mad.wide.u32(ga, gi, 4, g);
        mad.wide.u32(ya, c, 4, y);
        ld.global.f32(xv, [xa]);
        ld.global.f32(gv, [ga]);
        sigmoid(sg, gv);
        mul.rn.f32(xv, xv, sg);
        st.global.f32([ya], xv);
    }

    /// Final logits: y[i] = cap * tanh(x[i] / cap) for i < n.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn softcap(x: *const f32, y: *mut f32, n: u32, cap: f32) {
        let (bk, nt, t, c): (u32, u32, u32, u32);
        mov.u32(bk, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, bk, nt, t);
        let past: pred;
        setp.ge.u32(past, c, n);
        if past {
            ret();
        }
        let (xa, ya): (u64, u64);
        let (v, ic): (f32, f32);
        mad.wide.u32(xa, c, 4, x);
        mad.wide.u32(ya, c, 4, y);
        ld.global.f32(v, [xa]);
        rcp.rn.f32(ic, cap);
        mul.rn.f32(v, v, ic);
        tanh(v, v);
        mul.rn.f32(v, v, cap);
        st.global.f32([ya], v);
    }

    // ---------------------------------------------------------------- GPU <-> CPU expert handoff

    /// Posts the MoE job of one layer to the CPU through host-mapped `job`, and the slots of the 8 selected experts:
    /// slots[k] = map[ids[k]] (-1 if not cached). Job layout (bytes): seq u32 @0, layer @4, ids @8, slots @40, x @128.
    /// seq = params[3] * 64 + step + 1 is written last, after a system fence.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn moe_dispatch(
        ids: *const u32, map: *const i32, slots: *mut i32, x: *const f32, job: *mut u32, params: *const u32, step: u32, layer: u32, n: u32,
    ) {
        let t: u32;
        mov.u32(t, tid.x);
        let i: u32;
        mov.u32(i, t);
        label(top);
        let done: pred;
        setp.ge.u32(done, i, n);
        if done {
            bra(end);
        }
        let (xa, ja): (u64, u64);
        let v: f32;
        let ji: u32;
        mad.wide.u32(xa, i, 4, x);
        ld.global.f32(v, [xa]);
        add.u32(ji, i, 32);
        mad.wide.u32(ja, ji, 4, job);
        st.volatile.global.f32([ja], v);
        add.u32(i, i, 256);
        bra(top);
        label(end);
        let lt8: pred;
        setp.lt.u32(lt8, t, 8);
        if lt8 {
            let (ia, ma, sa, j1, j2): (u64, u64, u64, u64, u64);
            let (id, ji2, ji3): (u32, u32, u32);
            let sl: s32;
            mad.wide.u32(ia, t, 4, ids);
            ld.global.u32(id, [ia]);
            mad.wide.u32(ma, id, 4, map);
            ld.global.s32(sl, [ma]);
            mad.wide.u32(sa, t, 4, slots);
            st.global.s32([sa], sl);
            add.u32(ji2, t, 2);
            mad.wide.u32(j1, ji2, 4, job);
            st.volatile.global.u32([j1], id);
            add.u32(ji3, t, 10);
            mad.wide.u32(j2, ji3, 4, job);
            st.volatile.global.s32([j2], sl);
        }
        bar.sync(0);
        let first: pred;
        setp.eq.u32(first, t, 0);
        if first {
            let (tok, seq): (u32, u32);
            st.volatile.global.u32([job + 4], layer);
            ld.global.u32(tok, [params + 12]);
            mad.lo.u32(seq, tok, 64, step);
            add.u32(seq, seq, 1);
            membar.sys();
            st.volatile.global.u32([job], seq);
        }
    }

    /// Posts a job of `step` to the CPU team through host-mapped `job`, as moe_dispatch does: `word` @4, the token's
    /// position params[0] @8, from byte 128 on the na floats of a, then the nb of b and the nc of c; the seq last, after a
    /// system fence.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn job_post(
        a: *const f32, na: u32, b: *const f32, nb: u32, c: *const f32, nc: u32, job: *mut u32, params: *const u32, step: u32, word: u32,
    ) {
        let t: u32;
        mov.u32(t, tid.x);
        let (n1, n2, i): (u32, u32, u32);
        add.u32(n1, na, nb);
        add.u32(n2, n1, nc);
        mov.u32(i, t);
        label(top);
        let done: pred;
        setp.ge.u32(done, i, n2);
        if done {
            bra(end);
        }
        let (inb, inc): (pred, pred);
        setp.ge.u32(inb, i, na);
        setp.ge.u32(inc, i, n1);
        let (ib, ic, ji): (u32, u32, u32);
        sub.u32(ib, i, na);
        sub.u32(ic, i, n1);
        let (sa, sb, scc, src, ja): (u64, u64, u64, u64, u64);
        mad.wide.u32(sa, i, 4, a);
        mad.wide.u32(sb, ib, 4, b);
        mad.wide.u32(scc, ic, 4, c);
        selp.b64(src, sb, sa, inb);
        selp.b64(src, scc, src, inc);
        let v: f32;
        ld.global.f32(v, [src]);
        add.u32(ji, i, 32);
        mad.wide.u32(ja, ji, 4, job);
        st.volatile.global.f32([ja], v);
        add.u32(i, i, 256);
        bra(top);
        label(end);
        bar.sync(0);
        let first: pred;
        setp.eq.u32(first, t, 0);
        if first {
            let (tok, seq, pos): (u32, u32, u32);
            st.volatile.global.u32([job + 4], word);
            ld.global.u32(pos, [params]);
            st.volatile.global.u32([job + 8], pos);
            ld.global.u32(tok, [params + 12]);
            mad.lo.u32(seq, tok, 64, step);
            add.u32(seq, seq, 1);
            membar.sys();
            st.volatile.global.u32([job], seq);
        }
    }

    /// Waits until the CPU acknowledges the job of `step` (host-mapped `ack` == seq), at most about `timeout_ns`;
    /// on timeout writes the seq to `err`.
    #[reqntid(32)]
    pub unsafe extern "ptx-kernel" fn moe_wait(ack: *const u32, params: *const u32, step: u32, err: *mut u32, timeout_ns: u64) {
        let t: u32;
        mov.u32(t, tid.x);
        let first: pred;
        setp.eq.u32(first, t, 0);
        if !first {
            ret();
        }
        let (tok, seq): (u32, u32);
        ld.global.u32(tok, [params + 12]);
        mad.lo.u32(seq, tok, 64, step);
        add.u32(seq, seq, 1);
        let (t0, now, lim): (u64, u64, u64);
        mov.u64(t0, globaltimer);
        add.u64(lim, t0, timeout_ns);
        label(top);
        let v: u32;
        ld.acquire.sys.global.u32(v, [ack]);
        let ok: pred;
        setp.eq.u32(ok, v, seq);
        if ok {
            bra(end);
        }
        mov.u64(now, globaltimer);
        let late: pred;
        setp.gt.u64(late, now, lim);
        if late {
            st.volatile.global.u32([err], seq);
            bra(end);
        }
        nanosleep.u32(200);
        bra(top);
        label(end);
    }

    // ---------------------------------------------------------------- MoE experts on the GPU

    /// The SITU activation of the cached experts' gate_up ([8][2 nff], the checkpoint's rows: gate i is row 2i, up i
    /// row 2i + 1): beta tanh(g/beta) sigmoid(g) lb tanh(u/lb), f32 (`af` [8][nff]). A warp per block of 32, grid.y = 8
    /// experts.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn situ(gu: *const f32, slots: *const i32, af: *mut f32, nff: u32, beta: f32, lb: f32) {
        let ke: u32;
        mov.u32(ke, ctaid.y);
        let (sa, slot): (u64, s32);
        mad.wide.u32(sa, ke, 4, slots);
        ld.global.s32(slot, [sa]);
        let none: pred;
        setp.lt.s32(none, slot, 0);
        if none {
            ret();
        }
        let (lane, b): (u32, u32);
        warp_row(lane, b);
        let nb: u32;
        shr.u32(nb, nff, 5);
        let past: pred;
        setp.ge.u32(past, b, nb);
        if past {
            ret();
        }
        let (i, gi, base, ai): (u32, u32, u32, u32);
        shl.b32(i, b, 5);
        add.u32(i, i, lane);
        shl.b32(base, nff, 1);
        mul.lo.u32(base, base, ke);
        mad.lo.u32(gi, i, 2, base);
        let ga: u64;
        let (gv, uv, a, sg, tu, ib, il): (f32, f32, f32, f32, f32, f32, f32);
        mad.wide.u32(ga, gi, 4, gu);
        ld.global.v2.f32((gv, uv), [ga]);
        rcp.rn.f32(ib, beta);
        rcp.rn.f32(il, lb);
        mul.rn.f32(a, gv, ib);
        tanh(a, a);
        mul.rn.f32(a, a, beta);
        sigmoid(sg, gv);
        mul.rn.f32(a, a, sg);
        mul.rn.f32(tu, uv, il);
        tanh(tu, tu);
        mul.rn.f32(tu, tu, lb);
        mul.rn.f32(a, a, tu);
        let afa: u64;
        mad.lo.u32(ai, ke, nff, i);
        mad.wide.u32(afa, ai, 4, af);
        st.global.f32([afa], a);
    }

    /// y = W x for the 8 experts of `slots` (grid.y) on a bank of ModelOpt NVFP4 slots: the expert's codes as the
    /// checkpoint has them ([rows][k/2], element 2j of a row in the low nibble of byte j), then its E4M3 scales
    /// ([rows][k/16]). The expert's factor is amax[ids[ke]] / denom[0]. x is f32, the same for all experts (xstride 0)
    /// or each expert's at x + ke xstride. 8 lanes per row (16-byte chunks of 32 elements, two scale blocks: a warp
    /// reads 128 contiguous bytes of each of 4 rows), a thread takes 4 rows (r + 16i), 64 rows per block of 128
    /// threads; x from shared memory. k a multiple of 32, at most 2304.
    ///
    /// The values are decoded exactly. Codes k and k + 4 of a word (nibbles at bits 4k, 4k + 16) shift to an f16x2 of
    /// them times 2^-14: e1 e0 m to bits 11..9 of each half (f16's exponent bias of 15 against E2M1's 1, and with
    /// e1 e0 = 0 the subnormals of both), the sign from bit 12 to 15. Their f32 are exact, and so is the scale: (b & 0x7F)
    /// << 20 is the f32 of the E4M3 b times 2^-120 (subnormal ones too), times 2^120. The factor takes the 2^14.
    ///
    /// Four blocks an SM, 128 registers a thread: with fewer, ptxas moves the next chunk's loads down to where they are
    /// used (3 % slower on an RTX 3050).
    #[reqntid(128)]
    #[minnctapersm(4)]
    pub unsafe extern "ptx-kernel" fn moe_nvfp4_d(
        bank: *const u8, slot_bytes: u64, slots: *const i32, ids: *const u32, amax: *const f32, denom: *const f32, x: *const f32,
        xstride: u32, y: *mut f32, rows: u32, kdim: u32,
    ) {
        let ke: u32;
        mov.u32(ke, ctaid.y);
        let (sa0, slot): (u64, s32);
        mad.wide.u32(sa0, ke, 4, slots);
        ld.global.s32(slot, [sa0]);
        let none: pred;
        setp.lt.s32(none, slot, 0);
        if none {
            ret();
        }
        let aoff: u32;
        mul.lo.u32(aoff, ke, xstride);
        let xk: u64;
        mad.wide.u32(xk, aoff, 4, x);
        let (t, lr, bk, row0, last): (u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(lr, t, 7);
        mov.u32(bk, ctaid.x);
        shr.u32(row0, t, 3);
        mad.lo.u32(row0, bk, 64, row0);
        // x to shared memory, chunk c's 32 floats at byte 144 c: the 8 lanes of a row, which read 8 chunks at once,
        // then reach 8 different sets of banks. Float4 i goes to 144 (i / 8) + 16 (i % 8), by all the block's threads
        // before any leaves.
        let xs: shared<[f32; 2592], 16>;
        let (xb, i, n4): (u32, u32, u32);
        mov.u32(xb, xs);
        mov.u32(i, t);
        shr.u32(n4, kdim, 2);
        label(fill);
        let filled: pred;
        setp.ge.u32(filled, i, n4);
        if filled {
            bra(staged);
        }
        let (ga, so, hi, lo): (u64, u32, u32, u32);
        let (v0, v1, v2, v3): (f32, f32, f32, f32);
        mad.wide.u32(ga, i, 16, xk);
        ld.global.nc.v4.f32((v0, v1, v2, v3), [ga]);
        shr.u32(hi, i, 3);
        and.b32(lo, i, 7);
        mad.lo.u32(so, hi, 144, xb);
        mad.lo.u32(so, lo, 16, so);
        st.shared.v4.f32([so], (v0, v1, v2, v3));
        add.u32(i, i, 128);
        bra(fill);
        label(staged);
        bar.sync(0);
        let past: pred;
        setp.ge.u32(past, row0, rows);
        if past {
            ret();
        }
        sub.u32(last, rows, 1);
        let (sl, base): (u64, u64);
        cvt.u64.s32(sl, slot);
        mul.lo.u64(sl, sl, slot_bytes);
        add.u64(base, bank, sl);
        let (half, qbytes, sk, nc, c): (u32, u32, u32, u32, u32);
        shr.u32(half, kdim, 1);
        mul.lo.u32(qbytes, rows, half);
        shr.u32(sk, kdim, 4);
        shr.u32(nc, kdim, 5);
        mov.u32(c, lr);
        // the thread's rows row0 + 16 r (past the end: the last row, computed, not stored): chunk c of a row is bytes
        // 16c.., its scales bytes 2c, 2c + 1
        let (qa, sa): ([u64; 4], [u64; 4]);
        for r in 0..4 {
            let (rr, so): (u32, u32);
            add.u32(rr, row0, r * 16);
            min.u32(rr, rr, last);
            mad.wide.u32(qa[r], rr, half, base);
            mad.wide.u32(qa[r], lr, 16, qa[r]);
            mad.lo.u32(so, rr, sk, qbytes);
            mad.lo.u32(so, lr, 2, so);
            cvt.u64.u32(sa[r], so);
            add.u64(sa[r], sa[r], base);
        }
        // chunk c's x: elements 32c..32c + 32, at shared 144 c
        let xo: u32;
        mad.lo.u32(xo, lr, 144, xb);
        let (k120, f120): (u32, f32);
        mov.u32(k120, 0x7B80_0000);
        mov.b32(f120, k120);
        let acc: [f32; 4];
        for r in 0..4 {
            mov.f32(acc[r], 0.0);
        }
        let done: pred;
        setp.ge.u32(done, c, nc);
        if done {
            bra(end);
        }
        // Chunk c's codes and scales are in half h of w and s16 while half h of the loop sums them, and the next chunk
        // loads into the other half meanwhile: each load has a chunk's arithmetic to arrive in. Past the last chunk
        // the addresses stay, and the last half loads its own chunk again, which nothing sums.
        let (w, s16): ([u32; 32], [u16; 8]);
        for r in 0..4 {
            ld.global.nc.v4.u32((w[4 * r], w[4 * r + 1], w[4 * r + 2], w[4 * r + 3]), [qa[r]]);
        }
        for r in 0..4 {
            ld.global.nc.u16(s16[r], [sa[r]]);
        }
        label(top);
        for h in 0..2 {
            let (cn, more, sq, ss): (u32, pred, u64, u64);
            add.u32(cn, c, 8);
            setp.lt.u32(more, cn, nc);
            selp.b64(sq, 128, 0, more);
            selp.b64(ss, 16, 0, more);
            for r in 0..4 {
                add.u64(qa[r], qa[r], sq);
                add.u64(sa[r], sa[r], ss);
            }
            for r in 0..4 {
                ld.global.nc.v4.u32(
                    (w[16 - 16 * h + 4 * r], w[16 - 16 * h + 4 * r + 1], w[16 - 16 * h + 4 * r + 2], w[16 - 16 * h + 4 * r + 3]),
                    [qa[r]],
                );
            }
            for r in 0..4 {
                ld.global.nc.u16(s16[4 - 4 * h + r], [sa[r]]);
            }
            let xv: [f32; 32];
            for q in 0..8 {
                ld.shared.v4.f32((xv[4 * q], xv[4 * q + 1], xv[4 * q + 2], xv[4 * q + 3]), [xo + (q * 16)]);
            }
            for r in 0..4 {
                // the sums of the chunk's two blocks: words 0, 1 are elements 0..15, words 2, 3 elements 16..31
                let part: [f32; 2];
                mov.f32(part[0], 0.0);
                mov.f32(part[1], 0.0);
                for j in 0..4 {
                    for k in 0..4 {
                        // word j's codes k, k + 4 (elements 8j + k, 8j + k + 4): e1 e0 m to bits 9..11 (and 25..27)
                        let (sh, a, sg, hv): (u32, u32, u32, u32);
                        let (h0, h1): (f16, f16);
                        let (f0, f1): (f32, f32);
                        if k == 3 {
                            shr.u32(sh, w[16 * h + 4 * r + j], 3);
                        } else {
                            shl.b32(sh, w[16 * h + 4 * r + j], 9 - 4 * k);
                        }
                        // the sign then at bit 12: hv = a + 7 sg moves it to 15
                        and.b32(a, sh, 0x1E00_1E00);
                        and.b32(sg, sh, 0x1000_1000);
                        mad.lo.u32(hv, sg, 7, a);
                        mov.b32((h0, h1), hv);
                        cvt.f32.f16(f0, h0);
                        cvt.f32.f16(f1, h1);
                        fma.rn.f32(part[j / 2], f0, xv[8 * j + k], part[j / 2]);
                        fma.rn.f32(part[j / 2], f1, xv[8 * j + k + 4], part[j / 2]);
                    }
                }
                let (sc, e0, e1): (u32, u32, u32);
                let (s0, s1): (f32, f32);
                cvt.u32.u16(sc, s16[4 * h + r]);
                shl.b32(e0, sc, 20);
                and.b32(e0, e0, 0x07F0_0000);
                shl.b32(e1, sc, 12);
                and.b32(e1, e1, 0x07F0_0000);
                mov.b32(s0, e0);
                mov.b32(s1, e1);
                mul.rn.f32(s0, s0, f120);
                mul.rn.f32(s1, s1, f120);
                fma.rn.f32(acc[r], part[0], s0, acc[r]);
                fma.rn.f32(acc[r], part[1], s1, acc[r]);
            }
            add.u32(xo, xo, 1152);
            mov.u32(c, cn);
            if h == 0 {
                let last: pred;
                setp.ge.u32(last, c, nc);
                if last {
                    bra(end);
                }
            }
            if h == 1 {
                if more {
                    bra(top);
                }
            }
        }
        label(end);
        for sh in [4, 2, 1] {
            for r in 0..4 {
                let o: f32;
                shfl.sync.bfly.b32(o, acc[r], sh, 31, 0xffff_ffff);
                add.rn.f32(acc[r], acc[r], o);
            }
        }
        // the expert's factor, as the checkpoint gives it (amax / outer_scale_denominator), times the values' 2^14
        let (ia, aa, e): (u64, u64, u32);
        let (am, den, es): (f32, f32, f32);
        mad.wide.u32(ia, ke, 4, ids);
        ld.global.u32(e, [ia]);
        mad.wide.u32(aa, e, 4, amax);
        ld.global.nc.f32(am, [aa]);
        ld.global.nc.f32(den, [denom]);
        div.rn.f32(es, am, den);
        mul.rn.f32(es, es, 16384.0);
        let first: pred;
        setp.eq.u32(first, lr, 0);
        for r in 0..4 {
            let (rr, oi, oa, inr, store): (u32, u32, u64, pred, pred);
            add.u32(rr, row0, r * 16);
            setp.lt.u32(inr, rr, rows);
            and.pred(store, inr, first);
            mul.rn.f32(acc[r], acc[r], es);
            mad.lo.u32(oi, ke, rows, rr);
            mad.wide.u32(oa, oi, 4, y);
            if store {
                st.global.f32([oa], acc[r]);
            }
        }
    }

    /// acc += 4 e4m3 of `w` (element 0 in its low byte) times x0..x3. An e4m3 byte b as f16 is
    /// (b & 0x80) << 8 | (b & 0x7F) << 7: the e4m3 value times 2^-8, exactly (subnormals too); the f32 of that is exact.
    fn fma_e4m3x4(acc: f32, w: u32, x0: f32, x1: f32, x2: f32, x3: f32, zero: u32) {
        let (u, m, sg, h): (u32, u32, u32, u32);
        let (h0, h1): (f16, f16);
        let (f0, f1): (f32, f32);
        for pair in 0..2 {
            // bytes 2 pair, 2 pair + 1 into the high bytes of the two halves
            if pair == 0 {
                prmt.b32(u, w, zero, 0x1404);
            } else {
                prmt.b32(u, w, zero, 0x3424);
            }
            shr.u32(m, u, 1);
            and.b32(m, m, 0x3F80_3F80);
            and.b32(sg, u, 0x8000_8000);
            or.b32(h, m, sg);
            mov.b32((h0, h1), h);
            cvt.f32.f16(f0, h0);
            cvt.f32.f16(f1, h1);
            if pair == 0 {
                fma.rn.f32(acc, f0, x0, acc);
                fma.rn.f32(acc, f1, x1, acc);
            } else {
                fma.rn.f32(acc, f0, x2, acc);
                fma.rn.f32(acc, f1, x3, acc);
            }
        }
    }

    /// y = W x for the 8 experts of `slots` (grid.y) on a bank of MXFP8 slots: the expert's e4m3 values as the
    /// checkpoint has them ([rows][k]), then its UE8M0 scales ([rows][k/32], 2^(s-127) each). x is f32, the same for all
    /// experts (xstride 0) or each expert's at x + ke xstride. 8 lanes per row (16-byte chunks of 16 values: a warp reads
    /// 128 contiguous bytes of each of 4 rows), a thread takes 4 rows (r + 16i), 64 rows per block of 128 threads; x
    /// through L1. k a multiple of 128.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn moe_mxfp8_d(
        bank: *const u8, slot_bytes: u64, slots: *const i32, x: *const f32, xstride: u32, y: *mut f32, rows: u32, kdim: u32,
    ) {
        let ke: u32;
        mov.u32(ke, ctaid.y);
        let (sa0, slot): (u64, s32);
        mad.wide.u32(sa0, ke, 4, slots);
        ld.global.s32(slot, [sa0]);
        let none: pred;
        setp.lt.s32(none, slot, 0);
        if none {
            ret();
        }
        let aoff: u32;
        mul.lo.u32(aoff, ke, xstride);
        let xk: u64;
        mad.wide.u32(xk, aoff, 4, x);
        let (t, lr, bk, row0, last): (u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(lr, t, 7);
        mov.u32(bk, ctaid.x);
        shr.u32(row0, t, 3);
        mad.lo.u32(row0, bk, 64, row0);
        let past: pred;
        setp.ge.u32(past, row0, rows);
        if past {
            ret();
        }
        sub.u32(last, rows, 1);
        let (sl, base): (u64, u64);
        cvt.u64.s32(sl, slot);
        mul.lo.u64(sl, sl, slot_bytes);
        add.u64(base, bank, sl);
        let (qbytes, sk, nc, c, bsh): (u32, u32, u32, u32, u32);
        mul.lo.u32(qbytes, rows, kdim);
        shr.u32(sk, kdim, 5);
        shr.u32(nc, kdim, 4);
        mov.u32(c, lr);
        // chunk c = lr + 8i is in block c/2 = lr/2 + 4i: byte lr/2 of the row's scale word i
        shr.u32(bsh, lr, 1);
        shl.b32(bsh, bsh, 3);
        let (qa, sa): ([u64; 4], [u64; 4]);
        for r in 0..4 {
            let (rr, so): (u32, u32);
            add.u32(rr, row0, r * 16);
            min.u32(rr, rr, last);
            mad.wide.u32(qa[r], rr, kdim, base);
            mad.wide.u32(qa[r], lr, 16, qa[r]);
            mad.lo.u32(so, rr, sk, qbytes);
            cvt.u64.u32(sa[r], so);
            add.u64(sa[r], sa[r], base);
        }
        let xo: u64;
        mad.wide.u32(xo, lr, 64, xk);
        let zero: u32;
        mov.u32(zero, 0);
        let acc: [f32; 4];
        for r in 0..4 {
            mov.f32(acc[r], 0.0);
        }
        label(top);
        let done: pred;
        setp.ge.u32(done, c, nc);
        if done {
            bra(end);
        }
        let (w0, w1, w2, w3, sw): ([u32; 4], [u32; 4], [u32; 4], [u32; 4], [u32; 4]);
        for r in 0..4 {
            ld.global.nc.v4.u32((w0[r], w1[r], w2[r], w3[r]), [qa[r]]);
        }
        for r in 0..4 {
            ld.global.nc.u32(sw[r], [sa[r]]);
        }
        let xv: [f32; 16];
        for q in 0..4 {
            ld.global.nc.v4.f32((xv[4 * q], xv[4 * q + 1], xv[4 * q + 2], xv[4 * q + 3]), [xo + (q * 16)]);
        }
        for r in 0..4 {
            let part: f32;
            mov.f32(part, 0.0);
            fma_e4m3x4(part, w0[r], xv[0], xv[1], xv[2], xv[3], zero);
            fma_e4m3x4(part, w1[r], xv[4], xv[5], xv[6], xv[7], zero);
            fma_e4m3x4(part, w2[r], xv[8], xv[9], xv[10], xv[11], zero);
            fma_e4m3x4(part, w3[r], xv[12], xv[13], xv[14], xv[15], zero);
            // the block's 2^(s-127), times the 2^8 of the f16 values: the f32 of exponent s + 8
            let (sb, sf): (u32, f32);
            shr.u32(sb, sw[r], bsh);
            and.b32(sb, sb, 0xFF);
            add.u32(sb, sb, 8);
            shl.b32(sb, sb, 23);
            mov.b32(sf, sb);
            fma.rn.f32(acc[r], part, sf, acc[r]);
        }
        add.u32(c, c, 8);
        for r in 0..4 {
            add.u64(qa[r], qa[r], 128);
            add.u64(sa[r], sa[r], 4);
        }
        add.u64(xo, xo, 512);
        bra(top);
        label(end);
        for sh in [4, 2, 1] {
            for r in 0..4 {
                let o: f32;
                shfl.sync.bfly.b32(o, acc[r], sh, 31, 0xffff_ffff);
                add.rn.f32(acc[r], acc[r], o);
            }
        }
        let first: pred;
        setp.eq.u32(first, lr, 0);
        for r in 0..4 {
            let (rr, oi, oa, inr, store): (u32, u32, u64, pred, pred);
            add.u32(rr, row0, r * 16);
            setp.lt.u32(inr, rr, rows);
            and.pred(store, inr, first);
            mad.lo.u32(oi, ke, rows, rr);
            mad.wide.u32(oa, oi, 4, y);
            if store {
                st.global.f32([oa], acc[r]);
            }
        }
    }

    // ---------------------------------------------------------------- split-K decode attention

    /// One chunk of 2^cs keys (cs at most 8) of one kv head, for its 3 query heads: scores (scale q.k + relative bias -
    /// sink), their max m and sum l of exp(s - m), and acc = sum exp(s - m) v. Partials go to pm/pl [kvh][maxs][3] and
    /// pacc [kvh][maxs][3][128]. grid = (8 kv heads, chunks); chunks past the keys of the token do nothing. Only rows
    /// below `vram` count: the chunks start at the distance of row vram - 1 if the token is past it (the CPU takes the
    /// distances before, see attn_reduce).
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn attn_part(
        q: *const f32, r: *const f32, sink: *const f32, rb: *const u16, kc: *const u16, vc: *const u16, pm: *mut f32, pl: *mut f32,
        pacc: *mut f32, params: *const u32, ring: u32, extent: u32, scale: f32, maxs: u32, window: u32, cs: u32, vram: u32,
    ) {
        let qsh: shared<[f32; 384], 16>;
        let rsh: shared<[f32; 48], 16>;
        let ksh: shared<[f32; 4], 16>;
        let sc: shared<[f32; 768], 16>;
        let red: shared<[f32; 1536], 16>;
        let (kvh, sp, t): (u32, u32, u32);
        mov.u32(kvh, ctaid.x);
        mov.u32(sp, ctaid.y);
        mov.u32(t, tid.x);
        let (pos, nk, pmod, start, cnt, csz, d0): (u32, u32, u32, u32, u32, u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(nk, pos, 1);
        min.u32(nk, nk, window);
        // distances below d0 are rows past the first `vram`, in host memory: the CPU's
        max.u32(d0, nk, vram);
        sub.u32(d0, d0, vram);
        shl.b32(start, sp, cs);
        add.u32(start, start, d0);
        let none: pred;
        setp.ge.u32(none, start, nk);
        if none {
            ret();
        }
        sub.u32(cnt, nk, start);
        mov.u32(csz, 1);
        shl.b32(csz, csz, cs);
        min.u32(cnt, cnt, csz);
        // the row of the token's own position
        mov.u32(pmod, pos);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(pmod, pos, ring);
        }
        let (qb, rbs, kb, sb): (u32, u32, u32, u32);
        mov.u32(qb, qsh);
        mov.u32(rbs, rsh);
        mov.u32(kb, ksh);
        mov.u32(sb, sc);
        // q of heads 3 kvh .. 3 kvh + 3 (scaled), r, sinks
        let h0: u32;
        mul.lo.u32(h0, kvh, 3);
        for k in 0..3 {
            let (qi, qa2): (u32, u32);
            let qa: u64;
            let v: f32;
            mad.lo.u32(qi, h0, 128, t);
            add.u32(qi, qi, k * 128);
            mad.wide.u32(qa, qi, 4, q);
            ld.global.f32(v, [qa]);
            mul.rn.f32(v, v, scale);
            mad.lo.u32(qa2, t, 4, qb);
            st.shared.f32([qa2 + (k * 512)], v);
        }
        let lt48: pred;
        setp.lt.u32(lt48, t, 48);
        if lt48 {
            let (ri, ra2): (u32, u32);
            let ra: u64;
            let v: f32;
            mad.lo.u32(ri, h0, 16, t);
            mad.wide.u32(ra, ri, 4, r);
            ld.global.f32(v, [ra]);
            mad.lo.u32(ra2, t, 4, rbs);
            st.shared.f32([ra2], v);
        }
        let lt3: pred;
        setp.lt.u32(lt3, t, 3);
        if lt3 {
            let (si, sa2): (u32, u32);
            let sa: u64;
            let v: f32;
            add.u32(si, h0, t);
            mad.wide.u32(sa, si, 4, sink);
            ld.global.f32(v, [sa]);
            mad.lo.u32(sa2, t, 4, kb);
            st.shared.f32([sa2], v);
        }
        bar.sync(0);

        // scores: a key per thread
        let jj: u32;
        mov.u32(jj, t);
        label(stop);
        let sdone: pred;
        setp.ge.u32(sdone, jj, cnt);
        if sdone {
            bra(send);
        }
        score_dist(start, jj, kvh, pos, ring, extent, kc, rb, qb, rbs, kb, sb);
        add.u32(jj, jj, 128);
        bra(stop);
        label(send);
        bar.sync(0);

        // max and exp per head: warp h takes head h
        let (lane, w): (u32, u32);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        let no_head: pred;
        setp.ge.u32(no_head, w, 3);
        if no_head {
            bra(sm_skip);
        }
        softmax_part(w, lane, cnt, sb, kvh, sp, maxs, pm, pl);
        label(sm_skip);
        bar.sync(0);

        // acc: warp w takes keys w, w + 4, .. and lane l dimensions 4 l .. 4 l + 4 (8 bytes of each row), 8 keys at a time
        // so that 8 loads are in flight; past cnt a key repeats key cnt - 1 with p = 0. The row of distance d is pmod - d,
        // plus ring where that is below 0 (d < window <= ring).
        let acc: [f32; 12]; // [head][4 dimensions]
        for i in 0..12 {
            mov.f32(acc[i], 0.0);
        }
        let (jb, last): (u32, u32);
        mov.u32(jb, w);
        sub.u32(last, cnt, 1);
        label(vtop);
        let vdone: pred;
        setp.ge.u32(vdone, jb, cnt);
        if vdone {
            bra(vend);
        }
        let vh: [f16; 32];
        let pp: [f32; 24];
        for u in 0..8 {
            let (jj, jc, dist, row, rw, vo, pa): (u32, u32, u32, u32, u32, u32, u32);
            let (ok, wrap): (pred, pred);
            let va: u64;
            add.u32(jj, jb, u * 4);
            setp.lt.u32(ok, jj, cnt);
            selp.b32(jc, jj, last, ok);
            add.u32(dist, start, jc);
            sub.u32(row, pmod, dist);
            setp.gt.u32(wrap, dist, pmod);
            add.u32(rw, row, ring);
            selp.b32(row, rw, row, wrap);
            mad.lo.u32(vo, row, 8, kvh);
            shl.b32(vo, vo, 7);
            mad.lo.u32(vo, lane, 4, vo);
            mad.wide.u32(va, vo, 2, vc);
            ld.global.nc.v4.b16((vh[4 * u], vh[4 * u + 1], vh[4 * u + 2], vh[4 * u + 3]), [va]);
            mad.lo.u32(pa, jc, 4, sb);
            for h in 0..3 {
                ld.shared.f32(pp[3 * u + h], [pa + (h * 1024)]);
                selp.f32(pp[3 * u + h], pp[3 * u + h], 0.0, ok);
            }
        }
        for u in 0..8 {
            for i in 0..4 {
                let vf: f32;
                cvt.f32.f16(vf, vh[4 * u + i]);
                for h in 0..3 {
                    fma.rn.f32(acc[4 * h + i], pp[3 * u + h], vf, acc[4 * h + i]);
                }
            }
        }
        add.u32(jb, jb, 32);
        bra(vtop);
        label(vend);
        // the 4 warps' sums to shared memory, [warp][head][128], then thread t adds up dimension t of each head
        let (rdb, ra, rt): (u32, u32, u32);
        mov.u32(rdb, red);
        mad.lo.u32(ra, w, 1536, rdb);
        mad.lo.u32(ra, lane, 16, ra);
        for h in 0..3 {
            st.shared.v4.f32([ra + (h * 512)], (acc[4 * h], acc[4 * h + 1], acc[4 * h + 2], acc[4 * h + 3]));
        }
        bar.sync(0);
        mad.lo.u32(rt, t, 4, rdb);
        let a: [f32; 3];
        for h in 0..3 {
            let (x1, x2, x3): (f32, f32, f32);
            ld.shared.f32(a[h], [rt + (h * 512)]);
            ld.shared.f32(x1, [rt + (1536 + h * 512)]);
            ld.shared.f32(x2, [rt + (3072 + h * 512)]);
            ld.shared.f32(x3, [rt + (4608 + h * 512)]);
            add.rn.f32(a[h], a[h], x1);
            add.rn.f32(a[h], a[h], x2);
            add.rn.f32(a[h], a[h], x3);
        }
        let (po, pao): (u32, u64);
        mad.lo.u32(po, kvh, maxs, sp);
        mul.lo.u32(po, po, 384);
        add.u32(po, po, t);
        mad.wide.u32(pao, po, 4, pacc);
        st.global.f32([pao], a[0]);
        st.global.f32([pao + 512], a[1]);
        st.global.f32([pao + 1024], a[2]);
    }

    /// Warp `h`: the max m of the cnt scores of head h, p = exp(s - m) in place, and l = sum p; stores m and l.
    fn softmax_part(h: u32, lane: u32, cnt: u32, sb: u32, kvh: u32, sp: u32, maxs: u32, pm: u64, pl: u64) {
        let (hb, i): (u32, u32);
        mad.lo.u32(hb, h, 1024, sb);
        let m: f32;
        mov.f32(m, -3.0e38);
        mov.u32(i, lane);
        label(mt);
        let md: pred;
        setp.ge.u32(md, i, cnt);
        if md {
            bra(me);
        }
        let (a, v): (u32, f32);
        mad.lo.u32(a, i, 4, hb);
        ld.shared.f32(v, [a]);
        max.f32(m, m, v);
        add.u32(i, i, 32);
        bra(mt);
        label(me);
        warp_max(m);
        let l: f32;
        mov.f32(l, 0.0);
        mov.u32(i, lane);
        label(et);
        let ed: pred;
        setp.ge.u32(ed, i, cnt);
        if ed {
            bra(ee);
        }
        let (a2, v2): (u32, f32);
        mad.lo.u32(a2, i, 4, hb);
        ld.shared.f32(v2, [a2]);
        sub.rn.f32(v2, v2, m);
        exp(v2, v2);
        st.shared.f32([a2], v2);
        add.rn.f32(l, l, v2);
        add.u32(i, i, 32);
        bra(et);
        label(ee);
        warp_sum(l);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            let o: u32;
            let (ma, la): (u64, u64);
            mad.lo.u32(o, kvh, maxs, sp);
            mad.lo.u32(o, o, 3, h);
            mad.wide.u32(ma, o, 4, pm);
            mad.wide.u32(la, o, 4, pl);
            st.global.f32([ma], m);
            st.global.f32([la], l);
        }
    }

    /// Combines the chunks of 2^cs keys of attn_part, the CPU's partial of the rows past the first `vram` if the token
    /// is past them (cpu [24][130]: m, l, acc[128] of each head), and the zero sink logit: out[h][d], a block per query
    /// head, a thread per dimension.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn attn_reduce(
        pm: *const f32, pl: *const f32, pacc: *const f32, out: *mut f32, params: *const u32, window: u32, maxs: u32, cs: u32, cpu: *const f32, vram: u32,
    ) {
        let (h, t, kvh, hh): (u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(t, tid.x);
        div.u32(kvh, h, 3);
        rem.u32(hh, h, 3);
        let (pos, nk, pmod, ns, d0): (u32, u32, u32, u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(nk, pos, 1);
        min.u32(nk, nk, window);
        max.u32(d0, nk, vram);
        sub.u32(d0, d0, vram);
        mov.u32(pmod, 0);
        mov.u32(ns, 1);
        shl.b32(ns, ns, cs);
        sub.u32(ns, ns, 1);
        add.u32(ns, ns, nk);
        sub.u32(ns, ns, d0);
        shr.u32(ns, ns, cs);
        let has_cpu: pred;
        setp.ne.u32(has_cpu, d0, 0);
        let (cm, cl, ca): (f32, f32, f32);
        mov.f32(cm, 0.0);
        mov.f32(cl, 0.0);
        mov.f32(ca, 0.0);
        if has_cpu {
            let (ci, ct): (u32, u32);
            let (cma, caa): (u64, u64);
            mul.lo.u32(ci, h, 130);
            mad.wide.u32(cma, ci, 4, cpu);
            ld.volatile.global.f32(cm, [cma]);
            ld.volatile.global.f32(cl, [cma + 4]);
            add.u32(ct, ci, 2);
            add.u32(ct, ct, t);
            mad.wide.u32(caa, ct, 4, cpu);
            ld.volatile.global.f32(ca, [caa]);
        }
        let (mm, sp): (f32, u32);
        mov.f32(mm, 0.0);
        if has_cpu {
            max.f32(mm, mm, cm);
        }
        mov.u32(sp, 0);
        label(mt);
        let md: pred;
        setp.ge.u32(md, sp, ns);
        if md {
            bra(me);
        }
        let o: u32;
        let ma: u64;
        let mv: f32;
        mad.lo.u32(o, kvh, maxs, sp);
        mad.lo.u32(o, o, 3, hh);
        mad.wide.u32(ma, o, 4, pm);
        ld.global.f32(mv, [ma]);
        max.f32(mm, mm, mv);
        add.u32(sp, sp, 1);
        bra(mt);
        label(me);
        let (ll, acc): (f32, f32);
        sub.rn.f32(ll, 0.0, mm);
        exp(ll, ll);
        mov.f32(acc, 0.0);
        if has_cpu {
            let ce: f32;
            sub.rn.f32(ce, cm, mm);
            exp(ce, ce);
            fma.rn.f32(ll, cl, ce, ll);
            fma.rn.f32(acc, ca, ce, acc);
        }
        mov.u32(sp, 0);
        label(at);
        let ad: pred;
        setp.ge.u32(ad, sp, ns);
        if ad {
            bra(ae);
        }
        let (o2, o3): (u32, u32);
        let (ma2, la2, aa2): (u64, u64, u64);
        let (mv2, lv2, av2, e2): (f32, f32, f32, f32);
        mad.lo.u32(o2, kvh, maxs, sp);
        mad.lo.u32(o3, o2, 384, t);
        mad.lo.u32(o3, hh, 128, o3);
        mad.lo.u32(o2, o2, 3, hh);
        mad.wide.u32(ma2, o2, 4, pm);
        mad.wide.u32(la2, o2, 4, pl);
        mad.wide.u32(aa2, o3, 4, pacc);
        ld.global.f32(mv2, [ma2]);
        ld.global.f32(lv2, [la2]);
        ld.global.f32(av2, [aa2]);
        sub.rn.f32(e2, mv2, mm);
        exp(e2, e2);
        fma.rn.f32(ll, lv2, e2, ll);
        fma.rn.f32(acc, av2, e2, acc);
        add.u32(sp, sp, 1);
        bra(at);
        label(ae);
        div.rn.f32(acc, acc, ll);
        let (oo, oa): (u32, u64);
        mad.lo.u32(oo, h, 128, t);
        mad.wide.u32(oa, oo, 4, out);
        st.global.f32([oa], acc);
    }

    /// Top 8 of sigmoid(logits) + bias over 128 experts; weights = the sigmoids of the winners, normalized and scaled.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn router_topk(logits: *const f32, bias: *const f32, ids: *mut u32, wts: *mut f32, scale: f32) {
        let sel: shared<[f32; 128], 16>;
        let prb: shared<[f32; 128], 16>;
        let win: shared<[u32; 8], 16>;
        let (e, tk, te): (u32, u32, u32);
        mov.u32(e, tid.x);
        mov.u32(tk, ctaid.x);
        mad.lo.u32(te, tk, 128, e);
        let (la, ba): (u64, u64);
        let (lg, p, bv, sv): (f32, f32, f32, f32);
        mad.wide.u32(la, te, 4, logits);
        ld.global.f32(lg, [la]);
        sigmoid(p, lg);
        mad.wide.u32(ba, e, 4, bias);
        ld.global.nc.f32(bv, [ba]);
        add.rn.f32(sv, p, bv);
        let (sb, pb, wb, at): (u32, u32, u32, u32);
        mov.u32(sb, sel);
        mov.u32(pb, prb);
        mov.u32(wb, win);
        mad.lo.u32(at, e, 4, sb);
        st.shared.f32([at], sv);
        mad.lo.u32(at, e, 4, pb);
        st.shared.f32([at], p);
        bar.sync(0);
        let lane: u32;
        and.b32(lane, e, 31);
        let first: pred;
        setp.eq.u32(first, e, 0);
        for k in 0..8 {
            let (best, cand): (f32, f32);
            let (bi, ci): (u32, u32);
            mov.f32(best, -3.0e38);
            mov.u32(bi, 0);
            for q4 in 0..4 {
                let ca: u32;
                add.u32(ci, lane, q4 * 32);
                mad.lo.u32(ca, ci, 4, sb);
                ld.shared.f32(cand, [ca]);
                let better: pred;
                setp.gt.f32(better, cand, best);
                selp.f32(best, cand, best, better);
                selp.u32(bi, ci, bi, better);
            }
            for sh in [16, 8, 4, 2, 1] {
                let (ob, oi): (f32, u32);
                shfl.sync.bfly.b32(ob, best, sh, 31, 0xffff_ffff);
                shfl.sync.bfly.b32(oi, bi, sh, 31, 0xffff_ffff);
                let (gt, eq, lt, take): (pred, pred, pred, pred);
                setp.gt.f32(gt, ob, best);
                setp.eq.f32(eq, ob, best);
                setp.lt.u32(lt, oi, bi);
                and.pred(take, eq, lt);
                or.pred(take, take, gt);
                selp.f32(best, ob, best, take);
                selp.u32(bi, oi, bi, take);
            }
            bar.sync(0);
            if first {
                let kill: u32;
                st.shared.u32([wb + (k * 4)], bi);
                mad.lo.u32(kill, bi, 4, sb);
                st.shared.f32([kill], -3.0e38);
            }
            bar.sync(0);
        }
        let my: pred;
        setp.lt.u32(my, e, 8);
        if my {
            let (sum, pv, own): (f32, f32, f32);
            let (id, ia): (u32, u32);
            mov.f32(sum, 0.0);
            for k in 0..8 {
                let (wk, pa): (u32, u32);
                ld.shared.u32(wk, [wb + (k * 4)]);
                mad.lo.u32(pa, wk, 4, pb);
                ld.shared.f32(pv, [pa]);
                add.rn.f32(sum, sum, pv);
            }
            add.rn.f32(sum, sum, 0.000_000_000_000_000_000_01);
            mad.lo.u32(ia, e, 4, wb);
            ld.shared.u32(id, [ia]);
            let pa2: u32;
            mad.lo.u32(pa2, id, 4, pb);
            ld.shared.f32(own, [pa2]);
            div.rn.f32(own, own, sum);
            mul.rn.f32(own, own, scale);
            let (a1, a2): (u64, u64);
            let o8: u32;
            mad.lo.u32(o8, tk, 8, e);
            mad.wide.u32(a1, o8, 4, ids);
            mad.wide.u32(a2, o8, 4, wts);
            st.global.u32([a1], id);
            st.global.f32([a2], own);
        }
    }

    // ================================================================ batched (prefill) kernels

    /// Short convolution of `ntok` consecutive tokens of one sequence, a thread per channel (see conv4). x, y [ntok][c];
    /// taps [ntok][c / head_dim][4] where DYN.
    #[maxntid(256)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn conv4_seq<const DYN: u32>(
        x: *const f32, w: *const u16, state: *mut f32, taps: *const f32, y: *mut f32, c_total: u32, head_dim: u32, scale: f32, ntok: u32, taps_stride: u32,
        x_stride: u32,
    ) {
        let (b, nt, t, c): (u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, b, nt, t);
        let past: pred;
        setp.ge.u32(past, c, c_total);
        if past {
            ret();
        }
        let sa: u64;
        let (c3, hd): (u32, u32);
        mul.lo.u32(c3, c, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (h0, h1, h2): (f32, f32, f32);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        let (w0, w1, w2, w3): (f32, f32, f32, f32);
        conv_w::<DYN>(w0, w1, w2, w3, w, c, c_total);
        div.u32(hd, c, head_dim);
        shl.b32(hd, hd, 2);
        let tk: u32;
        mov.u32(tk, 0);
        label(top);
        let done: pred;
        setp.ge.u32(done, tk, ntok);
        if done {
            bra(end);
        }
        let (xi, yi, xa, ya): (u32, u32, u64, u64);
        let xv: f32;
        mad.lo.u32(xi, tk, x_stride, c);
        mad.lo.u32(yi, tk, c_total, c);
        mad.wide.u32(xa, xi, 4, x);
        mad.wide.u32(ya, yi, 4, y);
        ld.global.f32(xv, [xa]);
        let o: f32;
        mul.rn.f32(o, w0, h0);
        fma.rn.f32(o, w1, h1, o);
        fma.rn.f32(o, w2, h2, o);
        fma.rn.f32(o, w3, xv, o);
        if DYN == 1 {
            let (ti, ta): (u32, u64);
            mad.lo.u32(ti, tk, taps_stride, hd);
            mad.wide.u32(ta, ti, 4, taps);
            let (t0, t1, t2, t3): (f32, f32, f32, f32);
            ld.global.f32(t0, [ta]);
            ld.global.f32(t1, [ta + 4]);
            ld.global.f32(t2, [ta + 8]);
            ld.global.f32(t3, [ta + 12]);
            let dsum: f32;
            mul.rn.f32(dsum, xv, t0);
            fma.rn.f32(dsum, h2, t1, dsum);
            fma.rn.f32(dsum, h1, t2, dsum);
            fma.rn.f32(dsum, h0, t3, dsum);
            add.rn.f32(o, o, dsum);
        }
        add.rn.f32(o, o, xv);
        mul.rn.f32(o, o, scale);
        st.global.f32([ya], o);
        mov.f32(h0, h1);
        mov.f32(h1, h2);
        mov.f32(h2, xv);
        add.u32(tk, tk, 1);
        bra(top);
        label(end);
        st.global.f32([sa], h0);
        st.global.f32([sa + 4], h1);
        st.global.f32([sa + 8], h2);
    }

    /// K and V of `ntok` tokens at positions params[0] + t, as f16 at row pos % ring (pos where ring is 0).
    /// k, v: [ntok][stride] floats (the first nkv used). grid.y = token.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn kv_store_t(k: *const f32, v: *const f32, kc: *mut u16, vc: *mut u16, params: *const u32, ring: u32, nkv: u32, kstride: u32, vstride: u32) {
        let (b, nt, t, c, tk): (u32, u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mov.u32(tk, ctaid.y);
        mad.lo.u32(c, b, nt, t);
        let past: pred;
        setp.ge.u32(past, c, nkv);
        if past {
            ret();
        }
        let (pos, row): (u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(pos, pos, tk);
        mov.u32(row, pos);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(row, pos, ring);
        }
        let (ki, vi, o): (u32, u32, u32);
        let (ka, va, kca, vca): (u64, u64, u64, u64);
        mad.lo.u32(ki, tk, kstride, c);
        mad.lo.u32(vi, tk, vstride, c);
        mad.wide.u32(ka, ki, 4, k);
        mad.wide.u32(va, vi, 4, v);
        let (kv, vv): (f32, f32);
        let (kh, vh): (f16, f16);
        ld.global.f32(kv, [ka]);
        ld.global.f32(vv, [va]);
        cvt.rn.f16.f32(kh, kv);
        cvt.rn.f16.f32(vh, vv);
        mad.lo.u32(o, row, nkv, c);
        mad.wide.u32(kca, o, 2, kc);
        mad.wide.u32(vca, o, 2, vc);
        st.global.b16([kca], kh);
        st.global.b16([vca], vh);
    }

    /// Prefill attention: a block per (kv head, query token), all keys of the token in chunks of 256 (by distance
    /// 0, 1, ..), online softmax; the sink is the initial state (m = 0, l = 1). Token t is at position params[0] + t;
    /// it sees distances below `window` (SWA) or all; row of a position: pos % ring, or pos where ring is 0.
    /// q [t][q_stride] (3 heads of 128 per kv head), r [t][r_stride], sink [t][s_stride], out [t][3072].
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn attn_prefill(
        q: *const f32, r: *const f32, sink: *const f32, rb: *const u16, kc: *const u16, vc: *const u16, out: *mut f32, params: *const u32,
        ring: u32, window: u32, extent: u32, scale: f32, q_stride: u32, r_stride: u32, s_stride: u32,
    ) {
        let qsh: shared<[f32; 384], 16>;
        let rsh: shared<[f32; 48], 16>;
        let ksh: shared<[f32; 4], 16>;
        let sc: shared<[f32; 768], 16>;
        let ml: shared<[f32; 12], 16>; // m[3], l[3], alpha[3]
        let (kvh, tk, t): (u32, u32, u32);
        mov.u32(kvh, ctaid.x);
        mov.u32(tk, ctaid.y);
        mov.u32(t, tid.x);
        let (pos, nk): (u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(pos, pos, tk);
        add.u32(nk, pos, 1);
        min.u32(nk, nk, window);
        let (qb, rbs, kb, sb, mb): (u32, u32, u32, u32, u32);
        mov.u32(qb, qsh);
        mov.u32(rbs, rsh);
        mov.u32(kb, ksh);
        mov.u32(sb, sc);
        mov.u32(mb, ml);
        let h0: u32;
        mul.lo.u32(h0, kvh, 3);
        for k in 0..3 {
            let (qi, qa2): (u32, u32);
            let qa: u64;
            let v: f32;
            mad.lo.u32(qi, tk, q_stride, t);
            mad.lo.u32(qi, h0, 128, qi);
            add.u32(qi, qi, k * 128);
            mad.wide.u32(qa, qi, 4, q);
            ld.global.f32(v, [qa]);
            mul.rn.f32(v, v, scale);
            mad.lo.u32(qa2, t, 4, qb);
            st.shared.f32([qa2 + (k * 512)], v);
        }
        let lt48: pred;
        setp.lt.u32(lt48, t, 48);
        if lt48 {
            let (ri, ra2): (u32, u32);
            let ra: u64;
            let v: f32;
            mad.lo.u32(ri, tk, r_stride, t);
            mad.lo.u32(ri, h0, 16, ri);
            mad.wide.u32(ra, ri, 4, r);
            ld.global.f32(v, [ra]);
            mad.lo.u32(ra2, t, 4, rbs);
            st.shared.f32([ra2], v);
        }
        let lt3: pred;
        setp.lt.u32(lt3, t, 3);
        if lt3 {
            let (si, sa2, ma): (u32, u32, u32);
            let sa: u64;
            let v: f32;
            mad.lo.u32(si, tk, s_stride, h0);
            add.u32(si, si, t);
            mad.wide.u32(sa, si, 4, sink);
            ld.global.f32(v, [sa]);
            mad.lo.u32(sa2, t, 4, kb);
            st.shared.f32([sa2], v);
            mad.lo.u32(ma, t, 4, mb);
            st.shared.f32([ma], 0.0);
            st.shared.f32([ma + 12], 1.0);
        }
        bar.sync(0);
        let (a0, a1, a2): (f32, f32, f32);
        mov.f32(a0, 0.0);
        mov.f32(a1, 0.0);
        mov.f32(a2, 0.0);
        let (lane, w): (u32, u32);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        let start: u32;
        mov.u32(start, 0);
        label(ctop);
        let cdone: pred;
        setp.ge.u32(cdone, start, nk);
        if cdone {
            bra(cend);
        }
        let cnt: u32;
        sub.u32(cnt, nk, start);
        min.u32(cnt, cnt, 256);
        // scores of distances start + jj
        let jj: u32;
        mov.u32(jj, t);
        label(stop);
        let sdone: pred;
        setp.ge.u32(sdone, jj, cnt);
        if sdone {
            bra(send);
        }
        score_dist(start, jj, kvh, pos, ring, extent, kc, rb, qb, rbs, kb, sb);
        add.u32(jj, jj, 128);
        bra(stop);
        label(send);
        bar.sync(0);
        let no_head: pred;
        setp.ge.u32(no_head, w, 3);
        if no_head {
            bra(sm_skip);
        }
        softmax_online(w, lane, cnt, sb, mb);
        label(sm_skip);
        bar.sync(0);
        // acc = acc * alpha + sum p v
        let (al0, al1, al2): (f32, f32, f32);
        ld.shared.f32(al0, [mb + 24]);
        ld.shared.f32(al1, [mb + 28]);
        ld.shared.f32(al2, [mb + 32]);
        mul.rn.f32(a0, a0, al0);
        mul.rn.f32(a1, a1, al1);
        mul.rn.f32(a2, a2, al2);
        let j2: u32;
        mov.u32(j2, 0);
        label(vtop);
        let vdone: pred;
        setp.ge.u32(vdone, j2, cnt);
        if vdone {
            bra(vend);
        }
        let (dist, kp, row, vo, pa): (u32, u32, u32, u32, u32);
        add.u32(dist, start, j2);
        sub.u32(kp, pos, dist);
        mov.u32(row, kp);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(row, kp, ring);
        }
        mad.lo.u32(vo, row, 8, kvh);
        shl.b32(vo, vo, 7);
        add.u32(vo, vo, t);
        let va: u64;
        mad.wide.u32(va, vo, 2, vc);
        let vh: f16;
        let (vf, p0, p1, p2): (f32, f32, f32, f32);
        ld.global.nc.b16(vh, [va]);
        cvt.f32.f16(vf, vh);
        mad.lo.u32(pa, j2, 4, sb);
        ld.shared.f32(p0, [pa]);
        ld.shared.f32(p1, [pa + 1024]);
        ld.shared.f32(p2, [pa + 2048]);
        fma.rn.f32(a0, p0, vf, a0);
        fma.rn.f32(a1, p1, vf, a1);
        fma.rn.f32(a2, p2, vf, a2);
        add.u32(j2, j2, 1);
        bra(vtop);
        label(vend);
        bar.sync(0);
        add.u32(start, start, 256);
        bra(ctop);
        label(cend);
        let (l0, l1, l2): (f32, f32, f32);
        ld.shared.f32(l0, [mb + 12]);
        ld.shared.f32(l1, [mb + 16]);
        ld.shared.f32(l2, [mb + 20]);
        div.rn.f32(a0, a0, l0);
        div.rn.f32(a1, a1, l1);
        div.rn.f32(a2, a2, l2);
        let (oo, oa): (u32, u64);
        mad.lo.u32(oo, tk, 3072, t);
        mad.lo.u32(oo, h0, 128, oo);
        mad.wide.u32(oa, oo, 4, out);
        st.global.f32([oa], a0);
        st.global.f32([oa + 512], a1);
        st.global.f32([oa + 1024], a2);
    }

    /// The scores of distance start + jj (key position pos - dist) for the 3 heads in shared q, into sc[h][jj].
    fn score_dist(start: u32, jj: u32, kvh: u32, pos: u32, ring: u32, extent: u32, kc: u64, rb: u64, qb: u32, rbs: u32, kb: u32, sb: u32) {
        let (dist, kp, row, ko): (u32, u32, u32, u32);
        add.u32(dist, start, jj);
        sub.u32(kp, pos, dist);
        mov.u32(row, kp);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(row, kp, ring);
        }
        mad.lo.u32(ko, row, 8, kvh);
        shl.b32(ko, ko, 7);
        let ka: u64;
        mad.wide.u32(ka, ko, 2, kc);
        let (s0, s1, s2): (f32, f32, f32);
        mov.f32(s0, 0.0);
        mov.f32(s1, 0.0);
        mov.f32(s2, 0.0);
        for c in 0..32 {
            let (h0, h1, h2, h3): (f16, f16, f16, f16);
            ld.global.nc.v4.b16((h0, h1, h2, h3), [ka + (c * 8)]);
            let (k0, k1, k2, k3): (f32, f32, f32, f32);
            cvt.f32.f16(k0, h0);
            cvt.f32.f16(k1, h1);
            cvt.f32.f16(k2, h2);
            cvt.f32.f16(k3, h3);
            for hh in 0..3 {
                let (q0, q1, q2, q3): (f32, f32, f32, f32);
                ld.shared.v4.f32((q0, q1, q2, q3), [qb + (hh * 512 + c * 16)]);
                if hh == 0 {
                    fma.rn.f32(s0, q0, k0, s0);
                    fma.rn.f32(s0, q1, k1, s0);
                    fma.rn.f32(s0, q2, k2, s0);
                    fma.rn.f32(s0, q3, k3, s0);
                }
                if hh == 1 {
                    fma.rn.f32(s1, q0, k0, s1);
                    fma.rn.f32(s1, q1, k1, s1);
                    fma.rn.f32(s1, q2, k2, s1);
                    fma.rn.f32(s1, q3, k3, s1);
                }
                if hh == 2 {
                    fma.rn.f32(s2, q0, k0, s2);
                    fma.rn.f32(s2, q1, k1, s2);
                    fma.rn.f32(s2, q2, k2, s2);
                    fma.rn.f32(s2, q3, k3, s2);
                }
            }
        }
        let inb: pred;
        setp.lt.u32(inb, dist, extent);
        if inb {
            // the bias profiles at this distance: proj[r][dist], r = 4 c + i (rows of 1024 bf16)
            let ba: u64;
            mad.wide.u32(ba, dist, 2, rb);
            for c in 0..4 {
                let hb: [u16; 4];
                for i in 0..4 {
                    ld.global.nc.u16(hb[i], [ba + ((c * 4 + i) * 2048)]);
                }
                let (b0, b1, b2, b3): (f32, f32, f32, f32);
                bf16f(b0, hb[0]);
                bf16f(b1, hb[1]);
                bf16f(b2, hb[2]);
                bf16f(b3, hb[3]);
                for hh in 0..3 {
                    let (r0, r1, r2, r3): (f32, f32, f32, f32);
                    ld.shared.v4.f32((r0, r1, r2, r3), [rbs + (hh * 64 + c * 16)]);
                    if hh == 0 {
                        fma.rn.f32(s0, b0, r0, s0);
                        fma.rn.f32(s0, b1, r1, s0);
                        fma.rn.f32(s0, b2, r2, s0);
                        fma.rn.f32(s0, b3, r3, s0);
                    }
                    if hh == 1 {
                        fma.rn.f32(s1, b0, r0, s1);
                        fma.rn.f32(s1, b1, r1, s1);
                        fma.rn.f32(s1, b2, r2, s1);
                        fma.rn.f32(s1, b3, r3, s1);
                    }
                    if hh == 2 {
                        fma.rn.f32(s2, b0, r0, s2);
                        fma.rn.f32(s2, b1, r1, s2);
                        fma.rn.f32(s2, b2, r2, s2);
                        fma.rn.f32(s2, b3, r3, s2);
                    }
                }
            }
        }
        let (k0s, k1s, k2s): (f32, f32, f32);
        ld.shared.f32(k0s, [kb]);
        ld.shared.f32(k1s, [kb + 4]);
        ld.shared.f32(k2s, [kb + 8]);
        sub.rn.f32(s0, s0, k0s);
        sub.rn.f32(s1, s1, k1s);
        sub.rn.f32(s2, s2, k2s);
        let sa: u32;
        mad.lo.u32(sa, jj, 4, sb);
        st.shared.f32([sa], s0);
        st.shared.f32([sa + 1024], s1);
        st.shared.f32([sa + 2048], s2);
    }

    /// Warp h, online: m' = max(m, max s), p = exp(s - m') in place, l = l alpha + sum p with alpha = exp(m - m');
    /// ml holds m[3] @0, l[3] @12, alpha[3] @24.
    fn softmax_online(h: u32, lane: u32, cnt: u32, sb: u32, mb: u32) {
        let (hb, i, ma): (u32, u32, u32);
        mad.lo.u32(hb, h, 1024, sb);
        mad.lo.u32(ma, h, 4, mb);
        let (m, mo): (f32, f32);
        ld.shared.f32(mo, [ma]);
        mov.f32(m, mo);
        mov.u32(i, lane);
        label(mt);
        let md: pred;
        setp.ge.u32(md, i, cnt);
        if md {
            bra(me);
        }
        let (a, v): (u32, f32);
        mad.lo.u32(a, i, 4, hb);
        ld.shared.f32(v, [a]);
        max.f32(m, m, v);
        add.u32(i, i, 32);
        bra(mt);
        label(me);
        warp_max(m);
        let l: f32;
        mov.f32(l, 0.0);
        mov.u32(i, lane);
        label(et);
        let ed: pred;
        setp.ge.u32(ed, i, cnt);
        if ed {
            bra(ee);
        }
        let (a2, v2): (u32, f32);
        mad.lo.u32(a2, i, 4, hb);
        ld.shared.f32(v2, [a2]);
        sub.rn.f32(v2, v2, m);
        exp(v2, v2);
        st.shared.f32([a2], v2);
        add.rn.f32(l, l, v2);
        add.u32(i, i, 32);
        bra(et);
        label(ee);
        warp_sum(l);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            let (alpha, lo): (f32, f32);
            sub.rn.f32(alpha, mo, m);
            exp(alpha, alpha);
            ld.shared.f32(lo, [ma + 12]);
            fma.rn.f32(lo, lo, alpha, l);
            st.shared.f32([ma], m);
            st.shared.f32([ma + 12], lo);
            st.shared.f32([ma + 24], alpha);
        }
    }

    // ---------------------------------------------------------------- grouped experts (prefill)

    /// out[t][c] = sum_k wts[t][k] * (slots[t][k] >= 0 ? gpu[8t + k][c] : cpu[8t + k][c]); grid.y = token.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn moe_combine_t(gpu: *const f32, cpu: *const f32, slots: *const i32, wts: *const f32, out: *mut f32, n: u32) {
        let (b, nt, t, c, tk): (u32, u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mov.u32(tk, ctaid.y);
        mad.lo.u32(c, b, nt, t);
        let past: pred;
        setp.ge.u32(past, c, n);
        if past {
            ret();
        }
        let acc: f32;
        mov.f32(acc, 0.0);
        for k in 0..8 {
            let s: s32;
            let wv: f32;
            let (pi, sa): (u32, u64);
            let wa: u64;
            mad.lo.u32(pi, tk, 8, k);
            mad.wide.u32(sa, pi, 4, slots);
            mad.wide.u32(wa, pi, 4, wts);
            ld.global.s32(s, [sa]);
            ld.global.f32(wv, [wa]);
            let on_gpu: pred;
            setp.ge.s32(on_gpu, s, 0);
            let (o, ga, ca): (u32, u64, u64);
            let v: f32;
            mad.lo.u32(o, pi, n, c);
            mad.wide.u32(ga, o, 4, gpu);
            mad.wide.u32(ca, o, 4, cpu);
            if on_gpu {
                ld.global.f32(v, [ga]);
            } else {
                ld.global.f32(v, [ca]);
            }
            fma.rn.f32(acc, wv, v, acc);
        }
        let (oi, oa): (u32, u64);
        mad.lo.u32(oi, tk, n, c);
        mad.wide.u32(oa, oi, 4, out);
        st.global.f32([oa], acc);
    }

    // ================================================================ fused decode kernels

    /// Attention over depth: the states are `ptrs[0..ns]` (n floats each, ns <= 8), each RMS-normalized; query `q` of
    /// `qs[blockIdx.x]` scores them by `dot/sqrt(n)`, and `out[blockIdx.x]` is the softmax-weighted sum of the normalized
    /// states. One pass over the states: per state its sum of squares and dot with the query, one block reduction of the
    /// 16 numbers, then the weighted sum. grid (nq, ntok), 256 threads.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn mix8(ptrs: *const u64, ns: u32, qs: *const f32, out: *mut f32, n: u32) {
        let red: shared<[f32; 128], 16>;
        let tot: shared<[f32; 16], 16>;
        let (tk, qi, nq, t, lane, w): (u32, u32, u32, u32, u32, u32);
        mov.u32(tk, ctaid.y);
        mov.u32(qi, ctaid.x);
        mov.u32(nq, nctaid.x);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        let (tko, qoff, ooff): (u32, u32, u32);
        mul.lo.u32(tko, tk, n);
        mul.lo.u32(qoff, qi, n);
        mad.lo.u32(ooff, tk, nq, qi);
        mul.lo.u32(ooff, ooff, n);
        let (qv, ov): (u64, u64);
        mad.wide.u32(qv, qoff, 4, qs);
        mad.wide.u32(ov, ooff, 4, out);
        let sp: [u64; 8];
        let ss: [f32; 8];
        let dq: [f32; 8];
        for j in 0..8 {
            mov.f32(ss[j], 0.0);
            mov.f32(dq[j], 0.0);
            mov.u64(sp[j], 0);
            let pj: pred;
            setp.lt.u32(pj, ns, j + 1);
            if !pj {
                ld.global.u64(sp[j], [ptrs + (j * 8)]);
                mad.wide.u32(sp[j], tko, 4, sp[j]);
            }
        }
        let i: u32;
        mov.u32(i, t);
        label(top);
        let done: pred;
        setp.ge.u32(done, i, n);
        if done {
            bra(end);
        }
        let qa: u64;
        let qx: f32;
        mad.wide.u32(qa, i, 4, qv);
        ld.global.nc.f32(qx, [qa]);
        for j in 0..8 {
            let pj: pred;
            setp.lt.u32(pj, ns, j + 1);
            if !pj {
                let xa: u64;
                let xv: f32;
                mad.wide.u32(xa, i, 4, sp[j]);
                ld.global.f32(xv, [xa]);
                fma.rn.f32(ss[j], xv, xv, ss[j]);
                fma.rn.f32(dq[j], xv, qx, dq[j]);
            }
        }
        add.u32(i, i, 256);
        bra(top);
        label(end);
        let (rb, tb): (u32, u32);
        mov.u32(rb, red);
        mov.u32(tb, tot);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        let wa: u32;
        mad.lo.u32(wa, w, 64, rb);
        for j in 0..8 {
            warp_sum(ss[j]);
            warp_sum(dq[j]);
        }
        if first {
            for j in 0..8 {
                st.shared.f32([wa + (j * 4)], ss[j]);
                st.shared.f32([wa + (32 + j * 4)], dq[j]);
            }
        }
        bar.sync(0);
        let lt16: pred;
        setp.lt.u32(lt16, t, 16);
        if lt16 {
            let (sum, v): (f32, f32);
            let a: u32;
            mov.f32(sum, 0.0);
            mad.lo.u32(a, t, 4, rb);
            for k in 0..8 {
                ld.shared.f32(v, [a + (k * 64)]);
                add.rn.f32(sum, sum, v);
            }
            mad.lo.u32(a, t, 4, tb);
            st.shared.f32([a], sum);
        }
        bar.sync(0);
        // softmax over the states, in every thread: coef_j = p_j / rms_j
        let (nf, isn, m, l): (f32, f32, f32, f32);
        cvt.rn.f32.u32(nf, n);
        sqrt.rn.f32(isn, nf);
        rcp.rn.f32(isn, isn);
        mov.f32(m, -3.0e38);
        let sc: [f32; 8];
        let rs: [f32; 8];
        for j in 0..8 {
            let (a2, b2): (f32, f32);
            ld.shared.f32(a2, [tb + (j * 4)]);
            ld.shared.f32(b2, [tb + (32 + j * 4)]);
            div.rn.f32(a2, a2, nf);
            add.rn.f32(a2, a2, 0.000_001);
            sqrt.rn.f32(a2, a2);
            rcp.rn.f32(rs[j], a2);
            mul.rn.f32(sc[j], b2, rs[j]);
            mul.rn.f32(sc[j], sc[j], isn);
            let pj: pred;
            setp.lt.u32(pj, ns, j + 1);
            if pj {
                mov.f32(sc[j], -3.0e38);
            }
            max.f32(m, m, sc[j]);
        }
        mov.f32(l, 0.0);
        for j in 0..8 {
            sub.rn.f32(sc[j], sc[j], m);
            exp(sc[j], sc[j]);
            add.rn.f32(l, l, sc[j]);
        }
        rcp.rn.f32(l, l);
        for j in 0..8 {
            mul.rn.f32(sc[j], sc[j], l);
            mul.rn.f32(sc[j], sc[j], rs[j]);
        }
        mov.u32(i, t);
        label(otop);
        let odone: pred;
        setp.ge.u32(odone, i, n);
        if odone {
            bra(oend);
        }
        let acc: f32;
        mov.f32(acc, 0.0);
        for j in 0..8 {
            let pj: pred;
            setp.lt.u32(pj, ns, j + 1);
            if !pj {
                let xa: u64;
                let xv: f32;
                mad.wide.u32(xa, i, 4, sp[j]);
                ld.global.f32(xv, [xa]);
                fma.rn.f32(acc, sc[j], xv, acc);
            }
        }
        let oa: u64;
        mad.wide.u32(oa, i, 4, ov);
        st.global.f32([oa], acc);
        add.u32(i, i, 256);
        bra(otop);
        label(oend);
    }

    /// K and V of one token, a block per kv head (128 threads, a channel each): the short convolutions (with the taps of
    /// full attention where DYN), the K clip and the V norm, stored as f32 (k, v) and as f16 at the cache row of pos.
    #[reqntid(128)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn kv_prep<const DYN: u32>(
        gk: *const f32, gv: *const f32, kw: *const u16, vw: *const u16, state: *mut f32, k: *mut f32, v: *mut f32, kc: *mut u16, vc: *mut u16,
        params: *const u32, ring: u32, cscale: f32, thr: f32, eps: f32,
    ) {
        let red: shared<[f32; 32], 16>;
        let (h, t, c): (u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, h, 128, t);
        let (ko, vo): (f32, f32);
        conv_ch::<DYN, 0>(ko, gk, kw, state, c, h, cscale);
        conv_ch::<DYN, 1024>(vo, gv, vw, state, c, h, cscale);
        let r: u32;
        mov.u32(r, red);
        let (ss, f, norm, vs): (f32, f32, f32, f32);
        mul.rn.f32(ss, ko, ko);
        block_sum(ss, r);
        sqrt.rn.f32(norm, ss);
        add.rn.f32(norm, norm, 0.000_001);
        div.rn.f32(f, thr, norm);
        max.f32(f, f, 0.0);
        min.f32(f, f, 1.0);
        mul.rn.f32(ko, ko, f);
        mul.rn.f32(vs, vo, vo);
        block_sum(vs, r);
        div.rn.f32(vs, vs, 128.0);
        add.rn.f32(vs, vs, eps);
        sqrt.rn.f32(vs, vs);
        div.rn.f32(vo, vo, vs);
        let (ka, va): (u64, u64);
        mad.wide.u32(ka, c, 4, k);
        mad.wide.u32(va, c, 4, v);
        st.global.f32([ka], ko);
        st.global.f32([va], vo);
        let (pos, row, o): (u32, u32, u32);
        ld.global.u32(pos, [params]);
        mov.u32(row, pos);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(row, pos, ring);
        }
        mad.lo.u32(o, row, 1024, c);
        let (kca, vca): (u64, u64);
        let (kh, vh): (f16, f16);
        mad.wide.u32(kca, o, 2, kc);
        mad.wide.u32(vca, o, 2, vc);
        cvt.rn.f16.f32(kh, ko);
        cvt.rn.f16.f32(vh, vo);
        st.global.b16([kca], kh);
        st.global.b16([vca], vh);
    }

    /// The short convolution of channel c of x (taps of head h at x[1024 + 4 h] where DYN); state rows at `soff`. w is
    /// the checkpoint's BF16 weight (conv_w), of 1024 channels where DYN.
    fn conv_ch<const DYN: u32, const SOFF: u32>(o: f32, x: u64, w: u64, state: u64, c: u32, h: u32, scale: f32) {
        let (xa, sa): (u64, u64);
        let (c3, ct): (u32, u32);
        mad.wide.u32(xa, c, 4, x);
        mov.u32(ct, 1024);
        add.u32(c3, c, SOFF);
        mul.lo.u32(c3, c3, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (xv, h0, h1, h2): (f32, f32, f32, f32);
        ld.global.f32(xv, [xa]);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        let (w0, w1, w2, w3): (f32, f32, f32, f32);
        conv_w::<DYN>(w0, w1, w2, w3, w, c, ct);
        mul.rn.f32(o, w0, h0);
        fma.rn.f32(o, w1, h1, o);
        fma.rn.f32(o, w2, h2, o);
        fma.rn.f32(o, w3, xv, o);
        if DYN == 1 {
            let ti: u32;
            let ta: u64;
            mad.lo.u32(ti, h, 4, 1024);
            mad.wide.u32(ta, ti, 4, x);
            let (t0, t1, t2, t3): (f32, f32, f32, f32);
            ld.global.f32(t0, [ta]);
            ld.global.f32(t1, [ta + 4]);
            ld.global.f32(t2, [ta + 8]);
            ld.global.f32(t3, [ta + 12]);
            fma.rn.f32(o, xv, t0, o);
            fma.rn.f32(o, h2, t1, o);
            fma.rn.f32(o, h1, t2, o);
            fma.rn.f32(o, h0, t3, o);
        }
        add.rn.f32(o, o, xv);
        mul.rn.f32(o, o, scale);
        st.global.f32([sa], h1);
        st.global.f32([sa + 4], h2);
        st.global.f32([sa + 8], xv);
    }

    /// The Q clip (rows 0..n_head of 128) and the R clip (rows n_head..2 n_head of 16) of one token, times post[0].
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn clip_qr(
        qin: *const f32, qs: *const u16, qout: *mut f32, rin: *const f32, rs: *const u16, rout: *mut f32, n_head: u32, thr_q: f32, thr_r: f32, post: *const f32,
    ) {
        let (lane, row): (u32, u32);
        warp_row(lane, row);
        let (nr2, p): (u32, f32);
        shl.b32(nr2, n_head, 1);
        let past: pred;
        setp.ge.u32(past, row, nr2);
        if past {
            ret();
        }
        ld.global.f32(p, [post]);
        let is_r: pred;
        setp.ge.u32(is_r, row, n_head);
        let (d, hr, xoff): (u32, u32, u32);
        let (x, sc, y): (u64, u64, u64);
        let thr: f32;
        mov.u32(d, 128);
        mov.u32(hr, row);
        mov.u64(x, qin);
        mov.u64(sc, qs);
        mov.u64(y, qout);
        mov.f32(thr, thr_q);
        if is_r {
            mov.u32(d, 16);
            sub.u32(hr, row, n_head);
            mov.u64(x, rin);
            mov.u64(sc, rs);
            mov.u64(y, rout);
            mov.f32(thr, thr_r);
        }
        mul.lo.u32(xoff, hr, d);
        let ss: f32;
        mov.f32(ss, 0.0);
        for k in 0..4 {
            let idx: u32;
            add.u32(idx, lane, k * 32);
            let out: pred;
            setp.ge.u32(out, idx, d);
            if !out {
                let v: f32;
                load_scaled::<1>(v, x, sc, xoff, xoff, idx);
                fma.rn.f32(ss, v, v, ss);
            }
        }
        warp_sum(ss);
        let (norm, f): (f32, f32);
        sqrt.rn.f32(norm, ss);
        add.rn.f32(norm, norm, 0.000_001);
        div.rn.f32(f, thr, norm);
        max.f32(f, f, 0.0);
        min.f32(f, f, 1.0);
        mul.rn.f32(f, f, p);
        for k in 0..4 {
            let idx: u32;
            add.u32(idx, lane, k * 32);
            let out: pred;
            setp.ge.u32(out, idx, d);
            if !out {
                let v: f32;
                load_scaled::<1>(v, x, sc, xoff, xoff, idx);
                mul.rn.f32(v, v, f);
                let i: u32;
                let at: u64;
                add.u32(i, xoff, idx);
                mad.wide.u32(at, i, 4, y);
                st.global.f32([at], v);
            }
        }
    }

    /// y = conv(x) (+ res where RES): the short convolution of 4 taps of one token, a thread per channel.
    #[maxntid(256)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn conv_res<const RES: u32>(x: *const f32, w: *const u16, state: *mut f32, res: *const f32, y: *mut f32, n: u32, scale: f32) {
        let (b, nt, t, c): (u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, b, nt, t);
        let past: pred;
        setp.ge.u32(past, c, n);
        if past {
            ret();
        }
        let (o, zero): (f32, u32);
        mov.u32(zero, 0);
        conv_ch::<0, 0>(o, x, w, state, c, zero, scale);
        if RES == 1 {
            let ra: u64;
            let rv: f32;
            mad.wide.u32(ra, c, 4, res);
            ld.global.f32(rv, [ra]);
            add.rn.f32(o, o, rv);
        }
        let ya: u64;
        mad.wide.u32(ya, c, 4, y);
        st.global.f32([ya], o);
    }

    /// The end of a MoE layer for one token: m = sum_k w_k (slots[k] >= 0 ? gpu[k] : cpu[k]) per channel, then
    /// x += conv(m) (ffn post convolution).
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn moe_tail(gpu: *const f32, cpu: *const f32, slots: *const i32, wts: *const f32, w: *const u16, state: *mut f32, x: *mut f32, n: u32, scale: f32) {
        let (b, nt, t, c): (u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, b, nt, t);
        let past: pred;
        setp.ge.u32(past, c, n);
        if past {
            ret();
        }
        let m: f32;
        mov.f32(m, 0.0);
        for k in 0..8 {
            let s: s32;
            let wv: f32;
            ld.global.s32(s, [slots + (k * 4)]);
            ld.global.f32(wv, [wts + (k * 4)]);
            let on_gpu: pred;
            setp.ge.s32(on_gpu, s, 0);
            let (o, ga, ca): (u32, u64, u64);
            let v: f32;
            mad.lo.u32(o, n, k, c);
            mad.wide.u32(ga, o, 4, gpu);
            mad.wide.u32(ca, o, 4, cpu);
            if on_gpu {
                ld.global.f32(v, [ga]);
            } else {
                ld.volatile.global.f32(v, [ca]);
            }
            fma.rn.f32(m, wv, v, m);
        }
        let (sa, xa): (u64, u64);
        let c3: u32;
        mul.lo.u32(c3, c, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (h0, h1, h2, w0, w1, w2, w3, o2, xv): (f32, f32, f32, f32, f32, f32, f32, f32, f32);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        // the conv1d weights [n][4] BF16 (tap 3 the current token)
        conv_w::<0>(w0, w1, w2, w3, w, c, n);
        mul.rn.f32(o2, w0, h0);
        fma.rn.f32(o2, w1, h1, o2);
        fma.rn.f32(o2, w2, h2, o2);
        fma.rn.f32(o2, w3, m, o2);
        add.rn.f32(o2, o2, m);
        mul.rn.f32(o2, o2, scale);
        st.global.f32([sa], h1);
        st.global.f32([sa + 4], h2);
        st.global.f32([sa + 8], m);
        mad.wide.u32(xa, c, 4, x);
        ld.global.f32(xv, [xa]);
        add.rn.f32(xv, xv, o2);
        st.global.f32([xa], xv);
    }

    /// The router's top 8 (see router_topk) and the MoE job of the layer (see moe_dispatch), in one block.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn router_dispatch(
        logits: *const f32, bias: *const f32, ids: *mut u32, wts: *mut f32, scale: f32, map: *const i32, slots: *mut i32, x: *const f32,
        job: *mut u32, params: *const u32, step: u32, layer: u32, n: u32,
    ) {
        let sel: shared<[f32; 128], 16>;
        let prb: shared<[f32; 128], 16>;
        let win: shared<[u32; 8], 16>;
        let e: u32;
        mov.u32(e, tid.x);
        let (la, ba): (u64, u64);
        let (lg, p, bv, sv): (f32, f32, f32, f32);
        mad.wide.u32(la, e, 4, logits);
        ld.global.f32(lg, [la]);
        sigmoid(p, lg);
        mad.wide.u32(ba, e, 4, bias);
        ld.global.nc.f32(bv, [ba]);
        add.rn.f32(sv, p, bv);
        let (sb, pb, wb, at): (u32, u32, u32, u32);
        mov.u32(sb, sel);
        mov.u32(pb, prb);
        mov.u32(wb, win);
        mad.lo.u32(at, e, 4, sb);
        st.shared.f32([at], sv);
        mad.lo.u32(at, e, 4, pb);
        st.shared.f32([at], p);
        // the job's x meanwhile
        let i: u32;
        mov.u32(i, e);
        label(xtop);
        let xdone: pred;
        setp.ge.u32(xdone, i, n);
        if xdone {
            bra(xend);
        }
        let (xa, ja): (u64, u64);
        let (v, ji): (f32, u32);
        mad.wide.u32(xa, i, 4, x);
        ld.global.f32(v, [xa]);
        add.u32(ji, i, 32);
        mad.wide.u32(ja, ji, 4, job);
        st.volatile.global.f32([ja], v);
        add.u32(i, i, 128);
        bra(xtop);
        label(xend);
        bar.sync(0);
        let lane: u32;
        and.b32(lane, e, 31);
        let first: pred;
        setp.eq.u32(first, e, 0);
        for k in 0..8 {
            let (best, cand): (f32, f32);
            let (bi, ci): (u32, u32);
            mov.f32(best, -3.0e38);
            mov.u32(bi, 0);
            for q4 in 0..4 {
                let ca: u32;
                add.u32(ci, lane, q4 * 32);
                mad.lo.u32(ca, ci, 4, sb);
                ld.shared.f32(cand, [ca]);
                let better: pred;
                setp.gt.f32(better, cand, best);
                selp.f32(best, cand, best, better);
                selp.u32(bi, ci, bi, better);
            }
            for sh in [16, 8, 4, 2, 1] {
                let (ob, oi): (f32, u32);
                shfl.sync.bfly.b32(ob, best, sh, 31, 0xffff_ffff);
                shfl.sync.bfly.b32(oi, bi, sh, 31, 0xffff_ffff);
                let (gt, eq, lt, take): (pred, pred, pred, pred);
                setp.gt.f32(gt, ob, best);
                setp.eq.f32(eq, ob, best);
                setp.lt.u32(lt, oi, bi);
                and.pred(take, eq, lt);
                or.pred(take, take, gt);
                selp.f32(best, ob, best, take);
                selp.u32(bi, oi, bi, take);
            }
            bar.sync(0);
            if first {
                let kill: u32;
                st.shared.u32([wb + (k * 4)], bi);
                mad.lo.u32(kill, bi, 4, sb);
                st.shared.f32([kill], -3.0e38);
            }
            bar.sync(0);
        }
        let my: pred;
        setp.lt.u32(my, e, 8);
        if my {
            let (sum, pv, own): (f32, f32, f32);
            let (id, ia): (u32, u32);
            mov.f32(sum, 0.0);
            for k in 0..8 {
                let (wk, pa): (u32, u32);
                ld.shared.u32(wk, [wb + (k * 4)]);
                mad.lo.u32(pa, wk, 4, pb);
                ld.shared.f32(pv, [pa]);
                add.rn.f32(sum, sum, pv);
            }
            add.rn.f32(sum, sum, 0.000_000_000_000_000_000_01);
            mad.lo.u32(ia, e, 4, wb);
            ld.shared.u32(id, [ia]);
            let pa2: u32;
            mad.lo.u32(pa2, id, 4, pb);
            ld.shared.f32(own, [pa2]);
            div.rn.f32(own, own, sum);
            mul.rn.f32(own, own, scale);
            let (a1, a2, ma, sa, j1, j2): (u64, u64, u64, u64, u64, u64);
            let sl: s32;
            let (ji2, ji3): (u32, u32);
            mad.wide.u32(a1, e, 4, ids);
            mad.wide.u32(a2, e, 4, wts);
            st.global.u32([a1], id);
            st.global.f32([a2], own);
            mad.wide.u32(ma, id, 4, map);
            ld.global.s32(sl, [ma]);
            mad.wide.u32(sa, e, 4, slots);
            st.global.s32([sa], sl);
            add.u32(ji2, e, 2);
            mad.wide.u32(j1, ji2, 4, job);
            st.volatile.global.u32([j1], id);
            add.u32(ji3, e, 10);
            mad.wide.u32(j2, ji3, 4, job);
            st.volatile.global.s32([j2], sl);
        }
        bar.sync(0);
        if first {
            let (tok, seq): (u32, u32);
            st.volatile.global.u32([job + 4], layer);
            ld.global.u32(tok, [params + 12]);
            mad.lo.u32(seq, tok, 64, step);
            add.u32(seq, seq, 1);
            membar.sys();
            st.volatile.global.u32([job], seq);
        }
    }

    /// Profiling: writes the GPU's global timer (ns) to buf[i], from one thread.
    #[reqntid(32)]
    pub unsafe extern "ptx-kernel" fn stamp(buf: *mut u64, i: u32) {
        let t: u32;
        mov.u32(t, tid.x);
        let first: pred;
        setp.eq.u32(first, t, 0);
        if first {
            let now: u64;
            let a: u64;
            mov.u64(now, globaltimer);
            mad.wide.u32(a, i, 8, buf);
            st.global.u64([a], now);
        }
    }

    // ---------------------------------------------------------------- 768 threads: 3 values of 2304 each, in registers

    /// mix8 with 768 threads (n = 2304): the states in registers, one read of each. grid (nq, ntok).
    #[reqntid(768)]
    pub unsafe extern "ptx-kernel" fn mix768(ptrs: *const u64, ns: u32, qs: *const f32, out: *mut f32) {
        let red: shared<[f32; 384], 16>;
        let tot: shared<[f32; 16], 16>;
        let (tk, qi, nq, t, lane, w): (u32, u32, u32, u32, u32, u32);
        mov.u32(tk, ctaid.y);
        mov.u32(qi, ctaid.x);
        mov.u32(nq, nctaid.x);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        let (tko, qoff, ooff): (u32, u32, u32);
        mul.lo.u32(tko, tk, 2304);
        add.u32(tko, tko, t);
        mad.lo.u32(qoff, qi, 2304, t);
        mad.lo.u32(ooff, tk, nq, qi);
        mad.lo.u32(ooff, ooff, 2304, t);
        let (qv, ov): (u64, u64);
        mad.wide.u32(qv, qoff, 4, qs);
        mad.wide.u32(ov, ooff, 4, out);
        let (q0, q1, q2): (f32, f32, f32);
        ld.global.nc.f32(q0, [qv]);
        ld.global.nc.f32(q1, [qv + 3072]);
        ld.global.nc.f32(q2, [qv + 6144]);
        let xv: [f32; 24];
        let ss: [f32; 8];
        let dq: [f32; 8];
        for j in 0..8 {
            mov.f32(xv[j * 3], 0.0);
            mov.f32(xv[j * 3 + 1], 0.0);
            mov.f32(xv[j * 3 + 2], 0.0);
            let pj: pred;
            setp.lt.u32(pj, ns, j + 1);
            if !pj {
                let sp: u64;
                ld.global.u64(sp, [ptrs + (j * 8)]);
                mad.wide.u32(sp, tko, 4, sp);
                ld.global.f32(xv[j * 3], [sp]);
                ld.global.f32(xv[j * 3 + 1], [sp + 3072]);
                ld.global.f32(xv[j * 3 + 2], [sp + 6144]);
            }
            mul.rn.f32(ss[j], xv[j * 3], xv[j * 3]);
            fma.rn.f32(ss[j], xv[j * 3 + 1], xv[j * 3 + 1], ss[j]);
            fma.rn.f32(ss[j], xv[j * 3 + 2], xv[j * 3 + 2], ss[j]);
            mul.rn.f32(dq[j], xv[j * 3], q0);
            fma.rn.f32(dq[j], xv[j * 3 + 1], q1, dq[j]);
            fma.rn.f32(dq[j], xv[j * 3 + 2], q2, dq[j]);
        }
        let (rb, tb): (u32, u32);
        mov.u32(rb, red);
        mov.u32(tb, tot);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        let wa: u32;
        mad.lo.u32(wa, w, 64, rb);
        for j in 0..8 {
            warp_sum(ss[j]);
            warp_sum(dq[j]);
        }
        if first {
            for j in 0..8 {
                st.shared.f32([wa + (j * 4)], ss[j]);
                st.shared.f32([wa + (32 + j * 4)], dq[j]);
            }
        }
        bar.sync(0);
        let lt16: pred;
        setp.lt.u32(lt16, t, 16);
        if lt16 {
            let (sum, v): (f32, f32);
            let a: u32;
            mov.f32(sum, 0.0);
            mad.lo.u32(a, t, 4, rb);
            for k in 0..24 {
                ld.shared.f32(v, [a + (k * 64)]);
                add.rn.f32(sum, sum, v);
            }
            mad.lo.u32(a, t, 4, tb);
            st.shared.f32([a], sum);
        }
        bar.sync(0);
        let (m, l): (f32, f32);
        mov.f32(m, -3.0e38);
        let sc: [f32; 8];
        let rs: [f32; 8];
        for j in 0..8 {
            let (a2, b2): (f32, f32);
            ld.shared.f32(a2, [tb + (j * 4)]);
            ld.shared.f32(b2, [tb + (32 + j * 4)]);
            mul.rn.f32(a2, a2, 0.000_434_027_78);
            add.rn.f32(a2, a2, 0.000_001);
            sqrt.rn.f32(a2, a2);
            rcp.rn.f32(rs[j], a2);
            mul.rn.f32(sc[j], b2, rs[j]);
            mul.rn.f32(sc[j], sc[j], 0.020_833_334);
            let pj: pred;
            setp.lt.u32(pj, ns, j + 1);
            if pj {
                mov.f32(sc[j], -3.0e38);
            }
            max.f32(m, m, sc[j]);
        }
        mov.f32(l, 0.0);
        for j in 0..8 {
            sub.rn.f32(sc[j], sc[j], m);
            exp(sc[j], sc[j]);
            add.rn.f32(l, l, sc[j]);
        }
        rcp.rn.f32(l, l);
        let (o0, o1, o2): (f32, f32, f32);
        mov.f32(o0, 0.0);
        mov.f32(o1, 0.0);
        mov.f32(o2, 0.0);
        for j in 0..8 {
            mul.rn.f32(sc[j], sc[j], l);
            mul.rn.f32(sc[j], sc[j], rs[j]);
            fma.rn.f32(o0, sc[j], xv[j * 3], o0);
            fma.rn.f32(o1, sc[j], xv[j * 3 + 1], o1);
            fma.rn.f32(o2, sc[j], xv[j * 3 + 2], o2);
        }
        st.global.f32([ov], o0);
        st.global.f32([ov + 3072], o1);
        st.global.f32([ov + 6144], o2);
    }

    /// router_dispatch with one barrier: warp 0 picks the top 8 in registers (4 candidates a lane), while warps 1..3 copy
    /// x into the job; then lanes 0..7 of warp 0 write the routing and the job, and thread 0 posts it. 128 threads.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn router_dispatch2(
        logits: *const f32, bias: *const f32, ids: *mut u32, wts: *mut f32, scale: f32, map: *const i32, slots: *mut i32, x: *const f32,
        job: *mut u32, params: *const u32, step: u32, layer: u32, n: u32,
    ) {
        let sel: shared<[f32; 128], 16>;
        let prb: shared<[f32; 128], 16>;
        let e: u32;
        mov.u32(e, tid.x);
        let (la, ba): (u64, u64);
        let (lg, p, bv, sv): (f32, f32, f32, f32);
        mad.wide.u32(la, e, 4, logits);
        ld.global.f32(lg, [la]);
        sigmoid(p, lg);
        mad.wide.u32(ba, e, 4, bias);
        ld.global.nc.f32(bv, [ba]);
        add.rn.f32(sv, p, bv);
        let (sb, pb, at): (u32, u32, u32);
        mov.u32(sb, sel);
        mov.u32(pb, prb);
        mad.lo.u32(at, e, 4, sb);
        st.shared.f32([at], sv);
        mad.lo.u32(at, e, 4, pb);
        st.shared.f32([at], p);
        bar.sync(0);
        let (lane, w): (u32, u32);
        and.b32(lane, e, 31);
        shr.u32(w, e, 5);
        let copier: pred;
        setp.ne.u32(copier, w, 0);
        if copier {
            bra(copy);
        }
        // warp 0: the top 8
        let c: [f32; 4];
        for q4 in 0..4 {
            let ca: u32;
            mad.lo.u32(ca, lane, 4, sb);
            ld.shared.f32(c[q4], [ca + (q4 * 128)]);
        }
        let win: [u32; 8];
        for k in 0..8 {
            let (best, bi): (f32, u32);
            mov.f32(best, c[0]);
            mov.u32(bi, lane);
            for q4 in 1..4 {
                let (better, ci): (pred, u32);
                add.u32(ci, lane, q4 * 32);
                setp.gt.f32(better, c[q4], best);
                selp.f32(best, c[q4], best, better);
                selp.u32(bi, ci, bi, better);
            }
            for sh in [16, 8, 4, 2, 1] {
                let (ob, oi): (f32, u32);
                shfl.sync.bfly.b32(ob, best, sh, 31, 0xffff_ffff);
                shfl.sync.bfly.b32(oi, bi, sh, 31, 0xffff_ffff);
                let (gt, eq, lt, take): (pred, pred, pred, pred);
                setp.gt.f32(gt, ob, best);
                setp.eq.f32(eq, ob, best);
                setp.lt.u32(lt, oi, bi);
                and.pred(take, eq, lt);
                or.pred(take, take, gt);
                selp.f32(best, ob, best, take);
                selp.u32(bi, oi, bi, take);
            }
            mov.u32(win[k], bi);
            // the winner leaves the candidates
            for q4 in 0..4 {
                let (ci, hit): (u32, pred);
                add.u32(ci, lane, q4 * 32);
                setp.eq.u32(hit, ci, bi);
                selp.f32(c[q4], -3.0e38, c[q4], hit);
            }
        }
        let my: pred;
        setp.lt.u32(my, lane, 8);
        if my {
            let (sum, pv, own): (f32, f32, f32);
            let (id, pa): (u32, u32);
            mov.f32(sum, 0.0);
            mov.u32(id, 0);
            for k in 0..8 {
                mad.lo.u32(pa, win[k], 4, pb);
                ld.shared.f32(pv, [pa]);
                add.rn.f32(sum, sum, pv);
                let mine: pred;
                setp.eq.u32(mine, lane, k);
                selp.u32(id, win[k], id, mine);
            }
            add.rn.f32(sum, sum, 0.000_000_000_000_000_000_01);
            mad.lo.u32(pa, id, 4, pb);
            ld.shared.f32(own, [pa]);
            div.rn.f32(own, own, sum);
            mul.rn.f32(own, own, scale);
            let (a1, a2, ma, sa, j1, j2): (u64, u64, u64, u64, u64, u64);
            let sl: s32;
            let (ji2, ji3): (u32, u32);
            mad.wide.u32(a1, lane, 4, ids);
            mad.wide.u32(a2, lane, 4, wts);
            st.global.u32([a1], id);
            st.global.f32([a2], own);
            mad.wide.u32(ma, id, 4, map);
            ld.global.s32(sl, [ma]);
            mad.wide.u32(sa, lane, 4, slots);
            st.global.s32([sa], sl);
            add.u32(ji2, lane, 2);
            mad.wide.u32(j1, ji2, 4, job);
            st.global.u32([j1], id);
            add.u32(ji3, lane, 10);
            mad.wide.u32(j2, ji3, 4, job);
            st.global.s32([j2], sl);
        }
        bra(post);
        // warps 1..3: the job's x, 16 bytes a store; plain stores, ordered before the seq by the fence
        label(copy);
        let i: u32;
        sub.u32(i, e, 32);
        let n4: u32;
        shr.u32(n4, n, 2);
        label(xtop);
        let xdone: pred;
        setp.ge.u32(xdone, i, n4);
        if xdone {
            bra(post);
        }
        let (xa, ja): (u64, u64);
        let (v0, v1, v2, v3): (f32, f32, f32, f32);
        let ji: u32;
        mad.wide.u32(xa, i, 16, x);
        ld.global.v4.f32((v0, v1, v2, v3), [xa]);
        add.u32(ji, i, 8);
        mad.wide.u32(ja, ji, 16, job);
        st.global.v4.f32([ja], (v0, v1, v2, v3));
        add.u32(i, i, 96);
        bra(xtop);
        label(post);
        bar.sync(0);
        let first: pred;
        setp.eq.u32(first, e, 0);
        if first {
            let (tok, seq): (u32, u32);
            st.volatile.global.u32([job + 4], layer);
            ld.global.u32(tok, [params + 12]);
            mad.lo.u32(seq, tok, 64, step);
            add.u32(seq, seq, 1);
            // release at system scope: the job (all warps, ordered before by the barrier) is visible before seq
            st.release.sys.global.u32([job], seq);
        }
    }
}
