//! GPU kernels of the Inkling-2-Mini engine, written in kernite. The engine loads `gpu::PTX` with the driver API.
//!
//! Layout conventions: vectors are f32, one token at a time (grid.y walks tokens where a kernel takes several).
//! Q8_0 matrices are repacked at load: `qs` is [rows][k] int8, `d` is [rows][k/32] f16 (Marlin-style: aligned 16-byte loads).
//! Q4_K and Q5_0 experts keep the GGUF block layout, so that a cache upload is a plain copy.

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

    /// Sum of `v` over the 16 lanes of a half warp.
    fn half_sum(v: f32) {
        for sh in [8, 4, 2, 1] {
            let o: f32;
            shfl.sync.bfly.b32(o, v, sh, 31, 0xffff_ffff);
            add.rn.f32(v, v, o);
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

    /// acc += the 4 signed bytes of `w` times x0..x3
    fn fma_s8x4(acc: f32, w: u32, x0: f32, x1: f32, x2: f32, x3: f32) {
        let b: s32;
        let f: f32;
        bfe.s32(b, w, 0, 8);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x0, acc);
        bfe.s32(b, w, 8, 8);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x1, acc);
        bfe.s32(b, w, 16, 8);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x2, acc);
        bfe.s32(b, w, 24, 8);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x3, acc);
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

    // ---------------------------------------------------------------- Q8_0 GEMV

    /// y[r] = sum_c W[r][c] x[c] for a repacked Q8_0 matrix (`qs` [rows][k] int8, `d` [rows][k/32] f16).
    /// k is a multiple of 16 and at most 4608. A half warp takes a row, 16 rows per block of 256 threads.
    /// grid.y walks tokens: x advances by k floats, y by `y_stride`.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn gemv_q8(qs: *const u8, d: *const u16, x: *const f32, y: *mut f32, rows: u32, k: u32, y_stride: u32) {
        let xs: shared<[f32; 4608], 16>;
        let (base, tok, xoff): (u32, u32, u32);
        mov.u32(tok, ctaid.y);
        mul.lo.u32(xoff, tok, k);
        let xt: u64;
        mad.wide.u32(xt, xoff, 4, x);
        mov.u32(base, xs);
        stage(base, xt, k);

        let (t, hl, hw, row): (u32, u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(hl, t, 15);
        shr.u32(hw, t, 4);
        let b: u32;
        mov.u32(b, ctaid.x);
        mad.lo.u32(row, b, 16, hw);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }

        let (rowq, nblk, rowd): (u32, u32, u32);
        mul.lo.u32(rowq, row, k);
        shr.u32(nblk, k, 5);
        mul.lo.u32(rowd, row, nblk);

        let acc: f32;
        mov.f32(acc, 0.0);
        let c: u32;
        shl.b32(c, hl, 4);
        label(top);
        let done: pred;
        setp.ge.u32(done, c, k);
        if done {
            bra(end);
        }
        let i: u32;
        let (at, dat): (u64, u64);
        add.u32(i, rowq, c);
        mad.wide.u32(at, i, 1, qs);
        let (w0, w1, w2, w3): (u32, u32, u32, u32);
        ld.global.nc.v4.u32((w0, w1, w2, w3), [at]);
        let j: u32;
        let h: f16;
        shr.u32(j, c, 5);
        add.u32(j, j, rowd);
        mad.wide.u32(dat, j, 2, d);
        ld.global.nc.b16(h, [dat]);
        let sc: f32;
        cvt.f32.f16(sc, h);
        let xa: u32;
        mad.lo.u32(xa, c, 4, base);
        let (x0, x1, x2, x3): (f32, f32, f32, f32);
        let s: f32;
        mov.f32(s, 0.0);
        ld.shared.v4.f32((x0, x1, x2, x3), [xa]);
        fma_s8x4(s, w0, x0, x1, x2, x3);
        ld.shared.v4.f32((x0, x1, x2, x3), [xa + 16]);
        fma_s8x4(s, w1, x0, x1, x2, x3);
        ld.shared.v4.f32((x0, x1, x2, x3), [xa + 32]);
        fma_s8x4(s, w2, x0, x1, x2, x3);
        ld.shared.v4.f32((x0, x1, x2, x3), [xa + 48]);
        fma_s8x4(s, w3, x0, x1, x2, x3);
        fma.rn.f32(acc, s, sc, acc);
        add.u32(c, c, 256);
        bra(top);
        label(end);

        half_sum(acc);
        let first: pred;
        setp.eq.u32(first, hl, 0);
        if first {
            let o: u32;
            let yat: u64;
            mad.lo.u32(o, tok, y_stride, row);
            mad.wide.u32(yat, o, 4, y);
            st.global.f32([yat], acc);
        }
    }

    // ---------------------------------------------------------------- norms

    /// y = x / rms(x) * w (without w where `HAS_W` is 0), over n floats; grid.x walks vectors.
    #[maxntid(1024)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn rms_norm<const HAS_W: u32>(x: *const f32, w: *const f32, y: *mut f32, n: u32, eps: f32, xs: u32, ys: u32) {
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
            mad.wide.u32(wat, i, 4, w);
            ld.global.nc.f32(wv, [wat]);
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
        x: *const f32, s: *const f32, y: *mut f32, d: u32, rows: u32, s_rows: u32, thr: f32, eps: f32, post: *const f32,
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

    /// `v = x[xoff + idx] * s[soff + idx]`, or `x[xoff + idx]` alone where `HAS_S` is 0.
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
            mad.wide.u32(sat, j, 4, s);
            ld.global.nc.f32(w, [sat]);
            mul.rn.f32(v, v, w);
        }
    }

    // ---------------------------------------------------------------- depth mixing

    /// Attention over depth: the states are `ptrs[0..ns]` (n floats each), each RMS-normalized; query `q` of `qs[blockIdx.x]`
    /// scores them by `dot/sqrt(n)`, and `out[blockIdx.x]` is the softmax-weighted sum of the normalized states.
    #[maxntid(1024)]
    pub unsafe extern "ptx-kernel" fn mix(ptrs: *const u64, ns: u32, qs: *const f32, out: *mut f32, n: u32) {
        let (tk, nq, tko): (u32, u32, u32);
        mov.u32(tk, ctaid.y);
        mov.u32(nq, nctaid.x);
        mul.lo.u32(tko, tk, n);
        let red: shared<[f32; 32], 16>;
        let sc: shared<[f32; 32], 16>;
        let r: u32;
        mov.u32(r, red);
        let (qi, qoff): (u32, u32);
        mov.u32(qi, ctaid.x);
        mul.lo.u32(qoff, qi, n);
        let (qv, ov, ooff): (u64, u64, u32);
        mad.wide.u32(qv, qoff, 4, qs);
        mad.lo.u32(ooff, tk, nq, qi);
        mul.lo.u32(ooff, ooff, n);
        mad.wide.u32(ov, ooff, 4, out);
        let (t, nt): (u32, u32);
        mov.u32(t, tid.x);
        mov.u32(nt, ntid.x);
        let (nf, inv_sqrt_n): (f32, f32);
        cvt.rn.f32.u32(nf, n);
        sqrt.rn.f32(inv_sqrt_n, nf);
        rcp.rn.f32(inv_sqrt_n, inv_sqrt_n);

        // per state: rms scale and score, kept in shared memory: sc[2j] = 1/rms, sc[2j+1] = score
        let j: u32;
        mov.u32(j, 0);
        label(st_top);
        let st_done: pred;
        setp.ge.u32(st_done, j, ns);
        if st_done {
            bra(st_end);
        }
        let (pat, sv): (u64, u64);
        mad.wide.u32(pat, j, 8, ptrs);
        ld.global.u64(sv, [pat]);
        mad.wide.u32(sv, tko, 4, sv);
        let (ss, dq): (f32, f32);
        mov.f32(ss, 0.0);
        mov.f32(dq, 0.0);
        let i: u32;
        mov.u32(i, t);
        label(el_top);
        let el_done: pred;
        setp.ge.u32(el_done, i, n);
        if el_done {
            bra(el_end);
        }
        let (a, b): (u64, u64);
        let (xv, qv2): (f32, f32);
        mad.wide.u32(a, i, 4, sv);
        ld.global.f32(xv, [a]);
        mad.wide.u32(b, i, 4, qv);
        ld.global.nc.f32(qv2, [b]);
        fma.rn.f32(ss, xv, xv, ss);
        fma.rn.f32(dq, xv, qv2, dq);
        add.u32(i, i, nt);
        bra(el_top);
        label(el_end);
        block_sum(ss, r);
        block_sum(dq, r);
        let (rs, score): (f32, f32);
        div.rn.f32(rs, ss, nf);
        add.rn.f32(rs, rs, 0.000_001);
        sqrt.rn.f32(rs, rs);
        rcp.rn.f32(rs, rs);
        mul.rn.f32(score, dq, rs);
        mul.rn.f32(score, score, inv_sqrt_n);
        let first: pred;
        setp.eq.u32(first, t, 0);
        if first {
            let (base, at): (u32, u32);
            mov.u32(base, sc);
            mad.lo.u32(at, j, 8, base);
            st.shared.f32([at], rs);
            st.shared.f32([at + 4], score);
        }
        add.u32(j, j, 1);
        bra(st_top);
        label(st_end);
        bar.sync(0);

        // softmax over the states, in every thread
        let (m, l): (f32, f32);
        mov.f32(m, -3.0e38);
        let sbase: u32;
        mov.u32(sbase, sc);
        mov.u32(j, 0);
        label(mx_top);
        let mx_done: pred;
        setp.ge.u32(mx_done, j, ns);
        if mx_done {
            bra(mx_end);
        }
        let at3: u32;
        let s3: f32;
        mad.lo.u32(at3, j, 8, sbase);
        ld.shared.f32(s3, [at3 + 4]);
        max.f32(m, m, s3);
        add.u32(j, j, 1);
        bra(mx_top);
        label(mx_end);
        mov.f32(l, 0.0);
        mov.u32(j, 0);
        label(sm_top);
        let sm_done: pred;
        setp.ge.u32(sm_done, j, ns);
        if sm_done {
            bra(sm_end);
        }
        let at4: u32;
        let (s4, e4): (f32, f32);
        mad.lo.u32(at4, j, 8, sbase);
        ld.shared.f32(s4, [at4 + 4]);
        sub.rn.f32(s4, s4, m);
        exp(e4, s4);
        add.rn.f32(l, l, e4);
        add.u32(j, j, 1);
        bra(sm_top);
        label(sm_end);
        let il: f32;
        rcp.rn.f32(il, l);

        // out = sum_j p_j * x_j / rms_j
        mov.u32(i, t);
        label(o_top);
        let o_done: pred;
        setp.ge.u32(o_done, i, n);
        if o_done {
            bra(o_end);
        }
        let acc: f32;
        mov.f32(acc, 0.0);
        mov.u32(j, 0);
        label(w_top);
        let w_done: pred;
        setp.ge.u32(w_done, j, ns);
        if w_done {
            bra(w_end);
        }
        let at5: u32;
        let (r5, s5, p5, x5): (f32, f32, f32, f32);
        let (pat5, sv5, xa5): (u64, u64, u64);
        mad.lo.u32(at5, j, 8, sbase);
        ld.shared.f32(r5, [at5]);
        ld.shared.f32(s5, [at5 + 4]);
        sub.rn.f32(s5, s5, m);
        exp(p5, s5);
        mul.rn.f32(p5, p5, il);
        mul.rn.f32(p5, p5, r5);
        mad.wide.u32(pat5, j, 8, ptrs);
        ld.global.u64(sv5, [pat5]);
        mad.wide.u32(sv5, tko, 4, sv5);
        mad.wide.u32(xa5, i, 4, sv5);
        ld.global.f32(x5, [xa5]);
        fma.rn.f32(acc, p5, x5, acc);
        add.u32(j, j, 1);
        bra(w_top);
        label(w_end);
        let oat: u64;
        mad.wide.u32(oat, i, 4, ov);
        st.global.f32([oat], acc);
        add.u32(i, i, nt);
        bra(o_top);
        label(o_end);
    }

    // ---------------------------------------------------------------- short convolution

    /// Causal short convolution of 4 taps with a residual, for one token: per channel c,
    /// `y = (x + sum_k w[c][k] * h[k] + dyn) * scale` where h = (state[c][0..3], x), then state[c] = (h[1], h[2], x).
    /// With `DYN`, `dyn = sum_lag h[3 - lag] * taps[head(c)][lag]`, head(c) = c / (c_total / n_heads).
    #[maxntid(256)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn conv4<const DYN: u32>(
        x: *const f32, w: *const f32, state: *mut f32, taps: *const f32, y: *mut f32, c_total: u32, head_dim: u32, scale: f32,
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
        let (xa, wa, sa): (u64, u64, u64);
        let c4: u32;
        let c3: u32;
        mad.wide.u32(xa, c, 4, x);
        shl.b32(c4, c, 2);
        mad.wide.u32(wa, c4, 4, w);
        mul.lo.u32(c3, c, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (xv, h0, h1, h2): (f32, f32, f32, f32);
        ld.global.f32(xv, [xa]);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        let (w0, w1, w2, w3): (f32, f32, f32, f32);
        ld.global.nc.v4.f32((w0, w1, w2, w3), [wa]);
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

    /// The input of a controller's down projection: `g = (rms(cur), rms(msg))`, 2n floats.
    #[maxntid(1024)]
    pub unsafe extern "ptx-kernel" fn ctrl_in(cur: *const f32, msg: *const f32, g: *mut f32, n: u32) {
        let red: shared<[f32; 32], 16>;
        let (tk, to, to2): (u32, u32, u32);
        let (curt, msgt, gt): (u64, u64, u64);
        mov.u32(tk, ctaid.y);
        mul.lo.u32(to, tk, n);
        shl.b32(to2, to, 1);
        mad.wide.u32(curt, to, 4, cur);
        mad.wide.u32(msgt, to, 4, msg);
        mad.wide.u32(gt, to2, 4, g);
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
    /// repacked Q8_0 [n][64] (`gq`/`gd`, `fq`/`fd`) and gate is 64 floats.
    /// The body of ctrl_out, for token tk.
    fn ctrl_out_body(cur: u64, msg: u64, gate: u64, gq: u64, gd: u64, fq: u64, fd: u64, y: u64, n: u32, tk: u32) {
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
        q8_row64(gain, gq, gd, c, base);
        q8_row64(forget, fq, fd, c, base);
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
        cur: *const f32, msg: *const f32, gate: *const f32, gq: *const u8, gd: *const u16, fq: *const u8, fd: *const u16, y: *mut f32, n: u32,
    ) {
        let tk: u32;
        mov.u32(tk, ctaid.y);
        ctrl_out_body(cur, msg, gate, gq, gd, fq, fd, y, n, tk);
    }

    /// gemv_q8d of matrix ctaid.z, everything from tab[6 z ..]: qs, d, xq, xd, y, rows. Blocks past a matrix's rows return
    /// before they stage x: the grid is sized for the largest.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn gemv_q8dt(tab: *const u64, k: u32) {
        let (z, b, zi, b16): (u32, u32, u32, u32);
        mov.u32(z, ctaid.z);
        mov.u32(b, ctaid.x);
        let (ta, qz, dz, xqz, xdz, yz): (u64, u64, u64, u64, u64, u64);
        let rows: u32;
        mul.lo.u32(zi, z, 6);
        mad.wide.u32(ta, zi, 8, tab);
        ld.global.u32(rows, [ta + 40]);
        shl.b32(b16, b, 4);
        let idle: pred;
        setp.ge.u32(idle, b16, rows);
        if idle {
            ret();
        }
        ld.global.u64(qz, [ta]);
        ld.global.u64(dz, [ta + 8]);
        ld.global.u64(xqz, [ta + 16]);
        ld.global.u64(xdz, [ta + 24]);
        ld.global.u64(yz, [ta + 32]);
        gemv_q8d_body(qz, dz, xqz, xdz, yz, rows, k, b);
    }

    /// ctrl_out of controller ctaid.z (one token): cur, msg, gq, gd, fq, fd, y at tab[7 z ..]; its gate at gate + 64 z.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn ctrl_outz(tab: *const u64, gate: *const f32, n: u32) {
        let (z, zi, tk): (u32, u32, u32);
        mov.u32(z, ctaid.z);
        mov.u32(tk, 0);
        let (ta, cz, mz, gqz, gdz, fqz, fdz, yz, gz): (u64, u64, u64, u64, u64, u64, u64, u64, u64);
        mul.lo.u32(zi, z, 7);
        mad.wide.u32(ta, zi, 8, tab);
        ld.global.u64(cz, [ta]);
        ld.global.u64(mz, [ta + 8]);
        ld.global.u64(gqz, [ta + 16]);
        ld.global.u64(gdz, [ta + 24]);
        ld.global.u64(fqz, [ta + 32]);
        ld.global.u64(fdz, [ta + 40]);
        ld.global.u64(yz, [ta + 48]);
        shl.b32(zi, z, 6);
        mad.wide.u32(gz, zi, 4, gate);
        ctrl_out_body(cz, mz, gz, gqz, gdz, fqz, fdz, yz, n, tk);
    }

    /// v = row `r` of a repacked Q8_0 [*][64] matrix times the 64 floats at shared `xs`.
    fn q8_row64(v: f32, qs: u64, d: u64, r: u32, xs: u32) {
        mov.f32(v, 0.0);
        let (qa, da): (u64, u64);
        let r64: u32;
        let r2: u32;
        shl.b32(r64, r, 6);
        mad.wide.u32(qa, r64, 1, qs);
        shl.b32(r2, r, 1);
        mad.wide.u32(da, r2, 2, d);
        for blk in 0..2 {
            let s: f32;
            mov.f32(s, 0.0);
            for q4 in 0..2 {
                let (w0, w1, w2, w3): (u32, u32, u32, u32);
                ld.global.nc.v4.u32((w0, w1, w2, w3), [qa + (blk * 32 + q4 * 16)]);
                let (x0, x1, x2, x3): (f32, f32, f32, f32);
                ld.shared.v4.f32((x0, x1, x2, x3), [xs + ((blk * 32 + q4 * 16) * 4)]);
                fma_s8x4(s, w0, x0, x1, x2, x3);
                ld.shared.v4.f32((x0, x1, x2, x3), [xs + ((blk * 32 + q4 * 16) * 4 + 16)]);
                fma_s8x4(s, w1, x0, x1, x2, x3);
                ld.shared.v4.f32((x0, x1, x2, x3), [xs + ((blk * 32 + q4 * 16) * 4 + 32)]);
                fma_s8x4(s, w2, x0, x1, x2, x3);
                ld.shared.v4.f32((x0, x1, x2, x3), [xs + ((blk * 32 + q4 * 16) * 4 + 48)]);
                fma_s8x4(s, w3, x0, x1, x2, x3);
            }
            let h: f16;
            let sc: f32;
            ld.global.nc.b16(h, [da + (blk * 2)]);
            cvt.f32.f16(sc, h);
            fma.rn.f32(v, s, sc, v);
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

    // ---------------------------------------------------------------- attention

    /// Stores the K and V of one token (`nkv` floats each) as f16 at cache row `slot`, which is `pos % ring` for a ring
    /// buffer and `pos` where `ring` is 0; pos is `params[0]`.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn kv_store(k: *const f32, v: *const f32, kc: *mut u16, vc: *mut u16, params: *const u32, ring: u32, nkv: u32) {
        let (b, nt, t, c): (u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(nt, ntid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, b, nt, t);
        let past: pred;
        setp.ge.u32(past, c, nkv);
        if past {
            ret();
        }
        let (pos, slot): (u32, u32);
        ld.global.u32(pos, [params]);
        mov.u32(slot, pos);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(slot, pos, ring);
        }
        let (ka, va, kca, vca): (u64, u64, u64, u64);
        let (kv, vv): (f32, f32);
        let (kh, vh): (f16, f16);
        let o: u32;
        mad.wide.u32(ka, c, 4, k);
        mad.wide.u32(va, c, 4, v);
        ld.global.f32(kv, [ka]);
        ld.global.f32(vv, [va]);
        cvt.rn.f16.f32(kh, kv);
        cvt.rn.f16.f32(vh, vv);
        mad.lo.u32(o, slot, nkv, c);
        mad.wide.u32(kca, o, 2, kc);
        mad.wide.u32(vca, o, 2, vc);
        st.global.b16([kca], kh);
        st.global.b16([vca], vh);
    }

    /// Decode attention of one token, a block per query head (128 dims), 8 kv heads, `group` query heads per kv head.
    /// score_j = scale * q.k_j + rb[min(d_j, extent)] . r - sink, d_j = pos - pos_j; the pad row (d >= extent) is zero.
    /// The softmax has an extra zero logit (the sink, already subtracted from the scores). pos is `params[0]`.
    /// The cache holds pos_j = j (`ring` 0), or a ring of `ring` rows where row s holds the last position = s mod ring.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn attn_decode(
        q: *const f32, r: *const f32, sink: *const f32, rb: *const f32, kc: *const u16, vc: *const u16, out: *mut f32,
        params: *const u32, ring: u32, extent: u32, group: u32, scale: f32,
    ) {
        let ms: shared<[f32; 8], 16>;
        let ls: shared<[f32; 8], 16>;
        let accs: shared<[f32; 1024], 16>;
        let (h, t, lane, w, kvh): (u32, u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        div.u32(kvh, h, group);

        // q: 4 dims per lane; r: one dim per lane below 16
        let (qo, qa): (u32, u64);
        mad.lo.u32(qo, h, 128, 0);
        shl.b32(qo, h, 7);
        let l4: u32;
        shl.b32(l4, lane, 2);
        add.u32(qo, qo, l4);
        mad.wide.u32(qa, qo, 4, q);
        let (q0, q1, q2, q3): (f32, f32, f32, f32);
        ld.global.v4.f32((q0, q1, q2, q3), [qa]);
        mul.rn.f32(q0, q0, scale);
        mul.rn.f32(q1, q1, scale);
        mul.rn.f32(q2, q2, scale);
        mul.rn.f32(q3, q3, scale);
        let rv: f32;
        mov.f32(rv, 0.0);
        let rl: pred;
        setp.lt.u32(rl, lane, 16);
        let (ro, ra): (u32, u64);
        shl.b32(ro, h, 4);
        add.u32(ro, ro, lane);
        mad.wide.u32(ra, ro, 4, r);
        if rl {
            ld.global.f32(rv, [ra]);
        }
        let (sk, sa): (f32, u64);
        mad.wide.u32(sa, h, 4, sink);
        ld.global.f32(sk, [sa]);

        let (pos, nk): (u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(nk, pos, 1);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            min.u32(nk, nk, ring);
        }
        let pmod: u32;
        mov.u32(pmod, pos);
        if is_ring {
            rem.u32(pmod, pos, ring);
        }

        let (m, l, a0, a1, a2, a3): (f32, f32, f32, f32, f32, f32);
        mov.f32(m, -3.0e38);
        mov.f32(l, 0.0);
        mov.f32(a0, 0.0);
        mov.f32(a1, 0.0);
        mov.f32(a2, 0.0);
        mov.f32(a3, 0.0);
        let kvo: u32;
        shl.b32(kvo, kvh, 7);
        add.u32(kvo, kvo, l4);

        let j: u32;
        mov.u32(j, w);
        label(top);
        let done: pred;
        setp.ge.u32(done, j, nk);
        if done {
            bra(end);
        }
        // distance of row j
        let dist: u32;
        sub.u32(dist, pos, j);
        if is_ring {
            add.u32(dist, pmod, ring);
            sub.u32(dist, dist, j);
            rem.u32(dist, dist, ring);
        }
        let (ko, ka, va): (u32, u64, u64);
        mad.lo.u32(ko, j, 1024, kvo);
        mad.wide.u32(ka, ko, 2, kc);
        mad.wide.u32(va, ko, 2, vc);
        let (k0, k1, k2, k3): (f16, f16, f16, f16);
        ld.global.v4.b16((k0, k1, k2, k3), [ka]);
        let (kf0, kf1, kf2, kf3): (f32, f32, f32, f32);
        cvt.f32.f16(kf0, k0);
        cvt.f32.f16(kf1, k1);
        cvt.f32.f16(kf2, k2);
        cvt.f32.f16(kf3, k3);
        let s: f32;
        mul.rn.f32(s, q0, kf0);
        fma.rn.f32(s, q1, kf1, s);
        fma.rn.f32(s, q2, kf2, s);
        fma.rn.f32(s, q3, kf3, s);
        // relative bias: lanes below 16 add rb[d][lane] * r[lane]
        let inb: pred;
        setp.lt.u32(inb, dist, extent);
        let (bo, ba): (u32, u64);
        let bv: f32;
        mad.lo.u32(bo, dist, 16, lane);
        mad.wide.u32(ba, bo, 4, rb);
        and.pred(inb, inb, rl);
        if inb {
            ld.global.nc.f32(bv, [ba]);
            fma.rn.f32(s, bv, rv, s);
        }
        warp_sum(s);
        sub.rn.f32(s, s, sk);
        // online softmax
        let (mn, alpha, p): (f32, f32, f32);
        max.f32(mn, m, s);
        sub.rn.f32(alpha, m, mn);
        exp(alpha, alpha);
        sub.rn.f32(p, s, mn);
        exp(p, p);
        fma.rn.f32(l, l, alpha, p);
        mov.f32(m, mn);
        let (v0, v1, v2, v3): (f16, f16, f16, f16);
        ld.global.v4.b16((v0, v1, v2, v3), [va]);
        let (vf0, vf1, vf2, vf3): (f32, f32, f32, f32);
        cvt.f32.f16(vf0, v0);
        cvt.f32.f16(vf1, v1);
        cvt.f32.f16(vf2, v2);
        cvt.f32.f16(vf3, v3);
        mul.rn.f32(a0, a0, alpha);
        fma.rn.f32(a0, p, vf0, a0);
        mul.rn.f32(a1, a1, alpha);
        fma.rn.f32(a1, p, vf1, a1);
        mul.rn.f32(a2, a2, alpha);
        fma.rn.f32(a2, p, vf2, a2);
        mul.rn.f32(a3, a3, alpha);
        fma.rn.f32(a3, p, vf3, a3);
        add.u32(j, j, 8);
        bra(top);
        label(end);

        // combine the 8 warps and the zero sink logit
        let (mb, lb, ab): (u32, u32, u32);
        mov.u32(mb, ms);
        mov.u32(lb, ls);
        mov.u32(ab, accs);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        let wa: u32;
        mad.lo.u32(wa, w, 4, mb);
        if first {
            st.shared.f32([wa], m);
        }
        mad.lo.u32(wa, w, 4, lb);
        if first {
            st.shared.f32([wa], l);
        }
        let aa: u32;
        mad.lo.u32(aa, w, 128, lane);
        shl.b32(aa, aa, 2);
        add.u32(aa, aa, ab);
        // lane holds dims 4*lane..4*lane+3 of warp w: accs[w][4*lane..]
        mad.lo.u32(aa, w, 512, ab);
        mad.lo.u32(aa, lane, 16, aa);
        st.shared.v4.f32([aa], (a0, a1, a2, a3));
        bar.sync(0);
        let dimth: pred;
        setp.lt.u32(dimth, t, 128);
        if !dimth {
            ret();
        }
        let mm: f32;
        mov.f32(mm, 0.0);
        for k in 0..8 {
            let mk: f32;
            ld.shared.f32(mk, [mb + (k * 4)]);
            max.f32(mm, mm, mk);
        }
        let (ll, acc): (f32, f32);
        sub.rn.f32(ll, 0.0, mm);
        exp(ll, ll);
        mov.f32(acc, 0.0);
        for k in 0..8 {
            let (mk, lk, ak, ek): (f32, f32, f32, f32);
            ld.shared.f32(mk, [mb + (k * 4)]);
            ld.shared.f32(lk, [lb + (k * 4)]);
            sub.rn.f32(ek, mk, mm);
            exp(ek, ek);
            fma.rn.f32(ll, lk, ek, ll);
            let ao: u32;
            mad.lo.u32(ao, t, 4, ab);
            ld.shared.f32(ak, [ao + (k * 512)]);
            fma.rn.f32(acc, ak, ek, acc);
        }
        div.rn.f32(acc, acc, ll);
        let (oo, oa): (u32, u64);
        shl.b32(oo, h, 7);
        add.u32(oo, oo, t);
        mad.wide.u32(oa, oo, 4, out);
        st.global.f32([oa], acc);
    }

    // ---------------------------------------------------------------- MoE experts on the GPU

    /// out[c] = sum_k wts[k] * (slots[k] >= 0 ? gpu[k][c] : cpu[k][c]); `cpu` may be host-mapped memory.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn moe_combine(gpu: *const f32, cpu: *const f32, slots: *const i32, wts: *const f32, out: *mut f32, n: u32) {
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
        let acc: f32;
        mov.f32(acc, 0.0);
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
            fma.rn.f32(acc, wv, v, acc);
        }
        let oa: u64;
        mad.wide.u32(oa, c, 4, out);
        st.global.f32([oa], acc);
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

    // ================================================================ int8 (dp4a) kernels

    /// Sum of the s32 `v` over the warp.
    fn warp_isum(v: s32) {
        for sh in [16, 8, 4, 2, 1] {
            let o: s32;
            shfl.sync.bfly.b32(o, v, sh, 31, 0xffff_ffff);
            add.s32(v, v, o);
        }
    }

    /// Quantizes the f32 `v` of each lane as int8 of the warp's block of 32: q = round(v / d), d = max|v| / 127;
    /// lanes 0, 4, .. store 4 bytes each at `qa` (byte address of the block), lane 0 stores d and d * sum(q).
    fn q8_warp(v: f32, lane: u32, qa: u64, da: u64, sa: u64) {
        let (a, amax, d, id): (f32, f32, f32, f32);
        abs.f32(a, v);
        mov.f32(amax, a);
        warp_max(amax);
        div.rn.f32(d, amax, 127.0);
        let nz: pred;
        setp.gt.f32(nz, amax, 0.0);
        mov.f32(id, 0.0);
        if nz {
            rcp.rn.f32(id, d);
        }
        let qf: f32;
        mul.rn.f32(qf, v, id);
        let qi: s32;
        cvt.rni.s32.f32(qi, qf);
        let sum: s32;
        mov.s32(sum, qi);
        warp_isum(sum);
        // gather 4 bytes into lanes 0 mod 4
        let (b0, b1, b2, b3, w): (u32, u32, u32, u32, u32);
        and.b32(b0, qi, 255);
        shfl.sync.down.b32(b1, b0, 1, 31, 0xffff_ffff);
        shfl.sync.down.b32(b2, b0, 2, 31, 0xffff_ffff);
        shfl.sync.down.b32(b3, b0, 3, 31, 0xffff_ffff);
        shl.b32(b1, b1, 8);
        shl.b32(b2, b2, 16);
        shl.b32(b3, b3, 24);
        or.b32(w, b0, b1);
        or.b32(w, w, b2);
        or.b32(w, w, b3);
        let (l4, four): (u32, pred);
        and.b32(l4, lane, 3);
        setp.eq.u32(four, l4, 0);
        let wa: u64;
        cvt.u64.u32(wa, lane);
        add.u64(wa, wa, qa);
        if four {
            st.global.u32([wa], w);
        }
        let first: pred;
        setp.eq.u32(first, lane, 0);
        let (sf, ds): (f32, f32);
        cvt.rn.f32.s32(sf, sum);
        mul.rn.f32(ds, d, sf);
        if first {
            st.global.f32([da], d);
            st.global.f32([sa], ds);
        }
    }

    /// x (n f32) -> q8_1: q [n] int8, d [n/32], s [n/32] = d * sum(q). A warp per block of 32; grid.y walks vectors.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn quant_q8_1(x: *const f32, q: *mut u8, d: *mut f32, s: *mut f32, n: u32) {
        let (lane, b): (u32, u32);
        warp_row(lane, b);
        let (nb, vtok): (u32, u32);
        shr.u32(nb, n, 5);
        mov.u32(vtok, ctaid.y);
        let past: pred;
        setp.ge.u32(past, b, nb);
        if past {
            ret();
        }
        let (gb, i): (u32, u32);
        mad.lo.u32(gb, vtok, nb, b);
        shl.b32(i, gb, 5);
        add.u32(i, i, lane);
        let (xa, qa, da, sa): (u64, u64, u64, u64);
        let v: f32;
        mad.wide.u32(xa, i, 4, x);
        ld.global.f32(v, [xa]);
        let qb: u32;
        shl.b32(qb, gb, 5);
        mad.wide.u32(qa, qb, 1, q);
        mad.wide.u32(da, gb, 4, d);
        mad.wide.u32(sa, gb, 4, s);
        q8_warp(v, lane, qa, da, sa);
    }

    /// Copies `nbytes` (a multiple of 4) of global `src` to shared `dst`, by all threads of the block (no wait).
    fn stage_bytes(dst: u32, src: u64, nbytes: u32) {
        let (i, nt, nw): (u32, u32, u32);
        mov.u32(i, tid.x);
        mov.u32(nt, ntid.x);
        shr.u32(nw, nbytes, 2);
        label(top);
        let done: pred;
        setp.ge.u32(done, i, nw);
        if done {
            bra(end);
        }
        let at: u64;
        let sat: u32;
        let v: u32;
        mad.wide.u32(at, i, 4, src);
        ld.global.u32(v, [at]);
        mad.lo.u32(sat, i, 4, dst);
        st.shared.u32([sat], v);
        add.u32(i, i, nt);
        bra(top);
        label(end);
    }

    /// y[r] = W x for a repacked Q8_0 matrix (`qs` [rows][k] int8, `dw` [rows][k/32] f16) and q8_1 x (`xq`, `xd`).
    /// k is a multiple of 32 and at most 4608. A half warp per row, 16 rows per block.
    /// The body of gemv_q8d, for block b.
    fn gemv_q8d_body(qs: u64, dw: u64, xq: u64, xd: u64, y: u64, rows: u32, k: u32, b: u32) {
        let xs: shared<[u8; 4608], 16>;
        let ds: shared<[f32; 144], 16>;
        let (xb, db, nb): (u32, u32, u32);
        mov.u32(xb, xs);
        mov.u32(db, ds);
        shr.u32(nb, k, 5);
        stage_bytes(xb, xq, k);
        let nb4: u32;
        shl.b32(nb4, nb, 2);
        stage_bytes(db, xd, nb4);
        bar.sync(0);
        let (t, hl, hw, row): (u32, u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(hl, t, 15);
        shr.u32(hw, t, 4);
        mad.lo.u32(row, b, 16, hw);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (rowq, rowd): (u32, u32);
        mul.lo.u32(rowq, row, k);
        mul.lo.u32(rowd, row, nb);
        let acc: f32;
        mov.f32(acc, 0.0);
        let c: u32;
        shl.b32(c, hl, 4);
        label(top);
        let done: pred;
        setp.ge.u32(done, c, k);
        if done {
            bra(end);
        }
        let (i, j): (u32, u32);
        let (at, dat): (u64, u64);
        add.u32(i, rowq, c);
        mad.wide.u32(at, i, 1, qs);
        let (w0, w1, w2, w3): (u32, u32, u32, u32);
        ld.global.nc.v4.u32((w0, w1, w2, w3), [at]);
        let xa: u32;
        add.u32(xa, xb, c);
        let (x0, x1, x2, x3): (u32, u32, u32, u32);
        ld.shared.v4.u32((x0, x1, x2, x3), [xa]);
        let isum: s32;
        mov.s32(isum, 0);
        dp4a.s32.s32(isum, w0, x0, isum);
        dp4a.s32.s32(isum, w1, x1, isum);
        dp4a.s32.s32(isum, w2, x2, isum);
        dp4a.s32.s32(isum, w3, x3, isum);
        shr.u32(j, c, 5);
        let (h, ja, dxa): (f16, u32, u32);
        add.u32(ja, j, rowd);
        mad.wide.u32(dat, ja, 2, dw);
        ld.global.nc.b16(h, [dat]);
        let (sw, sx, f): (f32, f32, f32);
        cvt.f32.f16(sw, h);
        mad.lo.u32(dxa, j, 4, db);
        ld.shared.f32(sx, [dxa]);
        mul.rn.f32(sw, sw, sx);
        cvt.rn.f32.s32(f, isum);
        fma.rn.f32(acc, f, sw, acc);
        add.u32(c, c, 256);
        bra(top);
        label(end);
        half_sum(acc);
        let first: pred;
        setp.eq.u32(first, hl, 0);
        if first {
            let yat: u64;
            mad.wide.u32(yat, row, 4, y);
            st.global.f32([yat], acc);
        }
    }

    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn gemv_q8d(qs: *const u8, dw: *const u16, xq: *const u8, xd: *const f32, y: *mut f32, rows: u32, k: u32) {
        let b: u32;
        mov.u32(b, ctaid.x);
        gemv_q8d_body(qs, dw, xq, xd, y, rows, k, b);
    }

    /// gemv_q8d of matrix ctaid.z: qs and d at tab[2 z], tab[2 z + 1]; x of matrix z at xq + z k, xd + z k/32, y at y + z rows.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn gemv_q8dz(tab: *const u64, xq: *const u8, xd: *const f32, y: *mut f32, rows: u32, k: u32) {
        let (z, b, zi): (u32, u32, u32);
        mov.u32(z, ctaid.z);
        mov.u32(b, ctaid.x);
        let (ta, qz, dz, xqz, xdz, yz): (u64, u64, u64, u64, u64, u64);
        mul.lo.u32(zi, z, 2);
        mad.wide.u32(ta, zi, 8, tab);
        ld.global.u64(qz, [ta]);
        ld.global.u64(dz, [ta + 8]);
        mul.lo.u32(zi, z, k);
        mad.wide.u32(xqz, zi, 1, xq);
        shr.u32(zi, zi, 5);
        mad.wide.u32(xdz, zi, 4, xd);
        mul.lo.u32(zi, z, rows);
        mad.wide.u32(yz, zi, 4, y);
        gemv_q8d_body(qz, dz, xqz, xdz, yz, rows, k, b);
    }

    /// The SITU activation of the cached experts' gate_up ([8][2 nff]), quantized as q8_1: aq [8][nff], ad, as [8][nff/32].
    /// A warp per block of 32, grid.y = 8 experts.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn situ_q8(gu: *const f32, slots: *const i32, aq: *mut u8, ad: *mut f32, asum: *mut f32, nff: u32, beta: f32, lb: f32) {
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
        let (i, gi, ui, base): (u32, u32, u32, u32);
        shl.b32(i, b, 5);
        add.u32(i, i, lane);
        shl.b32(base, nff, 1);
        mul.lo.u32(base, base, ke);
        add.u32(gi, base, i);
        add.u32(ui, gi, nff);
        let (ga, ua): (u64, u64);
        let (gv, uv, a, sg, tu, ib, il): (f32, f32, f32, f32, f32, f32, f32);
        mad.wide.u32(ga, gi, 4, gu);
        mad.wide.u32(ua, ui, 4, gu);
        ld.global.f32(gv, [ga]);
        ld.global.f32(uv, [ua]);
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
        let (gb, qb): (u32, u32);
        mad.lo.u32(gb, ke, nb, b);
        shl.b32(qb, gb, 5);
        let (qa, da, s2): (u64, u64, u64);
        mad.wide.u32(qa, qb, 1, aq);
        mad.wide.u32(da, gb, 4, ad);
        mad.wide.u32(s2, gb, 4, asum);
        q8_warp(a, lane, qa, da, s2);
    }

    /// The UE4M3 scale byte `b` as f32, halved as ggml decodes it: 2^(e-8) (1 + m/8), or m 2^-10 when e = 0; 0x7F is 0.
    fn ue4m3(f: f32, b: u32) {
        let (e, bits, m): (u32, u32, u32);
        and.b32(e, b, 0x78);
        and.b32(bits, b, 0x7F);
        shl.b32(bits, bits, 20);
        add.u32(bits, bits, 0x3B80_0000);
        mov.b32(f, bits);
        let (sub, nan): (pred, pred);
        setp.eq.u32(sub, e, 0);
        if sub {
            let mf: f32;
            and.b32(m, b, 7);
            cvt.rn.f32.u32(mf, m);
            mul.rn.f32(f, mf, 0.0009765625);
        }
        setp.eq.u32(nan, bits, 0x4B70_0000);
        if nan {
            mov.f32(f, 0.0);
        }
    }

    /// The 8 E2M1 codes of w (its nibbles) as int8 of the doubled values, 13 instructions: `lo` those of the low
    /// nibbles of its 4 bytes, `hi` of the high ones. prmt looks up 4 nibbles at a time in the magnitudes `pa`/`pb`
    /// ({0,1,2,3}, {4,6,8,12}) and the negatives `na`/`nb` (bit 3 masked off: in a prmt selector it replicates the
    /// sign), the sign picks between the two (selector byte k, or 4 + k), the last two sort low and high nibbles apart.
    fn fp4x8(lo: u32, hi: u32, w: u32, pa: u32, pb: u32, na: u32, nb: u32) {
        let (m, s, m1, s1, p, n, a, b): (u32, u32, u32, u32, u32, u32, u32, u32);
        and.b32(m, w, 0x7777_7777);
        shr.u32(s, w, 1);
        and.b32(s, s, 0x4444_4444);
        or.b32(s, s, 0x3210_3210);
        prmt.b32(p, pa, pb, m);
        prmt.b32(n, na, nb, m);
        prmt.b32(a, p, n, s);
        shr.u32(m1, m, 16);
        shr.u32(s1, s, 16);
        prmt.b32(p, pa, pb, m1);
        prmt.b32(n, na, nb, m1);
        prmt.b32(b, p, n, s1);
        prmt.b32(lo, a, b, 0x6420);
        prmt.b32(hi, a, b, 0x7531);
    }

    /// acc += one 16-byte NVFP4 chunk (w0..w3, its group's scale word `sc`) times x: x0..x3 for its low nibbles, x4..x7
    /// for the high ones, `xlo`, `xhi` their scales; `ph` = 8 h, the shift to the chunk's two scale bytes h and 2 + h.
    /// UE4M3 byte b: (b & 0x7F) << 20 is the f32 2^(e-127) (1 + m/8), or m 2^-129 when e = 0 (a subnormal); times
    /// `f119` = 2^119, exactly, the scale halved as ggml decodes it (0x7F, NaN, is 0: experts::repack makes it 0x00).
    fn nv4_chunk(
        acc: f32, w0: u32, w1: u32, w2: u32, w3: u32, sc: u32, x0: u32, x1: u32, x2: u32, x3: u32, x4: u32, x5: u32, x6: u32, x7: u32,
        xlo: f32, xhi: f32, ph: u32, f119: f32, pa: u32, pb: u32, na: u32, nb: u32,
    ) {
        let (l0, l1, l2, l3, h0, h1, h2, h3): (u32, u32, u32, u32, u32, u32, u32, u32);
        fp4x8(l0, h0, w0, pa, pb, na, nb);
        fp4x8(l1, h1, w1, pa, pb, na, nb);
        fp4x8(l2, h2, w2, pa, pb, na, nb);
        fp4x8(l3, h3, w3, pa, pb, na, nb);
        let (dlo, dhi): (u32, u32);
        mov.u32(dlo, 0);
        mov.u32(dhi, 0);
        dp4a.s32.s32(dlo, l0, x0, dlo);
        dp4a.s32.s32(dlo, l1, x1, dlo);
        dp4a.s32.s32(dlo, l2, x2, dlo);
        dp4a.s32.s32(dlo, l3, x3, dlo);
        dp4a.s32.s32(dhi, h0, x4, dhi);
        dp4a.s32.s32(dhi, h1, x5, dhi);
        dp4a.s32.s32(dhi, h2, x6, dhi);
        dp4a.s32.s32(dhi, h3, x7, dhi);
        let (st, blo, bhi): (u32, u32, u32);
        shr.u32(st, sc, ph);
        shl.b32(blo, st, 20);
        and.b32(blo, blo, 0x07F0_0000);
        shl.b32(bhi, st, 4);
        and.b32(bhi, bhi, 0x07F0_0000);
        let (slo, shi, flo, fhi): (f32, f32, f32, f32);
        mov.b32(slo, blo);
        mov.b32(shi, bhi);
        mul.rn.f32(slo, slo, f119);
        mul.rn.f32(shi, shi, f119);
        cvt.rn.f32.s32(flo, dlo);
        cvt.rn.f32.s32(fhi, dhi);
        mul.rn.f32(slo, slo, xlo);
        mul.rn.f32(shi, shi, xhi);
        fma.rn.f32(acc, flo, slo, acc);
        fma.rn.f32(acc, fhi, shi, acc);
    }

    /// y = W x for the 8 experts of `slots` (grid.y) on an NVFP4 bank in the slot layout of experts.rs: rows of k/2
    /// bytes (value j of each 64 in the low nibble of byte j, 32 + j in the high one), the UE4M3 scales (k/16 a row),
    /// then the expert's f32 scale. x is q8_1 (int8 `xq`, `xd` per 32), the same for all experts (xstride 0) or each
    /// expert's at xq + ke xstride. 8 lanes per row (16-byte chunks: a warp reads 128 contiguous bytes of each of 4
    /// rows), a thread takes 4 rows (r + 16i), 64 rows per block of 128 threads: 8 experts of 2304 rows fit the GPU in
    /// one wave. x is read through L1, not staged. k a multiple of 64.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn moe_nv4_d(
        bank: *const u8, slot_bytes: u64, slots: *const i32, xq: *const u8, xd: *const f32, xstride: u32, y: *mut f32, rows: u32, kdim: u32,
    ) {
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
        // x straight from global memory: every block of the expert reads it, L1 keeps it
        let (aoff, boff): (u32, u32);
        mul.lo.u32(aoff, ke, xstride);
        shr.u32(boff, aoff, 5);
        let (xk, dk): (u64, u64);
        mad.wide.u32(xk, aoff, 1, xq);
        mad.wide.u32(dk, boff, 4, xd);
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
        let (sl, base, sca): (u64, u64, u64);
        cvt.u64.s32(sl, slot);
        mul.lo.u64(sl, sl, slot_bytes);
        add.u64(base, bank, sl);
        let (half, qbytes, sk, sbytes, o): (u32, u32, u32, u32, u32);
        shr.u32(half, kdim, 1);
        mul.lo.u32(qbytes, rows, half);
        shr.u32(sk, kdim, 4);
        mul.lo.u32(sbytes, rows, sk);
        add.u32(o, qbytes, sbytes);
        cvt.u64.u32(sca, o);
        add.u64(sca, sca, base);
        // a lane's chunks c = lr + 8i: group g = c/2 = lr/2 + 4i, half h = c%2 = lr%2. The low nibbles are values
        // 64g + 16h.., the high ones 64g + 32 + 16h..; their weight scales are bytes h and 2 + h of the group's 4, their
        // x scales those of blocks 2g, 2g + 1.
        let (nc, c, h, ph, g0, xoff): (u32, u32, u32, u32, u32, u32);
        let (xo, xda): (u64, u64);
        shr.u32(nc, kdim, 5);
        mov.u32(c, lr);
        and.b32(h, lr, 1);
        shl.b32(ph, h, 3);
        shr.u32(g0, lr, 1);
        // the thread's rows row0 + 16 r (past the end: the last row, computed, not stored), their chunk and scale words
        let (qa, sa): ([u64; 4], [u64; 4]);
        for r in 0..4 {
            let (rr, so): (u32, u32);
            add.u32(rr, row0, r * 16);
            min.u32(rr, rr, last);
            mad.wide.u32(qa[r], rr, half, base);
            mad.wide.u32(qa[r], lr, 16, qa[r]);
            mad.lo.u32(so, rr, sk, qbytes);
            mad.lo.u32(so, g0, 4, so);
            cvt.u64.u32(sa[r], so);
            add.u64(sa[r], sa[r], base);
        }
        shl.b32(xoff, g0, 6);
        mad.lo.u32(xoff, h, 16, xoff);
        mad.wide.u32(xo, xoff, 1, xk);
        mad.wide.u32(xda, g0, 8, dk);
        let (pa, pb, na, nbt): (u32, u32, u32, u32);
        mov.u32(pa, 0x0302_0100);
        mov.u32(pb, 0x0C08_0604);
        mov.u32(na, 0xFDFE_FF00);
        mov.u32(nbt, 0xF4F8_FAFC);
        let (k119, f119): (u32, f32);
        mov.u32(k119, 0x7B00_0000);
        mov.b32(f119, k119);
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
        let (w0, w1, w2, w3, sc): ([u32; 4], [u32; 4], [u32; 4], [u32; 4], [u32; 4]);
        for r in 0..4 {
            ld.global.nc.v4.u32((w0[r], w1[r], w2[r], w3[r]), [qa[r]]);
        }
        for r in 0..4 {
            ld.global.nc.u32(sc[r], [sa[r]]);
        }
        let (x0, x1, x2, x3, x4, x5, x6, x7): (u32, u32, u32, u32, u32, u32, u32, u32);
        ld.global.nc.v4.u32((x0, x1, x2, x3), [xo]);
        ld.global.nc.v4.u32((x4, x5, x6, x7), [xo + 32]);
        let (xlo, xhi): (f32, f32);
        ld.global.nc.v2.f32((xlo, xhi), [xda]);
        for r in 0..4 {
            nv4_chunk(acc[r], w0[r], w1[r], w2[r], w3[r], sc[r], x0, x1, x2, x3, x4, x5, x6, x7, xlo, xhi, ph, f119, pa, pb, na, nbt);
        }
        add.u32(c, c, 8);
        for r in 0..4 {
            add.u64(qa[r], qa[r], 128);
            add.u64(sa[r], sa[r], 16);
        }
        add.u64(xo, xo, 256);
        add.u64(xda, xda, 32);
        bra(top);
        label(end);
        for sh in [4, 2, 1] {
            for r in 0..4 {
                let o: f32;
                shfl.sync.bfly.b32(o, acc[r], sh, 31, 0xffff_ffff);
                add.rn.f32(acc[r], acc[r], o);
            }
        }
        let (first, es): (pred, f32);
        setp.eq.u32(first, lr, 0);
        ld.global.nc.f32(es, [sca]);
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

    /// y = W x for the 8 experts of `slots` (grid.y) on a Q8_0 bank in the slot layout of experts.rs: rows of k int8,
    /// then the f16 scales (k/32 a row). x as in moe_nv4_d. 4 lanes per row (16-byte chunks), 64 rows per block.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn moe_q8_d(
        bank: *const u8, slot_bytes: u64, slots: *const i32, xq: *const u8, xd: *const f32, xstride: u32, y: *mut f32, rows: u32, kdim: u32,
    ) {
        let xs: shared<[u8; 2304], 16>;
        let dsh: shared<[f32; 72], 16>;
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
        let (xb, db, nb, nb4, aoff, boff): (u32, u32, u32, u32, u32, u32);
        mov.u32(xb, xs);
        mov.u32(db, dsh);
        shr.u32(nb, kdim, 5);
        shl.b32(nb4, nb, 2);
        mul.lo.u32(aoff, ke, xstride);
        shr.u32(boff, aoff, 5);
        let (xk, dk): (u64, u64);
        mad.wide.u32(xk, aoff, 1, xq);
        mad.wide.u32(dk, boff, 4, xd);
        stage_bytes(xb, xk, kdim);
        stage_bytes(db, dk, nb4);
        bar.sync(0);
        let (t, part, row): (u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(part, t, 3);
        let bk: u32;
        mov.u32(bk, ctaid.x);
        shr.u32(row, t, 2);
        mad.lo.u32(row, bk, 64, row);
        let past: pred;
        setp.ge.u32(past, row, rows);
        if past {
            ret();
        }
        let (sl, base, qrow, drow): (u64, u64, u64, u64);
        cvt.u64.s32(sl, slot);
        mul.lo.u64(sl, sl, slot_bytes);
        add.u64(base, bank, sl);
        mad.wide.u32(qrow, row, kdim, base);
        let (qbytes, o): (u32, u32);
        mul.lo.u32(qbytes, rows, kdim);
        shr.u32(o, kdim, 4);
        mad.lo.u32(o, row, o, qbytes);
        cvt.u64.u32(drow, o);
        add.u64(drow, drow, base);
        let (nc, c): (u32, u32);
        shr.u32(nc, kdim, 4);
        mov.u32(c, part);
        let acc: f32;
        mov.f32(acc, 0.0);
        label(top);
        let done: pred;
        setp.ge.u32(done, c, nc);
        if done {
            bra(end);
        }
        // chunk c: values 16c.., in block c/2
        let (qa, xo, blk): (u64, u32, u32);
        mad.wide.u32(qa, c, 16, qrow);
        let (w0, w1, w2, w3): (u32, u32, u32, u32);
        ld.global.nc.v4.u32((w0, w1, w2, w3), [qa]);
        shl.b32(xo, c, 4);
        add.u32(xo, xo, xb);
        let (x0, x1, x2, x3): (u32, u32, u32, u32);
        ld.shared.v4.u32((x0, x1, x2, x3), [xo]);
        let isum: u32;
        mov.u32(isum, 0);
        dp4a.s32.s32(isum, w0, x0, isum);
        dp4a.s32.s32(isum, w1, x1, isum);
        dp4a.s32.s32(isum, w2, x2, isum);
        dp4a.s32.s32(isum, w3, x3, isum);
        shr.u32(blk, c, 1);
        let (da, hd, dw, dx, f, xda): (u64, f16, f32, f32, f32, u32);
        mad.wide.u32(da, blk, 2, drow);
        ld.global.nc.b16(hd, [da]);
        cvt.f32.f16(dw, hd);
        mad.lo.u32(xda, blk, 4, db);
        ld.shared.f32(dx, [xda]);
        mul.rn.f32(dw, dw, dx);
        cvt.rn.f32.s32(f, isum);
        fma.rn.f32(acc, f, dw, acc);
        add.u32(c, c, 4);
        bra(top);
        label(end);
        for sh in [2, 1] {
            let o: f32;
            shfl.sync.bfly.b32(o, acc, sh, 31, 0xffff_ffff);
            add.rn.f32(acc, acc, o);
        }
        let first: pred;
        setp.eq.u32(first, part, 0);
        if first {
            let (oi, oa): (u32, u64);
            mad.lo.u32(oi, ke, rows, row);
            mad.wide.u32(oa, oi, 4, y);
            st.global.f32([oa], acc);
        }
    }

    /// The (slot, count, offset) of group g and the (token, pair) of the 32 tokens of tile tz into shared `tb`:
    /// slot, the tile's first token and its token count (nt 0: nothing to do).
    fn group_tile(slot: u32, nt: u32, groups: u64, tb: u32) {
        let (g, tz, t, ga, ntok, off, t0): (u32, u32, u32, u64, u32, u32, u32);
        mov.u32(g, ctaid.y);
        mov.u32(tz, ctaid.z);
        mov.u32(t, tid.x);
        mul.lo.u32(off, g, 3);
        mad.wide.u32(ga, off, 4, groups);
        ld.global.u32(slot, [ga]);
        ld.global.u32(ntok, [ga + 4]);
        ld.global.u32(off, [ga + 8]);
        shl.b32(t0, tz, 5);
        mov.u32(nt, 0);
        let some: pred;
        setp.lt.u32(some, t0, ntok);
        if some {
            sub.u32(nt, ntok, t0);
            min.u32(nt, nt, 32);
        }
        let lt64: pred;
        setp.lt.u32(lt64, t, 64);
        and.pred(lt64, lt64, some);
        if lt64 {
            let (e, ta, v, sa2): (u32, u64, u32, u32);
            add.u32(e, off, t0);
            shl.b32(e, e, 1);
            add.u32(e, e, t);
            add.u32(e, e, 384);
            mad.wide.u32(ta, e, 4, groups);
            ld.global.u32(v, [ta]);
            mad.lo.u32(sa2, t, 4, tb);
            st.shared.u32([sa2], v);
        }
    }

    /// Stages values step*128.. of the tile's activations (128 threads): int8 into xb at a stride of 144 bytes a
    /// token, their 4 scales into db (4 a token). The activations of a token (BYPAIR 0) or of a pair (1).
    fn stage128<const BYPAIR: u32>(xb: u32, db: u32, tb: u32, xq: u64, xd: u64, kdim: u32, step: u32, nt: u32) {
        let t: u32;
        mov.u32(t, tid.x);
        for w8 in 0..8 {
            let (wi, tk, wo, src, dst, xr, tka): (u32, u32, u32, u32, u32, u32, u32);
            add.u32(wi, t, w8 * 128);
            shr.u32(tk, wi, 5);
            and.b32(wo, wi, 31);
            let v: u32;
            mov.u32(v, 0);
            let ok: pred;
            setp.lt.u32(ok, tk, nt);
            mad.lo.u32(tka, tk, 8, tb);
            ld.shared.u32(xr, [tka + (BYPAIR * 4)]);
            mul.lo.u32(src, xr, kdim);
            mad.lo.u32(src, step, 128, src);
            mad.lo.u32(src, wo, 4, src);
            let sa: u64;
            mad.wide.u32(sa, src, 1, xq);
            if ok {
                ld.global.u32(v, [sa]);
            }
            mad.lo.u32(dst, tk, 144, xb);
            mad.lo.u32(dst, wo, 4, dst);
            st.shared.u32([dst], v);
        }
        let (tk, bo, src, dst, xr, tka, nb): (u32, u32, u32, u32, u32, u32, u32);
        shr.u32(tk, t, 2);
        and.b32(bo, t, 3);
        let v: f32;
        mov.f32(v, 0.0);
        let ok: pred;
        setp.lt.u32(ok, tk, nt);
        mad.lo.u32(tka, tk, 8, tb);
        ld.shared.u32(xr, [tka + (BYPAIR * 4)]);
        shr.u32(nb, kdim, 5);
        mad.lo.u32(src, xr, nb, bo);
        mad.lo.u32(src, step, 4, src);
        let sa: u64;
        mad.wide.u32(sa, src, 4, xd);
        if ok {
            ld.global.f32(v, [sa]);
        }
        mad.lo.u32(dst, t, 4, db);
        st.shared.f32([dst], v);
    }

    /// Writes the 8 sums of a thread (row, tokens cc + 4i of the tile) to y[pair][rows], times `scale`.
    fn store8(acc: [f32; 8], cc: u32, nt: u32, tb: u32, y: u64, row: u32, rows: u32, scale: f32) {
        let rin: pred;
        setp.lt.u32(rin, row, rows);
        for i in 0..8 {
            let (tk, pair, o, tka, v): (u32, u32, u32, u32, f32);
            add.u32(tk, cc, i * 4);
            let tin: pred;
            setp.lt.u32(tin, tk, nt);
            and.pred(tin, tin, rin);
            mad.lo.u32(tka, tk, 8, tb);
            ld.shared.u32(pair, [tka + 4]);
            mad.lo.u32(o, pair, rows, row);
            let ya: u64;
            mad.wide.u32(ya, o, 4, y);
            mul.rn.f32(v, acc[i], scale);
            if tin {
                st.global.f32([ya], v);
            }
        }
    }

    /// Prefill: y[pair] = W x for the groups of (token, pair) of each cached expert (grid.y), on an NVFP4 bank (as
    /// moe_nv4_d); x of the token (BYPAIR 0, gate_up) or of the pair (1, down). 32 rows (grid.x) by 32 tokens
    /// (grid.z) a block of 128 threads: a thread takes a row and 8 tokens; k a multiple of 128.
    #[reqntid(128)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn moe_nv4_g<const BYPAIR: u32>(
        bank: *const u8, slot_bytes: u64, groups: *const u32, xq: *const u8, xd: *const f32, y: *mut f32, rows: u32, kdim: u32,
    ) {
        let xsh: shared<[u8; 4608], 16>;
        let dsh: shared<[f32; 128], 16>;
        let tsh: shared<[u32; 64], 16>;
        let (xb, db, tb, slot, nt): (u32, u32, u32, u32, u32);
        mov.u32(xb, xsh);
        mov.u32(db, dsh);
        mov.u32(tb, tsh);
        group_tile(slot, nt, groups, tb);
        let none: pred;
        setp.eq.u32(none, nt, 0);
        if none {
            ret();
        }
        bar.sync(0);
        let (t, rr, cc, bx, row, rows1, rowc): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(bx, ctaid.x);
        shr.u32(rr, t, 2);
        and.b32(cc, t, 3);
        mad.lo.u32(row, bx, 32, rr);
        sub.u32(rows1, rows, 1);
        min.u32(rowc, row, rows1);
        let (sl, base, qrow, srow, sca): (u64, u64, u64, u64, u64);
        cvt.u64.u32(sl, slot);
        mul.lo.u64(sl, sl, slot_bytes);
        add.u64(base, bank, sl);
        let (half, qbytes, sbytes, o): (u32, u32, u32, u32);
        shr.u32(half, kdim, 1);
        mul.lo.u32(qbytes, rows, half);
        mad.wide.u32(qrow, rowc, half, base);
        shr.u32(o, kdim, 4);
        mul.lo.u32(sbytes, rows, o);
        mad.lo.u32(o, rowc, o, qbytes);
        cvt.u64.u32(srow, o);
        add.u64(srow, srow, base);
        add.u32(o, qbytes, sbytes);
        cvt.u64.u32(sca, o);
        add.u64(sca, sca, base);
        let (pa, pb, na, nbt): (u32, u32, u32, u32);
        mov.u32(pa, 0x0302_0100);
        mov.u32(pb, 0x0C08_0604);
        mov.u32(na, 0xFDFE_FF00);
        mov.u32(nbt, 0xF4F8_FAFC);
        let acc: [f32; 8];
        for i in 0..8 {
            mov.f32(acc[i], 0.0);
        }
        let (step, nsteps): (u32, u32);
        mov.u32(step, 0);
        shr.u32(nsteps, kdim, 7);
        label(stop_);
        let sdone: pred;
        setp.ge.u32(sdone, step, nsteps);
        if sdone {
            bra(send_);
        }
        stage128::<BYPAIR>(xb, db, tb, xq, xd, kdim, step, nt);
        bar.sync(0);
        // the step's 8 weight scales (two groups of 64)
        let (qa, sa, s0, s1): (u64, u64, u32, u32);
        mad.wide.u32(qa, step, 64, qrow);
        mad.wide.u32(sa, step, 8, srow);
        ld.global.nc.v2.u32((s0, s1), [sa]);
        for j in 0..4 {
            // chunk j: group j/2 of the step, half j%2: low nibbles values 16h.. of it, high ones 32 + 16h..
            let (blo, bhi, slo, shi): (u32, u32, f32, f32);
            if j < 2 {
                bfe.u32(blo, s0, (j % 2) * 8, 8);
                bfe.u32(bhi, s0, 16 + (j % 2) * 8, 8);
            } else {
                bfe.u32(blo, s1, (j % 2) * 8, 8);
                bfe.u32(bhi, s1, 16 + (j % 2) * 8, 8);
            }
            ue4m3(slo, blo);
            ue4m3(shi, bhi);
            let (w0, w1, w2, w3): (u32, u32, u32, u32);
            ld.global.nc.v4.u32((w0, w1, w2, w3), [qa + (j * 16)]);
            let (l0, l1, l2, l3, h0, h1, h2, h3): (u32, u32, u32, u32, u32, u32, u32, u32);
            fp4x8(l0, h0, w0, pa, pb, na, nbt);
            fp4x8(l1, h1, w1, pa, pb, na, nbt);
            fp4x8(l2, h2, w2, pa, pb, na, nbt);
            fp4x8(l3, h3, w3, pa, pb, na, nbt);
            for i in 0..8 {
                let (tk, xa, da): (u32, u32, u32);
                add.u32(tk, cc, i * 4);
                mad.lo.u32(xa, tk, 144, xb);
                let (x0, x1, x2, x3, x4, x5, x6, x7): (u32, u32, u32, u32, u32, u32, u32, u32);
                ld.shared.v4.u32((x0, x1, x2, x3), [xa + ((j / 2) * 64 + (j % 2) * 16)]);
                ld.shared.v4.u32((x4, x5, x6, x7), [xa + ((j / 2) * 64 + 32 + (j % 2) * 16)]);
                let (dl, dh): (u32, u32);
                mov.u32(dl, 0);
                mov.u32(dh, 0);
                dp4a.s32.s32(dl, l0, x0, dl);
                dp4a.s32.s32(dl, l1, x1, dl);
                dp4a.s32.s32(dl, l2, x2, dl);
                dp4a.s32.s32(dl, l3, x3, dl);
                dp4a.s32.s32(dh, h0, x4, dh);
                dp4a.s32.s32(dh, h1, x5, dh);
                dp4a.s32.s32(dh, h2, x6, dh);
                dp4a.s32.s32(dh, h3, x7, dh);
                let (xl, xh, fl, fh): (f32, f32, f32, f32);
                mad.lo.u32(da, tk, 16, db);
                ld.shared.f32(xl, [da + ((j / 2) * 8)]);
                ld.shared.f32(xh, [da + ((j / 2) * 8 + 4)]);
                mul.rn.f32(xl, xl, slo);
                mul.rn.f32(xh, xh, shi);
                cvt.rn.f32.s32(fl, dl);
                cvt.rn.f32.s32(fh, dh);
                fma.rn.f32(acc[i], fl, xl, acc[i]);
                fma.rn.f32(acc[i], fh, xh, acc[i]);
            }
        }
        bar.sync(0);
        add.u32(step, step, 1);
        bra(stop_);
        label(send_);
        let es: f32;
        ld.global.nc.f32(es, [sca]);
        store8(acc, cc, nt, tb, y, row, rows, es);
    }

    /// Prefill on a Q8_0 bank (as moe_q8_d), as moe_nv4_g.
    #[reqntid(128)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn moe_q8_g<const BYPAIR: u32>(
        bank: *const u8, slot_bytes: u64, groups: *const u32, xq: *const u8, xd: *const f32, y: *mut f32, rows: u32, kdim: u32,
    ) {
        let xsh: shared<[u8; 4608], 16>;
        let dsh: shared<[f32; 128], 16>;
        let tsh: shared<[u32; 64], 16>;
        let (xb, db, tb, slot, nt): (u32, u32, u32, u32, u32);
        mov.u32(xb, xsh);
        mov.u32(db, dsh);
        mov.u32(tb, tsh);
        group_tile(slot, nt, groups, tb);
        let none: pred;
        setp.eq.u32(none, nt, 0);
        if none {
            ret();
        }
        bar.sync(0);
        let (t, rr, cc, bx, row, rows1, rowc): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        mov.u32(bx, ctaid.x);
        shr.u32(rr, t, 2);
        and.b32(cc, t, 3);
        mad.lo.u32(row, bx, 32, rr);
        sub.u32(rows1, rows, 1);
        min.u32(rowc, row, rows1);
        let (sl, base, qrow, drow): (u64, u64, u64, u64);
        cvt.u64.u32(sl, slot);
        mul.lo.u64(sl, sl, slot_bytes);
        add.u64(base, bank, sl);
        mad.wide.u32(qrow, rowc, kdim, base);
        let (qbytes, o): (u32, u32);
        mul.lo.u32(qbytes, rows, kdim);
        shr.u32(o, kdim, 4);
        mad.lo.u32(o, rowc, o, qbytes);
        cvt.u64.u32(drow, o);
        add.u64(drow, drow, base);
        let acc: [f32; 8];
        for i in 0..8 {
            mov.f32(acc[i], 0.0);
        }
        let (step, nsteps): (u32, u32);
        mov.u32(step, 0);
        shr.u32(nsteps, kdim, 7);
        label(stop_);
        let sdone: pred;
        setp.ge.u32(sdone, step, nsteps);
        if sdone {
            bra(send_);
        }
        stage128::<BYPAIR>(xb, db, tb, xq, xd, kdim, step, nt);
        bar.sync(0);
        // the step's 4 f16 weight scales
        let (qa, sa): (u64, u64);
        mad.wide.u32(qa, step, 128, qrow);
        mad.wide.u32(sa, step, 8, drow);
        let dws: [f32; 4];
        for b in 0..4 {
            let hf: f16;
            ld.global.nc.b16(hf, [sa + (b * 2)]);
            cvt.f32.f16(dws[b], hf);
        }
        for j in 0..8 {
            let (w0, w1, w2, w3): (u32, u32, u32, u32);
            ld.global.nc.v4.u32((w0, w1, w2, w3), [qa + (j * 16)]);
            for i in 0..8 {
                let (tk, xa, da): (u32, u32, u32);
                add.u32(tk, cc, i * 4);
                mad.lo.u32(xa, tk, 144, xb);
                let (x0, x1, x2, x3): (u32, u32, u32, u32);
                ld.shared.v4.u32((x0, x1, x2, x3), [xa + (j * 16)]);
                let isum: u32;
                mov.u32(isum, 0);
                dp4a.s32.s32(isum, w0, x0, isum);
                dp4a.s32.s32(isum, w1, x1, isum);
                dp4a.s32.s32(isum, w2, x2, isum);
                dp4a.s32.s32(isum, w3, x3, isum);
                let (dx, f): (f32, f32);
                mad.lo.u32(da, tk, 16, db);
                ld.shared.f32(dx, [da + ((j / 2) * 4)]);
                mul.rn.f32(dx, dx, dws[j / 2]);
                cvt.rn.f32.s32(f, isum);
                fma.rn.f32(acc[i], f, dx, acc[i]);
            }
        }
        bar.sync(0);
        add.u32(step, step, 1);
        bra(stop_);
        label(send_);
        let one: f32;
        mov.f32(one, 1.0);
        store8(acc, cc, nt, tb, y, row, rows, one);
    }

    /// Router logits: logits[e] = W[e] . x, W [n_expert][n] f32 (GGUF layout); a warp per expert.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn router_logits(w: *const f32, x: *const f32, logits: *mut f32, n_expert: u32, n: u32) {
        let (lane, e): (u32, u32);
        warp_row(lane, e);
        let past: pred;
        setp.ge.u32(past, e, n_expert);
        if past {
            ret();
        }
        let (ro, c, tk, xo): (u32, u32, u32, u32);
        mul.lo.u32(ro, e, n);
        shl.b32(c, lane, 2);
        mov.u32(tk, ctaid.y);
        mul.lo.u32(xo, tk, n);
        let xt: u64;
        mad.wide.u32(xt, xo, 4, x);
        let acc: f32;
        mov.f32(acc, 0.0);
        label(top);
        let done: pred;
        setp.ge.u32(done, c, n);
        if done {
            bra(end);
        }
        let i: u32;
        let (wa, xa): (u64, u64);
        add.u32(i, ro, c);
        mad.wide.u32(wa, i, 4, w);
        mad.wide.u32(xa, c, 4, xt);
        let (w0, w1, w2, w3, x0, x1, x2, x3): (f32, f32, f32, f32, f32, f32, f32, f32);
        ld.global.nc.v4.f32((w0, w1, w2, w3), [wa]);
        ld.global.v4.f32((x0, x1, x2, x3), [xa]);
        fma.rn.f32(acc, w0, x0, acc);
        fma.rn.f32(acc, w1, x1, acc);
        fma.rn.f32(acc, w2, x2, acc);
        fma.rn.f32(acc, w3, x3, acc);
        add.u32(c, c, 128);
        bra(top);
        label(end);
        warp_sum(acc);
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            let (la, lo): (u64, u32);
            mad.lo.u32(lo, tk, n_expert, e);
            mad.wide.u32(la, lo, 4, logits);
            st.global.f32([la], acc);
        }
    }

    // ---------------------------------------------------------------- split-K decode attention

    /// The number of keys of the token at params[0], and its position modulo the ring.
    fn n_keys(nk: u32, pmod: u32, pos: u32, ring: u32) {
        add.u32(nk, pos, 1);
        mov.u32(pmod, pos);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            min.u32(nk, nk, ring);
            rem.u32(pmod, pos, ring);
        }
    }

    /// One chunk of 256 keys of one kv head, for its 3 query heads: scores (scale q.k + relative bias - sink), their max m
    /// and sum l of exp(s - m), and acc = sum exp(s - m) v. Partials go to pm/pl [kvh][maxs][3] and pacc [kvh][maxs][3][128].
    /// grid = (8 kv heads, maxs chunks); chunks past the keys of the token do nothing.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn attn_part(
        q: *const f32, r: *const f32, sink: *const f32, rb: *const f32, kc: *const u16, vc: *const u16, pm: *mut f32, pl: *mut f32,
        pacc: *mut f32, params: *const u32, ring: u32, extent: u32, scale: f32, maxs: u32, window: u32,
    ) {
        let qsh: shared<[f32; 384], 16>;
        let rsh: shared<[f32; 48], 16>;
        let ksh: shared<[f32; 4], 16>;
        let sc: shared<[f32; 768], 16>;
        let (kvh, sp, t): (u32, u32, u32);
        mov.u32(kvh, ctaid.x);
        mov.u32(sp, ctaid.y);
        mov.u32(t, tid.x);
        let (pos, nk, pmod, start, cnt): (u32, u32, u32, u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(nk, pos, 1);
        min.u32(nk, nk, window);
        mov.u32(pmod, 0);
        shl.b32(start, sp, 8);
        let none: pred;
        setp.ge.u32(none, start, nk);
        if none {
            ret();
        }
        sub.u32(cnt, nk, start);
        min.u32(cnt, cnt, 256);
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

        // acc: a dimension per thread
        let (a0, a1, a2): (f32, f32, f32);
        mov.f32(a0, 0.0);
        mov.f32(a1, 0.0);
        mov.f32(a2, 0.0);
        let jj2: u32;
        mov.u32(jj2, 0);
        label(vtop);
        let vdone: pred;
        setp.ge.u32(vdone, jj2, cnt);
        if vdone {
            bra(vend);
        }
        let (dist, kp, j, vo, pa): (u32, u32, u32, u32, u32);
        add.u32(dist, start, jj2);
        sub.u32(kp, pos, dist);
        mov.u32(j, kp);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(j, kp, ring);
        }
        mad.lo.u32(vo, j, 8, kvh);
        shl.b32(vo, vo, 7);
        add.u32(vo, vo, t);
        let va: u64;
        mad.wide.u32(va, vo, 2, vc);
        let vh: f16;
        let (vf, p0, p1, p2): (f32, f32, f32, f32);
        ld.global.nc.b16(vh, [va]);
        cvt.f32.f16(vf, vh);
        mad.lo.u32(pa, jj2, 4, sb);
        ld.shared.f32(p0, [pa]);
        ld.shared.f32(p1, [pa + 1024]);
        ld.shared.f32(p2, [pa + 2048]);
        fma.rn.f32(a0, p0, vf, a0);
        fma.rn.f32(a1, p1, vf, a1);
        fma.rn.f32(a2, p2, vf, a2);
        add.u32(jj2, jj2, 1);
        bra(vtop);
        label(vend);
        let (po, pao): (u32, u64);
        mad.lo.u32(po, kvh, maxs, sp);
        mul.lo.u32(po, po, 384);
        add.u32(po, po, t);
        mad.wide.u32(pao, po, 4, pacc);
        st.global.f32([pao], a0);
        st.global.f32([pao + 512], a1);
        st.global.f32([pao + 1024], a2);
    }

    /// The scores of key `start + jj` for the 3 heads in shared q, into shared sc[h][jj].
    fn score_key(start: u32, jj: u32, kvh: u32, pos: u32, pmod: u32, ring: u32, extent: u32, kc: u64, rb: u64, qb: u32, rbs: u32, kb: u32, sb: u32) {
        let (j, ko): (u32, u32);
        add.u32(j, start, jj);
        mad.lo.u32(ko, j, 8, kvh);
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
        // distance of key j and its relative bias
        let dist: u32;
        sub.u32(dist, pos, j);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            add.u32(dist, pmod, ring);
            sub.u32(dist, dist, j);
            rem.u32(dist, dist, ring);
        }
        let inb: pred;
        setp.lt.u32(inb, dist, extent);
        if inb {
            let bo: u32;
            let ba: u64;
            shl.b32(bo, dist, 4);
            mad.wide.u32(ba, bo, 4, rb);
            for c in 0..4 {
                let (b0, b1, b2, b3): (f32, f32, f32, f32);
                ld.global.nc.v4.f32((b0, b1, b2, b3), [ba + (c * 16)]);
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

    /// Combines the chunks of attn_part and the zero sink logit: out[h][d], a block per query head, a thread per dimension.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn attn_reduce(pm: *const f32, pl: *const f32, pacc: *const f32, out: *mut f32, params: *const u32, window: u32, maxs: u32) {
        let (h, t, kvh, hh): (u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(t, tid.x);
        div.u32(kvh, h, 3);
        rem.u32(hh, h, 3);
        let (pos, nk, pmod, ns): (u32, u32, u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(nk, pos, 1);
        min.u32(nk, nk, window);
        mov.u32(pmod, 0);
        add.u32(ns, nk, 255);
        shr.u32(ns, ns, 8);
        let (mm, sp): (f32, u32);
        mov.f32(mm, 0.0);
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

    /// Y[t][r] = W x[t] for up to 32 tokens per tile (grid.y walks token tiles): W repacked Q8_0 [rows][k], x q8_1
    /// [ntok][k]. A tile is 32 rows x 32 tokens; thread (rr, cc) takes row rr and tokens cc, cc + 4, .., cc + 28.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn gemm_q8d_t(qs: *const u8, dw: *const u16, xq: *const u8, xd: *const f32, y: *mut f32, rows: u32, k: u32, ntok: u32, y_stride: u32) {
        let xsh: shared<[u8; 8704], 16>; // 32 tokens x 272 bytes (256 + pad)
        let dsh: shared<[f32; 256], 16>; // 32 tokens x 8 blocks
        let (t, rr, cc, bx, by, row, rowc, t0): (u32, u32, u32, u32, u32, u32, u32, u32);
        mov.u32(t, tid.x);
        shr.u32(rr, t, 2);
        and.b32(cc, t, 3);
        mov.u32(bx, ctaid.x);
        mov.u32(by, ctaid.y);
        mad.lo.u32(row, bx, 32, rr);
        shl.b32(t0, by, 5);
        let rows1: u32;
        sub.u32(rows1, rows, 1);
        min.u32(rowc, row, rows1);
        let (xb, db, nb): (u32, u32, u32);
        mov.u32(xb, xsh);
        mov.u32(db, dsh);
        shr.u32(nb, k, 5);
        let acc: [f32; 8];
        for i in 0..8 {
            mov.f32(acc[i], 0.0);
        }
        let (rowq, rowd): (u32, u32);
        mul.lo.u32(rowq, rowc, k);
        mul.lo.u32(rowd, rowc, nb);
        let kc: u32;
        mov.u32(kc, 0);
        label(ktop);
        let kdone: pred;
        setp.ge.u32(kdone, kc, k);
        if kdone {
            bra(kend);
        }
        // stage 256 bytes of x for 32 tokens: thread t copies words t, t + 128, .. (2048 words)
        for w8 in 0..16 {
            let (wi, tok, wo, gt, src, dst): (u32, u32, u32, u32, u32, u32);
            add.u32(wi, t, w8 * 128);
            shr.u32(tok, wi, 6);
            and.b32(wo, wi, 63);
            add.u32(gt, t0, tok);
            let v: u32;
            mov.u32(v, 0);
            let ok: pred;
            setp.lt.u32(ok, gt, ntok);
            mad.lo.u32(src, gt, k, kc);
            mad.lo.u32(src, wo, 4, src);
            let sa: u64;
            mad.wide.u32(sa, src, 1, xq);
            if ok {
                ld.global.u32(v, [sa]);
            }
            mad.lo.u32(dst, tok, 272, xb);
            mad.lo.u32(dst, wo, 4, dst);
            st.shared.u32([dst], v);
        }
        for w2 in 0..2 {
            let (wi, tok, bo, gt, src, dst): (u32, u32, u32, u32, u32, u32);
            add.u32(wi, t, w2 * 128);
            shr.u32(tok, wi, 3);
            and.b32(bo, wi, 7);
            add.u32(gt, t0, tok);
            let v: f32;
            mov.f32(v, 0.0);
            let ok: pred;
            setp.lt.u32(ok, gt, ntok);
            shr.u32(src, kc, 5);
            add.u32(src, src, bo);
            mad.lo.u32(src, gt, nb, src);
            let sa: u64;
            mad.wide.u32(sa, src, 4, xd);
            if ok {
                ld.global.f32(v, [sa]);
            }
            mad.lo.u32(dst, wi, 4, db);
            st.shared.f32([dst], v);
        }
        bar.sync(0);
        for blk in 0..8 {
            let (wa, da): (u64, u64);
            let (qi, di): (u32, u32);
            add.u32(qi, rowq, kc);
            add.u32(qi, qi, blk * 32);
            mad.wide.u32(wa, qi, 1, qs);
            let (w0, w1, w2, w3, w4, w5, w6, w7): (u32, u32, u32, u32, u32, u32, u32, u32);
            ld.global.nc.v4.u32((w0, w1, w2, w3), [wa]);
            ld.global.nc.v4.u32((w4, w5, w6, w7), [wa + 16]);
            shr.u32(di, kc, 5);
            add.u32(di, di, rowd);
            add.u32(di, di, blk);
            mad.wide.u32(da, di, 2, dw);
            let h: f16;
            let sw: f32;
            ld.global.nc.b16(h, [da]);
            cvt.f32.f16(sw, h);
            for i in 0..8 {
                let (tok, xa, xda): (u32, u32, u32);
                add.u32(tok, cc, i * 4);
                mad.lo.u32(xa, tok, 272, xb);
                let (x0, x1, x2, x3, x4, x5, x6, x7): (u32, u32, u32, u32, u32, u32, u32, u32);
                ld.shared.v4.u32((x0, x1, x2, x3), [xa + (blk * 32)]);
                ld.shared.v4.u32((x4, x5, x6, x7), [xa + (blk * 32 + 16)]);
                let isum: s32;
                mov.s32(isum, 0);
                dp4a.s32.s32(isum, w0, x0, isum);
                dp4a.s32.s32(isum, w1, x1, isum);
                dp4a.s32.s32(isum, w2, x2, isum);
                dp4a.s32.s32(isum, w3, x3, isum);
                dp4a.s32.s32(isum, w4, x4, isum);
                dp4a.s32.s32(isum, w5, x5, isum);
                dp4a.s32.s32(isum, w6, x6, isum);
                dp4a.s32.s32(isum, w7, x7, isum);
                let (sx, f): (f32, f32);
                mad.lo.u32(xda, tok, 32, db);
                ld.shared.f32(sx, [xda + (blk * 4)]);
                mul.rn.f32(sx, sx, sw);
                cvt.rn.f32.s32(f, isum);
                fma.rn.f32(acc[i], f, sx, acc[i]);
            }
        }
        bar.sync(0);
        add.u32(kc, kc, 256);
        bra(ktop);
        label(kend);
        let rin: pred;
        setp.lt.u32(rin, row, rows);
        for i in 0..8 {
            let (tok, gt, o): (u32, u32, u32);
            add.u32(tok, cc, i * 4);
            add.u32(gt, t0, tok);
            let tin: pred;
            setp.lt.u32(tin, gt, ntok);
            and.pred(tin, tin, rin);
            mad.lo.u32(o, gt, y_stride, row);
            let ya: u64;
            mad.wide.u32(ya, o, 4, y);
            if tin {
                st.global.f32([ya], acc[i]);
            }
        }
    }

    /// Short convolution of `ntok` consecutive tokens of one sequence, a thread per channel (see conv4). x, y [ntok][c];
    /// taps [ntok][c / head_dim][4] where DYN.
    #[maxntid(256)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn conv4_seq<const DYN: u32>(
        x: *const f32, w: *const f32, state: *mut f32, taps: *const f32, y: *mut f32, c_total: u32, head_dim: u32, scale: f32, ntok: u32, taps_stride: u32,
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
        let (wa, sa): (u64, u64);
        let (c4, c3, hd): (u32, u32, u32);
        shl.b32(c4, c, 2);
        mad.wide.u32(wa, c4, 4, w);
        mul.lo.u32(c3, c, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (h0, h1, h2): (f32, f32, f32);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        let (w0, w1, w2, w3): (f32, f32, f32, f32);
        ld.global.nc.v4.f32((w0, w1, w2, w3), [wa]);
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
        q: *const f32, r: *const f32, sink: *const f32, rb: *const f32, kc: *const u16, vc: *const u16, out: *mut f32, params: *const u32,
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
            let bo: u32;
            let ba: u64;
            shl.b32(bo, dist, 4);
            mad.wide.u32(ba, bo, 4, rb);
            for c in 0..4 {
                let (b0, b1, b2, b3): (f32, f32, f32, f32);
                ld.global.nc.v4.f32((b0, b1, b2, b3), [ba + (c * 16)]);
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

    /// The SITU activation of `gu` [pair][2 nff] for every pair (grid.y), quantized as q8_1: aq [pair][nff], ad, as [pair][nff/32].
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn situ_q8_n(gu: *const f32, aq: *mut u8, ad: *mut f32, asum: *mut f32, nff: u32, beta: f32, lb: f32) {
        let ke: u32;
        mov.u32(ke, ctaid.y);
        let (lane, b): (u32, u32);
        warp_row(lane, b);
        let nb: u32;
        shr.u32(nb, nff, 5);
        let past: pred;
        setp.ge.u32(past, b, nb);
        if past {
            ret();
        }
        let (i, gi, ui, base): (u32, u32, u32, u32);
        shl.b32(i, b, 5);
        add.u32(i, i, lane);
        shl.b32(base, nff, 1);
        mul.lo.u32(base, base, ke);
        add.u32(gi, base, i);
        add.u32(ui, gi, nff);
        let (ga, ua): (u64, u64);
        let (gv, uv, a, sg, tu, ib, il): (f32, f32, f32, f32, f32, f32, f32);
        mad.wide.u32(ga, gi, 4, gu);
        mad.wide.u32(ua, ui, 4, gu);
        ld.global.f32(gv, [ga]);
        ld.global.f32(uv, [ua]);
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
        let (gb, qb): (u32, u32);
        mad.lo.u32(gb, ke, nb, b);
        shl.b32(qb, gb, 5);
        let (qa, da, s2): (u64, u64, u64);
        mad.wide.u32(qa, qb, 1, aq);
        mad.wide.u32(da, gb, 4, ad);
        mad.wide.u32(s2, gb, 4, asum);
        q8_warp(a, lane, qa, da, s2);
    }

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

    /// Depth mix in one pass over the states: per state j < ns <= 8 its sum of squares and dot with the query,
    /// one block reduction of the 16 numbers, then the weighted sum (see mix). grid (nq, ntok), 256 threads.
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

    /// y = x / rms(x) * w, and y as q8_1 (q, d, s), for nvec vectors (grid.x) of n floats; x at stride xs.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn norm_q8(x: *const f32, w: *const f32, y: *mut f32, q: *mut u8, d: *mut f32, s: *mut f32, n: u32, eps: f32, xs: u32) {
        let red: shared<[f32; 32], 16>;
        let (v, xo, yo, t, lane, wp, nb): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(v, ctaid.x);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(wp, t, 5);
        mul.lo.u32(xo, v, xs);
        mul.lo.u32(yo, v, n);
        shr.u32(nb, n, 5);
        let (xv, yv): (u64, u64);
        mad.wide.u32(xv, xo, 4, x);
        mad.wide.u32(yv, yo, 4, y);
        let ss: f32;
        mov.f32(ss, 0.0);
        let i: u32;
        mov.u32(i, t);
        label(top);
        let done: pred;
        setp.ge.u32(done, i, n);
        if done {
            bra(end);
        }
        let a: u64;
        let f: f32;
        mad.wide.u32(a, i, 4, xv);
        ld.global.f32(f, [a]);
        fma.rn.f32(ss, f, f, ss);
        add.u32(i, i, 256);
        bra(top);
        label(end);
        let r: u32;
        mov.u32(r, red);
        block_sum(ss, r);
        let (nf, scale): (f32, f32);
        cvt.rn.f32.u32(nf, n);
        div.rn.f32(scale, ss, nf);
        add.rn.f32(scale, scale, eps);
        sqrt.rn.f32(scale, scale);
        rcp.rn.f32(scale, scale);
        let b: u32;
        mov.u32(b, wp);
        label(btop);
        let bdone: pred;
        setp.ge.u32(bdone, b, nb);
        if bdone {
            bra(bend);
        }
        let (j, gb, qb): (u32, u32, u32);
        shl.b32(j, b, 5);
        add.u32(j, j, lane);
        let (xa, wa, ya): (u64, u64, u64);
        let (xv2, wv): (f32, f32);
        mad.wide.u32(xa, j, 4, xv);
        mad.wide.u32(wa, j, 4, w);
        mad.wide.u32(ya, j, 4, yv);
        ld.global.f32(xv2, [xa]);
        ld.global.nc.f32(wv, [wa]);
        mul.rn.f32(xv2, xv2, scale);
        mul.rn.f32(xv2, xv2, wv);
        st.global.f32([ya], xv2);
        mad.lo.u32(gb, v, nb, b);
        shl.b32(qb, gb, 5);
        let (qa, da, sa): (u64, u64, u64);
        mad.wide.u32(qa, qb, 1, q);
        mad.wide.u32(da, gb, 4, d);
        mad.wide.u32(sa, gb, 4, s);
        q8_warp(xv2, lane, qa, da, sa);
        add.u32(b, b, 8);
        bra(btop);
        label(bend);
    }

    /// K and V of one token, a block per kv head (128 threads, a channel each): the short convolutions (with the taps of
    /// full attention where DYN), the K clip and the V norm, stored as f32 (k, v) and as f16 at the cache row of pos.
    #[reqntid(128)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn kv_prep<const DYN: u32>(
        gk: *const f32, gv: *const f32, kw: *const f32, vw: *const f32, state: *mut f32, k: *mut f32, v: *mut f32, kc: *mut u16, vc: *mut u16,
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

    /// The short convolution of channel c of x (taps of head h at x[1024 + 4 h] where DYN); state rows at `soff`.
    fn conv_ch<const DYN: u32, const SOFF: u32>(o: f32, x: u64, w: u64, state: u64, c: u32, h: u32, scale: f32) {
        let (xa, wa, sa): (u64, u64, u64);
        let (c4, c3): (u32, u32);
        mad.wide.u32(xa, c, 4, x);
        shl.b32(c4, c, 2);
        mad.wide.u32(wa, c4, 4, w);
        add.u32(c3, c, SOFF);
        mul.lo.u32(c3, c3, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (xv, h0, h1, h2): (f32, f32, f32, f32);
        ld.global.f32(xv, [xa]);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        let (w0, w1, w2, w3): (f32, f32, f32, f32);
        ld.global.nc.v4.f32((w0, w1, w2, w3), [wa]);
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
        qin: *const f32, qs: *const f32, qout: *mut f32, rin: *const f32, rs: *const f32, rout: *mut f32, n_head: u32, thr_q: f32, thr_r: f32, post: *const f32,
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

    /// a * sigmoid(g) over n floats (n a multiple of 32), as q8_1: a warp per block of 32.
    #[maxntid(256)]
    pub unsafe extern "ptx-kernel" fn gate_q8(a: *const f32, g: *const f32, q: *mut u8, d: *mut f32, s: *mut f32, n: u32) {
        let (lane, b): (u32, u32);
        warp_row(lane, b);
        let nb: u32;
        shr.u32(nb, n, 5);
        let past: pred;
        setp.ge.u32(past, b, nb);
        if past {
            ret();
        }
        let i: u32;
        shl.b32(i, b, 5);
        add.u32(i, i, lane);
        let (aa, ga): (u64, u64);
        let (av, gv, sg): (f32, f32, f32);
        mad.wide.u32(aa, i, 4, a);
        mad.wide.u32(ga, i, 4, g);
        ld.global.f32(av, [aa]);
        ld.global.f32(gv, [ga]);
        sigmoid(sg, gv);
        mul.rn.f32(av, av, sg);
        let qb: u32;
        shl.b32(qb, b, 5);
        let (qa, da, sa): (u64, u64, u64);
        mad.wide.u32(qa, qb, 1, q);
        mad.wide.u32(da, b, 4, d);
        mad.wide.u32(sa, b, 4, s);
        q8_warp(av, lane, qa, da, sa);
    }

    /// y = conv(x) (+ res where RES): the short convolution of 4 taps of one token, a thread per channel.
    #[maxntid(256)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn conv_res<const RES: u32>(x: *const f32, w: *const f32, state: *mut f32, res: *const f32, y: *mut f32, n: u32, scale: f32) {
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
    pub unsafe extern "ptx-kernel" fn moe_tail(gpu: *const f32, cpu: *const f32, slots: *const i32, wts: *const f32, w: *const f32, state: *mut f32, x: *mut f32, n: u32, scale: f32) {
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
        let (wa, sa, xa): (u64, u64, u64);
        let (c4, c3): (u32, u32);
        shl.b32(c4, c, 2);
        mad.wide.u32(wa, c4, 4, w);
        mul.lo.u32(c3, c, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (h0, h1, h2, w0, w1, w2, w3, o2, xv): (f32, f32, f32, f32, f32, f32, f32, f32, f32);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        ld.global.nc.v4.f32((w0, w1, w2, w3), [wa]);
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

    /// The controller input (rms(cur), rms(msg)) of one token, as q8_1 of 2n values (see ctrl_in).
    /// The body of ctrl_in_q.
    fn ctrl_in_body(cur: u64, msg: u64, q: u64, d: u64, s: u64, n: u32) {
        let red: shared<[f32; 32], 16>;
        let (t, lane, wp, nb): (u32, u32, u32, u32);
        mov.u32(t, tid.x);
        and.b32(lane, t, 31);
        shr.u32(wp, t, 5);
        shr.u32(nb, n, 5);
        let r: u32;
        mov.u32(r, red);
        let nf: f32;
        cvt.rn.f32.u32(nf, n);
        for half in 0..2 {
            let src: u64;
            if half == 0 {
                mov.u64(src, cur);
            } else {
                mov.u64(src, msg);
            }
            let ss: f32;
            mov.f32(ss, 0.0);
            let i: u32;
            mov.u32(i, t);
            label(atop);
            let adone: pred;
            setp.ge.u32(adone, i, n);
            if adone {
                bra(aend);
            }
            let a: u64;
            let f: f32;
            mad.wide.u32(a, i, 4, src);
            ld.global.f32(f, [a]);
            fma.rn.f32(ss, f, f, ss);
            add.u32(i, i, 256);
            bra(atop);
            label(aend);
            block_sum(ss, r);
            div.rn.f32(ss, ss, nf);
            add.rn.f32(ss, ss, 0.000_001);
            sqrt.rn.f32(ss, ss);
            rcp.rn.f32(ss, ss);
            let b: u32;
            mov.u32(b, wp);
            label(btop);
            let bdone: pred;
            setp.ge.u32(bdone, b, nb);
            if bdone {
                bra(bend);
            }
            let (j, gb, qb): (u32, u32, u32);
            shl.b32(j, b, 5);
            add.u32(j, j, lane);
            let xa: u64;
            let xv: f32;
            mad.wide.u32(xa, j, 4, src);
            ld.global.f32(xv, [xa]);
            mul.rn.f32(xv, xv, ss);
            add.u32(gb, b, half * 0);
            if half == 1 {
                add.u32(gb, b, nb);
            }
            shl.b32(qb, gb, 5);
            let (qa, da, sa): (u64, u64, u64);
            mad.wide.u32(qa, qb, 1, q);
            mad.wide.u32(da, gb, 4, d);
            mad.wide.u32(sa, gb, 4, s);
            q8_warp(xv, lane, qa, da, sa);
            add.u32(b, b, 8);
            bra(btop);
            label(bend);
        }
    }

    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn ctrl_in_q(cur: *const f32, msg: *const f32, q: *mut u8, d: *mut f32, s: *mut f32, n: u32) {
        ctrl_in_body(cur, msg, q, d, s, n);
    }

    /// ctrl_in_q of controller ctaid.z: cur and msg at tab[2 z], tab[2 z + 1]; q, d, s of controller z at z 2n, z 2n/32.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn ctrl_in_qz(tab: *const u64, q: *mut u8, d: *mut f32, s: *mut f32, n: u32) {
        let (z, n2, nb): (u32, u32, u32);
        mov.u32(z, ctaid.z);
        shl.b32(n2, n, 1);
        shr.u32(nb, n2, 5);
        let (ta, cz, mz, qz, dz, sz): (u64, u64, u64, u64, u64, u64);
        let (zq, zd): (u32, u32);
        mul.lo.u32(zq, z, 2);
        mad.wide.u32(ta, zq, 8, tab);
        ld.global.u64(cz, [ta]);
        ld.global.u64(mz, [ta + 8]);
        mul.lo.u32(zq, z, n2);
        mad.wide.u32(qz, zq, 1, q);
        mul.lo.u32(zd, z, nb);
        mad.wide.u32(dz, zd, 4, d);
        mad.wide.u32(sz, zd, 4, s);
        ctrl_in_body(cz, mz, qz, dz, sz, n);
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

    /// RMS norm (times w) of the vector at x, and its q8_1: 768 threads, element t + 768 i in v[i]; warp w quantizes blocks
    /// w + 24 i. Writes y [n] (if `y` is not 0), q, d, s.
    fn norm768(v0: f32, v1: f32, v2: f32, w: u64, y: u64, q: u64, d: u64, s: u64, t: u32, eps: f32, red: u32, wy: u32) {
        let (lane, wp): (u32, u32);
        and.b32(lane, t, 31);
        shr.u32(wp, t, 5);
        let ss: f32;
        mul.rn.f32(ss, v0, v0);
        fma.rn.f32(ss, v1, v1, ss);
        fma.rn.f32(ss, v2, v2, ss);
        block_sum(ss, red);
        let scale: f32;
        mul.rn.f32(scale, ss, 0.000_434_027_78);
        add.rn.f32(scale, scale, eps);
        sqrt.rn.f32(scale, scale);
        rcp.rn.f32(scale, scale);
        for i in 0..3 {
            let (j, b, qb): (u32, u32, u32);
            add.u32(j, t, i * 768);
            let (wa, ya, qa, da, sa): (u64, u64, u64, u64, u64);
            let (x, wv): (f32, f32);
            if i == 0 {
                mov.f32(x, v0);
            }
            if i == 1 {
                mov.f32(x, v1);
            }
            if i == 2 {
                mov.f32(x, v2);
            }
            mad.wide.u32(wa, j, 4, w);
            ld.global.nc.f32(wv, [wa]);
            mul.rn.f32(x, x, scale);
            mul.rn.f32(x, x, wv);
            let has_y: pred;
            setp.ne.u32(has_y, wy, 0);
            if has_y {
                mad.wide.u32(ya, j, 4, y);
                st.global.f32([ya], x);
            }
            add.u32(b, wp, i * 24);
            shl.b32(qb, b, 5);
            mad.wide.u32(qa, qb, 1, q);
            mad.wide.u32(da, b, 4, d);
            mad.wide.u32(sa, b, 4, s);
            q8_warp(x, lane, qa, da, sa);
        }
    }

    /// norm_q8 of vector ctaid.x (n = 2304) at x + ctaid.x xs: 768 threads, the values in registers.
    #[reqntid(768)]
    pub unsafe extern "ptx-kernel" fn norm_q8_768(x: *const f32, w: *const f32, y: *mut f32, q: *mut u8, d: *mut f32, s: *mut f32, eps: f32, xs: u32) {
        let red: shared<[f32; 32], 16>;
        let (vi, t, xo, r, wy): (u32, u32, u32, u32, u32);
        mov.u32(vi, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(r, red);
        mad.lo.u32(xo, vi, xs, t);
        let (xa, yv, qv, dv, sv): (u64, u64, u64, u64, u64);
        mad.wide.u32(xa, xo, 4, x);
        let (v0, v1, v2): (f32, f32, f32);
        ld.global.f32(v0, [xa]);
        ld.global.f32(v1, [xa + 3072]);
        ld.global.f32(v2, [xa + 6144]);
        let (yo, qo, bo): (u32, u32, u32);
        mul.lo.u32(yo, vi, 2304);
        mad.wide.u32(yv, yo, 4, y);
        mad.wide.u32(qv, yo, 1, q);
        mul.lo.u32(bo, vi, 72);
        mad.wide.u32(dv, bo, 4, d);
        mad.wide.u32(sv, bo, 4, s);
        mov.u32(wy, 1);
        norm768(v0, v1, v2, w, yv, qv, dv, sv, t, eps, r, wy);
    }

    /// The input of a controller, q8_1: [cur / rms(cur), msg / rms(msg)] (2 x 2304), at q, d, s; 768 threads.
    fn ctrl_in768(cur: u64, msg: u64, q: u64, d: u64, s: u64, t: u32, red: u32) {
        let (lane, wp): (u32, u32);
        and.b32(lane, t, 31);
        shr.u32(wp, t, 5);
        for half in 0..2 {
            let src: u64;
            if half == 0 {
                mov.u64(src, cur);
            }
            if half == 1 {
                mov.u64(src, msg);
            }
            let a: u64;
            let (v0, v1, v2, ss): (f32, f32, f32, f32);
            mad.wide.u32(a, t, 4, src);
            ld.global.f32(v0, [a]);
            ld.global.f32(v1, [a + 3072]);
            ld.global.f32(v2, [a + 6144]);
            mul.rn.f32(ss, v0, v0);
            fma.rn.f32(ss, v1, v1, ss);
            fma.rn.f32(ss, v2, v2, ss);
            block_sum(ss, red);
            mul.rn.f32(ss, ss, 0.000_434_027_78);
            add.rn.f32(ss, ss, 0.000_001);
            sqrt.rn.f32(ss, ss);
            rcp.rn.f32(ss, ss);
            for i in 0..3 {
                let x: f32;
                if i == 0 {
                    mul.rn.f32(x, v0, ss);
                }
                if i == 1 {
                    mul.rn.f32(x, v1, ss);
                }
                if i == 2 {
                    mul.rn.f32(x, v2, ss);
                }
                let (b, qb): (u32, u32);
                add.u32(b, wp, half * 72 + i * 24);
                shl.b32(qb, b, 5);
                let (qa, da, sa): (u64, u64, u64);
                mad.wide.u32(qa, qb, 1, q);
                mad.wide.u32(da, b, 4, d);
                mad.wide.u32(sa, b, 4, s);
                q8_warp(x, lane, qa, da, sa);
            }
        }
    }

    /// ctrl_in_qz with 768 threads (n = 2304): controller ctaid.z, cur and msg at tab[2 z], tab[2 z + 1].
    #[reqntid(768)]
    pub unsafe extern "ptx-kernel" fn ctrl_in_z768(tab: *const u64, q: *mut u8, d: *mut f32, s: *mut f32) {
        let red: shared<[f32; 32], 16>;
        let (z, t, r, zi): (u32, u32, u32, u32);
        mov.u32(z, ctaid.z);
        mov.u32(t, tid.x);
        mov.u32(r, red);
        let (ta, cz, mz, qz, dz, sz): (u64, u64, u64, u64, u64, u64);
        mul.lo.u32(zi, z, 2);
        mad.wide.u32(ta, zi, 8, tab);
        ld.global.u64(cz, [ta]);
        ld.global.u64(mz, [ta + 8]);
        mul.lo.u32(zi, z, 4608);
        mad.wide.u32(qz, zi, 1, q);
        mul.lo.u32(zi, z, 144);
        mad.wide.u32(dz, zi, 4, d);
        mad.wide.u32(sz, zi, 4, s);
        ctrl_in768(cz, mz, qz, dz, sz, t, r);
    }

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

/// The KV caches (int8 and int4) and the attention on them, a module of its own: kernite builds a module's PTX in const
/// evaluation, and one module with every kernel takes rustc past the steps it evaluates without a warning (and toward
/// kernite's limit of 1-2 MiB, see ~/kernite-repro). The helpers these kernels share with `gpu` are copies.
#[kernite_macros::kernels]
pub mod attn {
    // ---------------------------------------------------------------- helpers (copies of gpu's)

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

    /// Sum of the s32 `v` over the warp.
    fn warp_isum(v: s32) {
        for sh in [16, 8, 4, 2, 1] {
            let o: s32;
            shfl.sync.bfly.b32(o, v, sh, 31, 0xffff_ffff);
            add.s32(v, v, o);
        }
    }

    /// acc += the 4 signed bytes of `w` times x0..x3
    fn fma_s8x4(acc: f32, w: u32, x0: f32, x1: f32, x2: f32, x3: f32) {
        let b: s32;
        let f: f32;
        bfe.s32(b, w, 0, 8);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x0, acc);
        bfe.s32(b, w, 8, 8);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x1, acc);
        bfe.s32(b, w, 16, 8);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x2, acc);
        bfe.s32(b, w, 24, 8);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x3, acc);
    }

    /// The short convolution of channel c of x (taps of head h at x[1024 + 4 h] where DYN); state rows at `soff`.
    fn conv_ch<const DYN: u32, const SOFF: u32>(o: f32, x: u64, w: u64, state: u64, c: u32, h: u32, scale: f32) {
        let (xa, wa, sa): (u64, u64, u64);
        let (c4, c3): (u32, u32);
        mad.wide.u32(xa, c, 4, x);
        shl.b32(c4, c, 2);
        mad.wide.u32(wa, c4, 4, w);
        add.u32(c3, c, SOFF);
        mul.lo.u32(c3, c3, 3);
        mad.wide.u32(sa, c3, 4, state);
        let (xv, h0, h1, h2): (f32, f32, f32, f32);
        ld.global.f32(xv, [xa]);
        ld.global.f32(h0, [sa]);
        ld.global.f32(h1, [sa + 4]);
        ld.global.f32(h2, [sa + 8]);
        let (w0, w1, w2, w3): (f32, f32, f32, f32);
        ld.global.nc.v4.f32((w0, w1, w2, w3), [wa]);
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

    // ---------------------------------------------------------------- int8 KV cache
    //
    // A cache of R rows (positions) is [R][8 kv heads][128] int8 at kc, and the scales [R][8][4] f16 (a scale per 32
    // values, as q8_0) at ks: 1088 bytes a row, 53% of f16. Quantized as llama.cpp's q8_0: d = max|x| / 127.

    /// Quantizes the f32 `v` of each lane to int8 of the warp's 32 values: d = max|v| / 127 rounded to f16, q = round(v / d)
    /// within +-127; lanes 0, 4, .. store 4 bytes each at `qa` (the block's byte address), lane 0 stores d at `sa`.
    fn kv8_warp(v: f32, lane: u32, qa: u64, sa: u64) {
        let (a, amax, d, id): (f32, f32, f32, f32);
        abs.f32(a, v);
        mov.f32(amax, a);
        warp_max(amax);
        div.rn.f32(d, amax, 127.0);
        let dh: f16;
        cvt.rn.f16.f32(dh, d);
        cvt.f32.f16(d, dh);
        let nz: pred;
        setp.gt.f32(nz, d, 0.0);
        mov.f32(id, 0.0);
        if nz {
            rcp.rn.f32(id, d);
        }
        let qf: f32;
        mul.rn.f32(qf, v, id);
        let qi: s32;
        cvt.rni.s32.f32(qi, qf);
        min.s32(qi, qi, 127);
        max.s32(qi, qi, -127);
        let (b0, b1, b2, b3, w): (u32, u32, u32, u32, u32);
        and.b32(b0, qi, 255);
        shfl.sync.down.b32(b1, b0, 1, 31, 0xffff_ffff);
        shfl.sync.down.b32(b2, b0, 2, 31, 0xffff_ffff);
        shfl.sync.down.b32(b3, b0, 3, 31, 0xffff_ffff);
        shl.b32(b1, b1, 8);
        shl.b32(b2, b2, 16);
        shl.b32(b3, b3, 24);
        or.b32(w, b0, b1);
        or.b32(w, w, b2);
        or.b32(w, w, b3);
        let (l4, four): (u32, pred);
        and.b32(l4, lane, 3);
        setp.eq.u32(four, l4, 0);
        let wa: u64;
        cvt.u64.u32(wa, lane);
        add.u64(wa, wa, qa);
        if four {
            st.global.u32([wa], w);
        }
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            st.global.b16([sa], dh);
        }
    }

    /// The byte address of channel block (c - lane) of `row` in kc, and the address of its scale in ks.
    fn kv8_addr(qa: u64, sa: u64, row: u32, c: u32, lane: u32, kc: u64, ks: u64) {
        let (qo, so): (u32, u32);
        mad.lo.u32(qo, row, 1024, c);
        sub.u32(qo, qo, lane);
        cvt.u64.u32(qa, qo);
        add.u64(qa, qa, kc);
        shr.u32(so, c, 5);
        mad.lo.u32(so, row, 32, so);
        mad.wide.u32(sa, so, 2, ks);
    }

    /// kv_prep on the int8 cache: the convolutions, K clip, V norm, and the cache row of the token at params[0].
    #[reqntid(128)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn kv_prep8<const DYN: u32>(
        gk: *const f32, gv: *const f32, kw: *const f32, vw: *const f32, state: *mut f32, k: *mut f32, v: *mut f32, kc: *mut u8, ks: *mut u16, vc: *mut u8,
        vs: *mut u16, params: *const u32, ring: u32, cscale: f32, thr: f32, eps: f32,
    ) {
        let red: shared<[f32; 32], 16>;
        let (h, t, c, lane): (u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, h, 128, t);
        and.b32(lane, t, 31);
        let (ko, vo): (f32, f32);
        conv_ch::<DYN, 0>(ko, gk, kw, state, c, h, cscale);
        conv_ch::<DYN, 1024>(vo, gv, vw, state, c, h, cscale);
        let r: u32;
        mov.u32(r, red);
        let (ss, f, norm, vsq): (f32, f32, f32, f32);
        mul.rn.f32(ss, ko, ko);
        block_sum(ss, r);
        sqrt.rn.f32(norm, ss);
        add.rn.f32(norm, norm, 0.000_001);
        div.rn.f32(f, thr, norm);
        max.f32(f, f, 0.0);
        min.f32(f, f, 1.0);
        mul.rn.f32(ko, ko, f);
        mul.rn.f32(vsq, vo, vo);
        block_sum(vsq, r);
        div.rn.f32(vsq, vsq, 128.0);
        add.rn.f32(vsq, vsq, eps);
        sqrt.rn.f32(vsq, vsq);
        div.rn.f32(vo, vo, vsq);
        let (ka, va): (u64, u64);
        mad.wide.u32(ka, c, 4, k);
        mad.wide.u32(va, c, 4, v);
        st.global.f32([ka], ko);
        st.global.f32([va], vo);
        let (pos, row): (u32, u32);
        ld.global.u32(pos, [params]);
        mov.u32(row, pos);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(row, pos, ring);
        }
        let (kqa, ksa, vqa, vsa): (u64, u64, u64, u64);
        kv8_addr(kqa, ksa, row, c, lane, kc, ks);
        kv8_addr(vqa, vsa, row, c, lane, vc, vs);
        kv8_warp(ko, lane, kqa, ksa);
        kv8_warp(vo, lane, vqa, vsa);
    }

    /// The K and V rows of ubatch token ctaid.y (at position params[0] + ctaid.y) into the int8 cache; grid (8 kv heads, tokens).
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn kv_store8_t(
        k: *const f32, v: *const f32, kc: *mut u8, ks: *mut u16, vc: *mut u8, vs: *mut u16, params: *const u32, ring: u32, kstride: u32, vstride: u32,
    ) {
        let (h, tk, t, c, lane): (u32, u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(tk, ctaid.y);
        mov.u32(t, tid.x);
        mad.lo.u32(c, h, 128, t);
        and.b32(lane, t, 31);
        let (pos, row): (u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(pos, pos, tk);
        mov.u32(row, pos);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(row, pos, ring);
        }
        let (ki, vi): (u32, u32);
        let (ka, va): (u64, u64);
        mad.lo.u32(ki, tk, kstride, c);
        mad.lo.u32(vi, tk, vstride, c);
        mad.wide.u32(ka, ki, 4, k);
        mad.wide.u32(va, vi, 4, v);
        let (kv, vv): (f32, f32);
        ld.global.f32(kv, [ka]);
        ld.global.f32(vv, [va]);
        let (kqa, ksa, vqa, vsa): (u64, u64, u64, u64);
        kv8_addr(kqa, ksa, row, c, lane, kc, ks);
        kv8_addr(vqa, vsa, row, c, lane, vc, vs);
        kv8_warp(kv, lane, kqa, ksa);
        kv8_warp(vv, lane, vqa, vsa);
    }

    // ---------------------------------------------------------------- int4 KV cache (full attention)
    //
    // As the int8 cache with 4-bit values: [R][8 kv heads][64] bytes, value 2i in the low nibble of byte i and 2i + 1 in
    // the high one, symmetric (-7..7, d = max|x| / 7) and stored plus 8 (1..15), and the same f16 scale per 32 values:
    // 576 bytes a row, against 1088 int8 and 2048 f16. The full-attention caches are int4 (K and V) or int8; the
    // sliding-window rings stay int8.

    /// Quantizes the f32 `v` of each lane to 4 bits of the warp's 32 values (d = max|v| / 7 rounded to f16, q within
    /// +-7, stored as q + 8); lanes 0, 8, 16, 24 store 4 bytes (8 values) each at qa + lane / 2, lane 0 stores d at `sa`.
    fn kv4_warp(v: f32, lane: u32, qa: u64, sa: u64) {
        let (a, amax, d, id): (f32, f32, f32, f32);
        abs.f32(a, v);
        mov.f32(amax, a);
        warp_max(amax);
        div.rn.f32(d, amax, 7.0);
        let dh: f16;
        cvt.rn.f16.f32(dh, d);
        cvt.f32.f16(d, dh);
        let nz: pred;
        setp.gt.f32(nz, d, 0.0);
        mov.f32(id, 0.0);
        if nz {
            rcp.rn.f32(id, d);
        }
        let qf: f32;
        mul.rn.f32(qf, v, id);
        let qi: s32;
        cvt.rni.s32.f32(qi, qf);
        min.s32(qi, qi, 7);
        max.s32(qi, qi, -7);
        let (nib, w): (u32, u32);
        add.s32(qi, qi, 8);
        and.b32(nib, qi, 15);
        mov.u32(w, nib);
        for k in 1..8 {
            let o: u32;
            shfl.sync.down.b32(o, nib, k, 31, 0xffff_ffff);
            shl.b32(o, o, k * 4);
            or.b32(w, w, o);
        }
        let (l8, eight): (u32, pred);
        and.b32(l8, lane, 7);
        setp.eq.u32(eight, l8, 0);
        let (wa, half): (u64, u32);
        shr.u32(half, lane, 1);
        cvt.u64.u32(wa, half);
        add.u64(wa, wa, qa);
        if eight {
            st.global.u32([wa], w);
        }
        let first: pred;
        setp.eq.u32(first, lane, 0);
        if first {
            st.global.b16([sa], dh);
        }
    }

    /// The byte address of channel block (c - lane) of `row` in an int4 cache kc, and the address of its scale in ks.
    fn kv4_addr(qa: u64, sa: u64, row: u32, c: u32, lane: u32, kc: u64, ks: u64) {
        let (qo, so, cb): (u32, u32, u32);
        sub.u32(cb, c, lane);
        shr.u32(cb, cb, 1);
        mad.lo.u32(qo, row, 512, cb);
        cvt.u64.u32(qa, qo);
        add.u64(qa, qa, kc);
        shr.u32(so, c, 5);
        mad.lo.u32(so, row, 32, so);
        mad.wide.u32(sa, so, 2, ks);
    }

    /// kv_prep8 (with the input-dependent taps of full attention) for int4 caches.
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn kv_prep4(
        gk: *const f32, gv: *const f32, kw: *const f32, vw: *const f32, state: *mut f32, k: *mut f32, v: *mut f32, kc: *mut u8, ks: *mut u16, vc: *mut u8,
        vs: *mut u16, params: *const u32, cscale: f32, thr: f32, eps: f32,
    ) {
        let red: shared<[f32; 32], 16>;
        let (h, t, c, lane): (u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(t, tid.x);
        mad.lo.u32(c, h, 128, t);
        and.b32(lane, t, 31);
        let (ko, vo): (f32, f32);
        conv_ch::<1, 0>(ko, gk, kw, state, c, h, cscale);
        conv_ch::<1, 1024>(vo, gv, vw, state, c, h, cscale);
        let r: u32;
        mov.u32(r, red);
        let (ss, f, norm, vsq): (f32, f32, f32, f32);
        mul.rn.f32(ss, ko, ko);
        block_sum(ss, r);
        sqrt.rn.f32(norm, ss);
        add.rn.f32(norm, norm, 0.000_001);
        div.rn.f32(f, thr, norm);
        max.f32(f, f, 0.0);
        min.f32(f, f, 1.0);
        mul.rn.f32(ko, ko, f);
        mul.rn.f32(vsq, vo, vo);
        block_sum(vsq, r);
        div.rn.f32(vsq, vsq, 128.0);
        add.rn.f32(vsq, vsq, eps);
        sqrt.rn.f32(vsq, vsq);
        div.rn.f32(vo, vo, vsq);
        let (ka, va): (u64, u64);
        mad.wide.u32(ka, c, 4, k);
        mad.wide.u32(va, c, 4, v);
        st.global.f32([ka], ko);
        st.global.f32([va], vo);
        let row: u32;
        ld.global.u32(row, [params]);
        kv_quant4(ko, vo, row, c, lane, kc, ks, vc, vs);
    }

    /// The K and V values of channel c of `row` into int4 caches.
    fn kv_quant4(ko: f32, vo: f32, row: u32, c: u32, lane: u32, kc: u64, ks: u64, vc: u64, vs: u64) {
        let (kqa, ksa, vqa, vsa): (u64, u64, u64, u64);
        kv4_addr(kqa, ksa, row, c, lane, kc, ks);
        kv4_warp(ko, lane, kqa, ksa);
        kv4_addr(vqa, vsa, row, c, lane, vc, vs);
        kv4_warp(vo, lane, vqa, vsa);
    }

    /// kv_store8_t for int4 caches (full attention: no ring).
    #[reqntid(128)]
    pub unsafe extern "ptx-kernel" fn kv_store4_t(
        k: *const f32, v: *const f32, kc: *mut u8, ks: *mut u16, vc: *mut u8, vs: *mut u16, params: *const u32, kstride: u32, vstride: u32,
    ) {
        let (h, tk, t, c, lane): (u32, u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(tk, ctaid.y);
        mov.u32(t, tid.x);
        mad.lo.u32(c, h, 128, t);
        and.b32(lane, t, 31);
        let row: u32;
        ld.global.u32(row, [params]);
        add.u32(row, row, tk);
        let (ki, vi): (u32, u32);
        let (ka, va): (u64, u64);
        mad.lo.u32(ki, tk, kstride, c);
        mad.lo.u32(vi, tk, vstride, c);
        mad.wide.u32(ka, ki, 4, k);
        mad.wide.u32(va, vi, 4, v);
        let (kv, vv): (f32, f32);
        ld.global.f32(kv, [ka]);
        ld.global.f32(vv, [va]);
        kv_quant4(kv, vv, row, c, lane, kc, ks, vc, vs);
    }

    /// acc += the 8 signed nibbles of `w` (value j in bits 4j..) times x0..x7.
    fn fma_s4x8(acc: f32, w: u32, x0: f32, x1: f32, x2: f32, x3: f32, x4: f32, x5: f32, x6: f32, x7: f32) {
        let b: s32;
        let f: f32;
        bfe.s32(b, w, 0, 4);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x0, acc);
        bfe.s32(b, w, 4, 4);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x1, acc);
        bfe.s32(b, w, 8, 4);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x2, acc);
        bfe.s32(b, w, 12, 4);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x3, acc);
        bfe.s32(b, w, 16, 4);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x4, acc);
        bfe.s32(b, w, 20, 4);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x5, acc);
        bfe.s32(b, w, 24, 4);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x6, acc);
        bfe.s32(b, w, 28, 4);
        cvt.rn.f32.s32(f, b);
        fma.rn.f32(acc, f, x7, acc);
    }

    /// p0..p2 += the 8 nibbles of `w` times q of the 3 heads at shared `qa` (heads 512 bytes apart).
    fn dot3_s4x8(p0: f32, p1: f32, p2: f32, w: u32, qa: u32) {
        let (q0, q1, q2, q3, q4, q5, q6, q7): (f32, f32, f32, f32, f32, f32, f32, f32);
        ld.shared.v4.f32((q0, q1, q2, q3), [qa]);
        ld.shared.v4.f32((q4, q5, q6, q7), [qa + 16]);
        fma_s4x8(p0, w, q0, q1, q2, q3, q4, q5, q6, q7);
        ld.shared.v4.f32((q0, q1, q2, q3), [qa + 512]);
        ld.shared.v4.f32((q4, q5, q6, q7), [qa + 528]);
        fma_s4x8(p1, w, q0, q1, q2, q3, q4, q5, q6, q7);
        ld.shared.v4.f32((q0, q1, q2, q3), [qa + 1024]);
        ld.shared.v4.f32((q4, q5, q6, q7), [qa + 1040]);
        fma_s4x8(p2, w, q0, q1, q2, q3, q4, q5, q6, q7);
    }

    /// p0..p2 += the 4 signed bytes of `w` times q of the 3 heads at shared `qa` (heads 512 bytes apart).
    fn dot3_s8x4(p0: f32, p1: f32, p2: f32, w: u32, qa: u32) {
        let (q0, q1, q2, q3): (f32, f32, f32, f32);
        ld.shared.v4.f32((q0, q1, q2, q3), [qa]);
        fma_s8x4(p0, w, q0, q1, q2, q3);
        ld.shared.v4.f32((q0, q1, q2, q3), [qa + 512]);
        fma_s8x4(p1, w, q0, q1, q2, q3);
        ld.shared.v4.f32((q0, q1, q2, q3), [qa + 1024]);
        fma_s8x4(p2, w, q0, q1, q2, q3);
    }

    /// The query of the block's 3 heads (shared f32 qsh [3][128], scaled) as int8 for the dp4a scores: q8 [3][128] bytes
    /// in the order the K format reads them (K4: in each 8, the even dimensions, then the odd ones), d [3][4] per 32,
    /// and (K4) 8 times the sum of each 32 int8 at qs [3][4], what the +8 of the int4 keys adds. 128 threads, then a
    /// barrier.
    fn q_quant<const K4: u32>(t: u32, qb: u32, q8b: u32, qdb: u32, qsb: u32) {
        let (lane, w): (u32, u32);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        let (l8, eight, first): (u32, pred, pred);
        and.b32(l8, lane, 7);
        setp.eq.u32(eight, l8, 0);
        setp.eq.u32(first, lane, 0);
        for k in 0..3 {
            let (qa, v, a, amax, d, id): (u32, f32, f32, f32, f32, f32);
            mad.lo.u32(qa, t, 4, qb);
            ld.shared.f32(v, [qa + (k * 512)]);
            abs.f32(a, v);
            mov.f32(amax, a);
            warp_max(amax);
            div.rn.f32(d, amax, 127.0);
            let nz: pred;
            setp.gt.f32(nz, d, 0.0);
            mov.f32(id, 0.0);
            if nz {
                rcp.rn.f32(id, d);
            }
            let (qf, qi, sum): (f32, s32, s32);
            mul.rn.f32(qf, v, id);
            cvt.rni.s32.f32(qi, qf);
            min.s32(qi, qi, 127);
            max.s32(qi, qi, -127);
            mov.s32(sum, qi);
            warp_isum(sum);
            let b: u32;
            and.b32(b, qi, 255);
            let o: [u32; 8];
            mov.u32(o[0], b);
            for j in 1..8 {
                shfl.sync.down.b32(o[j], b, j, 31, 0xffff_ffff);
            }
            let (w0, w1, x): (u32, u32, u32);
            if K4 == 1 {
                mov.u32(w0, o[0]);
                mov.u32(w1, o[1]);
                for j in 1..4 {
                    shl.b32(x, o[2 * j], j * 8);
                    or.b32(w0, w0, x);
                    shl.b32(x, o[2 * j + 1], j * 8);
                    or.b32(w1, w1, x);
                }
            } else {
                mov.u32(w0, o[0]);
                mov.u32(w1, o[4]);
                for j in 1..4 {
                    shl.b32(x, o[j], j * 8);
                    or.b32(w0, w0, x);
                    shl.b32(x, o[j + 4], j * 8);
                    or.b32(w1, w1, x);
                }
            }
            if eight {
                let sa: u32;
                add.u32(sa, q8b, t);
                st.shared.v2.u32([sa + (k * 128)], (w0, w1));
            }
            if first {
                let (da, ssa): (u32, u32);
                mad.lo.u32(da, w, 4, qdb);
                st.shared.f32([da + (k * 16)], d);
                mul.lo.s32(sum, sum, 8);
                mad.lo.u32(ssa, w, 4, qsb);
                st.shared.s32([ssa + (k * 16)], sum);
            }
        }
        bar.sync(0);
    }

    /// score_dist on the int8 cache (K4 = 1: the int4 one), the dots by dp4a on the int8 query of q_quant; the key's row
    /// goes to shared ri[jj] for the V pass.
    fn score_dist8<const K4: u32>(
        start: u32, jj: u32, kvh: u32, pos: u32, ring: u32, extent: u32, kc: u64, ks: u64, rb: u64, q8b: u32, qdb: u32, qsb: u32, rbs: u32, kb: u32,
        sb: u32, rib: u32,
    ) {
        let (dist, kp, row, ko, ria): (u32, u32, u32, u32, u32);
        add.u32(dist, start, jj);
        sub.u32(kp, pos, dist);
        mov.u32(row, kp);
        let is_ring: pred;
        setp.ne.u32(is_ring, ring, 0);
        if is_ring {
            rem.u32(row, kp, ring);
        }
        mad.lo.u32(ria, jj, 4, rib);
        st.shared.u32([ria], row);
        mad.lo.u32(ko, row, 8, kvh);
        let (ka, sa): (u64, u64);
        let kq: u32;
        if K4 == 1 {
            shl.b32(kq, ko, 6);
        } else {
            shl.b32(kq, ko, 7);
        }
        cvt.u64.u32(ka, kq);
        add.u64(ka, ka, kc);
        mad.wide.u32(sa, ko, 8, ks);
        let (h0, h1, h2, h3): (f16, f16, f16, f16);
        ld.global.nc.v4.b16((h0, h1, h2, h3), [sa]);
        let (s0, s1, s2): (f32, f32, f32);
        mov.f32(s0, 0.0);
        mov.f32(s1, 0.0);
        mov.f32(s2, 0.0);
        for blk in 0..4 {
            let d: f32;
            let acc: [s32; 3];
            for h in 0..3 {
                mov.s32(acc[h], 0);
            }
            if K4 == 1 {
                let wk: [u32; 4];
                ld.global.nc.v4.u32((wk[0], wk[1], wk[2], wk[3]), [ka + (blk * 16)]);
                for i in 0..4 {
                    let (lo, hi): (u32, u32);
                    and.b32(lo, wk[i], 0x0f0f_0f0f);
                    shr.u32(hi, wk[i], 4);
                    and.b32(hi, hi, 0x0f0f_0f0f);
                    for h in 0..3 {
                        let (qe, qo): (u32, u32);
                        ld.shared.v2.u32((qe, qo), [q8b + (h * 128 + blk * 32 + i * 8)]);
                        dp4a.s32.s32(acc[h], lo, qe, acc[h]);
                        dp4a.s32.s32(acc[h], hi, qo, acc[h]);
                    }
                }
                for h in 0..3 {
                    let c8: s32;
                    ld.shared.s32(c8, [qsb + (h * 16 + blk * 4)]);
                    sub.s32(acc[h], acc[h], c8);
                }
            } else {
                for c in 0..2 {
                    let wk: [u32; 4];
                    ld.global.nc.v4.u32((wk[0], wk[1], wk[2], wk[3]), [ka + (blk * 32 + c * 16)]);
                    for h in 0..3 {
                        let xq: [u32; 4];
                        ld.shared.v4.u32((xq[0], xq[1], xq[2], xq[3]), [q8b + (h * 128 + blk * 32 + c * 16)]);
                        for i in 0..4 {
                            dp4a.s32.s32(acc[h], wk[i], xq[i], acc[h]);
                        }
                    }
                }
            }
            if blk == 0 {
                cvt.f32.f16(d, h0);
            }
            if blk == 1 {
                cvt.f32.f16(d, h1);
            }
            if blk == 2 {
                cvt.f32.f16(d, h2);
            }
            if blk == 3 {
                cvt.f32.f16(d, h3);
            }
            for h in 0..3 {
                let (f, dq): (f32, f32);
                cvt.rn.f32.s32(f, acc[h]);
                ld.shared.f32(dq, [qdb + (h * 16 + blk * 4)]);
                mul.rn.f32(dq, dq, d);
                if h == 0 {
                    fma.rn.f32(s0, f, dq, s0);
                }
                if h == 1 {
                    fma.rn.f32(s1, f, dq, s1);
                }
                if h == 2 {
                    fma.rn.f32(s2, f, dq, s2);
                }
            }
        }
        let inb: pred;
        setp.lt.u32(inb, dist, extent);
        if inb {
            let bo: u32;
            let ba: u64;
            shl.b32(bo, dist, 4);
            mad.wide.u32(ba, bo, 4, rb);
            for c in 0..4 {
                let (b0, b1, b2, b3): (f32, f32, f32, f32);
                ld.global.nc.v4.f32((b0, b1, b2, b3), [ba + (c * 16)]);
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
        let sa2: u32;
        mad.lo.u32(sa2, jj, 4, sb);
        st.shared.f32([sa2], s0);
        st.shared.f32([sa2 + 1024], s1);
        st.shared.f32([sa2 + 2048], s2);
    }

    /// The V pass of a chunk of `cnt` keys (V4: int4 values): lane group g = lane / 8 of warp w takes keys 4 w + g, + 16,
    /// .. (2 at a time), lane l8 = lane % 8 its 16 dimensions (one 16-byte load, 8 for int4), converted without cvt
    /// (the byte, plus 128 for int8 or the stored +8 for int4, into the low byte of 2^23); the 4 groups are summed by
    /// shuffles and each warp's sums go to shared red[w][h][128], then the block waits.
    fn v_chunk8<const V4: u32>(t: u32, cnt: u32, kvh: u32, vc: u64, vs: u64, sb: u32, rib: u32, redb: u32) {
        let (lane, w, g, l8, base): (u32, u32, u32, u32, u32);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        shr.u32(g, lane, 3);
        and.b32(l8, lane, 7);
        let acc: [f32; 48];
        for i in 0..48 {
            mov.f32(acc[i], 0.0);
        }
        let off: f32;
        if V4 == 1 {
            mov.f32(off, 8388616.0);
        } else {
            mov.f32(off, 8388736.0);
        }
        shl.b32(base, w, 2);
        label(vtop);
        let vdone: pred;
        setp.ge.u32(vdone, base, cnt);
        if vdone {
            bra(vend);
        }
        for u in 0..2 {
            let (jj, ok): (u32, pred);
            add.u32(jj, base, g);
            add.u32(jj, jj, u * 16);
            setp.lt.u32(ok, jj, cnt);
            let wv: [u32; 4];
            let (p0, p1, p2, d): (f32, f32, f32, f32);
            for k in 0..4 {
                mov.u32(wv[k], 0);
            }
            mov.f32(p0, 0.0);
            mov.f32(p1, 0.0);
            mov.f32(p2, 0.0);
            mov.f32(d, 0.0);
            if ok {
                let (pa, row, vo, so, b2, bo): (u32, u32, u32, u32, u32, u32);
                let (va, sa): (u64, u64);
                let dh: f16;
                mad.lo.u32(pa, jj, 4, rib);
                ld.shared.u32(row, [pa]);
                mad.lo.u32(vo, row, 8, kvh);
                shr.u32(b2, l8, 1);
                mad.lo.u32(so, vo, 4, b2);
                mad.wide.u32(sa, so, 2, vs);
                ld.global.nc.b16(dh, [sa]);
                cvt.f32.f16(d, dh);
                if V4 == 1 {
                    mad.lo.u32(bo, vo, 8, l8);
                    mad.wide.u32(va, bo, 8, vc);
                    ld.global.nc.v2.u32((wv[0], wv[1]), [va]);
                } else {
                    mad.lo.u32(bo, vo, 8, l8);
                    mad.wide.u32(va, bo, 16, vc);
                    ld.global.nc.v4.u32((wv[0], wv[1], wv[2], wv[3]), [va]);
                }
                mad.lo.u32(pa, jj, 4, sb);
                ld.shared.f32(p0, [pa]);
                ld.shared.f32(p1, [pa + 1024]);
                ld.shared.f32(p2, [pa + 2048]);
            }
            mul.rn.f32(p0, p0, d);
            mul.rn.f32(p1, p1, d);
            mul.rn.f32(p2, p2, d);
            let f: [f32; 16];
            if V4 == 1 {
                for k in 0..2 {
                    let (lo, hi, x): (u32, u32, u32);
                    and.b32(lo, wv[k], 0x0f0f_0f0f);
                    shr.u32(hi, wv[k], 4);
                    and.b32(hi, hi, 0x0f0f_0f0f);
                    for b in 0..4 {
                        prmt.b32(x, lo, 0x4B00_0000, 0x7540 + b);
                        mov.b32(f[8 * k + 2 * b], x);
                        prmt.b32(x, hi, 0x4B00_0000, 0x7540 + b);
                        mov.b32(f[8 * k + 2 * b + 1], x);
                    }
                }
            } else {
                for k in 0..4 {
                    let (tw, x): (u32, u32);
                    xor.b32(tw, wv[k], 0x8080_8080);
                    for b in 0..4 {
                        prmt.b32(x, tw, 0x4B00_0000, 0x7540 + b);
                        mov.b32(f[4 * k + b], x);
                    }
                }
            }
            for i in 0..16 {
                sub.rn.f32(f[i], f[i], off);
                fma.rn.f32(acc[i], p0, f[i], acc[i]);
                fma.rn.f32(acc[16 + i], p1, f[i], acc[16 + i]);
                fma.rn.f32(acc[32 + i], p2, f[i], acc[32 + i]);
            }
        }
        add.u32(base, base, 32);
        bra(vtop);
        label(vend);
        // the 4 lane groups hold the same dimensions: sum them
        for i in 0..48 {
            let o: f32;
            shfl.sync.bfly.b32(o, acc[i], 8, 31, 0xffff_ffff);
            add.rn.f32(acc[i], acc[i], o);
            shfl.sync.bfly.b32(o, acc[i], 16, 31, 0xffff_ffff);
            add.rn.f32(acc[i], acc[i], o);
        }
        let g0: pred;
        setp.eq.u32(g0, g, 0);
        if g0 {
            let ra: u32;
            mad.lo.u32(ra, w, 1536, redb);
            mad.lo.u32(ra, l8, 64, ra);
            for h in 0..3 {
                for m in 0..4 {
                    st.shared.v4.f32([ra + (h * 512 + m * 16)], (acc[h * 16 + 4 * m], acc[h * 16 + 4 * m + 1], acc[h * 16 + 4 * m + 2], acc[h * 16 + 4 * m + 3]));
                }
            }
        }
        bar.sync(0);
    }

    /// Sum over the 4 warps of red[w][h][t], for head h (byte offset of the head `ho`).
    fn red4(acc: f32, redb: u32, t: u32, ho: u32) {
        let (a, x): (u32, f32);
        mad.lo.u32(a, t, 4, redb);
        add.u32(a, a, ho);
        ld.shared.f32(acc, [a]);
        for w in 1..4 {
            ld.shared.f32(x, [a + (w * 1536)]);
            add.rn.f32(acc, acc, x);
        }
    }

    /// attn_part on the int8 cache (K4: the int4 one), chunks of 2^cs keys (at most 256): grid = (8 kv heads, chunks); a chunk past the
    /// keys of the token does nothing. Partials as attn_part: pm/pl [kvh][maxs][3], pacc [kvh][maxs][3][128].
    #[reqntid(128)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn attn_part8<const K4: u32>(
        q: *const f32, r: *const f32, sink: *const f32, rb: *const f32, kc: *const u8, ks: *const u16, vc: *const u8, vs: *const u16, pm: *mut f32,
        pl: *mut f32, pacc: *mut f32, params: *const u32, ring: u32, extent: u32, scale: f32, maxs: u32, window: u32, cs: u32,
    ) {
        let qsh: shared<[f32; 384], 16>;
        let q8sh: shared<[u32; 96], 16>;
        let qdsh: shared<[f32; 12], 16>;
        let qssh: shared<[u32; 12], 16>;
        let rsh: shared<[f32; 48], 16>;
        let ksh: shared<[f32; 4], 16>;
        let sc: shared<[f32; 768], 16>;
        let ri: shared<[u32; 256], 16>;
        let red: shared<[f32; 1536], 16>;
        let (kvh, sp, t): (u32, u32, u32);
        mov.u32(kvh, ctaid.x);
        mov.u32(sp, ctaid.y);
        mov.u32(t, tid.x);
        let (pos, nk, start, cnt, csz): (u32, u32, u32, u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(nk, pos, 1);
        min.u32(nk, nk, window);
        shl.b32(start, sp, cs);
        let none: pred;
        setp.ge.u32(none, start, nk);
        if none {
            ret();
        }
        shl.b32(csz, 1, cs);
        sub.u32(cnt, nk, start);
        min.u32(cnt, cnt, csz);
        let (qb, rbs, kb, sb, rib, redb): (u32, u32, u32, u32, u32, u32);
        mov.u32(qb, qsh);
        mov.u32(rbs, rsh);
        mov.u32(kb, ksh);
        mov.u32(sb, sc);
        mov.u32(rib, ri);
        mov.u32(redb, red);
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
            let (ri2, ra2): (u32, u32);
            let ra: u64;
            let v: f32;
            mad.lo.u32(ri2, h0, 16, t);
            mad.wide.u32(ra, ri2, 4, r);
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
        let (q8b, qdb, qsb): (u32, u32, u32);
        mov.u32(q8b, q8sh);
        mov.u32(qdb, qdsh);
        mov.u32(qsb, qssh);
        q_quant::<K4>(t, qb, q8b, qdb, qsb);

        let jj: u32;
        mov.u32(jj, t);
        label(stop);
        let sdone: pred;
        setp.ge.u32(sdone, jj, cnt);
        if sdone {
            bra(send);
        }
        score_dist8::<K4>(start, jj, kvh, pos, ring, extent, kc, ks, rb, q8b, qdb, qsb, rbs, kb, sb, rib);
        add.u32(jj, jj, 128);
        bra(stop);
        label(send);
        bar.sync(0);

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

        v_chunk8::<K4>(t, cnt, kvh, vc, vs, sb, rib, redb);
        let (po, pao): (u32, u64);
        mad.lo.u32(po, kvh, maxs, sp);
        mul.lo.u32(po, po, 384);
        add.u32(po, po, t);
        mad.wide.u32(pao, po, 4, pacc);
        let acc: f32;
        let ho: u32;
        for hh in 0..3 {
            mov.u32(ho, hh * 512);
            red4(acc, redb, t, ho);
            st.global.f32([pao + (hh * 512)], acc);
        }
    }

    /// Combines the chunks of attn_part8 (chunks of 2^cs keys) and the zero sink logit: out[h][d]; a block of 8 warps
    /// per query head; warp w takes chunks w, w + 8, .., lane its 4 dimensions.
    #[reqntid(256)]
    pub unsafe extern "ptx-kernel" fn attn_reduce8(pm: *const f32, pl: *const f32, pacc: *const f32, out: *mut f32, params: *const u32, window: u32, maxs: u32, cs: u32) {
        let red: shared<[f32; 1024], 16>;
        let lred: shared<[f32; 8], 16>;
        let mx: shared<[f32; 8], 16>;
        let (h, t, kvh, hh, lane, w): (u32, u32, u32, u32, u32, u32);
        mov.u32(h, ctaid.x);
        mov.u32(t, tid.x);
        div.u32(kvh, h, 3);
        rem.u32(hh, h, 3);
        and.b32(lane, t, 31);
        shr.u32(w, t, 5);
        let (pos, nk, ns, csz): (u32, u32, u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(nk, pos, 1);
        min.u32(nk, nk, window);
        shl.b32(csz, 1, cs);
        add.u32(ns, nk, csz);
        sub.u32(ns, ns, 1);
        shr.u32(ns, ns, cs);
        let base: u32;
        mul.lo.u32(base, kvh, maxs);
        // the max over the chunks and the sink's 0
        let (mm, sp): (f32, u32);
        mov.f32(mm, 0.0);
        mov.u32(sp, t);
        label(mt);
        let md: pred;
        setp.ge.u32(md, sp, ns);
        if md {
            bra(me);
        }
        let o: u32;
        let ma: u64;
        let mv: f32;
        add.u32(o, base, sp);
        mad.lo.u32(o, o, 3, hh);
        mad.wide.u32(ma, o, 4, pm);
        ld.global.f32(mv, [ma]);
        max.f32(mm, mm, mv);
        add.u32(sp, sp, 256);
        bra(mt);
        label(me);
        warp_max(mm);
        let (mxb, first): (u32, pred);
        mov.u32(mxb, mx);
        setp.eq.u32(first, lane, 0);
        if first {
            let wa: u32;
            mad.lo.u32(wa, w, 4, mxb);
            st.shared.f32([wa], mm);
        }
        bar.sync(0);
        let x: f32;
        ld.shared.f32(mm, [mxb]);
        for k in 1..8 {
            ld.shared.f32(x, [mxb + (k * 4)]);
            max.f32(mm, mm, x);
        }
        // the weighted sums
        let (a0, a1, a2, a3, ll): (f32, f32, f32, f32, f32);
        mov.f32(a0, 0.0);
        mov.f32(a1, 0.0);
        mov.f32(a2, 0.0);
        mov.f32(a3, 0.0);
        mov.f32(ll, 0.0);
        mov.u32(sp, w);
        label(at);
        let ad: pred;
        setp.ge.u32(ad, sp, ns);
        if ad {
            bra(ae);
        }
        let (o2, o3): (u32, u32);
        let (ma2, la2, aa2): (u64, u64, u64);
        let (mv2, lv2, e2, x0, x1, x2, x3): (f32, f32, f32, f32, f32, f32, f32);
        add.u32(o2, base, sp);
        mad.lo.u32(o2, o2, 3, hh);
        mad.lo.u32(o3, o2, 32, lane);
        mad.wide.u32(ma2, o2, 4, pm);
        mad.wide.u32(la2, o2, 4, pl);
        mad.wide.u32(aa2, o3, 16, pacc);
        ld.global.f32(mv2, [ma2]);
        ld.global.f32(lv2, [la2]);
        ld.global.v4.f32((x0, x1, x2, x3), [aa2]);
        sub.rn.f32(e2, mv2, mm);
        exp(e2, e2);
        fma.rn.f32(ll, lv2, e2, ll);
        fma.rn.f32(a0, x0, e2, a0);
        fma.rn.f32(a1, x1, e2, a1);
        fma.rn.f32(a2, x2, e2, a2);
        fma.rn.f32(a3, x3, e2, a3);
        add.u32(sp, sp, 8);
        bra(at);
        label(ae);
        let (redb, lb, ra): (u32, u32, u32);
        mov.u32(redb, red);
        mov.u32(lb, lred);
        mad.lo.u32(ra, w, 512, redb);
        mad.lo.u32(ra, lane, 16, ra);
        st.shared.v4.f32([ra], (a0, a1, a2, a3));
        if first {
            let la: u32;
            mad.lo.u32(la, w, 4, lb);
            st.shared.f32([la], ll);
        }
        bar.sync(0);
        let lt128: pred;
        setp.lt.u32(lt128, t, 128);
        if lt128 {
            let (acc, l, y): (f32, f32, f32);
            let (a, oo): (u32, u32);
            let oa: u64;
            mad.lo.u32(a, t, 4, redb);
            ld.shared.f32(acc, [a]);
            ld.shared.f32(l, [lb]);
            for k in 1..8 {
                ld.shared.f32(y, [a + (k * 512)]);
                add.rn.f32(acc, acc, y);
                ld.shared.f32(y, [lb + (k * 4)]);
                add.rn.f32(l, l, y);
            }
            sub.rn.f32(y, 0.0, mm);
            exp(y, y);
            add.rn.f32(l, l, y);
            div.rn.f32(acc, acc, l);
            mad.lo.u32(oo, h, 128, t);
            mad.wide.u32(oa, oo, 4, out);
            st.global.f32([oa], acc);
        }
    }

    /// attn_prefill on the int8 cache (K4: the int4 one): a block per (kv head, query token), the keys in chunks of 256 by distance, online
    /// softmax with the sink as the initial state (m = 0, l = 1); the V pass of a chunk split over the 4 warps.
    #[reqntid(128)]
    #[instances(0, 1)]
    pub unsafe extern "ptx-kernel" fn attn_prefill8<const K4: u32>(
        q: *const f32, r: *const f32, sink: *const f32, rb: *const f32, kc: *const u8, ks: *const u16, vc: *const u8, vs: *const u16, out: *mut f32,
        params: *const u32, ring: u32, window: u32, extent: u32, scale: f32, q_stride: u32, r_stride: u32, s_stride: u32,
    ) {
        let qsh: shared<[f32; 384], 16>;
        let q8sh: shared<[u32; 96], 16>;
        let qdsh: shared<[f32; 12], 16>;
        let qssh: shared<[u32; 12], 16>;
        let rsh: shared<[f32; 48], 16>;
        let ksh: shared<[f32; 4], 16>;
        let sc: shared<[f32; 768], 16>;
        let ml: shared<[f32; 12], 16>; // m[3], l[3], alpha[3]
        let ri: shared<[u32; 256], 16>;
        let red: shared<[f32; 1536], 16>;
        let (kvh, tk, t): (u32, u32, u32);
        mov.u32(kvh, ctaid.x);
        mov.u32(tk, ctaid.y);
        mov.u32(t, tid.x);
        let (pos, nk): (u32, u32);
        ld.global.u32(pos, [params]);
        add.u32(pos, pos, tk);
        add.u32(nk, pos, 1);
        min.u32(nk, nk, window);
        let (qb, rbs, kb, sb, mb, rib, redb): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(qb, qsh);
        mov.u32(rbs, rsh);
        mov.u32(kb, ksh);
        mov.u32(sb, sc);
        mov.u32(mb, ml);
        mov.u32(rib, ri);
        mov.u32(redb, red);
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
            let (ri2, ra2): (u32, u32);
            let ra: u64;
            let v: f32;
            mad.lo.u32(ri2, tk, r_stride, t);
            mad.lo.u32(ri2, h0, 16, ri2);
            mad.wide.u32(ra, ri2, 4, r);
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
        let (q8b, qdb, qsb): (u32, u32, u32);
        mov.u32(q8b, q8sh);
        mov.u32(qdb, qdsh);
        mov.u32(qsb, qssh);
        q_quant::<K4>(t, qb, q8b, qdb, qsb);
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
        let jj: u32;
        mov.u32(jj, t);
        label(stop);
        let sdone: pred;
        setp.ge.u32(sdone, jj, cnt);
        if sdone {
            bra(send);
        }
        score_dist8::<K4>(start, jj, kvh, pos, ring, extent, kc, ks, rb, q8b, qdb, qsb, rbs, kb, sb, rib);
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
        v_chunk8::<K4>(t, cnt, kvh, vc, vs, sb, rib, redb);
        // acc = acc alpha + the chunk's sum
        let (al0, al1, al2, c0, c1, c2): (f32, f32, f32, f32, f32, f32);
        let ho: u32;
        ld.shared.f32(al0, [mb + 24]);
        ld.shared.f32(al1, [mb + 28]);
        ld.shared.f32(al2, [mb + 32]);
        mov.u32(ho, 0);
        red4(c0, redb, t, ho);
        mov.u32(ho, 512);
        red4(c1, redb, t, ho);
        mov.u32(ho, 1024);
        red4(c2, redb, t, ho);
        fma.rn.f32(a0, a0, al0, c0);
        fma.rn.f32(a1, a1, al1, c1);
        fma.rn.f32(a2, a2, al2, c2);
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
}
