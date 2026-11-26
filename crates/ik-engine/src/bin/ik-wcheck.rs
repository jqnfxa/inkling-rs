//! The GPU kernels that read the checkpoint's weights, on real tensors, against references on the CPU (f64):
//! `ik-wcheck <checkpoint dir>`. Prints the largest error of each relative to the largest output, and fails past a
//! bound. The references are scalar Rust in f64, with no SIMD: they take the tensors' bytes from `Checkpoint` and
//! decode the stored formats themselves, and take the same f32 inputs as the kernels.
use ik_engine::checkpoint::{Checkpoint, Dtype};
use ik_engine::cpu::{dot_row, order_x};
use ik_engine::cuda::{Ctx, DevBuf, HostBuf, Stream};
use ik_engine::experts::{Fmt, Mat as XMat};
use ik_engine::kernels::Kernels;

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn bf16(b: &[u8], i: usize) -> f64 {
    f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16) as f64
}

/// 32 random bits
fn bits(seed: &mut u64) -> u32 {
    rnd(seed);
    (*seed >> 32) as u32
}

/// The f16 of bits h.
fn f16(h: u16) -> f64 {
    let (s, e, m) = (if h & 0x8000 != 0 { -1.0 } else { 1.0 }, ((h >> 10) & 31) as i32, (h & 1023) as f64);
    assert!(e != 31, "f16 infinity or NaN");
    s * if e == 0 { m * 2f64.powi(-24) } else { (1.0 + m / 1024.0) * 2f64.powi(e - 15) }
}

/// The f16 spacing at x: 2^-10 of its power of two, 2^-24 below 2^-14.
fn f16_ulp(x: f64) -> f64 {
    2f64.powi(((x.abs().to_bits() >> 52) as i32 - 1023).max(-14) - 10)
}

/// x rounded to the nearest f16, ties to the even one (as cvt.rn.f16.f32).
fn f16_round(x: f32) -> f64 {
    let (x, u) = (x as f64, f16_ulp(x as f64));
    let r = (x / u).round_ties_even() * u;
    assert!(r.abs() <= 65504.0, "f16 overflow");
    r
}

/// Random f16 bits: magnitudes 2^-6 to 8, or subnormal where `tiny`.
fn rnd_f16(seed: &mut u64, tiny: bool) -> u16 {
    let b = bits(seed);
    let exp = if tiny { 0 } else { 9 + (b >> 10) % 9 };
    (((b >> 31) << 15) | (exp << 10) | (b & 1023)) as u16
}

/// Decode attention of the token at `pos` in f64: query head h attends with kv head h / 3 to the keys at distances
/// d < min(pos + 1, window), at rows (pos - d) % ring (pos - d where ring is 0). A key's score is scale q.k, plus the
/// bias r[h] . rb[..][d] where d < extent, minus sink[h]; the softmax has a zero logit beside the scores.
/// q [24][128], r [24][16], sink [24], rb BF16 [16][1024], the caches f16 [rows][8][128]. Returns [24][128].
#[allow(clippy::too_many_arguments)]
fn attn_ref(q: &[f32], r: &[f32], sink: &[f32], rb: &[u8], kc: &[u16], vc: &[u16], pos: usize, ring: usize, window: usize, extent: usize, scale: f64) -> Vec<f64> {
    let nk = (pos + 1).min(window);
    let row = |d: usize| if ring > 0 { (pos - d) % ring } else { pos - d };
    let mut out = vec![0f64; 24 * 128];
    for h in 0..24 {
        let at = |d: usize, i: usize| (row(d) * 8 + h / 3) * 128 + i;
        let s: Vec<f64> = (0..nk)
            .map(|d| {
                let dot: f64 = (0..128).map(|i| q[h * 128 + i] as f64 * f16(kc[at(d, i)])).sum();
                let bias: f64 = if d < extent { (0..16).map(|j| r[h * 16 + j] as f64 * bf16(rb, j * 1024 + d)).sum() } else { 0.0 };
                scale * dot + bias - sink[h] as f64
            })
            .collect();
        let m = s.iter().fold(0f64, |a, &b| a.max(b));
        let den = (-m).exp() + s.iter().map(|&v| (v - m).exp()).sum::<f64>();
        for i in 0..128 {
            out[h * 128 + i] = (0..nk).map(|d| (s[d] - m).exp() * f16(vc[at(d, i)])).sum::<f64>() / den;
        }
    }
    out
}

/// y = W x on the CPU in f64, W a BF16 [rows][k].
fn gemv_ref(w: &[u8], x: &[f32], rows: usize, k: usize) -> Vec<f64> {
    (0..rows).map(|r| (0..k).map(|c| bf16(w, r * k + c) * x[c] as f64).sum()).collect()
}

const E2M1: [f64; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// float8 e4m3fn
fn e4m3(b: u8) -> f64 {
    let (s, e, m) = (if b & 0x80 != 0 { -1.0 } else { 1.0 }, ((b >> 3) & 15) as i32, (b & 7) as f64);
    assert!(b & 0x7F != 0x7F, "e4m3 NaN");
    s * if e == 0 { m / 8.0 * 2f64.powi(-6) } else { (1.0 + m / 8.0) * 2f64.powi(e - 7) }
}

/// Expert e's weights of `w` ("w13" or "w2") in layer l, dequantized as transformers does (in f64): [rows][k].
fn expert_ref(ck: &Checkpoint, l: usize, w: &str, e: usize) -> Result<(Vec<f64>, usize, usize), String> {
    let p = |s: &str| format!("model.llm.layers.{l}.mlp.experts.{w}_{s}");
    let t = ck.tensor(&p("weight"))?;
    let (rows, kb) = (t.shape[1], t.shape[2]);
    let codes = &ck.bytes(t)[e * rows * kb..(e + 1) * rows * kb];
    let st = ck.tensor(&p("weight_scale"))?;
    let ns = st.shape[2];
    let sc = &ck.bytes(st)[e * rows * ns..(e + 1) * rows * ns];
    if t.dtype == Dtype::U8 {
        // NVFP4: two codes a byte (the even element low), an E4M3 scale per 16, the factor amax / denominator
        let amax = ck.floats(&p("weight_amax"))?[e] as f64;
        let den = ck.floats(&p("weight_outer_scale_denominator"))?[0] as f64;
        let k = 2 * kb;
        let mut v = vec![0f64; rows * k];
        for r in 0..rows {
            for c in 0..k {
                let b = codes[r * kb + c / 2];
                let q = if c % 2 == 0 { b & 15 } else { b >> 4 };
                v[r * k + c] = E2M1[q as usize] * e4m3(sc[r * ns + c / 16]) * (amax / den);
            }
        }
        Ok((v, rows, k))
    } else {
        // MXFP8: e4m3 values, a power of two 2^(s-127) per 32
        let k = kb;
        let v = (0..rows * k).map(|i| e4m3(codes[i]) * 2f64.powi(sc[(i / k) * ns + (i % k) / 32] as i32 - 127)).collect();
        Ok((v, rows, k))
    }
}

/// A bank of slots of the experts `es` of layer l's `w`: each expert's codes as the checkpoint has them, then its
/// scales, the slot rounded up to 256 bytes. Returns the bank and the slot size.
fn bank(ck: &Checkpoint, l: usize, w: &str, es: &[usize]) -> Result<(Vec<u8>, usize), String> {
    let p = |s: &str| format!("model.llm.layers.{l}.mlp.experts.{w}_{s}");
    let (t, st) = (ck.tensor(&p("weight"))?, ck.tensor(&p("weight_scale"))?);
    let (qb, sb) = (t.nbytes() / t.shape[0], st.nbytes() / st.shape[0]);
    let slot = (qb + sb).next_multiple_of(256);
    let mut b = vec![0u8; slot * es.len()];
    for (i, &e) in es.iter().enumerate() {
        b[i * slot..i * slot + qb].copy_from_slice(&ck.bytes(t)[e * qb..(e + 1) * qb]);
        b[i * slot + qb..i * slot + qb + sb].copy_from_slice(&ck.bytes(st)[e * sb..(e + 1) * sb]);
    }
    Ok((b, slot))
}

/// The largest |a - b| over the largest |b|.
fn rel_err(a: &[f32], b: &[f64]) -> f64 {
    let scale = b.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-30);
    a.iter().zip(b).map(|(x, y)| (*x as f64 - y).abs()).fold(0.0, f64::max) / scale
}

fn check(name: &str, err: f64, bound: f64, fails: &mut Vec<String>) {
    println!("{name:40} {err:.2e}{}", if err > bound { "  FAIL" } else { "" });
    if err > bound || err.is_nan() {
        fails.push(name.to_string());
    }
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let ck = Checkpoint::open(std::path::Path::new(a.get(1).ok_or("usage: ik-wcheck <checkpoint dir>")?))?;
    let _ctx = Ctx::new(0)?;
    let k = Kernels::load()?;
    let s = Stream::new()?;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut fails = vec![];
    let n = 2304;

    // gemv_bf16: wo (k 3072), and the controller's down (k 4608), two tokens
    for (name, rows, kk) in [("model.llm.layers.3.attn.wo_ud.weight", n, 3072), ("model.llm.layers.7.controllers.k.down_weight", 64, 2 * n)] {
        let w = ck.get(name, Dtype::BF16, &[rows, kk])?;
        let x: Vec<f32> = (0..2 * kk).map(|_| rnd(&mut seed)).collect();
        let (bw, bx, by) = (DevBuf::from_slice(w)?, DevBuf::from_slice(&x)?, DevBuf::new(2 * rows * 4)?);
        k.gemv_bf16(&s, bw.ptr, bx.ptr, by.ptr, rows as u32, kk as u32, 2, rows as u32)?;
        s.sync()?;
        let y: Vec<f32> = by.download(2 * rows)?;
        for t in 0..2 {
            let want = gemv_ref(w, &x[t * kk..(t + 1) * kk], rows, kk);
            check(&format!("gemv_bf16 {} token {t}", name.rsplit("layers.").next().unwrap()), rel_err(&y[t * rows..(t + 1) * rows], &want), 1e-5, &mut fails);
        }
    }

    // gemv_bf16t: q, gate, rel, 23 rows of sink (an odd count) from one x, k from another
    let l = |s: &str| format!("model.llm.layers.5.attn.{s}.weight");
    let mats = [(l("wq_du"), 3072, 0), (l("gate_attn"), 3072, 0), (l("wr_du"), 384, 0), (l("attn_sink_proj"), 23, 0), (l("wk_dv"), 1024, 1)];
    let x: Vec<f32> = (0..2 * n).map(|_| rnd(&mut seed)).collect();
    let bx = DevBuf::from_slice(&x)?;
    let mut bufs = vec![];
    let mut tab: Vec<u64> = vec![];
    for (name, rows, xi) in &mats {
        let t = ck.tensor(name)?;
        let w = &ck.bytes(t)[..rows * n * 2];
        let (bw, by) = (DevBuf::from_slice(w)?, DevBuf::zeroed((rows + 1) * 4, &s)?);
        tab.extend([bw.ptr, bx.at(xi * n * 4), by.ptr, *rows as u64]);
        bufs.push((bw, by));
    }
    let bt = DevBuf::from_slice(&tab)?;
    k.gemv_bf16t(&s, bt.ptr, 3072, n as u32, mats.len() as u32)?;
    s.sync()?;
    for ((name, rows, xi), (_, by)) in mats.iter().zip(&bufs) {
        let w = &ck.bytes(ck.tensor(name)?)[..rows * n * 2];
        let y: Vec<f32> = by.download(rows + 1)?;
        let want = gemv_ref(w, &x[xi * n..(xi + 1) * n], *rows, n);
        check(&format!("gemv_bf16t {}", name.rsplit("attn.").next().unwrap()), rel_err(&y[..*rows], &want), 1e-5, &mut fails);
        if y[*rows] != 0.0 {
            fails.push(format!("gemv_bf16t wrote past {name}"));
        }
    }

    // gemm_bf16: wq on 130 tokens (a tile of tokens and 2 of the next), 200 rows of wo (a tile of rows and 72 of the
    // next) on 3 tokens, the controller's down (64 rows, k 4608) on 129 tokens; y's rows 5 floats apart, so that the
    // gaps, and a token past the last, must stay 0
    for (name, full, rows, nt) in [
        ("model.llm.layers.5.attn.wq_du.weight", [3072, n], 3072, 130),
        ("model.llm.layers.3.attn.wo_ud.weight", [n, 3072], 200, 3),
        ("model.llm.layers.7.controllers.k.down_weight", [64, 2 * n], 64, 129),
    ] {
        let kk = full[1];
        let w = &ck.get(name, Dtype::BF16, &full)?[..rows * kk * 2];
        let x: Vec<f32> = (0..nt * kk).map(|_| rnd(&mut seed)).collect();
        let ys = rows + 5;
        let (bw, bx, by) = (DevBuf::from_slice(w)?, DevBuf::from_slice(&x)?, DevBuf::zeroed((nt + 1) * ys * 4, &s)?);
        k.gemm_bf16(&s, bw.ptr, bx.ptr, by.ptr, rows as u32, kk as u32, nt as u32, ys as u32)?;
        s.sync()?;
        let y: Vec<f32> = by.download((nt + 1) * ys)?;
        let err = (0..nt).map(|t| rel_err(&y[t * ys..][..rows], &gemv_ref(w, &x[t * kk..][..kk], rows, kk))).fold(0.0, f64::max);
        check(&format!("gemm_bf16 {}, {rows} rows, {nt} tokens", name.rsplit("layers.").next().unwrap()), err, 1e-5, &mut fails);
        if y.chunks(ys).enumerate().any(|(t, yt)| yt.iter().enumerate().any(|(r, &v)| (t == nt || r >= rows) && v != 0.0)) {
            fails.push(format!("gemm_bf16 {name}: wrote past its rows or tokens"));
        }
    }

    // experts, decode: gate_up and down of 8 slots (the 4th one not cached: its outputs stay 0)
    let es = [3usize, 17, 64, 127, 0, 5, 9, 100];
    let slots: Vec<i32> = (0..8).map(|i| if i == 3 { -1 } else { i as i32 }).collect();
    let bslots = DevBuf::from_slice(&slots)?;
    let ids: Vec<u32> = es.iter().map(|&e| e as u32).collect();
    let bids = DevBuf::from_slice(&ids)?;
    for l in [0usize, 21] {
        for (w, xstride) in [("w13", 0usize), ("w2", 1152)] {
            let (b, slot) = bank(&ck, l, w, &es)?;
            let bb = DevBuf::from_slice(&b)?;
            let (_, rows, kk) = expert_ref(&ck, l, w, 0)?;
            let nx = if xstride == 0 { kk } else { 8 * kk };
            let x: Vec<f32> = (0..nx).map(|_| rnd(&mut seed) * 2.0).collect();
            let xs: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let bx = DevBuf::from_slice(&x)?;
            let by = DevBuf::zeroed(8 * rows * 4, &s)?;
            let nvfp4 = ck.tensor(&format!("model.llm.layers.{l}.mlp.experts.{w}_weight"))?.dtype == Dtype::U8;
            if nvfp4 {
                let p = |n: &str| format!("model.llm.layers.{l}.mlp.experts.{w}_weight_{n}");
                let (ba, bn) = (DevBuf::from_slice(&ck.floats(&p("amax"))?)?, DevBuf::from_slice(&ck.floats(&p("outer_scale_denominator"))?)?);
                k.moe_nvfp4_d(&s, bb.ptr, slot as u64, bslots.ptr, bids.ptr, ba.ptr, bn.ptr, bx.ptr, xstride as u32, 0, by.ptr, rows as u32, kk as u32, 8)?;
            } else {
                k.moe_mxfp8_d(&s, bb.ptr, slot as u64, bslots.ptr, bx.ptr, xstride as u32, 0, by.ptr, rows as u32, kk as u32, 8)?;
            }
            s.sync()?;
            let y: Vec<f32> = by.download(8 * rows)?;
            let mut err = 0f64;
            for (j, &e) in es.iter().enumerate() {
                let out = &y[j * rows..(j + 1) * rows];
                if slots[j] < 0 {
                    if out.iter().any(|&v| v != 0.0) {
                        fails.push(format!("layer {l} {w}: an uncached expert's outputs were written"));
                    }
                    continue;
                }
                let (wr, _, _) = expert_ref(&ck, l, w, e)?;
                let xe = &xs[if xstride == 0 { 0 } else { j * kk }..][..kk];
                let want: Vec<f64> = (0..rows).map(|r| (0..kk).map(|c| wr[r * kk + c] * xe[c]).sum()).collect();
                err = err.max(rel_err(out, &want));
            }
            check(&format!("moe_{}_d layer {l} {w}", if nvfp4 { "nvfp4" } else { "mxfp8" }), err, 1e-5, &mut fails);
        }
    }

    // experts, prefill: gate_up of the 16 pairs of 2 tokens, each token's x its own (x + (pair >> 3) k), the second
    // token's slots in reverse; pairs 3 and 12 not cached
    let sl16: Vec<i32> = (0..16).map(|j| if j == 3 || j == 12 { -1 } else if j < 8 { j } else { 15 - j }).collect();
    let id16: Vec<u32> = sl16.iter().map(|&s| es[s.max(0) as usize] as u32).collect();
    let (bsl16, bid16) = (DevBuf::from_slice(&sl16)?, DevBuf::from_slice(&id16)?);
    for l in [0usize, 21] {
        let (b, slot) = bank(&ck, l, "w13", &es)?;
        let bb = DevBuf::from_slice(&b)?;
        let (_, rows, kk) = expert_ref(&ck, l, "w13", 0)?;
        let x: Vec<f32> = (0..2 * kk).map(|_| rnd(&mut seed) * 2.0).collect();
        let (bx, by) = (DevBuf::from_slice(&x)?, DevBuf::zeroed(16 * rows * 4, &s)?);
        let nvfp4 = ck.tensor(&format!("model.llm.layers.{l}.mlp.experts.w13_weight"))?.dtype == Dtype::U8;
        if nvfp4 {
            let p = |n: &str| format!("model.llm.layers.{l}.mlp.experts.w13_weight_{n}");
            let (ba, bn) = (DevBuf::from_slice(&ck.floats(&p("amax"))?)?, DevBuf::from_slice(&ck.floats(&p("outer_scale_denominator"))?)?);
            k.moe_nvfp4_d(&s, bb.ptr, slot as u64, bsl16.ptr, bid16.ptr, ba.ptr, bn.ptr, bx.ptr, kk as u32, 3, by.ptr, rows as u32, kk as u32, 16)?;
        } else {
            k.moe_mxfp8_d(&s, bb.ptr, slot as u64, bsl16.ptr, bx.ptr, kk as u32, 3, by.ptr, rows as u32, kk as u32, 16)?;
        }
        s.sync()?;
        let y: Vec<f32> = by.download(16 * rows)?;
        let mut err = 0f64;
        for (j, &sj) in sl16.iter().enumerate() {
            let out = &y[j * rows..(j + 1) * rows];
            if sj < 0 {
                if out.iter().any(|&v| v != 0.0) {
                    fails.push(format!("layer {l} w13, 2 tokens: an uncached pair's outputs were written"));
                }
                continue;
            }
            let (wr, _, _) = expert_ref(&ck, l, "w13", es[sj as usize])?;
            let xe = &x[j / 8 * kk..][..kk];
            let want: Vec<f64> = (0..rows).map(|r| (0..kk).map(|c| wr[r * kk + c] * xe[c] as f64).sum()).collect();
            err = err.max(rel_err(out, &want));
        }
        check(&format!("moe_{}_d layer {l} w13, 2 tokens", if nvfp4 { "nvfp4" } else { "mxfp8" }), err, 1e-5, &mut fails);
    }

    // experts, grouped: gate_up and down of 3 NVFP4 experts in one launch, on 1, 33 and 70 tokens of their own (a
    // tile of tokens and 1 of the next, 2 tiles and 6 of the third); each expert's y has 1 token more, which must stay
    // 0 (its own seed: the checks after it keep their inputs)
    {
        let mut seed = 0x06e0_fa11u64;
        let (l, es3, ms) = (0usize, [17usize, 64, 127], [1usize, 33, 70]);
        for w in ["w13", "w2"] {
            let (b, slot) = bank(&ck, l, w, &es3)?;
            let bb = DevBuf::from_slice(&b)?;
            let (_, rows, kk) = expert_ref(&ck, l, w, 0)?;
            let p = |n: &str| format!("model.llm.layers.{l}.mlp.experts.{w}_weight_{n}");
            let (ba, bn) = (DevBuf::from_slice(&ck.floats(&p("amax"))?)?, DevBuf::from_slice(&ck.floats(&p("outer_scale_denominator"))?)?);
            let xs: Vec<Vec<f32>> = ms.iter().map(|&m| (0..m * kk).map(|_| rnd(&mut seed) * 2.0).collect()).collect();
            let (mut bufs, mut tab) = (vec![], vec![]);
            for (i, x) in xs.iter().enumerate() {
                let (bx, by) = (DevBuf::from_slice(x)?, DevBuf::zeroed((ms[i] + 1) * rows * 4, &s)?);
                tab.extend([bb.at(i * slot), bx.ptr, by.ptr, ms[i] as u64 | (es3[i] as u64) << 32]);
                bufs.push((bx, by));
            }
            let bt = DevBuf::from_slice(&tab)?;
            k.moe_nvfp4_g(&s, bt.ptr, ba.ptr, bn.ptr, rows as u32, kk as u32, 70, 3)?;
            s.sync()?;
            let mut err = 0f64;
            for (i, (_, by)) in bufs.iter().enumerate() {
                let y: Vec<f32> = by.download((ms[i] + 1) * rows)?;
                let (wr, _, _) = expert_ref(&ck, l, w, es3[i])?;
                for t in 0..ms[i] {
                    let xt = &xs[i][t * kk..][..kk];
                    let want: Vec<f64> = (0..rows).map(|r| (0..kk).map(|c| wr[r * kk + c] * xt[c] as f64).sum()).collect();
                    err = err.max(rel_err(&y[t * rows..][..rows], &want));
                }
                if y[ms[i] * rows..].iter().any(|&v| v != 0.0) {
                    fails.push(format!("moe_nvfp4_g layer {l} {w}: wrote past expert {i}'s tokens"));
                }
            }
            check(&format!("moe_nvfp4_g layer {l} {w}, 1, 33, 70 tokens"), err, 1e-5, &mut fails);
        }
    }

    // the prefill combine of 3 tokens: the GPU's pairs, then the CPU's added
    {
        let (nt, nn) = (3usize, 2304usize);
        let sl: Vec<i32> = (0..8 * nt as i32).map(|p| if p % 3 == 0 { -1 } else { p % 5 }).collect();
        let w: Vec<f32> = (0..8 * nt).map(|_| rnd(&mut seed).abs()).collect();
        let g: Vec<f32> = (0..8 * nt * nn).map(|_| rnd(&mut seed)).collect();
        let c: Vec<f32> = (0..8 * nt * nn).map(|_| rnd(&mut seed)).collect();
        let (bsl, bw, bg, bc) = (DevBuf::from_slice(&sl)?, DevBuf::from_slice(&w)?, DevBuf::from_slice(&g)?, DevBuf::from_slice(&c)?);
        let bo = DevBuf::zeroed(nt * nn * 4, &s)?;
        k.moe_combine_t(&s, bg.ptr, bsl.ptr, bw.ptr, bo.ptr, nn as u32, nt as u32, true, false)?;
        k.moe_combine_t(&s, bc.ptr, bsl.ptr, bw.ptr, bo.ptr, nn as u32, nt as u32, false, true)?;
        s.sync()?;
        let out: Vec<f32> = bo.download(nt * nn)?;
        let mut err = 0f64;
        for t in 0..nt {
            let want: Vec<f64> = (0..nn)
                .map(|i| (8 * t..8 * t + 8).map(|p| w[p] as f64 * if sl[p] >= 0 { g[p * nn + i] } else { c[p * nn + i] } as f64).sum())
                .collect();
            err = err.max(rel_err(&out[t * nn..(t + 1) * nn], &want));
        }
        check("moe_combine_t, the GPU's pairs then the CPU's", err, 1e-6, &mut fails);
    }

    // the grouped combine of 3 tokens: two chunks' rows (the first sets the output, the second adds), each pair in at
    // most one of them, some in neither
    {
        let (nt, nn, nr) = (3usize, 2304usize, 7usize);
        let mut rows = [vec![-1i32; 8 * nt], vec![-1i32; 8 * nt]];
        for p in 0..8 * nt {
            if p % 4 != 3 {
                rows[p % 2][p] = ((p * 5) % nr) as i32;
            }
        }
        let w: Vec<f32> = (0..8 * nt).map(|_| rnd(&mut seed).abs()).collect();
        let src: Vec<Vec<f32>> = (0..2).map(|_| (0..nr * nn).map(|_| rnd(&mut seed)).collect()).collect();
        let bw = DevBuf::from_slice(&w)?;
        let bo = DevBuf::from_slice(&vec![f32::NAN; nt * nn])?;
        for c in 0..2 {
            let (br, bs) = (DevBuf::from_slice(&rows[c])?, DevBuf::from_slice(&src[c])?);
            k.moe_combine_r(&s, bs.ptr, br.ptr, bw.ptr, bo.ptr, nn as u32, nt as u32, c > 0)?;
            s.sync()?;
        }
        let out: Vec<f32> = bo.download(nt * nn)?;
        let mut err = 0f64;
        for t in 0..nt {
            let mut want = vec![0f64; nn];
            for (c, rc) in rows.iter().enumerate() {
                for p in (8 * t..8 * t + 8).filter(|&p| rc[p] >= 0) {
                    for (i, v) in want.iter_mut().enumerate() {
                        *v += w[p] as f64 * src[c][rc[p] as usize * nn + i] as f64;
                    }
                }
            }
            err = err.max(rel_err(&out[t * nn..(t + 1) * nn], &want));
        }
        check("moe_combine_r, two chunks of rows", err, 1e-6, &mut fails);
    }

    // gather_rows on rows of 2304 floats, as the expert stream uses it: 5 rows of 9 (row 8 twice), each value its
    // source's bits (its own seed: the checks after it keep their inputs)
    {
        let nn = 2304usize;
        let mut seed = 0xc0b1_e5u64;
        let src: Vec<f32> = (0..9 * nn).map(|_| rnd(&mut seed)).collect();
        let sidx = [8u32, 0, 3, 8, 5];
        let (bs, bsi) = (DevBuf::from_slice(&src)?, DevBuf::from_slice(&sidx)?);
        let bg = DevBuf::zeroed(5 * nn * 4, &s)?;
        k.gather_rows(&s, bs.ptr, bsi.ptr, bg.ptr, nn as u32, 5)?;
        s.sync()?;
        let g: Vec<f32> = bg.download(5 * nn)?;
        let row = |v: &[f32], r: usize| v[r * nn..(r + 1) * nn].iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let wrong = (0..5).filter(|&i| row(&g, i) != row(&src, sidx[i] as usize)).count();
        check("gather_rows: rows not their source's", wrong as f64, 0.0, &mut fails);
    }

    // add_rows_w as the expert stream uses it: 5 rows of 2304 floats added, each weighted by its pair, to 5 rows of 12
    // (the other rows untouched), against f64
    {
        let nn = 2304usize;
        let mut seed = 0xadd_5eedu64;
        let src: Vec<f32> = (0..5 * nn).map(|_| rnd(&mut seed)).collect();
        let acc: Vec<f32> = (0..12 * nn).map(|_| rnd(&mut seed)).collect();
        let wts: Vec<f32> = (0..16).map(|_| rnd(&mut seed).abs()).collect();
        let (didx, widx) = ([11u32, 2, 7, 4, 0], [3u32, 15, 0, 8, 3]);
        let (bs, ba, bw) = (DevBuf::from_slice(&src)?, DevBuf::from_slice(&acc)?, DevBuf::from_slice(&wts)?);
        let (bdi, bwi) = (DevBuf::from_slice(&didx)?, DevBuf::from_slice(&widx)?);
        k.add_rows_w(&s, bs.ptr, ba.ptr, bdi.ptr, bw.ptr, bwi.ptr, nn as u32, 5)?;
        s.sync()?;
        let got: Vec<f32> = ba.download(12 * nn)?;
        let mut err = 0f64;
        for r in 0..12 {
            let at = didx.iter().position(|&x| x as usize == r);
            for c in 0..nn {
                let want = match at {
                    Some(i) => acc[r * nn + c] as f64 + wts[widx[i] as usize] as f64 * src[i * nn + c] as f64,
                    None => acc[r * nn + c] as f64,
                };
                err = err.max((got[r * nn + c] as f64 - want).abs() / want.abs().max(1e-3));
            }
        }
        check("add_rows_w, 5 rows into 12", err, 1e-6, &mut fails);
    }

    // NVFP4 on made-up slots, for what the real tensors may hold little of: every code byte in each row, and E4M3
    // scales of every magnitude, the subnormal ones among them (every 4th row has only those). Each row's error is over
    // its sum of |w x|, which one value decoded wrong moves by about 1/k.
    {
        let (rows, kk) = (64usize, 512usize);
        let (qb, sb) = (rows * kk / 2, rows * kk / 16);
        let slot = (qb + sb).next_multiple_of(256);
        let mut b = vec![0u8; 8 * slot];
        for e in 0..8 {
            for r in 0..rows {
                for i in 0..kk / 2 {
                    b[e * slot + r * kk / 2 + i] = ((i + 37 * r + 101 * e) % 256) as u8;
                }
                for j in 0..kk / 16 {
                    b[e * slot + qb + r * kk / 16 + j] = if r % 4 == 0 { 1 + (r + j + e) % 7 } else { 1 + (7 * r + 13 * j + e) % 0x7E } as u8;
                }
            }
        }
        let amax: Vec<f32> = (0..8).map(|e| 0.5 + e as f32).collect();
        let den = [3.0f32];
        let x: Vec<f32> = (0..kk).map(|_| rnd(&mut seed) * 2.0).collect();
        let (all, ids8): (Vec<i32>, Vec<u32>) = ((0..8).collect(), (0..8).collect());
        let (bb, ba, bn, bx) = (DevBuf::from_slice(&b)?, DevBuf::from_slice(&amax)?, DevBuf::from_slice(&den)?, DevBuf::from_slice(&x)?);
        let (bs8, bi8, by) = (DevBuf::from_slice(&all)?, DevBuf::from_slice(&ids8)?, DevBuf::zeroed(8 * rows * 4, &s)?);
        k.moe_nvfp4_d(&s, bb.ptr, slot as u64, bs8.ptr, bi8.ptr, ba.ptr, bn.ptr, bx.ptr, 0, 0, by.ptr, rows as u32, kk as u32, 8)?;
        s.sync()?;
        let y: Vec<f32> = by.download(8 * rows)?;
        let mut err = 0f64;
        for e in 0..8 {
            let f = amax[e] as f64 / den[0] as f64;
            for r in 0..rows {
                let (mut want, mut l1) = (0f64, 0f64);
                for c in 0..kk {
                    let byte = b[e * slot + r * kk / 2 + c / 2];
                    let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                    let wx = E2M1[code as usize] * e4m3(b[e * slot + qb + r * kk / 16 + c / 16]) * f * x[c] as f64;
                    want += wx;
                    l1 += wx.abs();
                }
                err = err.max((y[e * rows + r] as f64 - want).abs() / l1.max(1e-30));
            }
        }
        check("moe_nvfp4_d made up: codes, subnormal scales", err, 1e-5, &mut fails);
        // the same slots grouped: experts 2 and 5 on 3 and 40 tokens of their own. A tile's rows past the 64 read the
        // last one. (Its own seed: the checks after it keep their inputs.)
        let mut seed = 0x0005_eed6u64;
        let (eg, ms) = ([2usize, 5], [3usize, 40]);
        let xg: Vec<Vec<f32>> = ms.iter().map(|&m| (0..m * kk).map(|_| rnd(&mut seed) * 2.0).collect()).collect();
        let (mut bufs, mut tab) = (vec![], vec![]);
        for (i, x) in xg.iter().enumerate() {
            let (bx, by) = (DevBuf::from_slice(x)?, DevBuf::zeroed(ms[i] * rows * 4, &s)?);
            tab.extend([bb.at(eg[i] * slot), bx.ptr, by.ptr, ms[i] as u64 | (eg[i] as u64) << 32]);
            bufs.push((bx, by));
        }
        let bt = DevBuf::from_slice(&tab)?;
        k.moe_nvfp4_g(&s, bt.ptr, ba.ptr, bn.ptr, rows as u32, kk as u32, 40, 2)?;
        s.sync()?;
        let mut err = 0f64;
        for (i, (_, by)) in bufs.iter().enumerate() {
            let (e, y): (usize, Vec<f32>) = (eg[i], by.download(ms[i] * rows)?);
            let f = amax[e] as f64 / den[0] as f64;
            for t in 0..ms[i] {
                for r in 0..rows {
                    let (mut want, mut l1) = (0f64, 0f64);
                    for c in 0..kk {
                        let byte = b[e * slot + r * kk / 2 + c / 2];
                        let code = if c % 2 == 0 { byte & 15 } else { byte >> 4 };
                        let wx = E2M1[code as usize] * e4m3(b[e * slot + qb + r * kk / 16 + c / 16]) * f * xg[i][t * kk + c] as f64;
                        want += wx;
                        l1 += wx.abs();
                    }
                    err = err.max((y[t * rows + r] as f64 - want).abs() / l1.max(1e-30));
                }
            }
        }
        check("moe_nvfp4_g made up: codes, subnormal scales", err, 1e-5, &mut fails);
    }

    // experts on the CPU: every row of two experts' gate_up and down, the matrices views of the checkpoint's bytes
    for l in [0usize, 21] {
        for w in ["w13", "w2"] {
            let p = |s: &str| format!("model.llm.layers.{l}.mlp.experts.{w}_{s}");
            let (t, st) = (ck.tensor(&p("weight"))?, ck.tensor(&p("weight_scale"))?);
            let nvfp4 = t.dtype == Dtype::U8;
            let fmt = if nvfp4 { Fmt::NvFp4 } else { Fmt::MxFp8 };
            let (rows, kk) = (t.shape[1], if nvfp4 { 2 * t.shape[2] } else { t.shape[2] });
            let (qb, sb) = (t.nbytes() / t.shape[0], st.nbytes() / st.shape[0]);
            let x: Vec<f32> = (0..kk).map(|_| rnd(&mut seed) * 2.0).collect();
            let xs: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            let factors: Vec<f32> = if nvfp4 {
                let den = ck.floats(&p("weight_outer_scale_denominator"))?[0];
                ck.floats(&p("weight_amax"))?.iter().map(|a| a / den).collect()
            } else {
                vec![1.0; t.shape[0]]
            };
            let mut err = 0f64;
            for e in [3usize, 127] {
                let m = XMat { fmt, q: &ck.bytes(t)[e * qb..][..qb], s: &ck.bytes(st)[e * sb..][..sb], factor: factors[e], rows, k: kk };
                let mut xo = vec![0f32; kk];
                order_x(fmt, &x, &mut xo);
                let got: Vec<f32> = (0..rows).map(|r| dot_row(&m, r, &xo) * m.scale()).collect();
                let (wr, _, _) = expert_ref(&ck, l, w, e)?;
                let want: Vec<f64> = (0..rows).map(|r| (0..kk).map(|c| wr[r * kk + c] * xs[c]).sum()).collect();
                err = err.max(rel_err(&got, &want));
            }
            check(&format!("cpu dot_row {} layer {l} {w}", if nvfp4 { "nvfp4" } else { "mxfp8" }), err, 1e-5, &mut fails);
        }
    }

    // SITU of gate_up rows interleaved as the checkpoint has them: gate i is row 2i, up i row 2i + 1
    let nff = 1152;
    let gu: Vec<f32> = (0..8 * 2 * nff).map(|_| rnd(&mut seed) * 6.0).collect();
    let bgu = DevBuf::from_slice(&gu)?;
    let baf = DevBuf::zeroed(8 * nff * 4, &s)?;
    k.situ(&s, bgu.ptr, bslots.ptr, baf.ptr, nff as u32, 4.0, 25.0, 8)?;
    s.sync()?;
    let af: Vec<f32> = baf.download(8 * nff)?;
    let mut e_af = 0f64;
    for j in 0..8 {
        if slots[j] < 0 {
            if af[j * nff..(j + 1) * nff].iter().any(|&v| v != 0.0) {
                fails.push("situ: an uncached expert's activations were written".into());
            }
            continue;
        }
        let want: Vec<f64> = (0..nff)
            .map(|i| {
                let (g, u) = (gu[j * 2 * nff + 2 * i] as f64, gu[j * 2 * nff + 2 * i + 1] as f64);
                4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp()) * 25.0 * (u / 25.0).tanh()
            })
            .collect();
        e_af = e_af.max(rel_err(&af[j * nff..(j + 1) * nff], &want));
    }
    check("situ", e_af, 1e-5, &mut fails);

    // rms_norm with a BF16 gain, on two vectors
    let w = ck.get("model.llm.layers.3.mlp_norm.weight", Dtype::BF16, &[n])?;
    let x: Vec<f32> = (0..2 * n).map(|_| rnd(&mut seed) * 3.0).collect();
    let (bw, bx, by) = (DevBuf::from_slice(w)?, DevBuf::from_slice(&x)?, DevBuf::zeroed(2 * n * 4, &s)?);
    let want: Vec<f64> = (0..2 * n)
        .map(|i| {
            let v = &x[i / n * n..][..n];
            let ms = v.iter().map(|&a| a as f64 * a as f64).sum::<f64>() / n as f64;
            x[i] as f64 / (ms + 1e-6).sqrt() * bf16(w, i % n)
        })
        .collect();
    k.rms_norm(&s, bx.ptr, Some(bw.ptr), by.ptr, n as u32, 2, 1e-6)?;
    s.sync()?;
    check("rms_norm bf16 gain", rel_err(&by.download::<f32>(2 * n)?, &want), 1e-5, &mut fails);

    // the short convolutions over 5 steps from a zero state: a sliding layer's conv1d [c][1][4] (tap 3 the current
    // token), and a full layer's static taps [4][c] (lag 0 the current token) plus the input-dependent ones [head][lag]
    let kv = 1024;
    for full in [false, true] {
        let (name, shape) = if full { ("model.llm.layers.5.attn.k_sconv.static_w", vec![4, kv]) } else { ("model.llm.layers.0.attn.k_sconv.weight", vec![kv, 1, 4]) };
        let w = ck.get(name, Dtype::BF16, &shape)?;
        // the weight of lag j (0 the current token) for channel c
        let wl = |c: usize, j: usize| if full { bf16(w, j * kv + c) } else { bf16(w, c * 4 + 3 - j) };
        let (bw, bst, bx, by, btap) = (DevBuf::from_slice(w)?, DevBuf::zeroed(kv * 3 * 4, &s)?, DevBuf::new(kv * 4)?, DevBuf::new(kv * 4)?, DevBuf::new(8 * 4 * 4)?);
        let scale = if full { 1.0 / 3.0 } else { 1.0 / 5f32.sqrt() };
        let mut hist: Vec<Vec<f32>> = vec![];
        let mut err = 0f64;
        for _step in 0..5 {
            let x: Vec<f32> = (0..kv).map(|_| rnd(&mut seed)).collect();
            let taps: Vec<f32> = (0..32).map(|_| rnd(&mut seed) * 0.5).collect();
            bx.upload(&x)?;
            btap.upload(&taps)?;
            k.conv4(&s, bx.ptr, bw.ptr, bst.ptr, if full { Some(btap.ptr) } else { None }, by.ptr, kv as u32, 128, scale)?;
            s.sync()?;
            hist.push(x);
            let want: Vec<f64> = (0..kv)
                .map(|c| {
                    let mut o = hist.last().unwrap()[c] as f64;
                    for j in 0..4.min(hist.len()) {
                        let tap = if full { taps[c / 128 * 4 + j] as f64 } else { 0.0 };
                        o += (wl(c, j) + tap) * hist[hist.len() - 1 - j][c] as f64;
                    }
                    o * scale as f64
                })
                .collect();
            err = err.max(rel_err(&by.download::<f32>(kv)?, &want));
        }
        check(&format!("conv4 {}", if full { "static [4][c] + taps" } else { "conv1d [c][1][4]" }), err, 1e-5, &mut fails);
    }

    // the post convolutions over 4 steps from a zero state, conv1d [n][1][4] BF16: conv_res (attention, plus the
    // residual), and moe_tail (the weighted sum of 8 expert outputs, from the GPU where the slot is cached, else from
    // the CPU's, then x += its convolution)
    let scale = 1.0 / 5f32.sqrt();
    for tail in [false, true] {
        let name = if tail { "model.llm.layers.3.mlp_sconv.weight" } else { "model.llm.layers.3.attn_sconv.weight" };
        let w = ck.get(name, Dtype::BF16, &[n, 1, 4])?;
        let (bw, bst, bx, by, bres) = (DevBuf::from_slice(w)?, DevBuf::zeroed(n * 3 * 4, &s)?, DevBuf::new(8 * n * 4)?, DevBuf::new(n * 4)?, DevBuf::new(n * 4)?);
        let (bcpu, bwts) = (DevBuf::new(8 * n * 4)?, DevBuf::new(8 * 4)?);
        let mut hist: Vec<Vec<f64>> = vec![];
        let mut err = 0f64;
        for _step in 0..4 {
            let res: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
            bres.upload(&res)?;
            by.upload(&res)?;
            let cur: Vec<f64> = if tail {
                let (g, c): (Vec<f32>, Vec<f32>) = ((0..8 * n).map(|_| rnd(&mut seed)).collect(), (0..8 * n).map(|_| rnd(&mut seed)).collect());
                let wts: Vec<f32> = (0..8).map(|_| rnd(&mut seed).abs()).collect();
                bx.upload(&g)?;
                bcpu.upload(&c)?;
                bwts.upload(&wts)?;
                k.moe_tail(&s, bx.ptr, bcpu.ptr, bslots.ptr, bwts.ptr, bw.ptr, bst.ptr, by.ptr, n as u32, scale)?;
                (0..n).map(|ch| (0..8).map(|j| wts[j] as f64 * if slots[j] >= 0 { g[j * n + ch] } else { c[j * n + ch] } as f64).sum()).collect()
            } else {
                let x: Vec<f32> = (0..n).map(|_| rnd(&mut seed)).collect();
                bx.upload(&x)?;
                k.conv_res(&s, bx.ptr, bw.ptr, bst.ptr, Some(bres.ptr), by.ptr, n as u32, scale)?;
                x.iter().map(|&v| v as f64).collect()
            };
            s.sync()?;
            hist.push(cur);
            let want: Vec<f64> = (0..n)
                .map(|ch| {
                    let mut o = hist.last().unwrap()[ch];
                    for j in 0..4.min(hist.len()) {
                        o += bf16(w, ch * 4 + 3 - j) * hist[hist.len() - 1 - j][ch];
                    }
                    res[ch] as f64 + o * scale as f64
                })
                .collect();
            err = err.max(rel_err(&by.download::<f32>(n)?, &want));
        }
        check(if tail { "moe_tail conv1d [n][1][4]" } else { "conv_res conv1d [n][1][4]" }, err, 1e-5, &mut fails);
    }

    // the q clip: 24 heads of 128 times the BF16 scale [24][128], clipped to an L2 norm of 22.627
    let sq = ck.get("model.llm.layers.5.attn.q_l2_clip.scale", Dtype::BF16, &[24, 128])?;
    let x: Vec<f32> = (0..24 * 128).map(|_| rnd(&mut seed) * 4.0).collect();
    let (bsq, bx, by, bpost) = (DevBuf::from_slice(sq)?, DevBuf::from_slice(&x)?, DevBuf::new(24 * 128 * 4)?, DevBuf::from_slice(&[1f32])?);
    k.clip_rows(&s, bx.ptr, Some(bsq.ptr), by.ptr, 128, 24, 24, 22.627, bpost.ptr)?;
    s.sync()?;
    let mut want = vec![0f64; 24 * 128];
    for h in 0..24 {
        let v: Vec<f64> = (0..128).map(|i| x[h * 128 + i] as f64 * bf16(sq, h * 128 + i)).collect();
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        let f = (22.627 / (norm + 1e-6)).min(1.0);
        for i in 0..128 {
            want[h * 128 + i] = v[i] * f;
        }
    }
    check("clip_rows bf16 scale", rel_err(&by.download::<f32>(24 * 128)?, &want), 1e-5, &mut fails);

    // a controller's output with layer 7's BF16 k gain and forget [n][64]
    let p = |s: &str| format!("model.llm.layers.7.controllers.k.{s}_weight");
    let (gw, fw) = (ck.get(&p("gain"), Dtype::BF16, &[n, 64])?, ck.get(&p("forget"), Dtype::BF16, &[n, 64])?);
    let (cur, msg): (Vec<f32>, Vec<f32>) = ((0..n).map(|_| rnd(&mut seed)).collect(), (0..n).map(|_| rnd(&mut seed)).collect());
    let gate: Vec<f32> = (0..64).map(|_| rnd(&mut seed) * 2.0).collect();
    let (bgw, bfw, bc, bm, bg, by) = (DevBuf::from_slice(gw)?, DevBuf::from_slice(fw)?, DevBuf::from_slice(&cur)?, DevBuf::from_slice(&msg)?, DevBuf::from_slice(&gate)?, DevBuf::new(n * 4)?);
    k.ctrl_out(&s, bc.ptr, bm.ptr, bg.ptr, bgw.ptr, bfw.ptr, by.ptr, n as u32)?;
    s.sync()?;
    let want: Vec<f64> = (0..n)
        .map(|c| {
            let dot = |w: &[u8]| (0..64).map(|r| bf16(w, c * 64 + r) * gate[r] as f64).sum::<f64>().tanh();
            (1.0 - dot(fw)) * cur[c] as f64 + dot(gw) * msg[c] as f64
        })
        .collect();
    check("ctrl_out bf16 gain, forget", rel_err(&by.download::<f32>(n)?, &want), 1e-5, &mut fails);

    // attention on made-up f16 caches (every 97th row subnormal) with the BF16 bias profiles [16][1024] of layer 0
    // (sliding: a ring of 1280 rows, a window of 1024, 18 chunks of 64) and of layer 5 (full, 8 chunks of 256): decode
    // (attn_part, then attn_reduce, the last chunks past the keys) at a position before the window or the profiles'
    // extent and at one past it, then prefill (attn_prefill) of the 37 tokens up to the second (tiles of 16, 16 and 5),
    // their rows of q, r and sink padded
    let scale = 1.0 / 128f32.sqrt();
    for (l, ring, window, cs, positions) in [(0usize, 1280usize, 1024usize, 6u32, [100usize, 2000]), (5, 0, usize::MAX, 8, [37, 1300])] {
        let rb = ck.get(&format!("model.llm.layers.{l}.attn.rel_logits_proj.proj"), Dtype::BF16, &[16, 1024])?;
        let rows = if ring > 0 { ring } else { positions[1] + 1 };
        let kc: Vec<u16> = (0..rows * 1024).map(|i| rnd_f16(&mut seed, i / 1024 % 97 == 0)).collect();
        let vc: Vec<u16> = (0..rows * 1024).map(|i| rnd_f16(&mut seed, i / 1024 % 97 == 0)).collect();
        let (brb, bkc, bvc) = (DevBuf::from_slice(rb)?, DevBuf::from_slice(&kc)?, DevBuf::from_slice(&vc)?);
        let w = window.min(u32::MAX as usize) as u32;
        let what = if ring > 0 { "sliding, ring" } else { "full" };
        let maxs = if ring > 0 { 18 } else { 8 };
        let (bpm, bpl, bpacc, bout) = (DevBuf::new(8 * maxs * 3 * 4)?, DevBuf::new(8 * maxs * 3 * 4)?, DevBuf::new(8 * maxs * 3 * 128 * 4)?, DevBuf::new(24 * 128 * 4)?);
        for pos in positions {
            let q: Vec<f32> = (0..24 * 128).map(|_| rnd(&mut seed) * 3.0).collect();
            let r: Vec<f32> = (0..24 * 16).map(|_| rnd(&mut seed)).collect();
            let sink: Vec<f32> = (0..24).map(|_| rnd(&mut seed) * 2.0).collect();
            let (bq, br, bs, bp) = (DevBuf::from_slice(&q)?, DevBuf::from_slice(&r)?, DevBuf::from_slice(&sink)?, DevBuf::from_slice(&[pos as u32, 0, 0, 0])?);
            k.attn_part(&s, bq.ptr, br.ptr, bs.ptr, brb.ptr, bkc.ptr, bvc.ptr, bpm.ptr, bpl.ptr, bpacc.ptr, bp.ptr, ring as u32, 1024, scale, maxs as u32, maxs as u32, w, cs, u32::MAX)?;
            k.attn_reduce(&s, bpm.ptr, bpl.ptr, bpacc.ptr, bout.ptr, bp.ptr, w, maxs as u32, cs, 24, 0, u32::MAX)?;
            s.sync()?;
            let want = attn_ref(&q, &r, &sink, rb, &kc, &vc, pos, ring, window, 1024, scale as f64);
            check(&format!("attn_part + attn_reduce {what}, pos {pos}"), rel_err(&bout.download::<f32>(24 * 128)?, &want), 1e-5, &mut fails);
            if ring == 0 && pos > 600 {
                // the rows past the first `vram` on the CPU (cpu_attn), the rest in chunks of the GPU from distance
                // pos + 1 - vram on: past the profiles' extent, within it, and one row
                for vram in [pos / 3, pos - 600, pos] {
                    let proj: Vec<u16> = rb.chunks(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
                    let qy = ik_engine::cpu_attn::Query { q: &q, r: &r, sink: &sink, proj: &proj, extent: 1024, scale, pos };
                    let mut part = [ik_engine::cpu_attn::Partial::EMPTY; 24];
                    unsafe { ik_engine::cpu_attn::rows(&qy, kc.as_ptr(), vc.as_ptr(), 0, pos + 1 - vram, &mut part) };
                    let flat: Vec<f32> = part.iter().flat_map(|p| [p.m, p.l].into_iter().chain(p.acc)).collect();
                    let bcpu = DevBuf::from_slice(&flat)?;
                    k.attn_part(&s, bq.ptr, br.ptr, bs.ptr, brb.ptr, bkc.ptr, bvc.ptr, bpm.ptr, bpl.ptr, bpacc.ptr, bp.ptr, 0, 1024, scale, maxs as u32, maxs as u32, w, cs, vram as u32)?;
                    k.attn_reduce(&s, bpm.ptr, bpl.ptr, bpacc.ptr, bout.ptr, bp.ptr, w, maxs as u32, cs, 24, bcpu.ptr, vram as u32)?;
                    s.sync()?;
                    check(&format!("attn_part + cpu_attn rows + attn_reduce full, pos {pos}, rows past {vram} on the CPU"), rel_err(&bout.download::<f32>(24 * 128)?, &want), 1e-5, &mut fails);
                }
            }
        }
        let (p0, nt, qs, rs, ss) = (positions[1] - 36, 37, 3072 + 64, 384 + 16, 24 + 8);
        let q: Vec<f32> = (0..nt * qs).map(|_| rnd(&mut seed) * 3.0).collect();
        let r: Vec<f32> = (0..nt * rs).map(|_| rnd(&mut seed)).collect();
        let sink: Vec<f32> = (0..nt * ss).map(|_| rnd(&mut seed) * 2.0).collect();
        let (bq, br, bs, bp, bo) = (DevBuf::from_slice(&q)?, DevBuf::from_slice(&r)?, DevBuf::from_slice(&sink)?, DevBuf::from_slice(&[p0 as u32, 0, 0, 0])?, DevBuf::new(nt * 3072 * 4)?);
        k.attn_prefill(&s, bq.ptr, br.ptr, bs.ptr, brb.ptr, bkc.ptr, bvc.ptr, bo.ptr, bp.ptr, ring as u32, w, 1024, scale, qs as u32, rs as u32, ss as u32, nt as u32, (0, u32::MAX), 0, 0)?;
        s.sync()?;
        let wants: Vec<Vec<f64>> =
            (0..nt).map(|t| attn_ref(&q[t * qs..][..3072], &r[t * rs..][..384], &sink[t * ss..][..24], rb, &kc, &vc, p0 + t, ring, window, 1024, scale as f64)).collect();
        let err_of = |out: &[f32]| (0..nt).map(|t| rel_err(&out[t * 3072..][..3072], &wants[t])).fold(0.0, f64::max);
        check(&format!("attn_prefill {what}, pos {p0}..{}", p0 + nt), err_of(&bo.download::<f32>(nt * 3072)?), 1e-5, &mut fails);
        if ring == 0 {
            // in passes, the state carried: rows 0..700 in place, then 700..1299 and 1299.. each from a copy of its own
            // (as prefill reads host rows); the last pass has rows of the last 2 tokens alone
            let state = DevBuf::new(nt * 8 * 390 * 4)?;
            k.attn_prefill(&s, bq.ptr, br.ptr, bs.ptr, brb.ptr, bkc.ptr, bvc.ptr, bo.ptr, bp.ptr, 0, w, 1024, scale, qs as u32, rs as u32, ss as u32, nt as u32, (0, 700), state.ptr, 2)?;
            let mut copies = vec![];
            for (r0, r1, mode) in [(700usize, 1299usize, 3u32), (1299, rows, 1)] {
                let (kcopy, vcopy) = (DevBuf::from_slice(&kc[r0 * 1024..r1 * 1024])?, DevBuf::from_slice(&vc[r0 * 1024..r1 * 1024])?);
                let off = (r0 * 2048) as u64;
                let (kp, vp) = (kcopy.ptr.wrapping_sub(off), vcopy.ptr.wrapping_sub(off));
                k.attn_prefill(&s, bq.ptr, br.ptr, bs.ptr, brb.ptr, kp, vp, bo.ptr, bp.ptr, 0, w, 1024, scale, qs as u32, rs as u32, ss as u32, nt as u32, (r0 as u32, r1 as u32), state.ptr, mode)?;
                copies.push((kcopy, vcopy));
            }
            s.sync()?;
            check(&format!("attn_prefill full in 3 passes, pos {p0}..{}", p0 + nt), err_of(&bo.download::<f32>(nt * 3072)?), 1e-5, &mut fails);
        }
    }

    // an attention job for the CPU: word, position, then q [24][128], r [24][16] and the sink [24] copied whole into
    // host-mapped memory, the seq of the token's step last (its own seed: the checks after it keep their inputs)
    {
        let (na, nb, nc) = (3072usize, 384usize, 24usize);
        let mut seed = 0x5eed_0b5u64;
        let x: Vec<f32> = (0..na + nb + nc).map(|_| rnd(&mut seed)).collect();
        let (ba, bb, bc) = (DevBuf::from_slice(&x[..na])?, DevBuf::from_slice(&x[na..na + nb])?, DevBuf::from_slice(&x[na + nb..])?);
        let job = HostBuf::new(4 * (32 + na + nb + nc))?;
        let bp = DevBuf::from_slice(&[1234u32, 0, 0, 77])?;
        k.job_post(&s, ba.ptr, na as u32, bb.ptr, nb as u32, bc.ptr, nc as u32, job.dev, bp.ptr, 50, 41)?;
        s.sync()?;
        // SAFETY: the host side of the mapped buffer, after the stream is done
        let got = unsafe { std::slice::from_raw_parts(job.ptr as *const u32, 32 + na + nb + nc) };
        let head_ok = got[0] == 77 * 64 + 50 + 1 && got[1] == 41 && got[2] == 1234;
        let wrong = (0..x.len()).filter(|&i| f32::from_bits(got[32 + i]) != x[i]).count();
        check("job_post header and floats (wrong)", (wrong + !head_ok as usize) as f64, 0.0, &mut fails);
    }

    // the prefill cache writer: K and V of 5 tokens at positions 1278.. of a ring of 1280 rows, then of 3 tokens at 7..
    // of a plain cache, their rows of 1024 padded to 1056: each value the nearest f16 (every 16th a tie, every 16th
    // after it subnormal) at its row, the other rows untouched
    for (ring, p0, nt) in [(1280usize, 1278usize, 5usize), (0, 7, 3)] {
        let (rows, stride) = (if ring > 0 { ring } else { p0 + nt + 2 }, 1056);
        let mut val = |i: usize| match i % 16 {
            0 => (2048 + 2 * (bits(&mut seed) % 1024) + 1) as f32 / 2048.0 * 2f32.powi((bits(&mut seed) % 6) as i32 - 2) * if i % 32 == 0 { -1.0 } else { 1.0 },
            1 => rnd(&mut seed) * 2f32.powi(-15),
            _ => rnd(&mut seed) * 8.0,
        };
        let (kx, vx): (Vec<f32>, Vec<f32>) = ((0..nt * stride).map(&mut val).collect(), (0..nt * stride).map(&mut val).collect());
        let (bk, bv, bp) = (DevBuf::from_slice(&kx)?, DevBuf::from_slice(&vx)?, DevBuf::from_slice(&[p0 as u32, 0, 0, 0])?);
        let (bkc, bvc) = (DevBuf::zeroed(rows * 2048, &s)?, DevBuf::zeroed(rows * 2048, &s)?);
        k.kv_store_t(&s, bk.ptr, bv.ptr, bkc.ptr, bvc.ptr, bp.ptr, ring as u32, 1024, stride as u32, stride as u32, nt as u32)?;
        s.sync()?;
        let (kc, vc): (Vec<u16>, Vec<u16>) = (bkc.download(rows * 1024)?, bvc.download(rows * 1024)?);
        let mut wrong = 0;
        for row in 0..rows {
            let tok = (0..nt).find(|&t| (if ring > 0 { (p0 + t) % ring } else { p0 + t }) == row);
            for (x, c) in [(&kx, &kc), (&vx, &vc)] {
                for i in 0..1024 {
                    let got = c[row * 1024 + i];
                    wrong += !match tok {
                        Some(t) => f16(got) == f16_round(x[t * stride + i]),
                        None => got == 0,
                    } as usize;
                }
            }
        }
        check(&format!("kv_store_t {}: values not the nearest f16", if ring > 0 { "ring" } else { "plain" }), wrong as f64, 0.0, &mut fails);
    }

    // the decode cache writer over 5 steps from a zero state: layer 0 (sliding: conv1d [c][1][4], a ring of 1280 rows
    // at positions 1278..) and layer 5 (full: static taps [4][c] plus the input-dependent ones, a plain cache); the
    // convolutions, the K clip to an L2 norm of 22.627 per head (inputs larger head by head, so some heads are clipped
    // and some not), the V RMS norm without a gain; as f32, and as f16 rows (each value within half an f16 spacing of
    // the f64 one), the other rows untouched
    for (l, ring, p0) in [(0usize, 1280usize, 1278usize), (5, 0, 3)] {
        let full = ring == 0;
        let (wn, shape) = if full { ("sconv.static_w", vec![4, kv]) } else { ("sconv.weight", vec![kv, 1, 4]) };
        let kw = ck.get(&format!("model.llm.layers.{l}.attn.k_{wn}"), Dtype::BF16, &shape)?;
        let vw = ck.get(&format!("model.llm.layers.{l}.attn.v_{wn}"), Dtype::BF16, &shape)?;
        // the weight of lag j (0 the current token) for channel c
        let wl = |w: &[u8], c: usize, j: usize| if full { bf16(w, j * kv + c) } else { bf16(w, c * 4 + 3 - j) };
        let (nx, rows) = (if full { kv + 32 } else { kv }, if ring > 0 { ring } else { p0 + 7 });
        let (bkw, bvw, bst) = (DevBuf::from_slice(kw)?, DevBuf::from_slice(vw)?, DevBuf::zeroed(2 * kv * 3 * 4, &s)?);
        let (bgk, bgv, bk, bv, bp) = (DevBuf::new(nx * 4)?, DevBuf::new(nx * 4)?, DevBuf::new(kv * 4)?, DevBuf::new(kv * 4)?, DevBuf::new(16)?);
        let (bkc, bvc) = (DevBuf::zeroed(rows * 2048, &s)?, DevBuf::zeroed(rows * 2048, &s)?);
        let (cscale, thr, eps) = (if full { 1.0 / 3.0 } else { 1.0 / 5f32.sqrt() }, 22.627f32, 1e-6f32);
        let mut hist: Vec<[Vec<f32>; 2]> = vec![];
        let (mut err, mut e16, mut clipped) = (0f64, 0f64, 0);
        for step in 0..5 {
            let gk: Vec<f32> = (0..nx).map(|i| rnd(&mut seed) * if i < kv { 0.5 + (i / 128) as f32 } else { 0.5 }).collect();
            let gv: Vec<f32> = (0..nx).map(|i| rnd(&mut seed) * if i < kv { 2.0 } else { 0.5 }).collect();
            let pos = p0 + step;
            bgk.upload(&gk)?;
            bgv.upload(&gv)?;
            bp.upload(&[pos as u32, 0, 0, 0])?;
            k.kv_prep(&s, full, bgk.ptr, bgv.ptr, bkw.ptr, bvw.ptr, bst.ptr, bk.ptr, bv.ptr, bkc.ptr, bvc.ptr, bp.ptr, ring as u32, cscale, thr, eps)?;
            s.sync()?;
            hist.push([gk, gv]);
            let conv = |x: usize, w: &[u8], c: usize| {
                let cur = &hist[hist.len() - 1][x];
                let mut o = cur[c] as f64;
                for j in 0..4.min(hist.len()) {
                    let tap = if full { cur[kv + c / 128 * 4 + j] as f64 } else { 0.0 };
                    o += (wl(w, c, j) + tap) * hist[hist.len() - 1 - j][x][c] as f64;
                }
                o * cscale as f64
            };
            let (mut kr, mut vr): (Vec<f64>, Vec<f64>) = ((0..kv).map(|c| conv(0, kw, c)).collect(), (0..kv).map(|c| conv(1, vw, c)).collect());
            for h in 0..8 {
                let ks = &mut kr[h * 128..][..128];
                let f = (thr as f64 / (ks.iter().map(|a| a * a).sum::<f64>().sqrt() + 1e-6)).clamp(0.0, 1.0);
                clipped += (f < 1.0) as usize;
                ks.iter_mut().for_each(|a| *a *= f);
                let vs = &mut vr[h * 128..][..128];
                let rms = (vs.iter().map(|a| a * a).sum::<f64>() / 128.0 + eps as f64).sqrt();
                vs.iter_mut().for_each(|a| *a /= rms);
            }
            let (kf, vf) = (bk.download::<f32>(kv)?, bv.download::<f32>(kv)?);
            err = err.max(rel_err(&kf, &kr)).max(rel_err(&vf, &vr));
            // the f16 rows: the kernel's f32 values (checked above) rounded to nearest, exactly. Against f64 they would
            // also carry the f32 error, which near 0 is many f16 spacings.
            let row = if ring > 0 { pos % ring } else { pos };
            for (c, got) in [(bkc.download::<u16>(rows * 1024)?, &kf), (bvc.download::<u16>(rows * 1024)?, &vf)] {
                for (i, &w) in got.iter().enumerate() {
                    e16 = e16.max((f16(c[row * 1024 + i]) - f16_round(w)).abs() / f16_ulp(w as f64));
                }
            }
        }
        let written: Vec<usize> = (0..5).map(|t| if ring > 0 { (p0 + t) % ring } else { p0 + t }).collect();
        for c in [bkc.download::<u16>(rows * 1024)?, bvc.download::<u16>(rows * 1024)?] {
            if (0..rows).filter(|r| !written.contains(r)).any(|r| c[r * 1024..][..1024].iter().any(|&v| v != 0)) {
                fails.push(format!("kv_prep layer {l}: a row not of the 5 positions was written"));
            }
        }
        if clipped == 0 || clipped == 40 {
            fails.push(format!("kv_prep layer {l}: {clipped} of 40 heads clipped; the check wants both kinds"));
        }
        check(&format!("kv_prep layer {l} k, v"), err, 1e-5, &mut fails);
        check(&format!("kv_prep layer {l} f16 rows: its f32 rounded (f16 spacings)"), e16, 0.0, &mut fails);
    }

    if fails.is_empty() {
        println!("all good");
        Ok(())
    } else {
        Err(format!("failed: {}", fails.join(", ")))
    }
}
