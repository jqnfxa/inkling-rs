//! The GPU kernels that read the checkpoint's weights, on real tensors, against references on the CPU (f64):
//! `ik-wcheck <checkpoint dir>`. Prints the largest error of each relative to the largest output, and fails past a
//! bound. The references are scalar Rust in f64, with no SIMD: they take the tensors' bytes from `Checkpoint` and
//! decode the stored formats themselves, and take the same f32 inputs as the kernels.
use ik_engine::checkpoint::{Checkpoint, Dtype};
use ik_engine::cpu::dot_row;
use ik_engine::cuda::{Ctx, DevBuf, Stream};
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
                k.moe_nvfp4_d(&s, bb.ptr, slot as u64, bslots.ptr, bids.ptr, ba.ptr, bn.ptr, bx.ptr, xstride as u32, by.ptr, rows as u32, kk as u32)?;
            } else {
                k.moe_mxfp8_d(&s, bb.ptr, slot as u64, bslots.ptr, bx.ptr, xstride as u32, by.ptr, rows as u32, kk as u32)?;
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
        k.moe_nvfp4_d(&s, bb.ptr, slot as u64, bs8.ptr, bi8.ptr, ba.ptr, bn.ptr, bx.ptr, 0, by.ptr, rows as u32, kk as u32)?;
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
                let got: Vec<f32> = (0..rows).map(|r| dot_row(&m, r, &x) * m.scale()).collect();
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
    k.situ(&s, bgu.ptr, bslots.ptr, baf.ptr, nff as u32, 4.0, 25.0)?;
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

    if fails.is_empty() {
        println!("all good");
        Ok(())
    } else {
        Err(format!("failed: {}", fails.join(", ")))
    }
}
