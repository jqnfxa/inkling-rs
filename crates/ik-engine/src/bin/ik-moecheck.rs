//! The GPU expert kernels against the CPU ones, on random banks in the slot layout of experts.rs: `ik-moecheck`.
//! Each format and shape, 8 experts (some not cached), x shared (gate_up) or one per expert (down). Prints the
//! largest difference relative to the largest output.
use ik_engine::cpu::{Q8_0, dot_row, quantize_q8_0};
use ik_engine::cuda::{Ctx, DevBuf, Stream};
use ik_engine::experts::{Fmt, Mat};
use ik_engine::kernels::Kernels;

fn rnd(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

/// A random matrix in the slot layout, scales like the model's (NVFP4 scales near 2^-6, f16 Q8_0 scales near 0.002).
fn random_mat(fmt: Fmt, rows: usize, k: usize, seed: &mut u64) -> Vec<u8> {
    let mut b = vec![0u8; fmt.bytes(rows, k)];
    match fmt {
        Fmt::Nv4 => {
            for v in &mut b[..rows * k / 2] {
                *v = rnd(seed) as u8;
            }
            for v in &mut b[rows * k / 2..rows * k / 2 + rows * k / 16] {
                *v = 0x08 + (rnd(seed) % 40) as u8;
            }
            let o = rows * k / 2 + rows * k / 16;
            b[o..o + 4].copy_from_slice(&(0.5 + (rnd(seed) % 100) as f32 / 100.0).to_le_bytes());
        }
        Fmt::Q8 => {
            for v in &mut b[..rows * k] {
                *v = rnd(seed) as u8;
            }
            for c in b[rows * k..rows * k + rows * k / 16].chunks_exact_mut(2) {
                c.copy_from_slice(&half::f16::from_f32(0.001 + (rnd(seed) % 100) as f32 * 2e-5).to_le_bytes());
            }
        }
    }
    b
}

fn main() -> Result<(), String> {
    let _ctx = Ctx::new(0)?;
    let k = Kernels::load()?;
    let s = Stream::new()?;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let slots: [i32; 8] = [2, -1, 0, 1, -1, -1, 2, 0];
    let dslots = DevBuf::from_slice(&slots)?;
    let mut worst = 0f64;
    for fmt in [Fmt::Nv4, Fmt::Q8] {
        for (rows, kd, per_expert) in [(2304usize, 2304usize, false), (2304, 1152, true)] {
            let sb = fmt.bytes(rows, kd);
            let mats: Vec<Vec<u8>> = (0..3).map(|_| random_mat(fmt, rows, kd, &mut seed)).collect();
            let bank: Vec<u8> = mats.concat();
            let dbank = DevBuf::from_slice(&bank)?;
            // activations: one, or one per expert
            let nx = if per_expert { 8 } else { 1 };
            let xs: Vec<Q8_0> = (0..nx)
                .map(|_| {
                    let x: Vec<f32> = (0..kd).map(|_| (rnd(&mut seed) % 2000) as f32 / 1000.0 - 1.0).collect();
                    let mut q = Q8_0::default();
                    quantize_q8_0(&x, &mut q);
                    q
                })
                .collect();
            let xq: Vec<u8> = xs.iter().flat_map(|q| q.qs.iter().map(|&v| v as u8)).collect();
            let xd: Vec<f32> = xs.iter().flat_map(|q| q.d.iter().copied()).collect();
            let (dxq, dxd) = (DevBuf::from_slice(&xq)?, DevBuf::from_slice(&xd)?);
            let y = DevBuf::zeroed(8 * rows * 4, &s)?;
            let stride = if per_expert { kd as u32 } else { 0 };
            k.moe_d(&s, fmt, dbank.ptr, sb as u64, dslots.ptr, dxq.ptr, dxd.ptr, stride, y.ptr, rows as u32, kd as u32)?;
            s.sync()?;
            let got: Vec<f32> = y.download(8 * rows)?;
            let (mut big, mut err) = (0f32, 0f32);
            for (ke, &slot) in slots.iter().enumerate() {
                if slot < 0 {
                    continue;
                }
                let m = Mat { fmt, b: &mats[slot as usize], rows, k: kd };
                let x = &xs[if per_expert { ke } else { 0 }];
                for r in 0..rows {
                    let want = dot_row(&m, r, x) * m.scale();
                    big = big.max(want.abs());
                    err = err.max((got[ke * rows + r] - want).abs());
                }
            }
            let rel = err as f64 / big.max(1e-20) as f64;
            worst = worst.max(rel);
            println!("{fmt:?} {rows}x{kd} x {}: max |y| {big:.3}, max diff {err:.2e} ({rel:.2e} relative)", if per_expert { "per expert" } else { "shared" });
        }
    }
    // prefill: groups of (token, pair) per cached expert, 40 tokens in one group to cross a tile of 32
    for fmt in [Fmt::Nv4, Fmt::Q8] {
        for (rows, kd, by_pair) in [(2304usize, 2304usize, false), (2304, 1152, true)] {
            let sb = fmt.bytes(rows, kd);
            let mats: Vec<Vec<u8>> = (0..3).map(|_| random_mat(fmt, rows, kd, &mut seed)).collect();
            let dbank = DevBuf::from_slice(&mats.concat())?;
            let groups: [(u32, usize); 3] = [(2, 5), (0, 40), (1, 1)];
            let (ntok, mut pair) = (48usize, 0usize);
            let mut gbuf = vec![0u32; 384 + 2 * 64];
            let mut members = vec![]; // (slot, token, pair)
            let mut off = 0;
            for (g, &(slot, n)) in groups.iter().enumerate() {
                gbuf[3 * g] = slot;
                gbuf[3 * g + 1] = n as u32;
                gbuf[3 * g + 2] = off as u32;
                for i in 0..n {
                    let tok = (i * 7 + g) % ntok;
                    gbuf[384 + 2 * (off + i)] = tok as u32;
                    gbuf[384 + 2 * (off + i) + 1] = pair as u32;
                    members.push((slot, tok, pair));
                    pair += 1;
                }
                off += n;
            }
            let nx = if by_pair { pair } else { ntok };
            let xs: Vec<Q8_0> = (0..nx)
                .map(|_| {
                    let x: Vec<f32> = (0..kd).map(|_| (rnd(&mut seed) % 2000) as f32 / 1000.0 - 1.0).collect();
                    let mut q = Q8_0::default();
                    quantize_q8_0(&x, &mut q);
                    q
                })
                .collect();
            let xq: Vec<u8> = xs.iter().flat_map(|q| q.qs.iter().map(|&v| v as u8)).collect();
            let xd: Vec<f32> = xs.iter().flat_map(|q| q.d.iter().copied()).collect();
            let (dxq, dxd, dg) = (DevBuf::from_slice(&xq)?, DevBuf::from_slice(&xd)?, DevBuf::from_slice(&gbuf)?);
            let y = DevBuf::zeroed(pair * rows * 4, &s)?;
            k.moe_g(&s, fmt, by_pair, dbank.ptr, sb as u64, dg.ptr, dxq.ptr, dxd.ptr, y.ptr, rows as u32, kd as u32, 3, 2)?;
            s.sync()?;
            let got: Vec<f32> = y.download(pair * rows)?;
            let (mut big, mut err) = (0f32, 0f32);
            for &(slot, tok, p) in &members {
                let m = Mat { fmt, b: &mats[slot as usize], rows, k: kd };
                let x = &xs[if by_pair { p } else { tok }];
                for r in 0..rows {
                    let want = dot_row(&m, r, x) * m.scale();
                    big = big.max(want.abs());
                    err = err.max((got[p * rows + r] - want).abs());
                }
            }
            let rel = err as f64 / big.max(1e-20) as f64;
            worst = worst.max(rel);
            println!("grouped {fmt:?} {rows}x{kd} x by {}: {} pairs, max |y| {big:.3}, max diff {err:.2e} ({rel:.2e} relative)", if by_pair { "pair" } else { "token" }, members.len());
        }
    }
    println!("{}", if worst < 1e-5 { "OK" } else { "MISMATCH" });
    Ok(())
}
