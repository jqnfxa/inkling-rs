//! Kernel equivalence: each 768-thread kernel against the general one it replaces, on the same random inputs.
//! Prints the largest relative difference of each output. `ik-kcmp`
use ik_engine::cuda::{Ctx, DevBuf, Stream};
use ik_engine::kernels::Kernels;

fn rnd(seed: &mut u64) -> f32 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn diff(a: &[f32], b: &[f32]) -> f64 {
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-20) as f64;
    a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).fold(0.0, f64::max) / scale
}

fn main() -> Result<(), String> {
    let _ctx = Ctx::new(0)?;
    let k = Kernels::load()?;
    let s = Stream::new()?;
    let n = 2304usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut v = |len: usize| -> Vec<f32> { (0..len).map(|_| rnd(&mut seed) * 3.0).collect() };
    let x = v(8 * n);
    let w = v(n);
    let qv = v(3 * n);
    let bx = DevBuf::from_slice(&x)?;
    let bw = DevBuf::from_slice(&w)?;
    let bq = DevBuf::from_slice(&qv)?;
    let outs = |m: usize| -> Result<Vec<DevBuf>, String> { (0..m).map(|_| DevBuf::zeroed(4 * 3 * 2 * n, &s)).collect() };
    let (o1, o2) = (outs(4)?, outs(4)?);
    let get = |b: &DevBuf, len: usize| -> Result<Vec<f32>, String> { b.download(len) };
    let qf = |b: &DevBuf| -> Result<Vec<f32>, String> {
        let raw: Vec<u8> = b.download(n)?;
        Ok(raw.iter().map(|&c| c as i8 as f32).collect())
    };

    // norm_q8: n = 2304 picks norm_q8_768; the general kernel via n = 2304 with IK_OLD_NORM is the same launcher, so call
    // both kernels through the two n values the launcher dispatches on
    for (name, b) in [("norm_q8 (256 threads)", &o1), ("norm_q8_768", &o2)] {
        if name.contains("768") {
            unsafe { std::env::remove_var("IK_OLD_NORM") };
        } else {
            unsafe { std::env::set_var("IK_OLD_NORM", "1") };
        }
        k.norm_q8(&s, bx.ptr, bw.ptr, b[0].ptr, b[1].ptr, b[2].ptr, b[3].ptr, n as u32, 1e-6, 2, n as u32)?;
    }
    s.sync()?;
    let (y1, y2) = (get(&o1[0], 2 * n)?, get(&o2[0], 2 * n)?);
    println!("norm_q8 samples: old {:?} new {:?}; equal bits {}", &y1[..3], &y2[..3], y1.iter().zip(&y2).filter(|(a, b)| a.to_bits() == b.to_bits()).count());
    println!("norm_q8: y {:.2e}, q {:.2e}, d {:.2e}, s {:.2e}", diff(&get(&o1[0], 2 * n)?, &get(&o2[0], 2 * n)?), diff(&qf(&o1[1])?, &qf(&o2[1])?), diff(&get(&o1[2], 144)?, &get(&o2[2], 144)?), diff(&get(&o1[3], 144)?, &get(&o2[3], 144)?));

    // mix: 5 states, 3 queries
    let ptrs: Vec<u64> = (0..5).map(|j| bx.ptr + (j * n * 4) as u64).collect();
    let bp = DevBuf::from_slice(&ptrs)?;
    for (old, b) in [(true, &o1), (false, &o2)] {
        if old {
            unsafe { std::env::set_var("IK_OLD_MIX", "1") };
        } else {
            unsafe { std::env::remove_var("IK_OLD_MIX") };
        }
        k.mix_t(&s, bp.ptr, 5, bq.ptr, b[0].ptr, 3, n as u32, 1)?;
    }
    s.sync()?;
    println!("mix: out {:.2e}", diff(&get(&o1[0], 3 * n)?, &get(&o2[0], 3 * n)?));

    // controller input: 2 controllers
    let tab: Vec<u64> = vec![bx.ptr, bx.ptr + (n * 4) as u64, bx.ptr + (2 * n * 4) as u64, bx.ptr + (3 * n * 4) as u64];
    let bt = DevBuf::from_slice(&tab)?;
    for (old, b) in [(true, &o1), (false, &o2)] {
        if old {
            unsafe { std::env::set_var("IK_OLD_CTRL", "1") };
        } else {
            unsafe { std::env::remove_var("IK_OLD_CTRL") };
        }
        k.ctrl_in_qz(&s, bt.ptr, b[1].ptr, b[2].ptr, b[3].ptr, n as u32, 2)?;
    }
    s.sync()?;
    let q1: Vec<u8> = o1[1].download(4 * n)?;
    let q2: Vec<u8> = o2[1].download(4 * n)?;
    let qd = q1.iter().zip(&q2).filter(|(a, b)| a != b).count();
    println!("ctrl_in: q bytes differing {qd} of {}, d {:.2e}, s {:.2e}", 4 * n, diff(&get(&o1[2], 288)?, &get(&o2[2], 288)?), diff(&get(&o1[3], 288)?, &get(&o2[3], 288)?));
    Ok(())
}
