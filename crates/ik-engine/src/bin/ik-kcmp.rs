//! Kernel equivalence: mix768, the depth mix of n = 2304 in 768 threads, against mix8, the general kernel it replaces
//! (IK_OLD_MIX), on the same random inputs. Prints the largest relative difference of the output. `ik-kcmp`
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

    // 5 states, 3 queries
    let x = v(5 * n);
    let qv = v(3 * n);
    let bx = DevBuf::from_slice(&x)?;
    let bq = DevBuf::from_slice(&qv)?;
    let ptrs: Vec<u64> = (0..5).map(|j| bx.ptr + (j * n * 4) as u64).collect();
    let bp = DevBuf::from_slice(&ptrs)?;
    let (o1, o2) = (DevBuf::zeroed(4 * 3 * n, &s)?, DevBuf::zeroed(4 * 3 * n, &s)?);
    for (old, o) in [(true, &o1), (false, &o2)] {
        if old {
            unsafe { std::env::set_var("IK_OLD_MIX", "1") };
        } else {
            unsafe { std::env::remove_var("IK_OLD_MIX") };
        }
        k.mix_t(&s, bp.ptr, 5, bq.ptr, o.ptr, 3, n as u32, 1)?;
    }
    s.sync()?;
    let (y1, y2): (Vec<f32>, Vec<f32>) = (o1.download(3 * n)?, o2.download(3 * n)?);
    println!("mix: out {:.2e}", diff(&y1, &y2));
    Ok(())
}
