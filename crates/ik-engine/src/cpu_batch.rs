//! CPU expert rows against many tokens (prefill): each row is decoded once, then multiplied with each token's
//! activations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::{Act, dot_row_scalar, simd};
use crate::experts::{Fmt, Mat};

/// out[i] = row r of `m` . xs[i] for each token i, the expert's factor left out.
pub fn dot_tokens(m: &Mat, r: usize, xs: &[&Act], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if simd() && m.k <= 4096 && m.k % 128 == 0 {
        // SAFETY: the features are present; each x holds k values
        return unsafe {
            match m.fmt {
                Fmt::NvFp4 => dot_tokens_nvfp4(m.qs(r), m.sc(r), xs, out),
                Fmt::MxFp8 => dot_tokens_mxfp8(m.qs(r), m.sc(r), xs, out),
            }
        };
    }
    for (o, x) in out.iter_mut().zip(xs) {
        *o = dot_row_scalar(m, r, x);
    }
}

/// NVFP4: the row's weights (FP4 + 12) and the scales of the lanes, per 128 values as the row kernel has them, then per
/// token its sums.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_tokens_nvfp4(q: &[u8], s: &[u8], xs: &[&Act], out: &mut [f32]) {
    use crate::cpu::{e4m3h_8, fp4_lut_u, hsum, nvfp4_sums, nvfp4_weights, scale_lanes};
    unsafe {
        let nt = q.len() / 64;
        let lut = fp4_lut_u();
        let mut w = [[_mm256_setzero_si256(); 4]; 32];
        let mut ws = [_mm256_setzero_ps(); 32];
        for it in 0..nt {
            w[it] = nvfp4_weights(q.as_ptr().add(64 * it), lut);
            ws[it] = e4m3h_8(scale_lanes(s.as_ptr().add(8 * it)));
        }
        let ones = _mm256_set1_epi16(1);
        for (o, x) in out.iter_mut().zip(xs) {
            let x = &x.q;
            let (xp, dp, cp) = (x.qs.as_ptr(), x.dv.as_ptr(), x.c12.as_ptr());
            let mut acc = _mm256_setzero_ps();
            for it in 0..nt {
                let h = nvfp4_sums!(w[it], xp.add(128 * it), cp.add(8 * it), |a, b| _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), ones));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(h), _mm256_mul_ps(ws[it], _mm256_loadu_ps(dp.add(8 * it))), acc);
            }
            *o = hsum(acc);
        }
    }
}

/// MXFP8: the row as f32 (each value times its block's scale, exactly), then per token an f32 dot.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_tokens_mxfp8(q: &[u8], s: &[u8], xs: &[&Act], out: &mut [f32]) {
    use crate::cpu::{e4m3x32, hsum, ue8m0_x256};
    unsafe {
        let k = q.len();
        let mut w = [_mm256_setzero_ps(); 512];
        for (b, &sc) in s.iter().enumerate() {
            let v = e4m3x32(q.as_ptr().add(32 * b));
            let f = _mm256_set1_ps(ue8m0_x256(sc));
            for j in 0..4 {
                w[4 * b + j] = _mm256_mul_ps(v[j], f);
            }
        }
        for (o, x) in out.iter_mut().zip(xs) {
            let xp = x.f.as_ptr();
            let mut acc = [_mm256_setzero_ps(); 4];
            for v in 0..k / 32 {
                for (j, a) in acc.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(w[4 * v + j], _mm256_loadu_ps(xp.add(32 * v + 8 * j)), *a);
                }
            }
            *o = hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::tests::{act, random_mat, random_x};

    #[test]
    fn batched_rows_match_single_dots() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for (fmt, k) in [(Fmt::NvFp4, 2304), (Fmt::NvFp4, 1152), (Fmt::MxFp8, 2304), (Fmt::MxFp8, 1152)] {
            let (q, s) = random_mat(fmt, 2, k, true, &mut seed);
            let m = Mat { fmt, q: &q, s: &s, factor: 1.0, rows: 2, k };
            let xs: Vec<Act> = (0..7).map(|_| act(&random_x(k, &mut seed), fmt)).collect();
            let refs: Vec<&Act> = xs.iter().collect();
            let mut out = vec![0f32; 7];
            dot_tokens(&m, 1, &refs, &mut out);
            for (i, x) in xs.iter().enumerate() {
                let want = dot_row_scalar(&m, 1, x);
                assert!((out[i] - want).abs() <= 1e-5 * want.abs().max(1.0), "{fmt:?} k {k} token {i}: {} vs {want}", out[i]);
            }
        }
    }
}
