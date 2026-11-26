//! CPU expert rows against many tokens (prefill): each row is decoded once to f32, then multiplied with each token's
//! activations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::{dot_row_scalar, simd};
use crate::experts::{Fmt, Mat};

/// out[i] = row r of `m` . xs[i] for each token i (k values each), the expert's factor left out.
pub fn dot_tokens(m: &Mat, r: usize, xs: &[&[f32]], out: &mut [f32]) {
    assert!(out.len() == xs.len() && xs.iter().all(|x| x.len() >= m.k), "{} outputs for {} tokens of {} values", out.len(), xs.len(), m.k);
    #[cfg(target_arch = "x86_64")]
    if simd() && m.k <= 4096 && m.k % 64 == 0 {
        // SAFETY: the features are present; each x holds k values
        return unsafe {
            let mut buf = std::mem::MaybeUninit::<[__m256; 512]>::uninit();
            let wp = buf.as_mut_ptr() as *mut __m256;
            match m.fmt {
                Fmt::NvFp4 => row_nvfp4(m.qs(r), m.sc(r), wp),
                Fmt::MxFp8 => row_mxfp8(m.qs(r), m.sc(r), wp),
            }
            // the k / 8 vectors written
            dot_f32(std::slice::from_raw_parts(wp, m.k / 8), xs, out)
        };
    }
    for (o, x) in out.iter_mut().zip(xs) {
        *o = dot_row_scalar(m, r, x);
    }
}

/// An NVFP4 row as f32 at w (k / 8 vectors): each value times its block's scale, which is exact (2 significant bits
/// times 4).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_nvfp4(q: &[u8], s: &[u8], w: *mut __m256) {
    use crate::cpu::{E4M3_X2_14, e2m1_lut, e2m1x64};
    unsafe {
        let lut = e2m1_lut();
        for c in 0..q.len() / 32 {
            let v = e2m1x64(q.as_ptr().add(32 * c), lut);
            for (j, v) in v.iter().enumerate() {
                w.add(8 * c + j).write(_mm256_mul_ps(*v, _mm256_set1_ps(E4M3_X2_14[(s[4 * c + j / 2] & 0x7F) as usize])));
            }
        }
    }
}

/// An MXFP8 row as f32 at w (k / 8 vectors): each value times its block's scale, exactly.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn row_mxfp8(q: &[u8], s: &[u8], w: *mut __m256) {
    use crate::cpu::{e4m3x32, ue8m0_x256};
    unsafe {
        for (b, &sc) in s.iter().enumerate() {
            let v = e4m3x32(q.as_ptr().add(32 * b));
            let f = _mm256_set1_ps(ue8m0_x256(sc));
            for (j, v) in v.iter().enumerate() {
                w.add(4 * b + j).write(_mm256_mul_ps(*v, f));
            }
        }
    }
}

/// out[i] = w . xs[i], w a row of k values as f32 (k a multiple of 32): 4 tokens at a time, each vector of w read once
/// for them, 2 sums a token; then the rest one at a time, 4 sums each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32(w: &[__m256], xs: &[&[f32]], out: &mut [f32]) {
    use crate::cpu::hsum;
    unsafe {
        let mut t = 0;
        while t + 4 <= xs.len() {
            let xp: [*const f32; 4] = std::array::from_fn(|j| xs[t + j].as_ptr());
            let mut acc = [[_mm256_setzero_ps(); 2]; 4];
            for v in 0..w.len() / 2 {
                let (w0, w1) = (w[2 * v], w[2 * v + 1]);
                for (a, x) in acc.iter_mut().zip(xp) {
                    a[0] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x.add(16 * v)), a[0]);
                    a[1] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x.add(16 * v + 8)), a[1]);
                }
            }
            for (j, a) in acc.iter().enumerate() {
                out[t + j] = hsum(_mm256_add_ps(a[0], a[1]));
            }
            t += 4;
        }
        for (o, x) in out[t..].iter_mut().zip(&xs[t..]) {
            let xp = x.as_ptr();
            let mut acc = [_mm256_setzero_ps(); 4];
            for v in 0..w.len() / 4 {
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
    use crate::cpu::tests::{dot_f64, random_mat, random_x};

    #[test]
    fn batched_rows_match_dequant() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for (fmt, k) in [(Fmt::NvFp4, 2304), (Fmt::NvFp4, 1152), (Fmt::MxFp8, 2304), (Fmt::MxFp8, 1152)] {
            let (q, s) = random_mat(fmt, 2, k, true, &mut seed);
            let m = Mat { fmt, q: &q, s: &s, factor: 1.0, rows: 2, k };
            let xs: Vec<Vec<f32>> = (0..7).map(|_| random_x(k, &mut seed)).collect();
            let refs: Vec<&[f32]> = xs.iter().map(|x| &x[..]).collect();
            let mut out = vec![0f32; 7];
            for r in 0..2 {
                dot_tokens(&m, r, &refs, &mut out);
                for (i, x) in xs.iter().enumerate() {
                    let (want, l1) = dot_f64(&m, r, x);
                    assert!((out[i] as f64 - want).abs() <= 1e-5 * l1, "{fmt:?} k {k} row {r} token {i}: {} vs f64 {want} (sum of |w x| {l1})", out[i]);
                }
            }
        }
    }
}
