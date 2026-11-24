//! CPU expert rows against many tokens (prefill): each row is decoded once into int8 vectors and its scales, then
//! multiplied with each token's Q8_0 activations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::{Q8_0, dot_row_scalar};
use crate::experts::{Fmt, Mat};

/// out[i] = row r of `m` . xs[i] for each token i, the expert's scale left out.
pub fn dot_tokens(m: &Mat, r: usize, xs: &[&Q8_0], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") && m.k <= 4096 {
        // SAFETY: the features are present; each x holds k values
        return unsafe { dot_tokens_avx2(m, r, xs, out) };
    }
    for (o, x) in out.iter_mut().zip(xs) {
        *o = dot_row_scalar(m, r, x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_tokens_avx2(m: &Mat, r: usize, xs: &[&Q8_0], out: &mut [f32]) {
    use crate::cpu::{UE4M3, dot4, fp4_lut, h8, hsum, nv4_weights, pair};
    unsafe {
        let (q, s) = (m.qs(r), m.sc(r));
        let nb = m.k / 32;
        // the row's weights, 32 a vector, and the weight scale of each lane
        let mut w = [_mm256_setzero_si256(); 128];
        let mut ws = [_mm256_setzero_ps(); 128];
        match m.fmt {
            Fmt::Nv4 => {
                let lut = fp4_lut();
                for g in 0..nb / 2 {
                    let (lo, hi) = nv4_weights(q.as_ptr().add(g * 32), lut);
                    w[2 * g] = lo;
                    w[2 * g + 1] = hi;
                    let sc = |i: usize| UE4M3[s[4 * g + i] as usize];
                    ws[2 * g] = pair(sc(0), sc(1));
                    ws[2 * g + 1] = pair(sc(2), sc(3));
                }
            }
            Fmt::Q8 => {
                for b in 0..nb {
                    w[b] = _mm256_loadu_si256(q.as_ptr().add(b * 32) as *const __m256i);
                    ws[b] = h8(u16::from_le_bytes([s[2 * b], s[2 * b + 1]]));
                }
            }
        }
        for (o, x) in out.iter_mut().zip(xs) {
            let mut acc = _mm256_setzero_ps();
            for b in 0..nb {
                let xv = _mm256_loadu_si256(x.qs.as_ptr().add(b * 32) as *const __m256i);
                let p = _mm256_cvtepi32_ps(dot4(w[b], xv));
                acc = _mm256_fmadd_ps(p, _mm256_mul_ps(ws[b], _mm256_set1_ps(*x.d.get_unchecked(b))), acc);
            }
            *o = hsum(acc);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::tests::{random_mat, random_x};

    #[test]
    fn batched_rows_match_single_dots() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for (fmt, k) in [(Fmt::Nv4, 2304), (Fmt::Nv4, 1152), (Fmt::Q8, 2304), (Fmt::Q8, 1152)] {
            let b = random_mat(fmt, 2, k, &mut seed);
            let m = Mat { fmt, b: &b, rows: 2, k };
            let xs: Vec<Q8_0> = (0..7).map(|_| random_x(k, &mut seed).1).collect();
            let refs: Vec<&Q8_0> = xs.iter().collect();
            let mut out = vec![0f32; 7];
            dot_tokens(&m, 1, &refs, &mut out);
            for (i, x) in xs.iter().enumerate() {
                let want = dot_row_scalar(&m, 1, x);
                assert!((out[i] - want).abs() <= 1e-4 * want.abs().max(1.0), "{fmt:?} k {k} token {i}: {} vs {want}", out[i]);
            }
        }
    }
}
