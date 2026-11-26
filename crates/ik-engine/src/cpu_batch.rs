//! CPU expert rows against many tokens (prefill): each row is decoded once to f32, then multiplied with each token's
//! activations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::{dot_row_scalar, simd};
use crate::experts::{Fmt, Mat};

/// Vectors of k in a block of dot_rows: the block of 3 decoded rows (6 KB) stays in L1 while each group of 4 tokens
/// passes it.
const KB: usize = 64;

/// out[j * xs.len() + i] = row r0 + j of `m` . xs[i], for the rows j in 0..nr and each token i (k values each), the
/// expert's factor left out. The rows are decoded 3 at a time, and taken against 4 tokens at a time with the 12 sums
/// in registers: 7 loads for 12 FMAs, and each token's vector loaded once for the 3 rows.
pub fn dot_rows(m: &Mat, r0: usize, nr: usize, xs: &[&[f32]], out: &mut [f32]) {
    let nt = xs.len();
    assert!(r0 + nr <= m.rows && out.len() >= nr * nt && xs.iter().all(|x| x.len() >= m.k), "rows {r0}+{nr} of {}, {} outputs for {nt} tokens", m.rows, out.len());
    #[cfg(target_arch = "x86_64")]
    if simd() && m.k <= 4096 && m.k % 64 == 0 {
        // SAFETY: the features are present; each x holds k values, and each decoded row is k / 8 vectors of the 512
        unsafe {
            let mut buf = std::mem::MaybeUninit::<[[__m256; 512]; 3]>::uninit();
            let wp = buf.as_mut_ptr() as *mut __m256;
            let mut acc = vec![[[_mm256_setzero_ps(); 4]; 3]; nt.div_ceil(4)];
            for j in (0..nr).step_by(3) {
                let g = (nr - j).min(3);
                for i in 0..g {
                    let (q, s, w) = (m.qs(r0 + j + i), m.sc(r0 + j + i), wp.add(512 * i));
                    match m.fmt {
                        Fmt::NvFp4 => row_nvfp4(q, s, w),
                        Fmt::MxFp8 => row_mxfp8(q, s, w),
                    }
                }
                rows_dot(wp, g, m.k / 8, xs, &mut acc, &mut out[j * nt..(j + g) * nt]);
            }
            return;
        }
    }
    for j in 0..nr {
        for (i, x) in xs.iter().enumerate() {
            out[j * nt + i] = dot_row_scalar(m, r0 + j, x);
        }
    }
}

/// The g (1 to 3) rows of nv vectors at w (512 vectors apart) against every token of xs, into out [g][xs.len()]: by
/// blocks of KB vectors, and in each block by tiles of the g rows and 4 tokens, whose sums wait in acc between blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_dot(w: *const __m256, g: usize, nv: usize, xs: &[&[f32]], acc: &mut [[[__m256; 4]; 3]], out: &mut [f32]) {
    use crate::cpu::hsum;
    let nt = xs.len();
    unsafe {
        acc.fill([[_mm256_setzero_ps(); 4]; 3]);
        for v0 in (0..nv).step_by(KB) {
            let v1 = (v0 + KB).min(nv);
            for (c, a) in acc.iter_mut().enumerate() {
                let x = &xs[4 * c..(4 * c + 4).min(nt)];
                match g {
                    1 => tile_t::<1>(w, x, v0, v1, a),
                    2 => tile_t::<2>(w, x, v0, v1, a),
                    _ => tile_t::<3>(w, x, v0, v1, a),
                }
            }
        }
        for (j, oj) in out.chunks_mut(nt).take(g).enumerate() {
            for (i, o) in oj.iter_mut().enumerate() {
                *o = hsum(acc[i / 4][j][i % 4]);
            }
        }
    }
}

/// tile::<R, T> for the T = x.len() (1 to 4) tokens.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn tile_t<const R: usize>(w: *const __m256, x: &[&[f32]], v0: usize, v1: usize, a: &mut [[__m256; 4]; 3]) {
    unsafe {
        match x.len() {
            1 => tile::<R, 1>(w, x, v0, v1, a),
            2 => tile::<R, 2>(w, x, v0, v1, a),
            3 => tile::<R, 3>(w, x, v0, v1, a),
            _ => tile::<R, 4>(w, x, v0, v1, a),
        }
    }
}

/// a[j][i] += row j . token i over the vectors v0..v1, for the rows j < R at w (512 vectors apart) and the tokens
/// i < T of x, the R T sums in registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn tile<const R: usize, const T: usize>(w: *const __m256, x: &[&[f32]], v0: usize, v1: usize, a: &mut [[__m256; 4]; 3]) {
    unsafe {
        let xp: [*const f32; T] = std::array::from_fn(|i| x[i].as_ptr());
        let mut s: [[__m256; T]; R] = std::array::from_fn(|j| std::array::from_fn(|i| a[j][i]));
        for v in v0..v1 {
            let wv: [__m256; R] = std::array::from_fn(|j| *w.add(512 * j + v));
            for (i, p) in xp.iter().enumerate() {
                let xv = _mm256_loadu_ps(p.add(8 * v));
                for (sj, wj) in s.iter_mut().zip(&wv) {
                    sj[i] = _mm256_fmadd_ps(*wj, xv, sj[i]);
                }
            }
        }
        for (aj, sj) in a.iter_mut().zip(&s) {
            aj[..T].copy_from_slice(sj);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::tests::{dot_f64, random_mat, random_x};

    /// dot_rows on 1 to 8 rows (groups of 3 and what is left) and 1 to 9 tokens (groups of 4 and what is left), rows
    /// from an offset, against f64.
    #[test]
    fn row_tiles_match_dequant() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for (fmt, k) in [(Fmt::NvFp4, 2304), (Fmt::NvFp4, 1152), (Fmt::MxFp8, 2304), (Fmt::MxFp8, 1152)] {
            let (q, s) = random_mat(fmt, 10, k, true, &mut seed);
            let m = Mat { fmt, q: &q, s: &s, factor: 1.0, rows: 10, k };
            let xs: Vec<Vec<f32>> = (0..9).map(|_| random_x(k, &mut seed)).collect();
            for (nr, nt) in [(1, 1), (2, 3), (3, 4), (4, 5), (5, 8), (8, 9), (7, 2)] {
                let refs: Vec<&[f32]> = xs[..nt].iter().map(|x| &x[..]).collect();
                let mut out = vec![f32::NAN; nr * nt];
                dot_rows(&m, 2, nr, &refs, &mut out);
                for j in 0..nr {
                    for (i, x) in refs.iter().enumerate() {
                        let (want, l1) = dot_f64(&m, 2 + j, x);
                        let got = out[j * nt + i] as f64;
                        assert!((got - want).abs() <= 1e-5 * l1, "{fmt:?} k {k}, {nr} rows {nt} tokens: row {j} token {i}: {got} vs f64 {want} (sum of |w x| {l1})");
                    }
                }
            }
        }
    }
}
