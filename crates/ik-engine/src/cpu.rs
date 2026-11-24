//! CPU kernels of the experts that miss the GPU cache: NVFP4 and Q8_0 rows (in the slot layout of experts.rs) against
//! Q8_0 activations, AVX2.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use crate::experts::{Fmt, Mat};
use crate::quant::FP4;

/// Activations quantized as Q8_0: per 32 values, d, 32 int8 and their sum.
#[derive(Default)]
pub struct Q8_0 {
    pub d: Vec<f32>,
    pub qs: Vec<i8>,
    pub sums: Vec<i32>,
}

/// ggml quantize_row_q8_0_ref (d rounded to f16 as ggml stores it)
pub fn quantize_q8_0(x: &[f32], out: &mut Q8_0) {
    let nb = x.len() / 32;
    out.d.resize(nb, 0.0);
    out.qs.resize(nb * 32, 0);
    out.sums.resize(nb, 0);
    for b in 0..nb {
        let (d, s) = quantize_q8_0_block(&x[b * 32..(b + 1) * 32], &mut out.qs[b * 32..(b + 1) * 32]);
        out.d[b] = d;
        out.sums[b] = s;
    }
}

/// One block of [`quantize_q8_0`]: writes the 32 int8 and returns d and their sum.
pub fn quantize_q8_0_block(xb: &[f32], qs: &mut [i8]) -> (f32, i32) {
    let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    let mut s = 0;
    for (q, &v) in qs.iter_mut().zip(xb) {
        *q = (v * id).round() as i8;
        s += *q as i32;
    }
    (f16::from_f32(d).to_f32(), s)
}

/// The UE4M3 scales as f32, halved as ggml decodes them (quant::ue4m3), by their byte.
pub static UE4M3: [f32; 256] = {
    let mut t = [0f32; 256];
    let mut x = 0;
    while x < 256 {
        let (e, m) = ((x >> 3) & 0xF, x & 7);
        t[x] = if x == 0 || x == 0x7F {
            0.0
        } else if e == 0 {
            m as f32 * f32::from_bits(117 << 23) // m 2^-9, halved
        } else {
            f32::from_bits((((e + 119) << 23) | (m << 20)) as u32)
        };
        x += 1;
    }
    t
};

#[inline]
fn simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    return is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
    #[allow(unreachable_code)]
    false
}

/// Row r of `m` . x, the expert's scale left out (the caller applies m.scale()).
pub fn dot_row(m: &Mat, r: usize, x: &Q8_0) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the features are present; x holds k values
        return unsafe {
            match m.fmt {
                Fmt::Nv4 => dot_nv4_avx2(m.qs(r), m.sc(r), x),
                Fmt::Q8 => dot_q8_avx2(m.qs(r), m.sc(r), x),
            }
        };
    }
    dot_row_scalar(m, r, x)
}

/// The scalar reference of [`dot_row`].
pub fn dot_row_scalar(m: &Mat, r: usize, x: &Q8_0) -> f32 {
    let (q, s) = (m.qs(r), m.sc(r));
    let mut acc = 0f32;
    match m.fmt {
        Fmt::Nv4 => {
            for g in 0..m.k / 64 {
                for half in 0..2 {
                    // values 64g + 32 half + i: nibble `half` of byte i of the group, activation block 2g + half
                    let b = 2 * g + half;
                    for sub in 0..2 {
                        let mut sum = 0i32;
                        for i in sub * 16..sub * 16 + 16 {
                            let byte = q[g * 32 + i];
                            let w = FP4[if half == 0 { byte & 0xF } else { byte >> 4 } as usize] as i32;
                            sum += w * x.qs[b * 32 + i] as i32;
                        }
                        acc += sum as f32 * UE4M3[s[b * 2 + sub] as usize] * x.d[b];
                    }
                }
            }
        }
        Fmt::Q8 => {
            for b in 0..m.k / 32 {
                let sum: i32 = (0..32).map(|i| q[b * 32 + i] as i8 as i32 * x.qs[b * 32 + i] as i32).sum();
                acc += sum as f32 * f16::from_le_bytes([s[2 * b], s[2 * b + 1]]).to_f32() * x.d[b];
            }
        }
    }
    acc
}

/// The FP4 table in both 128-bit lanes, for pshufb.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn fp4_lut() -> __m256i {
    let t = |i: usize| FP4[i] as i8;
    _mm256_setr_epi8(
        t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12), t(13), t(14), t(15),
        t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12), t(13), t(14), t(15),
    )
}

/// The 32 int8 weights of the low (values 0..31 of a group) and high (32..63) nibbles of 32 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn nv4_weights(q: *const u8, lut: __m256i) -> (__m256i, __m256i) {
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let raw = _mm256_loadu_si256(q as *const __m256i);
        let lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(raw, m4));
        let hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4));
        (lo, hi)
    }
}

/// Sums of 4 consecutive products of int8 w and x, in 8 lanes. |w| through sign (-128 stays 128 as unsigned) and
/// |x| <= 127 keep each pair of products within i16.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn dot4(w: __m256i, x: __m256i) -> __m256i {
    _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w)), _mm256_set1_epi16(1))
}

/// Two scales, each in 4 lanes: [a a a a b b b b].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn pair(a: f32, b: f32) -> __m256 {
    _mm256_set_m128(_mm_set1_ps(b), _mm_set1_ps(a))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_nv4_avx2(q: &[u8], s: &[u8], x: &Q8_0) -> f32 {
    unsafe {
        let lut = fp4_lut();
        let mut acc = _mm256_setzero_ps();
        for g in 0..q.len() / 32 {
            let (lo, hi) = nv4_weights(q.as_ptr().add(g * 32), lut);
            let xl = _mm256_loadu_si256(x.qs.as_ptr().add(g * 64) as *const __m256i);
            let xh = _mm256_loadu_si256(x.qs.as_ptr().add(g * 64 + 32) as *const __m256i);
            let (d0, d1) = (*x.d.get_unchecked(2 * g), *x.d.get_unchecked(2 * g + 1));
            let sc = |i: usize| *UE4M3.get_unchecked(*s.get_unchecked(4 * g + i) as usize);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot4(lo, xl)), _mm256_mul_ps(pair(sc(0), sc(1)), _mm256_set1_ps(d0)), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot4(hi, xh)), _mm256_mul_ps(pair(sc(2), sc(3)), _mm256_set1_ps(d1)), acc);
        }
        hsum(acc)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q8_avx2(q: &[u8], s: &[u8], x: &Q8_0) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for b in 0..q.len() / 32 {
            let w = _mm256_loadu_si256(q.as_ptr().add(b * 32) as *const __m256i);
            let xv = _mm256_loadu_si256(x.qs.as_ptr().add(b * 32) as *const __m256i);
            let dw = h8(u16::from_le_bytes([*s.get_unchecked(2 * b), *s.get_unchecked(2 * b + 1)]));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot4(w, xv)), _mm256_mul_ps(dw, _mm256_set1_ps(*x.d.get_unchecked(b))), acc);
        }
        hsum(acc)
    }
}

/// An f16 in each of 8 lanes, by F16C. A broadcast, not an insert: an insert into a register merges with what was in
/// it, and LLVM picks the accumulator, which chains every block of a dot to the last one.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
pub(crate) fn h8(bits: u16) -> __m256 {
    _mm256_cvtph_ps(_mm_set1_epi16(bits as i16))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hsum(v: __m256) -> f32 {
    hsum128(_mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum128(s: __m128) -> f32 {
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::quant::{dot, ue4m3};

    pub fn rnd(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// A random matrix of `rows` rows of k in the slot layout of `fmt`, with sane scales.
    pub fn random_mat(fmt: Fmt, rows: usize, k: usize, seed: &mut u64) -> Vec<u8> {
        let mut b = vec![0u8; fmt.bytes(rows, k)];
        match fmt {
            Fmt::Nv4 => {
                for v in &mut b[..rows * k / 2] {
                    *v = rnd(seed) as u8;
                }
                for v in &mut b[rows * k / 2..rows * k / 2 + rows * k / 16] {
                    *v = 0x20 + (rnd(seed) % 16) as u8; // 2^-3 .. 2^-2 or so
                }
                let o = rows * k / 2 + rows * k / 16;
                b[o..o + 4].copy_from_slice(&1.5f32.to_le_bytes());
            }
            Fmt::Q8 => {
                for v in &mut b[..rows * k] {
                    *v = (rnd(seed) % 255) as u8; // not -128
                    if *v == 0x80 {
                        *v = 0;
                    }
                }
                for c in b[rows * k..rows * k + rows * k / 16].chunks_exact_mut(2) {
                    c.copy_from_slice(&f16::from_f32(0.002 + (rnd(seed) % 100) as f32 * 1e-5).to_le_bytes());
                }
            }
        }
        b
    }

    pub fn random_x(k: usize, seed: &mut u64) -> (Vec<f32>, Q8_0) {
        let x: Vec<f32> = (0..k).map(|_| (rnd(seed) % 2000) as f32 / 1000.0 - 1.0).collect();
        let mut q = Q8_0::default();
        quantize_q8_0(&x, &mut q);
        (x, q)
    }

    #[test]
    fn ue4m3_table() {
        for x in 0..=255u8 {
            assert_eq!(UE4M3[x as usize], ue4m3(x), "{x:#x}");
        }
    }

    #[test]
    fn dots_match_dequant() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for (fmt, k) in [(Fmt::Nv4, 2304), (Fmt::Nv4, 1152), (Fmt::Q8, 2304), (Fmt::Q8, 1152)] {
            let b = random_mat(fmt, 3, k, &mut seed);
            let m = Mat { fmt, b: &b, rows: 3, k };
            let (x, q) = random_x(k, &mut seed);
            for r in 0..3 {
                let mut w = vec![0f32; k];
                m.dequant_row(r, &mut w);
                let want = dot(&w, &x);
                let (a, s) = (dot_row(&m, r, &q), dot_row_scalar(&m, r, &q));
                assert!((a - s).abs() <= 1e-4 * s.abs().max(1.0), "{fmt:?} k {k}: avx2 {a} scalar {s}");
                assert!((a - want).abs() <= 0.02 * want.abs().max(1.0), "{fmt:?} k {k}: {a} vs f32 {want}");
            }
        }
    }
}
