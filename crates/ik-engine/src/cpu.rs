//! CPU kernels of the experts that miss the GPU cache: Q4_K x Q8_K and Q5_0 x Q8_0 row dots (AVX2), as ggml computes them.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

/// Activations quantized for Q4_K rows: per 256 values, d, 256 int8, and 16 sums of 16.
#[derive(Default)]
pub struct Q8K {
    pub d: Vec<f32>,
    pub qs: Vec<i8>,
    pub bsums: Vec<i16>,
}

/// Activations quantized for Q5_0 rows: per 32 values, d, 32 int8 and their sum.
#[derive(Default)]
pub struct Q8_0 {
    pub d: Vec<f32>,
    pub qs: Vec<i8>,
    pub sums: Vec<i32>,
}

/// ggml quantize_row_q8_K_ref (AVX2 when present: rounding half to even there, half away from zero in the scalar).
pub fn quantize_q8k(x: &[f32], out: &mut Q8K) {
    let nb = x.len() / 256;
    out.d.resize(nb, 0.0);
    out.qs.resize(nb * 256, 0);
    out.bsums.resize(nb * 16, 0);
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the features are present; out is sized for x.
        unsafe { quantize_q8k_avx2(x, out) };
        return;
    }
    for b in 0..nb {
        let xb = &x[b * 256..(b + 1) * 256];
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xb {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let qs = &mut out.qs[b * 256..(b + 1) * 256];
        if amax == 0.0 {
            out.d[b] = 0.0;
            qs.fill(0);
            out.bsums[b * 16..(b + 1) * 16].fill(0);
            continue;
        }
        let iscale = -127.0 / max;
        for (q, &v) in qs.iter_mut().zip(xb) {
            *q = ((iscale * v).round() as i32).min(127) as i8;
        }
        for j in 0..16 {
            out.bsums[b * 16 + j] = qs[j * 16..(j + 1) * 16].iter().map(|&q| q as i16).sum();
        }
        out.d[b] = 1.0 / iscale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn quantize_q8k_avx2(x: &[f32], out: &mut Q8K) {
    unsafe {
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let bias = _mm_set1_epi8(-128);
        for b in 0..x.len() / 256 {
            let xb = x.as_ptr().add(b * 256);
            let (mut mx, mut mn) = (_mm256_set1_ps(f32::NEG_INFINITY), _mm256_set1_ps(f32::INFINITY));
            for i in 0..32 {
                let v = _mm256_loadu_ps(xb.add(8 * i));
                mx = _mm256_max_ps(mx, v);
                mn = _mm256_min_ps(mn, v);
            }
            let mx = hmax(mx);
            let mn = -hmax(_mm256_sub_ps(_mm256_setzero_ps(), mn));
            let max = if -mn > mx { mn } else { mx };
            let qs = out.qs.as_mut_ptr().add(b * 256);
            if max == 0.0 {
                out.d[b] = 0.0;
                std::ptr::write_bytes(qs, 0, 256);
                out.bsums[b * 16..(b + 1) * 16].fill(0);
                continue;
            }
            let iscale = -127.0 / max;
            let vs = _mm256_set1_ps(iscale);
            for i in 0..8 {
                let q = |k: usize| _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(xb.add(32 * i + 8 * k))));
                let ab = _mm256_packs_epi32(q(0), q(1));
                let cd = _mm256_packs_epi32(q(2), q(3));
                let r = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(ab, cd), perm);
                _mm256_storeu_si256(qs.add(32 * i) as *mut __m256i, r);
            }
            // sums of 16: psadbw of the bytes biased to unsigned, less 16 * 128
            for j in 0..16 {
                let v = _mm_xor_si128(_mm_loadu_si128(qs.add(16 * j) as *const __m128i), bias);
                let sad = _mm_sad_epu8(v, _mm_setzero_si128());
                let sum = _mm_cvtsi128_si32(sad) + _mm_extract_epi16(sad, 4);
                out.bsums[b * 16 + j] = (sum - 2048) as i16;
            }
            out.d[b] = 1.0 / iscale;
        }
    }
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

#[inline]
fn hf(b: &[u8], o: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32()
}

fn scales_mins(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut s = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..8 {
        if j < 4 {
            s[j] = sc[j] & 63;
            m[j] = sc[j + 4] & 63;
        } else {
            s[j] = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4);
            m[j] = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        }
    }
    (s, m)
}

/// Whether the AVX2 kernels can run: AVX2, FMA and F16C.
#[cfg(target_arch = "x86_64")]
fn simd() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
}

/// Dot of a Q4_K row with Q8_K activations.
pub fn dot_q4k(row: &[u8], y: &Q8K) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the features are present; the row is whole blocks and y covers it.
        return unsafe { dot_q4k_avx2(row, y) };
    }
    dot_q4k_scalar(row, y)
}

pub fn dot_q4k_scalar(row: &[u8], y: &Q8K) -> f32 {
    let mut sum = 0f32;
    for (b, blk) in row.chunks_exact(144).enumerate() {
        let d = hf(blk, 0) * y.d[b];
        let dmin = hf(blk, 2) * y.d[b];
        let (s, m) = scales_mins(&blk[4..16]);
        let q4 = &blk[16..144];
        let q8 = &y.qs[b * 256..(b + 1) * 256];
        let mut sumi = 0i32;
        let mut summ = 0i32;
        for j in 0..8 {
            let chunk = &q4[(j / 2) * 32..(j / 2) * 32 + 32];
            let shift = (j % 2) * 4;
            let p: i32 = (0..32).map(|i| ((chunk[i] >> shift) & 0xF) as i32 * q8[j * 32 + i] as i32).sum();
            sumi += s[j] as i32 * p;
            summ += m[j] as i32 * (y.bsums[b * 16 + 2 * j] as i32 + y.bsums[b * 16 + 2 * j + 1] as i32);
        }
        sum += d * sumi as f32 - dmin * summ as f32;
    }
    sum
}

/// As ggml's AVX2 `vec_dot_q4_K_q8_K`: scales and mins unpacked in vector registers, the mins dotted with the
/// activation sums by `madd`, d and dmin converted by F16C. No scalar work per block but the 12 bytes of scales.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q4k_avx2(row: &[u8], y: &Q8K) -> f32 {
    const K1: u32 = 0x3f3f_3f3f;
    const K2: u32 = 0x0f0f_0f0f;
    const K3: u32 = 0x0303_0303;
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();
        let nb = row.len() / 144;
        let (yd, yq, ys) = (y.d.as_ptr(), y.qs.as_ptr(), y.bsums.as_ptr());
        for b in 0..nb {
            let blk = row.as_ptr().add(b * 144);
            let dy = *yd.add(b);
            let d = _mm256_mul_ps(h8((blk as *const u16).read_unaligned()), _mm256_set1_ps(dy));
            let dmin = _mm_mul_ps(h4((blk.add(2) as *const u16).read_unaligned()), _mm_set1_ps(-dy));
            let sc = blk.add(4) as *const u32;
            let (u0, u1, u2) = (sc.read_unaligned(), sc.add(1).read_unaligned(), sc.add(2).read_unaligned());
            let m_hi = ((u2 >> 4) & K2) | (((u1 >> 6) & K3) << 4);
            let m_lo = u1 & K1;
            let s_hi = (u2 & K2) | (((u0 >> 6) & K3) << 4);
            let s_lo = u0 & K1;
            // 8 scales then 8 mins, as 16-bit
            let ms = _mm256_cvtepu8_epi16(_mm_set_epi32(m_hi as i32, m_lo as i32, s_hi as i32, s_lo as i32));
            let bs = _mm256_loadu_si256(ys.add(b * 16) as *const __m256i);
            let bs = _mm_hadd_epi16(_mm256_castsi256_si128(bs), _mm256_extracti128_si256(bs, 1));
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(ms, 1), bs);
            acc_m = _mm_fmadd_ps(dmin, _mm_cvtepi32_ps(prod), acc_m);
            let s128 = _mm256_castsi256_si128(ms);
            let scales = _mm256_set_m128i(s128, s128);
            let q4 = blk.add(16);
            let q8 = yq.add(b * 256);
            let mut sumi = _mm256_setzero_si256();
            for c in 0..4 {
                let sl = _mm256_shuffle_epi8(scales, _mm256_set1_epi16((((4 * c + 1) << 8) | (4 * c)) as i16));
                let sh = _mm256_shuffle_epi8(scales, _mm256_set1_epi16((((4 * c + 3) << 8) | (4 * c + 2)) as i16));
                let bits = _mm256_loadu_si256(q4.add(c * 32) as *const __m256i);
                let lo = _mm256_and_si256(bits, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
                let y0 = _mm256_loadu_si256(q8.add(c * 64) as *const __m256i);
                let y1 = _mm256_loadu_si256(q8.add(c * 64 + 32) as *const __m256i);
                let p0 = _mm256_madd_epi16(sl, _mm256_maddubs_epi16(lo, y0));
                let p1 = _mm256_madd_epi16(sh, _mm256_maddubs_epi16(hi, y1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
            }
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum(acc) + hsum128(acc_m)
    }
}

/// Dot of a Q5_0 row with Q8_0 activations: sum d_x d_y (sum q y - 16 sum y), q in 0..31.
pub fn dot_q50(row: &[u8], y: &Q8_0) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the features are present; the row is whole blocks and y covers it.
        return unsafe { dot_q50_avx2(row, y) };
    }
    dot_q50_scalar(row, y)
}

pub fn dot_q50_scalar(row: &[u8], y: &Q8_0) -> f32 {
    let mut sum = 0f32;
    for (b, blk) in row.chunks_exact(22).enumerate() {
        let d = hf(blk, 0) * y.d[b];
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let mut s = 0i32;
        for j in 0..16 {
            let q0 = (blk[6 + j] & 0xF) as i32 | ((((qh >> j) & 1) as i32) << 4);
            let q1 = (blk[6 + j] >> 4) as i32 | ((((qh >> (j + 16)) & 1) as i32) << 4);
            s += q0 * y.qs[b * 32 + j] as i32 + q1 * y.qs[b * 32 + j + 16] as i32;
        }
        sum += d * (s - 16 * y.sums[b]) as f32;
    }
    sum
}

/// As ggml's AVX2 `vec_dot_q5_0_q8_0`: the fifth bit makes each value q - 16 as a signed byte (0xF0 where it is
/// clear), and the signed product is `maddubs` of |q - 16| with y carrying q's sign.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q50_avx2(row: &[u8], y: &Q8_0) -> f32 {
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
        let bitm = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe_u64 as i64);
        let all = _mm256_set1_epi64x(-1);
        let f0 = _mm256_set1_epi8(0xF0u8 as i8);
        let ones16 = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        let nb = row.len() / 22;
        let (yd, yq) = (y.d.as_ptr(), y.qs.as_ptr());
        for b in 0..nb {
            let blk = row.as_ptr().add(b * 22);
            let d = _mm256_mul_ps(h8((blk as *const u16).read_unaligned()), _mm256_set1_ps(*yd.add(b)));
            let qh = (blk.add(2) as *const u32).read_unaligned();
            let raw = _mm_loadu_si128(blk.add(6) as *const __m128i);
            let q = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), m4);
            let set = _mm256_cmpeq_epi8(_mm256_or_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), shuf), bitm), all);
            let q = _mm256_or_si256(q, _mm256_andnot_si256(set, f0));
            let yy = _mm256_loadu_si256(yq.add(b * 32) as *const __m256i);
            let p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(q, q), _mm256_sign_epi8(yy, q)), ones16);
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p), acc);
        }
        hsum(acc)
    }
}

/// An f16 in each of 8 lanes, by F16C. A broadcast, not an insert: an insert into a register merges with what was in
/// it, and LLVM picks the accumulator, which chains every block of a dot to the last one.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
fn h8(bits: u16) -> __m256 {
    _mm256_cvtph_ps(_mm_set1_epi16(bits as i16))
}

/// An f16 in each of 4 lanes, as [`h8`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
fn h4(bits: u16) -> __m128 {
    _mm_cvtph_ps(_mm_set1_epi16(bits as i16))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256) -> f32 {
    hsum128(_mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hmax(v: __m256) -> f32 {
    let s = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(_mm_max_ss(s, _mm_movehdup_ps(s)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum128(s: __m128) -> f32 {
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{dequant_q4k_row, dequant_q50_row, dot};

    fn rnd(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn q4k_and_q50_dots_match_dequant() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        // random Q4_K row of 2304 with sane scales
        let mut row = vec![0u8; 9 * 144];
        for blk in row.chunks_exact_mut(144) {
            blk[0..2].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
            blk[2..4].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
            for b in &mut blk[4..] {
                *b = rnd(&mut seed) as u8;
            }
        }
        let x: Vec<f32> = (0..2304).map(|_| (rnd(&mut seed) % 2000) as f32 / 1000.0 - 1.0).collect();
        let mut w = vec![0f32; 2304];
        dequant_q4k_row(&row, &mut w);
        let want = dot(&w, &x);
        let mut q = Q8K { d: vec![], qs: vec![], bsums: vec![] };
        quantize_q8k(&x, &mut q);
        // the AVX2 quantizer rounds ties to even: at most one step on a value, and the same sums
        let mut qs_ref = vec![0i8; 2304];
        for b in 0..9 {
            let iscale = 1.0 / q.d[b];
            for i in 0..256 {
                qs_ref[b * 256 + i] = ((iscale * x[b * 256 + i]).round() as i32).min(127) as i8;
            }
            for j in 0..16 {
                let want: i16 = q.qs[b * 256 + j * 16..b * 256 + j * 16 + 16].iter().map(|&v| v as i16).sum();
                assert_eq!(q.bsums[b * 16 + j], want);
            }
        }
        assert!(q.qs.iter().zip(&qs_ref).all(|(a, b)| (*a as i32 - *b as i32).abs() <= 1));
        let (a, s) = (dot_q4k(&row, &q), dot_q4k_scalar(&row, &q));
        assert!((a - s).abs() <= 1e-3 * s.abs().max(1.0), "avx2 {a} scalar {s}");
        assert!((a - want).abs() <= 0.02 * want.abs().max(1.0), "q4k {a} vs f32 {want}");

        let mut row5 = vec![0u8; 36 * 22];
        for blk in row5.chunks_exact_mut(22) {
            blk[0..2].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
            for b in &mut blk[2..] {
                *b = rnd(&mut seed) as u8;
            }
        }
        let x5: Vec<f32> = (0..1152).map(|_| (rnd(&mut seed) % 2000) as f32 / 1000.0 - 1.0).collect();
        let mut w5 = vec![0f32; 1152];
        dequant_q50_row(&row5, &mut w5);
        let want5 = dot(&w5, &x5);
        let mut q5 = Q8_0 { d: vec![], qs: vec![], sums: vec![] };
        quantize_q8_0(&x5, &mut q5);
        let (a5, s5) = (dot_q50(&row5, &q5), dot_q50_scalar(&row5, &q5));
        assert!((a5 - s5).abs() <= 1e-3 * s5.abs().max(1.0), "avx2 {a5} scalar {s5}");
        assert!((a5 - want5).abs() <= 0.02 * want5.abs().max(1.0), "q50 {a5} vs f32 {want5}");
    }
}
