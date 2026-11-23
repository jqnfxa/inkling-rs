//! CPU expert rows against many tokens (prefill): each piece of a packed row is decoded once, in registers, and
//! multiplied with the tokens 4 at a time.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

use crate::cpu::{Q8_0, Q8K};

/// Per-token data that the batched dots precompute once: for Q8_K, the sums of each sub-block of 32 (bsums paired);
/// for Q8_0, 16 * d * sum(q) of each block, the -16 offset of Q5_0.
pub struct TokQ8K<'a> {
    pub x: &'a Q8K,
    pub s32: Vec<i32>, // [n/32]
}

impl<'a> TokQ8K<'a> {
    pub fn new(x: &'a Q8K) -> TokQ8K<'a> {
        TokQ8K { x, s32: x.bsums.chunks_exact(2).map(|p| p[0] as i32 + p[1] as i32).collect() }
    }
}

pub struct TokQ8_0<'a> {
    pub x: &'a Q8_0,
    pub corr: Vec<f32>, // [n/32] 16 * d * sum
}

impl<'a> TokQ8_0<'a> {
    pub fn new(x: &'a Q8_0) -> TokQ8_0<'a> {
        TokQ8_0 { x, corr: x.d.iter().zip(&x.sums).map(|(&d, &s)| 16.0 * d * s as f32).collect() }
    }
}

/// out[i] = row . x[i] for each token i (Q8_K activations), on the packed Q4_K row: the nibbles are split and the
/// scales decoded once per super-block, for all the tokens, 4 at a time.
pub fn q4k_dot_tokens_packed(row: &[u8], xs: &[TokQ8K], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 is present
        return unsafe { q4k_packed_avx2(row, xs, out) };
    }
    for (o, x) in out.iter_mut().zip(xs) {
        *o = crate::cpu::dot_q4k_scalar(row, x.x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q4k_packed_avx2(row: &[u8], xs: &[TokQ8K], out: &mut [f32]) {
    unsafe {
        let mut i = 0;
        while i < xs.len() {
            let left = xs.len() - i;
            // the widest block that fits (IK_QBLK caps it), so that most rows take one pass
            static MAXW: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
            let maxw = *MAXW.get_or_init(|| std::env::var("IK_QBLK").ok().and_then(|v| v.parse().ok()).unwrap_or(8));
            let w = if left >= 8 && maxw >= 8 { 8 } else if left >= 4 && maxw >= 4 { 4 } else if left >= 2 && maxw >= 2 { 2 } else { 1 };
            match w {
                8 => q4k_block::<8>(row, &xs[i..i + 8], &mut out[i..i + 8]),
                4 => q4k_block::<4>(row, &xs[i..i + 4], &mut out[i..i + 4]),
                2 => q4k_block::<2>(row, &xs[i..i + 2], &mut out[i..i + 2]),
                _ => q4k_block::<1>(row, &xs[i..i + 1], &mut out[i..i + 1]),
            }
            i += w;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn q4k_block<const N: usize>(row: &[u8], xs: &[TokQ8K], out: &mut [f32]) {
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let nb = row.len() / 144;
        let mut acc = [_mm256_setzero_ps(); N];
        for b in 0..nb {
            let blk = row.as_ptr().add(b * 144);
            let d = f16::from_bits(u16::from_le_bytes([*blk, *blk.add(1)])).to_f32();
            let dmin = f16::from_bits(u16::from_le_bytes([*blk.add(2), *blk.add(3)])).to_f32();
            let s = std::slice::from_raw_parts(blk.add(4), 12);
            let mut sc = [0i16; 8];
            let mut mn = [0i32; 8];
            for j in 0..8 {
                let (a, m) = if j < 4 { (s[j] & 63, s[j + 4] & 63) } else { ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)) };
                sc[j] = a as i16;
                mn[j] = m as i32;
            }
            let mv = _mm256_loadu_si256(mn.as_ptr() as *const __m256i);
            let mut sumi = [_mm256_setzero_si256(); N];
            for c in 0..4 {
                let bits = _mm256_loadu_si256(blk.add(16 + c * 32) as *const __m256i);
                let lo = _mm256_and_si256(bits, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
                let (s0, s1) = (_mm256_set1_epi16(sc[2 * c]), _mm256_set1_epi16(sc[2 * c + 1]));
                for t in 0..N {
                    let xq = xs[t].x.qs.as_ptr().add(b * 256 + c * 64);
                    let y0 = _mm256_loadu_si256(xq as *const __m256i);
                    let y1 = _mm256_loadu_si256(xq.add(32) as *const __m256i);
                    let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), s0);
                    let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), s1);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_add_epi32(p0, p1));
                }
            }
            for t in 0..N {
                let x = &xs[t];
                let dx = x.x.d[b];
                let mins = _mm256_mullo_epi32(mv, _mm256_loadu_si256(x.s32.as_ptr().add(b * 8) as *const __m256i));
                acc[t] = _mm256_fmadd_ps(_mm256_set1_ps(d * dx), _mm256_cvtepi32_ps(sumi[t]), acc[t]);
                acc[t] = _mm256_fnmadd_ps(_mm256_set1_ps(dmin * dx), _mm256_cvtepi32_ps(mins), acc[t]);
            }
        }
        for t in 0..N {
            out[t] = hsum(acc[t]);
        }
    }
}

/// out[i] = row . x[i] for each token i (Q8_0 activations), on the packed Q5_0 row: the values of a block are
/// assembled once for all the tokens; the scales apply to 4 blocks at once.
pub fn q50_dot_tokens_packed(row: &[u8], xs: &[TokQ8_0], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && (row.len() / 22) % 4 == 0 {
        // SAFETY: AVX2 is present
        return unsafe { q50_packed_avx2(row, xs, out) };
    }
    for (o, x) in out.iter_mut().zip(xs) {
        *o = crate::cpu::dot_q50_scalar(row, x.x);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn q50_packed_avx2(row: &[u8], xs: &[TokQ8_0], out: &mut [f32]) {
    unsafe {
        let m4 = _mm_set1_epi8(0xF);
        let shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
        let bitm = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe_u64 as i64);
        let ones = _mm256_set1_epi16(1);
        let nb = row.len() / 22;
        // the row's values and scales, assembled once
        let mut qv = [_mm256_setzero_si256(); 64];
        let mut dw = [0f32; 64];
        for b in 0..nb {
            let blk = row.as_ptr().add(b * 22);
            dw[b] = f16::from_bits(u16::from_le_bytes([*blk, *blk.add(1)])).to_f32();
            let qh = u32::from_le_bytes([*blk.add(2), *blk.add(3), *blk.add(4), *blk.add(5)]);
            let raw = _mm_loadu_si128(blk.add(6) as *const __m128i);
            let q = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(raw, 4), m4), _mm_and_si128(raw, m4));
            let v = _mm256_or_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), shuf), bitm);
            let hb = _mm256_cmpeq_epi8(v, _mm256_set1_epi64x(-1));
            qv[b] = _mm256_or_si256(q, _mm256_and_si256(hb, _mm256_set1_epi8(0x10)));
        }
        for (t, x) in xs.iter().enumerate() {
            let mut acc = _mm256_setzero_ps();
            let mut b = 0;
            while b < nb {
                // 4 blocks: their 8-lane sums, reduced to 4 sums, scaled by d_w d_x
                let mut p = [_mm256_setzero_si256(); 4];
                for k in 0..4 {
                    let xv = _mm256_loadu_si256(x.x.qs.as_ptr().add((b + k) * 32) as *const __m256i);
                    p[k] = _mm256_madd_epi16(_mm256_maddubs_epi16(qv[b + k], xv), ones);
                }
                let h01 = _mm256_hadd_epi32(p[0], p[1]);
                let h23 = _mm256_hadd_epi32(p[2], p[3]);
                let h = _mm256_hadd_epi32(h01, h23); // lanes: b0 b1 b2 b3 (low half) + b0 b1 b2 b3 (high half)
                let h4 = _mm_add_epi32(_mm256_castsi256_si128(h), _mm256_extracti128_si256(h, 1));
                let sw = _mm_loadu_ps(dw.as_ptr().add(b));
                let sx = _mm_loadu_ps(x.x.d.as_ptr().add(b));
                let cr = _mm_loadu_ps(x.corr.as_ptr().add(b));
                // d_w (d_x sum - 16 d_x s)
                let v = _mm_fmsub_ps(_mm_cvtepi32_ps(h4), sx, cr);
                acc = _mm256_add_ps(acc, _mm256_castps128_ps256(_mm_mul_ps(v, sw)));
                b += 4;
            }
            out[t] = hsum(_mm256_blend_ps(acc, _mm256_setzero_ps(), 0xF0));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::{dot_q4k_scalar, dot_q50_scalar, quantize_q8_0, quantize_q8k};

    #[test]
    fn batched_rows_match_single_dots() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut row = vec![0u8; 9 * 144];
        for blk in row.chunks_exact_mut(144) {
            blk[0..2].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
            blk[2..4].copy_from_slice(&f16::from_f32(0.02).to_bits().to_le_bytes());
            for b in &mut blk[4..] {
                *b = rnd() as u8;
            }
        }
        let xs: Vec<Q8K> = (0..7)
            .map(|_| {
                let x: Vec<f32> = (0..2304).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
                let mut q = Q8K { d: vec![], qs: vec![], bsums: vec![] };
                quantize_q8k(&x, &mut q);
                q
            })
            .collect();
        let refs: Vec<TokQ8K> = xs.iter().map(TokQ8K::new).collect();
        let mut out = vec![0f32; 7];
        q4k_dot_tokens_packed(&row, &refs, &mut out);
        for (i, x) in xs.iter().enumerate() {
            let want = dot_q4k_scalar(&row, x);
            assert!((out[i] - want).abs() <= 1e-3 * want.abs().max(1.0), "q4k token {i}: {} vs {want}", out[i]);
        }
        let mut row5 = vec![0u8; 36 * 22];
        for blk in row5.chunks_exact_mut(22) {
            blk[0..2].copy_from_slice(&f16::from_f32(0.01).to_bits().to_le_bytes());
            for b in &mut blk[2..] {
                *b = rnd() as u8;
            }
        }
        let x5: Vec<Q8_0> = (0..5)
            .map(|_| {
                let x: Vec<f32> = (0..1152).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
                let mut q = Q8_0 { d: vec![], qs: vec![], sums: vec![] };
                quantize_q8_0(&x, &mut q);
                q
            })
            .collect();
        let refs5: Vec<TokQ8_0> = x5.iter().map(TokQ8_0::new).collect();
        let mut out5 = vec![0f32; 5];
        q50_dot_tokens_packed(&row5, &refs5, &mut out5);
        for (i, x) in x5.iter().enumerate() {
            let want = dot_q50_scalar(&row5, x);
            assert!((out5[i] - want).abs() <= 1e-3 * want.abs().max(1.0), "q50 token {i}: {} vs {want}", out5[i]);
        }
    }
}
