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
    /// 12 times the sum of each 16 int8, for the NVFP4 kernel (its weights are FP4 + 12), in the order it reduces them:
    /// per 128 values, the first halves of the 4 blocks, then their second halves (see `c12_at`)
    pub c12: Vec<i32>,
}

/// Where block b's two halves go in Q8_0::c12.
#[inline]
pub fn c12_at(b: usize) -> (usize, usize) {
    (8 * (b / 4) + b % 4, 8 * (b / 4) + 4 + b % 4)
}

/// ggml quantize_row_q8_0_ref (d rounded to f16 as ggml stores it)
pub fn quantize_q8_0(x: &[f32], out: &mut Q8_0) {
    let nb = x.len() / 32;
    out.d.resize(nb, 0.0);
    out.qs.resize(nb * 32, 0);
    out.sums.resize(nb, 0);
    out.c12.resize(2 * nb.div_ceil(4) * 4, 0);
    for b in 0..nb {
        let (d, s, s16) = quantize_q8_0_block16(&x[b * 32..(b + 1) * 32], &mut out.qs[b * 32..(b + 1) * 32]);
        out.d[b] = d;
        out.sums[b] = s;
        let (lo, hi) = c12_at(b);
        (out.c12[lo], out.c12[hi]) = (12 * s16[0], 12 * s16[1]);
    }
}

/// quantize_q8_0_block, also returning the sums of each 16 int8; AVX2 where the CPU has it, the same values.
#[inline]
pub fn quantize_q8_0_block16(xb: &[f32], qs: &mut [i8]) -> (f32, i32, [i32; 2]) {
    #[cfg(target_arch = "x86_64")]
    if simd() && xb.len() == 32 && qs.len() == 32 {
        // SAFETY: the features are present; 32 values each
        return unsafe { quantize_block_avx2(xb, qs) };
    }
    let (d, s) = quantize_q8_0_block(xb, qs);
    let a: i32 = qs[..16].iter().map(|&v| v as i32).sum();
    (d, s, [a, s - a])
}

/// quantize_q8_0_block of 32 values, AVX2: the same rounding as `round` (halves away from zero), so the same int8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn quantize_block_avx2(xb: &[f32], qs: &mut [i8]) -> (f32, i32, [i32; 2]) {
    unsafe {
        let abs = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFF));
        let v: [__m256; 4] = std::array::from_fn(|i| _mm256_loadu_ps(xb.as_ptr().add(8 * i)));
        let m = _mm256_max_ps(_mm256_max_ps(_mm256_and_ps(v[0], abs), _mm256_and_ps(v[1], abs)), _mm256_max_ps(_mm256_and_ps(v[2], abs), _mm256_and_ps(v[3], abs)));
        let m = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let amax = _mm_cvtss_f32(_mm_max_ss(m, _mm_movehdup_ps(m)));
        let d = amax / 127.0;
        let id = _mm256_set1_ps(if d != 0.0 { 1.0 / d } else { 0.0 });
        let (half, one, sign) = (_mm256_set1_ps(0.5), _mm256_set1_ps(1.0), _mm256_set1_ps(-0.0));
        let q: [__m256i; 4] = std::array::from_fn(|i| {
            let y = _mm256_mul_ps(v[i], id);
            let t = _mm256_round_ps(y, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            let up = _mm256_cmp_ps(_mm256_and_ps(_mm256_sub_ps(y, t), abs), half, _CMP_GE_OQ);
            let step = _mm256_and_ps(up, _mm256_or_ps(_mm256_and_ps(y, sign), one));
            _mm256_cvttps_epi32(_mm256_add_ps(t, step))
        });
        let packed = _mm256_permutevar8x32_epi32(_mm256_packs_epi16(_mm256_packs_epi32(q[0], q[1]), _mm256_packs_epi32(q[2], q[3])), _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256(qs.as_mut_ptr() as *mut __m256i, packed);
        let hs = |a: __m256i| {
            let s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
            _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01)))
        };
        let (a, b) = (hs(_mm256_add_epi32(q[0], q[1])), hs(_mm256_add_epi32(q[2], q[3])));
        (f16::from_f32(d).to_f32(), a + b, [a, b])
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

/// The FP4 table plus 12 (0..24, as u8) in both 128-bit lanes, for pshufb.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn fp4_lut_u() -> __m256i {
    let t = |i: usize| (FP4[i] + 12) as i8;
    _mm256_setr_epi8(
        t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12), t(13), t(14), t(15),
        t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12), t(13), t(14), t(15),
    )
}

/// 8 UE4M3 scale bytes (one in each i32 lane) as f32, as the UE4M3 table has them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn ue4m3_8(v: __m256i) -> __m256 {
    let m7f = _mm256_set1_epi32(0x7F);
    let normal = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(_mm256_and_si256(v, m7f), 20), _mm256_set1_epi32(0x3B80_0000)));
    let sub = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, _mm256_set1_epi32(7))), _mm256_set1_ps(1.0 / 1024.0));
    let is_sub = _mm256_cmpeq_epi32(_mm256_and_si256(v, _mm256_set1_epi32(0x78)), _mm256_setzero_si256());
    let f = _mm256_blendv_ps(normal, sub, _mm256_castsi256_ps(is_sub));
    _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(v, m7f)), f)
}

/// The body of the 128-value NVFP4 kernels: the weights as FP4 + 12 (unsigned, for the u8 x i8 dot) and the 12 x sums
/// taken back (x.c12), the sums of each 16 reduced to one lane each by hadd, the 8 scales of the step converted at
/// once (`$fast`: all normal, 3 instructions; else ue4m3_8). `$dot(w, x)`: the 8 sums of 4 products.
macro_rules! nv4_body {
    ($q:expr, $s:expr, $x:expr, $fast:expr, $dot:expr) => {{
        let (q, s, x): (&[u8], &[u8], &Q8_0) = ($q, $s, $x);
        let lut = fp4_lut_u();
        let m4 = _mm256_set1_epi8(0xF);
        let order = _mm_setr_epi8(0, 2, 4, 6, 1, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1);
        let mut acc = _mm256_setzero_ps();
        let (qp, xp, sp, dp, cp) = (q.as_ptr(), x.qs.as_ptr(), s.as_ptr(), x.d.as_ptr(), x.c12.as_ptr());
        for it in 0..q.len() / 64 {
            let r0 = _mm256_loadu_si256(qp.add(64 * it) as *const __m256i);
            let r1 = _mm256_loadu_si256(qp.add(64 * it + 32) as *const __m256i);
            let w0 = _mm256_shuffle_epi8(lut, _mm256_and_si256(r0, m4));
            let w1 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(r0, 4), m4));
            let w2 = _mm256_shuffle_epi8(lut, _mm256_and_si256(r1, m4));
            let w3 = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(r1, 4), m4));
            let xb = xp.add(128 * it);
            let q0 = $dot(w0, _mm256_loadu_si256(xb as *const __m256i));
            let q1 = $dot(w1, _mm256_loadu_si256(xb.add(32) as *const __m256i));
            let q2 = $dot(w2, _mm256_loadu_si256(xb.add(64) as *const __m256i));
            let q3 = $dot(w3, _mm256_loadu_si256(xb.add(96) as *const __m256i));
            // [block 0..3 first 16 | block 0..3 second 16]
            let h = _mm256_hadd_epi32(_mm256_hadd_epi32(q0, q1), _mm256_hadd_epi32(q2, q3));
            let h = _mm256_sub_epi32(h, _mm256_loadu_si256(cp.add(8 * it) as *const __m256i));
            let v = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(_mm_loadl_epi64(sp.add(8 * it) as *const __m128i), order));
            let sc = if $fast { _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(v, 20), _mm256_set1_epi32(0x3B80_0000))) } else { ue4m3_8(v) };
            let d = _mm256_broadcast_ps(&*(dp.add(4 * it) as *const __m128));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(h), _mm256_mul_ps(sc, d), acc);
        }
        hsum(acc)
    }};
}

/// Whether the UE4M3 scales `s` are all normal numbers (exponent above 0, not the 0x7F that reads as 0, top bit clear):
/// then the kernels convert them with a shift and an add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn normal_scales(s: &[u8]) -> bool {
    unsafe {
        let (e, z, m7f) = (_mm256_set1_epi8(0x78), _mm256_setzero_si256(), _mm256_set1_epi8(0x7F));
        let mut bad = _mm256_setzero_si256();
        let n = s.len() / 32;
        for i in 0..n {
            let v = _mm256_loadu_si256(s.as_ptr().add(32 * i) as *const __m256i);
            let sub = _mm256_cmpeq_epi8(_mm256_and_si256(v, e), z);
            bad = _mm256_or_si256(bad, _mm256_or_si256(sub, _mm256_or_si256(_mm256_cmpeq_epi8(v, m7f), v)));
        }
        // the top bit of a byte of `bad` marks a bad scale (cmpeq gives 0xFF, a scale with bit 7 set has it itself)
        _mm256_movemask_epi8(bad) == 0 && s[32 * n..].iter().all(|&b| b & 0x78 != 0 && b != 0x7F && b & 0x80 == 0)
    }
}

static VNNI: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    return is_x86_feature_detected!("avxvnni") && std::env::var("IK_NO_VNNI").is_err();
    #[allow(unreachable_code)]
    false
});

/// NVFP4 row . x, 128 values a step (see nv4_body), by AVX-VNNI where the CPU has it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_nv4_avx2(q: &[u8], s: &[u8], x: &Q8_0) -> f32 {
    unsafe {
        if q.len() % 64 != 0 || x.c12.len() * 32 < q.len() * 2 {
            return dot_nv4_avx2_64(q, s, x);
        }
        let fast = normal_scales(s);
        match (*VNNI, fast) {
            (true, true) => dot_nv4_vnni::<true>(q, s, x),
            (true, false) => dot_nv4_vnni::<false>(q, s, x),
            (false, true) => dot_nv4_maddubs::<true>(q, s, x),
            (false, false) => dot_nv4_maddubs::<false>(q, s, x),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
unsafe fn dot_nv4_vnni<const FAST: bool>(q: &[u8], s: &[u8], x: &Q8_0) -> f32 {
    unsafe { nv4_body!(q, s, x, FAST, |w, xv| _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w, xv)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_nv4_maddubs<const FAST: bool>(q: &[u8], s: &[u8], x: &Q8_0) -> f32 {
    let ones = _mm256_set1_epi16(1);
    unsafe { nv4_body!(q, s, x, FAST, |w, xv| _mm256_madd_epi16(_mm256_maddubs_epi16(w, xv), ones)) }
}

/// dot_nv4_avx2 a group of 64 values a step, for rows the 128-value kernel does not take.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_nv4_avx2_64(q: &[u8], s: &[u8], x: &Q8_0) -> f32 {
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
                    *v = match rnd(seed) % 64 {
                        0 => (rnd(seed) % 8) as u8, // subnormal
                        1 => 0x7F,                  // read as 0
                        _ => 0x20 + (rnd(seed) % 16) as u8, // 2^-3 .. 2^-2 or so
                    };
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
    fn quantize_simd_matches() {
        let mut seed = 0x5eed_0001_u64;
        for case in 0..200 {
            let xb: Vec<f32> = match case {
                // ties: amax 127 makes the scale 1, and halves round away from zero
                0 => (0..32).map(|i| if i == 0 { 127.0 } else { (i as f32 - 16.0) + 0.5 }).collect(),
                1 => vec![0.0; 32],
                _ => (0..32).map(|_| (rnd(&mut seed) % 20001) as f32 / 1000.0 - 10.0).collect(),
            };
            let (mut a, mut b) = (vec![0i8; 32], vec![0i8; 32]);
            let (d, s) = quantize_q8_0_block(&xb, &mut a);
            let (d2, s2, s16) = quantize_q8_0_block16(&xb, &mut b);
            assert_eq!((a.clone(), d.to_bits(), s), (b, d2.to_bits(), s2), "case {case}");
            assert_eq!(s16[0], a[..16].iter().map(|&v| v as i32).sum::<i32>());
        }
    }

    #[test]
    fn nv4_kernels_agree() {
        if !simd() {
            return;
        }
        let mut seed = 0x0bad_cafe_1234_5678u64;
        for k in [2304, 1152] {
            let mut b = random_mat(Fmt::Nv4, 4, k, &mut seed);
            // rows 0 and 1 with normal scales only, 2 and 3 as random_mat made them (some subnormal, some 0x7F)
            for v in &mut b[4 * k / 2..4 * k / 2 + 2 * k / 16] {
                *v = 0x20 + (rnd(&mut seed) % 16) as u8;
            }
            b[4 * k / 2 + 2 * k / 16] = 3;
            b[4 * k / 2 + 3 * k / 16] = 0x7F;
            let m = Mat { fmt: Fmt::Nv4, b: &b, rows: 4, k };
            let (_, x) = random_x(k, &mut seed);
            for r in 0..4 {
                let (q, sc) = (m.qs(r), m.sc(r));
                let want = dot_row_scalar(&m, r, &x);
                // SAFETY: avx2 is present (and avx-vnni where used)
                let normal = unsafe { normal_scales(sc) };
                assert_eq!(normal, r < 2, "row {r}");
                let mut got = vec![unsafe { dot_nv4_maddubs::<false>(q, sc, &x) }, unsafe { dot_nv4_avx2_64(q, sc, &x) }];
                if normal {
                    got.push(unsafe { dot_nv4_maddubs::<true>(q, sc, &x) });
                }
                if is_x86_feature_detected!("avxvnni") {
                    got.push(unsafe { dot_nv4_vnni::<false>(q, sc, &x) });
                    if normal {
                        got.push(unsafe { dot_nv4_vnni::<true>(q, sc, &x) });
                    }
                }
                for g in got {
                    assert!((g - want).abs() <= 1e-4 * want.abs().max(1.0), "k {k} row {r}: {g} vs scalar {want}");
                }
            }
        }
    }

    #[test]
    fn ue4m3_simd() {
        if !simd() {
            return;
        }
        for x in (0..256).step_by(8) {
            let mut got = [0f32; 8];
            // SAFETY: avx2 is present
            unsafe {
                let v = _mm256_setr_epi32(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
                _mm256_storeu_ps(got.as_mut_ptr(), ue4m3_8(v));
            }
            for (i, g) in got.iter().enumerate() {
                assert_eq!(g.to_bits(), UE4M3[x as usize + i].to_bits(), "{:#x}", x as usize + i);
            }
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
                let l1: f32 = w.iter().zip(&x).map(|(a, b)| (a * b).abs()).sum();
                let (a, s) = (dot_row(&m, r, &q), dot_row_scalar(&m, r, &q));
                assert!((a - s).abs() <= 1e-4 * s.abs().max(1.0), "{fmt:?} k {k}: avx2 {a} scalar {s}");
                assert!((a - want).abs() <= 0.01 * l1, "{fmt:?} k {k}: {a} vs f32 {want} (sum of |w x| {l1})");
            }
        }
    }
}
