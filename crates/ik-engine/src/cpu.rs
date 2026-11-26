//! CPU kernels of the experts that miss the GPU cache, AVX2: rows as the checkpoint has them (experts.rs), ModelOpt
//! NVFP4 ones against int8 activations and MXFP8 ones against f32 activations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::experts::{FP4X2, Fmt, Mat, e4m3, e4m3_abs, ue8m0};

/// Where element i of a vector is in Q8::qs: each 64 values as their 16-value chunks 0, 2, 1, 3, the order in which the
/// NVFP4 kernels unpack the nibbles of a row's 32 bytes of them.
#[inline]
pub fn pos(i: usize) -> usize {
    const SW: [usize; 4] = [0, 2, 1, 3];
    (i & !63) | (SW[(i >> 4) & 3] << 4) | (i & 15)
}

/// The first of the two lanes (of the 8 of each 128 values) of the NVFP4 kernels' sums that block b (32 values) goes to.
#[inline]
fn lane(b: usize) -> usize {
    8 * (b / 4) + [0, 4, 2, 6][b % 4]
}

/// A vector quantized for the NVFP4 kernels: per 32 values a scale and 32 int8 (`d`, and `qs` at `pos`), and per 128
/// values the scales of the 8 lanes of the kernels' sums (`dv`: of blocks 0, 0, 2, 2, 1, 1, 3, 3) and 12 times the sums
/// of their int8 (`c12`: the 16-value chunks 0, 1, 4, 5, 2, 3, 6, 7), which the kernels take back from their weights,
/// FP4 + 12. k is a multiple of 128.
#[derive(Default, Clone)]
pub struct Q8 {
    pub d: Vec<f32>,
    pub qs: Vec<i8>,
    pub dv: Vec<f32>,
    pub c12: Vec<i32>,
}

impl Q8 {
    pub fn with_len(k: usize) -> Q8 {
        assert!(k % 128 == 0, "a Q8 of {k} values");
        Q8 { d: vec![0.0; k / 32], qs: vec![0; k], dv: vec![0.0; k / 16], c12: vec![0; k / 16] }
    }

    /// The int8 of element i.
    pub fn q(&self, i: usize) -> i8 {
        self.qs[pos(i)]
    }

    /// Quantizes block b (values 32b..32b + 32) from the 32 values `x`.
    /// SAFETY: the vector holds block b, and no other thread writes block b or reads the vector meanwhile.
    pub unsafe fn put_block(&self, b: usize, x: &[f32]) {
        let mut q = [0i8; 32];
        let (d, s16) = quantize_block(x, &mut q);
        let l = lane(b);
        unsafe {
            let qs = self.qs.as_ptr() as *mut i8;
            std::ptr::copy_nonoverlapping(q.as_ptr(), qs.add(pos(32 * b)), 16);
            std::ptr::copy_nonoverlapping(q.as_ptr().add(16), qs.add(pos(32 * b + 16)), 16);
            *(self.d.as_ptr() as *mut f32).add(b) = d;
            let (dv, c12) = (self.dv.as_ptr() as *mut f32, self.c12.as_ptr() as *mut i32);
            (*dv.add(l), *dv.add(l + 1)) = (d, d);
            (*c12.add(l), *c12.add(l + 1)) = (12 * s16[0], 12 * s16[1]);
        }
    }
}

/// x quantized for the NVFP4 kernels.
pub fn quantize(x: &[f32], out: &mut Q8) {
    if out.qs.len() != x.len() {
        *out = Q8::with_len(x.len());
    }
    for b in 0..x.len() / 32 {
        // SAFETY: `out` holds x.len() values, and only this thread has it
        unsafe { out.put_block(b, &x[32 * b..32 * b + 32]) };
    }
}

/// An activation vector as the expert kernels read it: quantized for NVFP4 rows (`q`), as it is for MXFP8 rows (`f`).
/// The format of the layer's experts says which of the two is filled.
#[derive(Default, Clone)]
pub struct Act {
    pub q: Q8,
    pub f: Vec<f32>,
}

impl Act {
    /// A vector of k values for the rows of `fmt`, to be written a block at a time.
    pub fn with_len(k: usize, fmt: Fmt) -> Act {
        match fmt {
            Fmt::NvFp4 => Act { q: Q8::with_len(k), f: vec![] },
            Fmt::MxFp8 => Act { q: Q8::default(), f: vec![0.0; k] },
        }
    }

    /// Sets the vector to x, as the rows of `fmt` read it.
    pub fn set(&mut self, x: &[f32], fmt: Fmt) {
        match fmt {
            Fmt::NvFp4 => quantize(x, &mut self.q),
            Fmt::MxFp8 => {
                self.f.clear();
                self.f.extend_from_slice(x);
            }
        }
    }

    /// Sets block b (values 32b..32b + 32) to the 32 values x.
    /// SAFETY: as Q8::put_block; the vector is one of with_len's for `fmt`.
    pub unsafe fn put_block(&self, b: usize, x: &[f32], fmt: Fmt) {
        match fmt {
            Fmt::NvFp4 => unsafe { self.q.put_block(b, x) },
            Fmt::MxFp8 => unsafe { std::ptr::copy_nonoverlapping(x.as_ptr(), (self.f.as_ptr() as *mut f32).add(32 * b), 32) },
        }
    }
}

/// The 32 values of `xb` as int8 times a scale (absmax / 127, halves rounded away from zero), and the sums of each 16
/// int8; AVX2 where the CPU has it, the same values.
#[inline]
pub fn quantize_block(xb: &[f32], qs: &mut [i8; 32]) -> (f32, [i32; 2]) {
    #[cfg(target_arch = "x86_64")]
    if simd() && xb.len() == 32 {
        // SAFETY: the features are present; 32 values each
        return unsafe { quantize_block_avx2(xb, qs) };
    }
    quantize_block_scalar(xb, qs)
}

fn quantize_block_scalar(xb: &[f32], qs: &mut [i8; 32]) -> (f32, [i32; 2]) {
    let amax = xb.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for (q, &v) in qs.iter_mut().zip(xb) {
        *q = (v * id).round() as i8;
    }
    let s = |h: &[i8]| h.iter().map(|&v| v as i32).sum::<i32>();
    (d, [s(&qs[..16]), s(&qs[16..])])
}

/// quantize_block_scalar, AVX2: the same rounding as `round` (halves away from zero), so the same int8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn quantize_block_avx2(xb: &[f32], qs: &mut [i8; 32]) -> (f32, [i32; 2]) {
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
        (d, [hs(_mm256_add_epi32(q[0], q[1])), hs(_mm256_add_epi32(q[2], q[3]))])
    }
}

#[inline]
pub(crate) fn simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    return is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
    #[allow(unreachable_code)]
    false
}

/// Row r of `m` . x, the expert's factor left out (the caller applies m.scale()).
pub fn dot_row(m: &Mat, r: usize, x: &Act) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if simd() && m.k % 128 == 0 {
        // SAFETY: the features are present; x holds k values
        return unsafe {
            match m.fmt {
                Fmt::NvFp4 => dot_nvfp4_avx2(m.qs(r), m.sc(r), &x.q),
                Fmt::MxFp8 => dot_mxfp8_avx2(m.qs(r), m.sc(r), &x.f),
            }
        };
    }
    dot_row_scalar(m, r, x)
}

/// The scalar reference of [`dot_row`].
pub fn dot_row_scalar(m: &Mat, r: usize, x: &Act) -> f32 {
    match m.fmt {
        Fmt::NvFp4 => dot_nvfp4_ref(m.qs(r), m.sc(r), &x.q),
        Fmt::MxFp8 => dot_mxfp8_ref(m.qs(r), m.sc(r), &x.f),
    }
}

/// An NVFP4 row (codes q, E4M3 scales s) . x: per 16 values, the int dot of the doubled E2M1 values and the int8, times
/// half the scale and x's scale.
pub fn dot_nvfp4_ref(q: &[u8], s: &[u8], x: &Q8) -> f32 {
    let mut acc = 0f32;
    for (c, &sc) in s.iter().enumerate() {
        let mut sum = 0i32;
        for i in 16 * c..16 * c + 16 {
            let code = (q[i / 2] >> (4 * (i % 2))) & 15;
            sum += FP4X2[code as usize] as i32 * x.q(i) as i32;
        }
        acc += sum as f32 * (0.5 * e4m3_abs(sc)) * x.d[c / 2];
    }
    acc
}

/// An MXFP8 row (E4M3 values q, UE8M0 scales s) . x.
pub fn dot_mxfp8_ref(q: &[u8], s: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, &sc) in s.iter().enumerate() {
        let sum: f32 = (32 * b..32 * b + 32).map(|i| e4m3(q[i]) * x[i]).sum();
        acc += sum * ue8m0(sc);
    }
    acc
}

/// The doubled E2M1 values plus 12 (0..24, as u8) in both 128-bit lanes, for pshufb.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn fp4_lut_u() -> __m256i {
    let t = |i: usize| (FP4X2[i] + 12) as i8;
    _mm256_setr_epi8(
        t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12), t(13), t(14), t(15),
        t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12), t(13), t(14), t(15),
    )
}

/// The 8 scale bytes of 128 values to the lanes of the NVFP4 kernels' sums: those of the 16-value chunks 0, 1, 4, 5,
/// 2, 3, 6, 7, one byte in each i32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn scale_lanes(s: *const u8) -> __m256i {
    unsafe { _mm256_cvtepu8_epi32(_mm_shuffle_epi8(_mm_loadl_epi64(s as *const __m128i), _mm_setr_epi8(0, 1, 4, 5, 2, 3, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1))) }
}

/// 8 E4M3 scale bytes (one in each i32 lane) as f32, halved for the doubled E2M1 values: sign bit ignored, as e4m3_abs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn e4m3h_8(v: __m256i) -> __m256 {
    let v = _mm256_and_si256(v, _mm256_set1_epi32(0x7F));
    let normal = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(v, 20), _mm256_set1_epi32(0x3B80_0000)));
    let sub = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, _mm256_set1_epi32(7))), _mm256_set1_ps(1.0 / 1024.0));
    let is_sub = _mm256_cmpeq_epi32(_mm256_and_si256(v, _mm256_set1_epi32(0x78)), _mm256_setzero_si256());
    _mm256_blendv_ps(normal, sub, _mm256_castsi256_ps(is_sub))
}

/// The 4 vectors of 32 weights (FP4 + 12) of 64 code bytes, 128 values: each 32 bytes' nibbles as the elements in order
/// [0..16 | 32..48] and [16..32 | 48..64] (the two lanes), which Q8's order of x matches.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn nvfp4_weights(q: *const u8, lut: __m256i) -> [__m256i; 4] {
    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let r0 = _mm256_loadu_si256(q as *const __m256i);
        let r1 = _mm256_loadu_si256(q.add(32) as *const __m256i);
        let (a0, b0) = (_mm256_shuffle_epi8(lut, _mm256_and_si256(r0, m4)), _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(r0, 4), m4)));
        let (a1, b1) = (_mm256_shuffle_epi8(lut, _mm256_and_si256(r1, m4)), _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(r1, 4), m4)));
        [_mm256_unpacklo_epi8(a0, b0), _mm256_unpackhi_epi8(a0, b0), _mm256_unpacklo_epi8(a1, b1), _mm256_unpackhi_epi8(a1, b1)]
    }
}

/// The sums of the 4 dots of 128 values (`$dot(w, x)`: 8 sums of 4 products each) reduced to the 8 lanes of 16 values
/// (chunks 0, 1, 4, 5, 2, 3, 6, 7), with x's sums times 12 taken back.
macro_rules! nvfp4_sums {
    ($w:expr, $xb:expr, $c12:expr, $dot:expr) => {{
        let (w, xb): ([__m256i; 4], *const i8) = ($w, $xb);
        let q0 = $dot(w[0], _mm256_loadu_si256(xb as *const __m256i));
        let q1 = $dot(w[1], _mm256_loadu_si256(xb.add(32) as *const __m256i));
        let q2 = $dot(w[2], _mm256_loadu_si256(xb.add(64) as *const __m256i));
        let q3 = $dot(w[3], _mm256_loadu_si256(xb.add(96) as *const __m256i));
        let h = _mm256_hadd_epi32(_mm256_hadd_epi32(q0, q1), _mm256_hadd_epi32(q2, q3));
        _mm256_sub_epi32(h, _mm256_loadu_si256($c12 as *const __m256i))
    }};
}
pub(crate) use nvfp4_sums;

/// The body of the NVFP4 row kernels, 128 values a step: the scales of the step converted at once (`$fast`: all
/// normal, 2 instructions; else e4m3h_8).
macro_rules! nvfp4_body {
    ($q:expr, $s:expr, $x:expr, $fast:expr, $dot:expr) => {{
        let (q, s, x): (&[u8], &[u8], &Q8) = ($q, $s, $x);
        let lut = fp4_lut_u();
        let mut acc = _mm256_setzero_ps();
        let (qp, xp, sp, dp, cp) = (q.as_ptr(), x.qs.as_ptr(), s.as_ptr(), x.dv.as_ptr(), x.c12.as_ptr());
        for it in 0..q.len() / 64 {
            let w = nvfp4_weights(qp.add(64 * it), lut);
            let h = nvfp4_sums!(w, xp.add(128 * it), cp.add(8 * it), $dot);
            let v = scale_lanes(sp.add(8 * it));
            let sc = if $fast { _mm256_castsi256_ps(_mm256_add_epi32(_mm256_slli_epi32(v, 20), _mm256_set1_epi32(0x3B80_0000))) } else { e4m3h_8(v) };
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(h), _mm256_mul_ps(sc, _mm256_loadu_ps(dp.add(8 * it))), acc);
        }
        hsum(acc)
    }};
}

/// Whether the E4M3 scales `s` are all normal and positive (exponent above 0, sign clear): then the kernels convert
/// them with a shift and an add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn normal_scales(s: &[u8]) -> bool {
    unsafe {
        let (e, z, top) = (_mm256_set1_epi8(0x78), _mm256_setzero_si256(), _mm256_set1_epi8(0x80u8 as i8));
        let mut bad = _mm256_setzero_si256();
        let n = s.len() / 32;
        for i in 0..n {
            let v = _mm256_loadu_si256(s.as_ptr().add(32 * i) as *const __m256i);
            bad = _mm256_or_si256(bad, _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(v, e), z), _mm256_and_si256(v, top)));
        }
        // the top bit of a byte of `bad` marks a bad scale
        _mm256_movemask_epi8(bad) == 0 && s[32 * n..].iter().all(|&b| b & 0x78 != 0 && b & 0x80 == 0)
    }
}

static VNNI: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    return is_x86_feature_detected!("avxvnni") && std::env::var("IK_NO_VNNI").is_err();
    #[allow(unreachable_code)]
    false
});

/// NVFP4 row . x (k a multiple of 128), by AVX-VNNI where the CPU has it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_nvfp4_avx2(q: &[u8], s: &[u8], x: &Q8) -> f32 {
    unsafe {
        let fast = normal_scales(s);
        match (*VNNI, fast) {
            (true, true) => dot_nvfp4_vnni::<true>(q, s, x),
            (true, false) => dot_nvfp4_vnni::<false>(q, s, x),
            (false, true) => dot_nvfp4_maddubs::<true>(q, s, x),
            (false, false) => dot_nvfp4_maddubs::<false>(q, s, x),
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c,avxvnni")]
unsafe fn dot_nvfp4_vnni<const FAST: bool>(q: &[u8], s: &[u8], x: &Q8) -> f32 {
    unsafe { nvfp4_body!(q, s, x, FAST, |w, xv| _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w, xv)) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_nvfp4_maddubs<const FAST: bool>(q: &[u8], s: &[u8], x: &Q8) -> f32 {
    let ones = _mm256_set1_epi16(1);
    unsafe { nvfp4_body!(q, s, x, FAST, |w, xv| _mm256_madd_epi16(_mm256_maddubs_epi16(w, xv), ones)) }
}

/// The 32 E4M3 values at q as 4 vectors of 8 f32, each the value times 2^-8 (exactly: an E4M3 byte in the high byte of
/// an f16, shifted right by one with its sign, is that f16), in the order of elements 0..8, 8..16, 16..24, 24..32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
pub(crate) unsafe fn e4m3x32(q: *const u8) -> [__m256; 4] {
    unsafe {
        let (z, mask) = (_mm256_setzero_si256(), _mm256_set1_epi16(0xBF80u16 as i16));
        let raw = _mm256_loadu_si256(q as *const __m256i);
        // [0..8 | 16..24] and [8..16 | 24..32] as f16
        let lo = _mm256_and_si256(_mm256_srai_epi16(_mm256_unpacklo_epi8(z, raw), 1), mask);
        let hi = _mm256_and_si256(_mm256_srai_epi16(_mm256_unpackhi_epi8(z, raw), 1), mask);
        [
            _mm256_cvtph_ps(_mm256_castsi256_si128(lo)),
            _mm256_cvtph_ps(_mm256_castsi256_si128(hi)),
            _mm256_cvtph_ps(_mm256_extracti128_si256(lo, 1)),
            _mm256_cvtph_ps(_mm256_extracti128_si256(hi, 1)),
        ]
    }
}

/// A UE8M0 scale times the 2^8 of e4m3x32: the f32 of exponent s + 8 (s is at most 246).
#[inline]
pub(crate) fn ue8m0_x256(s: u8) -> f32 {
    f32::from_bits((s as u32 + 8) << 23)
}

/// MXFP8 row . x (k a multiple of 64): two blocks a step, each in its own sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_mxfp8_avx2(q: &[u8], s: &[u8], x: &[f32]) -> f32 {
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 2];
        let (qp, xp) = (q.as_ptr(), x.as_ptr());
        for b2 in 0..s.len() / 2 {
            for (j, a) in acc.iter_mut().enumerate() {
                let b = 2 * b2 + j;
                let w = e4m3x32(qp.add(32 * b));
                let xb = xp.add(32 * b);
                let mut p = _mm256_mul_ps(w[0], _mm256_loadu_ps(xb));
                p = _mm256_fmadd_ps(w[1], _mm256_loadu_ps(xb.add(8)), p);
                p = _mm256_fmadd_ps(w[2], _mm256_loadu_ps(xb.add(16)), p);
                p = _mm256_fmadd_ps(w[3], _mm256_loadu_ps(xb.add(24)), p);
                *a = _mm256_fmadd_ps(p, _mm256_set1_ps(ue8m0_x256(*s.get_unchecked(b))), *a);
            }
        }
        hsum(_mm256_add_ps(acc[0], acc[1]))
    }
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

    pub fn rnd(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// Codes and scales of a random matrix of `rows` rows of k in `fmt`, with sane scales; NVFP4 ones all normal unless
    /// `odd`, then about one in 16 subnormal.
    pub fn random_mat(fmt: Fmt, rows: usize, k: usize, odd: bool, seed: &mut u64) -> (Vec<u8>, Vec<u8>) {
        let (qb, sb) = fmt.parts(rows, k);
        match fmt {
            Fmt::NvFp4 => {
                let q = (0..qb).map(|_| rnd(seed) as u8).collect();
                let mut s: Vec<u8> = (0..sb).map(|_| if odd && rnd(seed) % 16 == 0 { (rnd(seed) % 8) as u8 } else { 0x20 + (rnd(seed) % 24) as u8 }).collect();
                if odd {
                    // one subnormal scale in each row at least
                    for r in 0..rows {
                        s[r * sb / rows] = 3;
                    }
                }
                (q, s)
            }
            Fmt::MxFp8 => {
                let q = (0..qb)
                    .map(|_| loop {
                        let b = rnd(seed) as u8;
                        if b & 0x7F != 0x7F {
                            break b;
                        }
                    })
                    .collect();
                let s = (0..sb).map(|_| 110 + (rnd(seed) % 12) as u8).collect();
                (q, s)
            }
        }
    }

    pub fn random_x(k: usize, seed: &mut u64) -> Vec<f32> {
        (0..k).map(|_| (rnd(seed) % 2000) as f32 / 1000.0 - 1.0).collect()
    }

    pub fn act(x: &[f32], fmt: Fmt) -> Act {
        let mut a = Act::default();
        a.set(x, fmt);
        a
    }

    #[test]
    fn q8_layout() {
        // pos is a permutation of each 64
        let mut seen = [false; 128];
        for i in 0..128 {
            assert!(!seen[pos(i)]);
            seen[pos(i)] = true;
        }
        let mut seed = 0x5eed_0001_u64;
        let x = random_x(256, &mut seed);
        let mut q = Q8::default();
        quantize(&x, &mut q);
        for i in 0..256 {
            let v = q.q(i) as f32 * q.d[i / 32];
            assert!((v - x[i]).abs() <= q.d[i / 32] * 0.5 + 1e-6, "element {i}: {v} vs {}", x[i]);
        }
        for b in 0..8 {
            let s16 = |h: usize| (32 * b + 16 * h..32 * b + 16 * h + 16).map(|i| q.q(i) as i32).sum::<i32>();
            assert_eq!((q.dv[lane(b)], q.dv[lane(b) + 1]), (q.d[b], q.d[b]));
            assert_eq!((q.c12[lane(b)], q.c12[lane(b) + 1]), (12 * s16(0), 12 * s16(1)));
        }
    }

    #[test]
    fn quantize_simd_matches() {
        if !simd() {
            return;
        }
        let mut seed = 0x5eed_0002_u64;
        for case in 0..200 {
            let xb: Vec<f32> = match case {
                // ties: amax 127 makes the scale 1, and halves round away from zero
                0 => (0..32).map(|i| if i == 0 { 127.0 } else { (i as f32 - 16.0) + 0.5 }).collect(),
                1 => vec![0.0; 32],
                _ => (0..32).map(|_| (rnd(&mut seed) % 20001) as f32 / 1000.0 - 10.0).collect(),
            };
            let (mut a, mut b) = ([0i8; 32], [0i8; 32]);
            let r0 = quantize_block_scalar(&xb, &mut a);
            // SAFETY: the features are present
            let r1 = unsafe { quantize_block_avx2(&xb, &mut b) };
            assert_eq!((a, r0.0.to_bits(), r0.1), (b, r1.0.to_bits(), r1.1), "case {case}");
        }
    }

    #[test]
    fn e4m3_simd() {
        if !simd() {
            return;
        }
        for x in (0..256).step_by(8) {
            let mut got = [0f32; 8];
            // SAFETY: avx2 is present
            unsafe {
                let v = _mm256_setr_epi32(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
                _mm256_storeu_ps(got.as_mut_ptr(), e4m3h_8(v));
            }
            for (i, g) in got.iter().enumerate() {
                assert_eq!(*g, 0.5 * e4m3_abs((x as usize + i) as u8), "{:#x}", x as usize + i);
            }
        }
        // MXFP8 values: all 256 bytes, the NaNs too (as 480, as e4m3 has them)
        let q: Vec<u8> = (0..=255u8).collect();
        for c in 0..8 {
            let mut got = [0f32; 32];
            // SAFETY: as above, 32 bytes
            unsafe {
                let v = e4m3x32(q.as_ptr().add(32 * c));
                for (j, w) in v.iter().enumerate() {
                    _mm256_storeu_ps(got.as_mut_ptr().add(8 * j), *w);
                }
            }
            for (i, g) in got.iter().enumerate() {
                let b = (32 * c + i) as u8;
                assert_eq!(g.to_bits(), (e4m3(b) / 256.0).to_bits(), "{b:#x}");
            }
        }
    }

    #[test]
    fn nvfp4_kernels_agree() {
        if !simd() {
            return;
        }
        let mut seed = 0x0bad_cafe_1234_5678u64;
        for k in [2304, 1152] {
            for odd in [false, true] {
                let (q, s) = random_mat(Fmt::NvFp4, 4, k, odd, &mut seed);
                let m = Mat { fmt: Fmt::NvFp4, q: &q, s: &s, factor: 1.0, rows: 4, k };
                let x = act(&random_x(k, &mut seed), Fmt::NvFp4);
                for r in 0..4 {
                    let (rq, rs) = (m.qs(r), m.sc(r));
                    let want = dot_nvfp4_ref(rq, rs, &x.q);
                    // SAFETY: avx2 is present (and avx-vnni where used)
                    let normal = unsafe { normal_scales(rs) };
                    assert_eq!(normal, !odd, "row {r}");
                    let mut got = vec![unsafe { dot_nvfp4_maddubs::<false>(rq, rs, &x.q) }, dot_row(&m, r, &x)];
                    if normal {
                        got.push(unsafe { dot_nvfp4_maddubs::<true>(rq, rs, &x.q) });
                    }
                    if is_x86_feature_detected!("avxvnni") {
                        got.push(unsafe { dot_nvfp4_vnni::<false>(rq, rs, &x.q) });
                    }
                    for g in got {
                        assert!((g - want).abs() <= 1e-5 * want.abs().max(1.0), "k {k} row {r}: {g} vs scalar {want}");
                    }
                }
            }
        }
    }

    #[test]
    fn dots_match_dequant() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for (fmt, k) in [(Fmt::NvFp4, 2304), (Fmt::NvFp4, 1152), (Fmt::MxFp8, 2304), (Fmt::MxFp8, 1152)] {
            let (q, s) = random_mat(fmt, 3, k, true, &mut seed);
            let m = Mat { fmt, q: &q, s: &s, factor: 1.0, rows: 3, k };
            let x = random_x(k, &mut seed);
            let a = act(&x, fmt);
            for r in 0..3 {
                let mut w = vec![0f32; k];
                m.dequant_row(r, &mut w);
                let want: f64 = w.iter().zip(&x).map(|(a, b)| *a as f64 * *b as f64).sum();
                let l1: f64 = w.iter().zip(&x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
                let (got, sc) = (dot_row(&m, r, &a) as f64, dot_row_scalar(&m, r, &a) as f64);
                assert!((got - sc).abs() <= 1e-5 * sc.abs().max(1.0), "{fmt:?} k {k}: avx2 {got} scalar {sc}");
                // int8 x for NVFP4 (within 1% of the sum of |w x|), f32 for MXFP8 (rounding only)
                let tol = if fmt == Fmt::NvFp4 { 0.01 } else { 1e-5 };
                assert!((got - want).abs() <= tol * l1, "{fmt:?} k {k}: {got} vs f64 {want} (sum of |w x| {l1})");
            }
        }
    }
}
