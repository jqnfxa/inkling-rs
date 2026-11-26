//! CPU kernels of the experts that miss the GPU cache, AVX2: rows as the checkpoint has them (experts.rs), ModelOpt
//! NVFP4 and MXFP8, against f32 activations. The kernels decode the stored values exactly (through f16) and sum in f32.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::experts::{FP4X2, Fmt, Mat, e4m3, e4m3_abs, ue8m0};

#[inline]
pub(crate) fn simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    return is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
    #[allow(unreachable_code)]
    false
}

/// Where element i of the activations goes in the order that [`dot_row`] takes them for rows of `fmt`. NVFP4: in each
/// block of 16 its 8 even elements, then its 8 odd ones, as the kernel decodes the low and the high nibbles of a
/// block's bytes into vectors of their own. MXFP8: as they are.
#[inline]
pub fn x_pos(fmt: Fmt, i: usize) -> usize {
    match fmt {
        Fmt::NvFp4 => (i & !15) | ((i & 1) << 3) | ((i & 15) >> 1),
        Fmt::MxFp8 => i,
    }
}

/// `x` in the order that [`dot_row`] takes it for rows of `fmt` (see [`x_pos`]), into out[..x.len()].
pub fn order_x(fmt: Fmt, x: &[f32], out: &mut [f32]) {
    for (i, &v) in x.iter().enumerate() {
        out[x_pos(fmt, i)] = v;
    }
}

/// Row r of `m` . x (k values), the expert's factor left out (the caller applies m.scale()). x is in the order of
/// [`order_x`].
pub fn dot_row(m: &Mat, r: usize, x: &[f32]) -> f32 {
    assert!(x.len() >= m.k, "x of {} values for rows of {}", x.len(), m.k);
    #[cfg(target_arch = "x86_64")]
    if simd() && m.k % 64 == 0 {
        // SAFETY: the features are present; x holds k values
        return unsafe {
            match m.fmt {
                Fmt::NvFp4 => dot_nvfp4_avx2(m.qs(r), m.sc(r), x),
                Fmt::MxFp8 => dot_mxfp8_avx2(m.qs(r), m.sc(r), x),
            }
        };
    }
    let xn: Vec<f32> = (0..m.k).map(|i| x[x_pos(m.fmt, i)]).collect();
    dot_row_scalar(m, r, &xn)
}

/// The scalar reference of [`dot_row`].
pub fn dot_row_scalar(m: &Mat, r: usize, x: &[f32]) -> f32 {
    match m.fmt {
        Fmt::NvFp4 => dot_nvfp4_ref(m.qs(r), m.sc(r), x),
        Fmt::MxFp8 => dot_mxfp8_ref(m.qs(r), m.sc(r), x),
    }
}

/// An NVFP4 row (E2M1 codes q, two a byte, the low nibble first; E4M3 scales s, one per 16 values) . x.
pub fn dot_nvfp4_ref(q: &[u8], s: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0f32;
    for (b, &sc) in s.iter().enumerate() {
        let sum: f32 = (16 * b..16 * b + 16).map(|i| FP4X2[((q[i / 2] >> (4 * (i % 2))) & 15) as usize] as f32 * 0.5 * x[i]).sum();
        acc += sum * e4m3_abs(sc);
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

/// For pshufb, in both 128-bit lanes: the high byte of the f16 of each E2M1 code times 2^-14, which is exact. The code
/// s e1 e0 m goes to s 0 0 0 e1 e0 m 0: the exponent e1 e0 over f16's bias of 15 against E2M1's 1, and with e1 e0 = 0
/// the subnormals of both (m/2 against m/2 2^-14).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn e2m1_lut() -> __m256i {
    let t = |c: u8| (((c & 8) << 4) | ((c & 7) << 1)) as i8;
    _mm256_setr_epi8(
        t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12), t(13), t(14), t(15),
        t(0), t(1), t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9), t(10), t(11), t(12), t(13), t(14), t(15),
    )
}

/// An NVFP4 scale byte's magnitude (sign ignored, as e4m3_abs) times the 2^14 that e2m1x64 takes off the values, so a
/// block's sum times it is the block's dot: normal ones the f32 of exponent e - 7 + 14 over their 3 mantissa bits,
/// subnormal ones m 2^-9 2^14.
pub(crate) const E4M3_X2_14: [f32; 128] = {
    let mut t = [0f32; 128];
    let mut v = 0;
    while v < 128 {
        t[v] = if v < 8 { (32 * v) as f32 } else { f32::from_bits(((v as u32) << 20) + (134 << 23)) };
        v += 1;
    }
    t
};

/// The 64 E2M1 values of 32 code bytes at q, each times 2^-14 (exactly, see e2m1_lut), as 8 vectors of 8 f32 in the
/// order of elements 0..8, 8..16, .., 56..64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
#[inline]
pub(crate) unsafe fn e2m1x64(q: *const u8, lut: __m256i) -> [__m256; 8] {
    unsafe {
        let (z, m4) = (_mm256_setzero_si256(), _mm256_set1_epi8(0xF));
        let r = _mm256_loadu_si256(q as *const __m256i);
        // the high bytes of the even elements and of the odd ones
        let ev = _mm256_shuffle_epi8(lut, _mm256_and_si256(r, m4));
        let od = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(r, 4), m4));
        // in order: elements [0..16 | 32..48] and [16..32 | 48..64]
        let (a, b) = (_mm256_unpacklo_epi8(ev, od), _mm256_unpackhi_epi8(ev, od));
        // f16 of [0..8 | 32..40], [8..16 | 40..48], [16..24 | 48..56], [24..32 | 56..64]
        let (h0, h1) = (_mm256_unpacklo_epi8(z, a), _mm256_unpackhi_epi8(z, a));
        let (h2, h3) = (_mm256_unpacklo_epi8(z, b), _mm256_unpackhi_epi8(z, b));
        [
            _mm256_cvtph_ps(_mm256_castsi256_si128(h0)),
            _mm256_cvtph_ps(_mm256_castsi256_si128(h1)),
            _mm256_cvtph_ps(_mm256_castsi256_si128(h2)),
            _mm256_cvtph_ps(_mm256_castsi256_si128(h3)),
            _mm256_cvtph_ps(_mm256_extracti128_si256(h0, 1)),
            _mm256_cvtph_ps(_mm256_extracti128_si256(h1, 1)),
            _mm256_cvtph_ps(_mm256_extracti128_si256(h2, 1)),
            _mm256_cvtph_ps(_mm256_extracti128_si256(h3, 1)),
        ]
    }
}

/// The bf16 of each E2M1 code (sign c & 8, magnitude c & 7), which is its value exactly: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
const E2M1_BF16: [u16; 16] = {
    const MAG: [u16; 8] = [0x0000, 0x3F00, 0x3F80, 0x3FC0, 0x4000, 0x4040, 0x4080, 0x40C0];
    let mut t = [0u16; 16];
    let mut c = 0;
    while c < 16 {
        t[c] = MAG[c & 7] | if c & 8 != 0 { 0x8000 } else { 0 };
        c += 1;
    }
    t
};

/// For pshufb, in both 128-bit lanes: the low (`hi` false) or the high byte of E2M1_BF16 of each code.
const fn e2m1_bf16_bytes(hi: bool) -> [u8; 32] {
    let mut t = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        let v = E2M1_BF16[i & 15];
        t[i] = if hi { (v >> 8) as u8 } else { v as u8 };
        i += 1;
    }
    t
}
const E2M1_LO: [u8; 32] = e2m1_bf16_bytes(false);
const E2M1_HI: [u8; 32] = e2m1_bf16_bytes(true);

/// An NVFP4 scale byte's magnitude (sign ignored, as e4m3_abs): normal ones the f32 of exponent e - 7 over their 3
/// mantissa bits, subnormal ones m 2^-9.
pub(crate) const E4M3_ABS: [f32; 128] = {
    let mut t = [0f32; 128];
    let mut v = 0;
    while v < 128 {
        t[v] = if v < 8 { v as f32 / 512.0 } else { f32::from_bits(((v as u32) << 20) + (120 << 23)) };
        v += 1;
    }
    t
};

/// NVFP4 row . x (k a multiple of 64, x in the order of x_pos): 64 values a step, each block of 16 summed before its
/// scale, the 4 blocks of a step in sums of their own. The codes go to the bf16 of their values by two table lookups
/// (pshufb) and to f32 by putting the bf16 in the high half of a lane; the step's dwords are first permuted so that
/// each vector of 8 holds the even or the odd elements of one block.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_nvfp4_avx2(q: &[u8], s: &[u8], x: &[f32]) -> f32 {
    unsafe {
        let (lo_t, hi_t) = (_mm256_loadu_si256(E2M1_LO.as_ptr() as *const __m256i), _mm256_loadu_si256(E2M1_HI.as_ptr() as *const __m256i));
        let (z, m4) = (_mm256_setzero_si256(), _mm256_set1_epi8(0xF));
        // dwords [0 2 4 6 | 1 3 5 7]: each lane holds one half of every block (a block is 2 dwords)
        let perm = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
        let mut acc = [_mm256_setzero_ps(); 4];
        let (qp, sp, xp) = (q.as_ptr(), s.as_ptr(), x.as_ptr());
        for c in 0..q.len() / 32 {
            let r = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(qp.add(32 * c) as *const __m256i), perm);
            let (ev, od) = (_mm256_and_si256(r, m4), _mm256_and_si256(_mm256_srli_epi16(r, 4), m4));
            let (el, eh) = (_mm256_shuffle_epi8(lo_t, ev), _mm256_shuffle_epi8(hi_t, ev));
            let (ol, oh) = (_mm256_shuffle_epi8(lo_t, od), _mm256_shuffle_epi8(hi_t, od));
            // bf16 of the even (odd) elements: e01 holds blocks 0 and 1, e23 blocks 2 and 3
            let (e01, e23) = (_mm256_unpacklo_epi8(el, eh), _mm256_unpackhi_epi8(el, eh));
            let (o01, o23) = (_mm256_unpacklo_epi8(ol, oh), _mm256_unpackhi_epi8(ol, oh));
            let w = [
                (_mm256_unpacklo_epi16(z, e01), _mm256_unpacklo_epi16(z, o01)),
                (_mm256_unpackhi_epi16(z, e01), _mm256_unpackhi_epi16(z, o01)),
                (_mm256_unpacklo_epi16(z, e23), _mm256_unpacklo_epi16(z, o23)),
                (_mm256_unpackhi_epi16(z, e23), _mm256_unpackhi_epi16(z, o23)),
            ];
            let xb = xp.add(64 * c);
            for (b, a) in acc.iter_mut().enumerate() {
                let p = _mm256_mul_ps(_mm256_castsi256_ps(w[b].0), _mm256_loadu_ps(xb.add(16 * b)));
                let p = _mm256_fmadd_ps(_mm256_castsi256_ps(w[b].1), _mm256_loadu_ps(xb.add(16 * b + 8)), p);
                *a = _mm256_fmadd_ps(p, _mm256_set1_ps(E4M3_ABS[(*sp.add(4 * c + b) & 0x7F) as usize]), *a);
            }
        }
        hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])))
    }
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

/// dst = src by non-temporal stores, which skip the read of dst's lines and leave them out of the caches: for host
/// copies that the GPU then reads by DMA while the other cores keep DRAM busy, where they copy about 1.7 times as fast
/// as memcpy. Call [`store_fence`] before the GPU may read dst.
pub fn copy_nt(dst: &mut [u8], src: &[u8]) {
    assert_eq!(dst.len(), src.len(), "copy_nt of {} bytes to {}", src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if simd() && (dst.as_ptr() as usize).is_multiple_of(32) {
        // SAFETY: AVX2 is there, dst is aligned for the stores, and the lengths are equal
        unsafe { copy_nt_avx2(dst, src) };
        return;
    }
    dst.copy_from_slice(src);
}

/// copy_nt's 128 bytes a step: 4 loads, then 4 streaming stores (dst 32-byte aligned); the tail by memcpy.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_nt_avx2(dst: &mut [u8], src: &[u8]) {
    let n = dst.len() / 128 * 128;
    let (dp, sp) = (dst.as_mut_ptr(), src.as_ptr());
    for i in (0..n).step_by(128) {
        // SAFETY: i + 128 <= n <= the lengths, and dp + i is 32-byte aligned
        unsafe {
            let (s, d) = (sp.add(i) as *const __m256i, dp.add(i) as *mut __m256i);
            let (a, b, c, e) = (_mm256_loadu_si256(s), _mm256_loadu_si256(s.add(1)), _mm256_loadu_si256(s.add(2)), _mm256_loadu_si256(s.add(3)));
            _mm256_stream_si256(d, a);
            _mm256_stream_si256(d.add(1), b);
            _mm256_stream_si256(d.add(2), c);
            _mm256_stream_si256(d.add(3), e);
        }
    }
    dst[n..].copy_from_slice(&src[n..]);
}

/// Orders the stores of [`copy_nt`] before the stores after it, such as the driver's for a DMA from dst.
pub fn store_fence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: sfence has no operands
    unsafe {
        _mm_sfence()
    };
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// copy_nt gives each byte of the source, at lengths around its 128-byte steps, and leaves the bytes past dst
    #[test]
    fn copy_nt_bytes() {
        let src: Vec<u8> = (0..5000u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
        let mut buf = vec![0u8; 5000 + 64];
        let base = buf.as_ptr().align_offset(32);
        for (off, len) in [(0, 0), (0, 31), (0, 128), (0, 129), (0, 4095), (32, 4096), (64, 4999), (3, 300)] {
            buf.fill(0xA5);
            copy_nt(&mut buf[base + off..base + off + len], &src[..len]);
            store_fence();
            assert_eq!(&buf[base + off..base + off + len], &src[..len], "off {off} len {len}");
            assert!(buf[base + off + len..].iter().all(|&b| b == 0xA5), "past the copy: off {off} len {len}");
        }
    }

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

    /// Row r of m . x in f64 from dequant_row, and the sum of |w x|, the scale of an f32 sum's rounding: within 1e-5 of
    /// it the kernels' sums are rounding, and one value decoded wrong is about 1/k of it.
    pub fn dot_f64(m: &Mat, r: usize, x: &[f32]) -> (f64, f64) {
        let mut w = vec![0f32; m.k];
        m.dequant_row(r, &mut w);
        let want = w.iter().zip(x).map(|(a, b)| *a as f64 * *b as f64).sum();
        let l1 = w.iter().zip(x).map(|(a, b)| (*a as f64 * *b as f64).abs()).sum();
        (want, l1)
    }

    #[test]
    fn scale_table() {
        for v in 0..128u8 {
            assert_eq!(E4M3_X2_14[v as usize].to_bits(), (e4m3_abs(v) * 16384.0).to_bits(), "{v:#x}");
            assert_eq!(E4M3_ABS[v as usize].to_bits(), e4m3_abs(v).to_bits(), "{v:#x}");
        }
    }

    #[test]
    fn e2m1_bf16_table() {
        for c in 0..16 {
            assert_eq!(f32::from_bits((E2M1_BF16[c] as u32) << 16), FP4X2[c] as f32 * 0.5, "code {c}");
        }
    }

    #[test]
    fn x_order() {
        for fmt in [Fmt::NvFp4, Fmt::MxFp8] {
            let mut seen = vec![false; 2304];
            for i in 0..2304 {
                let p = x_pos(fmt, i);
                assert!(p / 16 == i / 16 && !seen[p], "{fmt:?} {i} -> {p}");
                seen[p] = true;
            }
        }
    }

    #[test]
    fn decode_simd() {
        if !simd() {
            return;
        }
        let q: Vec<u8> = (0..=255u8).collect();
        // E2M1: all 256 code bytes, low nibble first (code 8 as -0)
        for c in 0..8 {
            let mut got = [0f32; 64];
            // SAFETY: the features are present; 32 bytes
            unsafe {
                let v = e2m1x64(q.as_ptr().add(32 * c), e2m1_lut());
                for (j, w) in v.iter().enumerate() {
                    _mm256_storeu_ps(got.as_mut_ptr().add(8 * j), *w);
                }
            }
            for (i, g) in got.iter().enumerate() {
                let b = q[32 * c + i / 2];
                let code = (b >> (4 * (i % 2))) & 15;
                assert_eq!(*g * 16384.0, FP4X2[code as usize] as f32 * 0.5, "byte {b:#x} nibble {}", i % 2);
            }
        }
        // MXFP8 values: all 256 bytes, the NaNs too (as 480, as e4m3 has them)
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
    fn dots_match_dequant() {
        let mut seed = 0x1234_5678_9abc_def1u64;
        for (fmt, k) in [(Fmt::NvFp4, 2304), (Fmt::NvFp4, 1152), (Fmt::MxFp8, 2304), (Fmt::MxFp8, 1152)] {
            for odd in [false, true] {
                let (q, s) = random_mat(fmt, 3, k, odd, &mut seed);
                let m = Mat { fmt, q: &q, s: &s, factor: 1.0, rows: 3, k };
                let x = random_x(k, &mut seed);
                let mut xo = vec![0f32; k];
                order_x(fmt, &x, &mut xo);
                for r in 0..3 {
                    let (want, l1) = dot_f64(&m, r, &x);
                    for (name, got) in [("dot_row", dot_row(&m, r, &xo)), ("scalar", dot_row_scalar(&m, r, &x))] {
                        assert!((got as f64 - want).abs() <= 1e-5 * l1, "{fmt:?} k {k} row {r} {name}: {got} vs f64 {want} (sum of |w x| {l1})");
                    }
                }
            }
        }
    }
}
