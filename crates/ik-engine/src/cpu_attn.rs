//! Decode attention on the CPU over the full-attention KV rows that are in host memory, for attn_reduce to merge with
//! the GPU's chunks of the rows in VRAM. As attn_part: query head h (3 a KV head) scores the key of the row at
//! distance d as scale q.k, plus its relative bias where d is under the extent, less its sink; the partial of a set
//! of distances is m = max s, l = sum exp(s - m) and acc = sum exp(s - m) v.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::cpu::simd;

pub const HEADS: usize = 24;
pub const KV_HEADS: usize = 8;
pub const DIM: usize = 128;
pub const N_REL: usize = 16;
/// f16 values of a K or V row: 8 KV heads of 128
pub const ROW: usize = KV_HEADS * DIM;

/// One token's queries.
pub struct Query<'a> {
    /// [24][128], clipped and times tau (clip_qr)
    pub q: &'a [f32],
    /// [24][16]
    pub r: &'a [f32],
    /// [24]
    pub sink: &'a [f32],
    /// the relative bias profiles, [16][extent] bf16
    pub proj: &'a [u16],
    pub extent: usize,
    pub scale: f32,
    /// the token's position: the row at distance d is pos - d
    pub pos: usize,
}

/// The softmax partial of a set of distances, for one head.
#[derive(Clone, Copy, Debug)]
pub struct Partial {
    pub m: f32,
    pub l: f32,
    pub acc: [f32; DIM],
}

impl Partial {
    pub const EMPTY: Partial = Partial { m: f32::NEG_INFINITY, l: 0.0, acc: [0.0; DIM] };

    /// self and o as one partial.
    pub fn merge(&mut self, o: &Partial) {
        if o.l == 0.0 {
            return;
        }
        if self.l == 0.0 {
            *self = *o;
            return;
        }
        let m = self.m.max(o.m);
        let (a, b) = ((self.m - m).exp(), (o.m - m).exp());
        self.l = self.l * a + o.l * b;
        for (x, y) in self.acc.iter_mut().zip(&o.acc) {
            *x = *x * a + *y * b;
        }
        self.m = m;
    }
}

pub fn f16_f32(h: u16) -> f32 {
    let (s, e, m) = (((h >> 15) as u32) << 31, ((h >> 10) & 31) as u32, (h & 1023) as u32);
    let mag = match e {
        // subnormal: m 2^-24
        0 => (m as f32 * 2f32.powi(-24)).to_bits(),
        31 => 0x7F80_0000 | (m << 13),
        _ => ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(s | mag)
}

pub fn bf16_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Rows scored at a time: their scores, then one rescale of each head's sums, then their values.
const BLOCK: usize = 32;
/// While a row is scored, its V row is fetched for the block's values, and the K row this many rows on for the scores:
/// two rows fill a 4 KiB page, where the hardware prefetcher stops. ik-cpuattn with 8 threads: 30.5 GB/s without the
/// fetches, 33.5 with K 2 rows ahead (31.8 at 1, 32.9 at 4, 31.5 at 16); DRAM gives 33 to 34.
const AHEAD: usize = 2;

/// The partials of the 24 heads over distances lo..hi, merged into `out`. k and v are row 0 of the caches.
///
/// # Safety
/// Rows pos - hi + 1 ..= pos - lo of k and v hold ROW f16 values each, readable.
pub unsafe fn rows(qy: &Query, k: *const u16, v: *const u16, lo: usize, hi: usize, out: &mut [Partial; HEADS]) {
    assert!(lo <= hi && hi <= qy.pos + 1 && qy.q.len() >= HEADS * DIM && qy.r.len() >= HEADS * N_REL && qy.sink.len() >= HEADS && qy.proj.len() >= N_REL * qy.extent);
    #[cfg(target_arch = "x86_64")]
    if simd() {
        // SAFETY: the CPU has AVX2, FMA and F16C; the rows as the caller says
        return unsafe { rows_avx2(qy, k, v, lo, hi, out) };
    }
    // SAFETY: as the caller says
    unsafe { rows_scalar(qy, k, v, lo, hi, out) }
}

/// The relative bias of head h at distance d: under the extent, its r times the profiles at d.
fn bias(qy: &Query, h: usize, d: usize) -> f32 {
    if d >= qy.extent {
        return 0.0;
    }
    (0..N_REL).map(|rr| qy.r[h * N_REL + rr] * bf16_f32(qy.proj[rr * qy.extent + d])).sum()
}

/// rows, one value at a time.
unsafe fn rows_scalar(qy: &Query, k: *const u16, v: *const u16, lo: usize, hi: usize, out: &mut [Partial; HEADS]) {
    let qs: Vec<f32> = qy.q[..HEADS * DIM].iter().map(|&x| x * qy.scale).collect();
    let mut s = [[0f32; HEADS]; BLOCK];
    let mut d0 = lo;
    while d0 < hi {
        let nb = (hi - d0).min(BLOCK);
        for (b, sb) in s.iter_mut().enumerate().take(nb) {
            let d = d0 + b;
            for (h, x) in sb.iter_mut().enumerate() {
                // SAFETY: the row is readable (the caller)
                let kh = unsafe { k.add((qy.pos - d) * ROW + h / 3 * DIM) };
                let dot: f32 = (0..DIM).map(|i| qs[h * DIM + i] * f16_f32(unsafe { *kh.add(i) })).sum();
                *x = dot + bias(qy, h, d) - qy.sink[h];
            }
        }
        // each head: the new max, its sums rescaled once, then the block's weights
        for (h, o) in out.iter_mut().enumerate() {
            let bm = s[..nb].iter().map(|sb| sb[h]).fold(f32::NEG_INFINITY, f32::max);
            let m = o.m.max(bm);
            if o.l != 0.0 && m > o.m {
                let a = (o.m - m).exp();
                o.l *= a;
                o.acc.iter_mut().for_each(|x| *x *= a);
            }
            o.m = m;
            for sb in s[..nb].iter_mut() {
                sb[h] = (sb[h] - m).exp();
                o.l += sb[h];
            }
        }
        for (b, sb) in s.iter().enumerate().take(nb) {
            for (h, o) in out.iter_mut().enumerate() {
                // SAFETY: as above
                let vh = unsafe { v.add((qy.pos - d0 - b) * ROW + h / 3 * DIM) };
                for (i, a) in o.acc.iter_mut().enumerate() {
                    *a += sb[h] * f16_f32(unsafe { *vh.add(i) });
                }
            }
        }
        d0 += nb;
    }
}

/// rows with AVX2: a row's 24 scores as 3 vectors of 8 heads, their exps 8 at a time, and the values of a block in
/// tiles of 3 heads by 32 dims that stay in registers over its rows. The rows go in rising addresses, from distance
/// hi - 1 down: in falling ones the DRAM gave 30.7 GB/s, not 33.5.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn rows_avx2(qy: &Query, k: *const u16, v: *const u16, lo: usize, hi: usize, out: &mut [Partial; HEADS]) {
    let qs: Vec<f32> = qy.q[..HEADS * DIM].iter().map(|&x| x * qy.scale).collect();
    // r as [16][24]: a row's bias is the sum over rr of its profile value times rt[rr]
    let rt: Vec<f32> = (0..N_REL * HEADS).map(|i| qy.r[i % HEADS * N_REL + i / HEADS]).collect();
    let sink = [ld8(qy.sink, 0), ld8(qy.sink, 8), ld8(qy.sink, 16)];
    let (mut om, mut ol) = ([0f32; HEADS], [0f32; HEADS]);
    for (h, o) in out.iter().enumerate() {
        (om[h], ol[h]) = (o.m, o.l);
    }
    let (mut m, mut l) = ([ld8(&om, 0), ld8(&om, 8), ld8(&om, 16)], [ld8(&ol, 0), ld8(&ol, 8), ld8(&ol, 16)]);
    // the K or V row at distance d
    let row = |p: *const u16, d: usize| p.wrapping_add((qy.pos - d) * ROW);
    let fetch = |p: *const u16| {
        for c in 0..ROW * 2 / 64 {
            _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(c * 32) as *const i8);
        }
    };
    // the rows in rising addresses: blocks from distance hi - 1 down, row b of a block at distance d1 - 1 - b
    for d in (hi.saturating_sub(AHEAD).max(lo)..hi).rev() {
        fetch(row(k, d));
    }
    let mut s = [[0f32; HEADS]; BLOCK];
    let mut d1 = hi;
    while d1 > lo {
        let nb = (d1 - lo).min(BLOCK);
        for (b, sb) in s.iter_mut().enumerate().take(nb) {
            let d = d1 - 1 - b;
            if d >= lo + AHEAD {
                fetch(row(k, d - AHEAD));
            }
            fetch(row(v, d));
            // SAFETY: the row is readable (the caller)
            let mut x = unsafe { scores(&qs, row(k, d)) };
            if d < qy.extent {
                for rr in 0..N_REL {
                    let p = _mm256_set1_ps(bf16_f32(qy.proj[rr * qy.extent + d]));
                    for (c, x) in x.iter_mut().enumerate() {
                        *x = _mm256_fmadd_ps(ld8(&rt, rr * HEADS + 8 * c), p, *x);
                    }
                }
            }
            for (c, x) in x.iter().enumerate() {
                st8(sb, 8 * c, _mm256_sub_ps(*x, sink[c]));
            }
        }
        // the new maxes, the sums rescaled once where they grew, then the block's weights
        let mut alpha = [0f32; HEADS];
        let mut grew = 0u32;
        for c in 0..3 {
            let mut mn = m[c];
            for sb in &s[..nb] {
                mn = _mm256_max_ps(mn, ld8(sb, 8 * c));
            }
            grew |= (_mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(mn, m[c])) as u32) << (8 * c);
            let a = exp8(_mm256_sub_ps(m[c], mn));
            st8(&mut alpha, 8 * c, a);
            (m[c], l[c]) = (mn, _mm256_mul_ps(l[c], a));
        }
        for (h, o) in out.iter_mut().enumerate().filter(|(h, _)| grew >> h & 1 != 0) {
            let a = _mm256_set1_ps(alpha[h]);
            for i in (0..DIM).step_by(8) {
                let x = _mm256_mul_ps(ld8(&o.acc, i), a);
                st8(&mut o.acc, i, x);
            }
        }
        for sb in s[..nb].iter_mut() {
            for c in 0..3 {
                let p = exp8(_mm256_sub_ps(ld8(sb, 8 * c), m[c]));
                st8(sb, 8 * c, p);
                l[c] = _mm256_add_ps(l[c], p);
            }
        }
        for g in 0..KV_HEADS {
            for t in 0..DIM / 32 {
                // a[i][j]: dims 32 t + 8 i .. of head 3 g + j
                let mut a = [[_mm256_setzero_ps(); 3]; 4];
                for (b, sb) in s.iter().enumerate().take(nb) {
                    let vr = row(v, d1 - 1 - b).wrapping_add(g * DIM + 32 * t);
                    let p = [_mm256_broadcast_ss(&sb[3 * g]), _mm256_broadcast_ss(&sb[3 * g + 1]), _mm256_broadcast_ss(&sb[3 * g + 2])];
                    for (i, a) in a.iter_mut().enumerate() {
                        // SAFETY: the row is readable (the caller)
                        let x = _mm256_cvtph_ps(unsafe { _mm_loadu_si128(vr.add(8 * i) as *const __m128i) });
                        for (a, p) in a.iter_mut().zip(p) {
                            *a = _mm256_fmadd_ps(p, x, *a);
                        }
                    }
                }
                for (i, a) in a.iter().enumerate() {
                    for (j, a) in a.iter().enumerate() {
                        let o = &mut out[3 * g + j].acc;
                        let x = _mm256_add_ps(ld8(o, 32 * t + 8 * i), *a);
                        st8(o, 32 * t + 8 * i, x);
                    }
                }
            }
        }
        d1 -= nb;
    }
    for c in 0..3 {
        st8(&mut om, 8 * c, m[c]);
        st8(&mut ol, 8 * c, l[c]);
    }
    for (h, o) in out.iter_mut().enumerate() {
        (o.m, o.l) = (om[h], ol[h]);
    }
}

/// The 24 heads' dots of the scaled queries qs [24][128] with the K row at k, as 3 vectors of 8 heads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn scores(qs: &[f32], k: *const u16) -> [__m256; 3] {
    assert!(qs.len() >= HEADS * DIM);
    let mut hv = [_mm256_setzero_ps(); HEADS];
    for g in 0..KV_HEADS {
        let mut a = [[_mm256_setzero_ps(); 2]; 3];
        for i in 0..DIM / 16 {
            // SAFETY: the row has 1024 values (the caller)
            let kv = unsafe { [_mm256_cvtph_ps(_mm_loadu_si128(k.add(g * DIM + 16 * i) as *const __m128i)), _mm256_cvtph_ps(_mm_loadu_si128(k.add(g * DIM + 16 * i + 8) as *const __m128i))] };
            for (j, a) in a.iter_mut().enumerate() {
                for (h, a) in a.iter_mut().enumerate() {
                    // SAFETY: qs has 24 * 128 floats
                    *a = _mm256_fmadd_ps(unsafe { _mm256_loadu_ps(qs.as_ptr().add((3 * g + j) * DIM + 16 * i + 8 * h)) }, kv[h], *a);
                }
            }
        }
        for (j, a) in a.iter().enumerate() {
            hv[3 * g + j] = _mm256_add_ps(a[0], a[1]);
        }
    }
    [sum8(&hv[..8]), sum8(&hv[8..16]), sum8(&hv[16..])]
}

/// The 8 floats of p from i.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn ld8(p: &[f32], i: usize) -> __m256 {
    let p = &p[i..i + 8];
    // SAFETY: 8 floats
    unsafe { _mm256_loadu_ps(p.as_ptr()) }
}

/// x into the 8 floats of p from i.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn st8(p: &mut [f32], i: usize, x: __m256) {
    let p = &mut p[i..i + 8];
    // SAFETY: 8 floats
    unsafe { _mm256_storeu_ps(p.as_mut_ptr(), x) }
}

/// Lane i: the sum of the lanes of v[i], i = 0..8.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
fn sum8(v: &[__m256]) -> __m256 {
    let t = [_mm256_hadd_ps(v[0], v[1]), _mm256_hadd_ps(v[2], v[3]), _mm256_hadd_ps(v[4], v[5]), _mm256_hadd_ps(v[6], v[7])];
    // lanes of v0 .. v3 summed in fours: the low halves, then the high ones
    let (u0, u1) = (_mm256_hadd_ps(t[0], t[1]), _mm256_hadd_ps(t[2], t[3]));
    _mm256_add_ps(_mm256_permute2f128_ps::<0x20>(u0, u1), _mm256_permute2f128_ps::<0x31>(u0, u1))
}

/// e^x in 8 lanes for x up to 0, within 2 ulp; below -87.3 as at -87.3 (1.2e-38). Cephes' expf: x = n ln 2 + r,
/// e^r by its polynomial.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
fn exp8(x: __m256) -> __m256 {
    let x = _mm256_max_ps(x, _mm256_set1_ps(-87.3));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)));
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693_359_4), x);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.121_944_4e-4), r);
    let mut y = _mm256_set1_ps(1.987_569_1e-4);
    for c in [1.398_2e-3, 8.333_452e-3, 4.166_579_6e-2, 0.166_666_65, 0.5] {
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(c));
    }
    let y = _mm256_add_ps(_mm256_fmadd_ps(y, _mm256_mul_ps(r, r), r), _mm256_set1_ps(1.0));
    let e = _mm256_slli_epi32::<23>(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)));
    _mm256_mul_ps(y, _mm256_castsi256_ps(e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::tests::rnd;

    /// A random normal f16 of magnitude 2^-4 .. 2^2.
    fn rnd_f16(seed: &mut u64) -> u16 {
        let x = rnd(seed);
        ((x as u16) & 0x8000) | ((11 + (x >> 16) % 6) as u16) << 10 | ((x >> 32) as u16 & 1023)
    }

    #[test]
    fn f16_decodes() {
        for (h, want) in [(0x3C00u16, 1.0f32), (0xC000, -2.0), (0x0001, 2f32.powi(-24)), (0x8001, -(2f32.powi(-24))), (0x03FF, 1023.0 * 2f32.powi(-24)), (0x7BFF, 65504.0), (0x7C00, f32::INFINITY), (0x3555, 0.33325195)] {
            assert_eq!(f16_f32(h), want, "{h:#x}");
        }
        assert!(f16_f32(0x7E00).is_nan());
    }

    /// The 24 heads' normalized outputs and log-sum-exps over distances 0..=pos in f64, with nothing but scalar
    /// arithmetic.
    fn reference(qy: &Query, k: &[u16], v: &[u16]) -> Vec<(f64, Vec<f64>)> {
        (0..HEADS)
            .map(|h| {
                let g = h / 3;
                let sc: Vec<f64> = (0..=qy.pos)
                    .map(|d| {
                        let row = qy.pos - d;
                        let mut s: f64 = (0..DIM).map(|i| qy.q[h * DIM + i] as f64 * qy.scale as f64 * f16_f32(k[row * ROW + g * DIM + i]) as f64).sum();
                        if d < qy.extent {
                            s += (0..N_REL).map(|r| qy.r[h * N_REL + r] as f64 * bf16_f32(qy.proj[r * qy.extent + d]) as f64).sum::<f64>();
                        }
                        s - qy.sink[h] as f64
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let l: f64 = sc.iter().map(|s| (s - m).exp()).sum();
                let out = (0..DIM).map(|i| sc.iter().enumerate().map(|(d, s)| (s - m).exp() * f16_f32(v[(qy.pos - d) * ROW + g * DIM + i]) as f64).sum::<f64>() / l).collect();
                (m + l.ln(), out)
            })
            .collect()
    }

    #[test]
    fn rows_match_f64() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let (pos, extent) = (300, 64);
        let k: Vec<u16> = (0..(pos + 1) * ROW).map(|_| rnd_f16(&mut seed)).collect();
        let v: Vec<u16> = (0..(pos + 1) * ROW).map(|_| rnd_f16(&mut seed)).collect();
        let f = |seed: &mut u64, n: usize, a: f32| -> Vec<f32> { (0..n).map(|_| ((rnd(seed) % 2001) as f32 / 1000.0 - 1.0) * a).collect() };
        let (q, r, sink) = (f(&mut seed, HEADS * DIM, 0.3), f(&mut seed, HEADS * N_REL, 1.0), f(&mut seed, HEADS, 2.0));
        let proj: Vec<u16> = (0..N_REL * extent).map(|_| (f(&mut seed, 1, 1.0)[0].to_bits() >> 16) as u16).collect();
        let qy = Query { q: &q, r: &r, sink: &sink, proj: &proj, extent, scale: 1.0 / (DIM as f32).sqrt(), pos };
        let want = reference(&qy, &k, &v);
        // in uneven pieces, merged: the scalar way and the one this CPU takes
        type Rows = unsafe fn(&Query, *const u16, *const u16, usize, usize, &mut [Partial; HEADS]);
        for (way, f) in [("scalar", rows_scalar as Rows), ("rows", rows as Rows)] {
            let mut all = [Partial::EMPTY; HEADS];
            for (lo, hi) in [(0, 5), (5, 70), (70, 71), (71, pos + 1)] {
                let mut p = [Partial::EMPTY; HEADS];
                unsafe { f(&qy, k.as_ptr(), v.as_ptr(), lo, hi, &mut p) };
                for (a, b) in all.iter_mut().zip(&p) {
                    a.merge(b);
                }
            }
            for (h, (p, (lse, out))) in all.iter().zip(&want).enumerate() {
                let got = p.m as f64 + (p.l as f64).ln();
                assert!((got - lse).abs() < 1e-5 * lse.abs().max(1.0), "{way}, head {h}: log-sum-exp {got} vs f64 {lse}");
                for i in 0..DIM {
                    let g = p.acc[i] as f64 / p.l as f64;
                    assert!((g - out[i]).abs() < 1e-5, "{way}, head {h} dim {i}: {g} vs f64 {}", out[i]);
                }
            }
        }
    }
}
