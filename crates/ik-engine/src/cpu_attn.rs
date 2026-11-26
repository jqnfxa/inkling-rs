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

/// The partials of the 24 heads over distances lo..hi, merged into `out`. k and v are row 0 of the caches.
///
/// # Safety
/// Rows pos - hi + 1 ..= pos - lo of k and v hold ROW f16 values each, readable.
pub unsafe fn rows(qy: &Query, k: *const u16, v: *const u16, lo: usize, hi: usize, out: &mut [Partial; HEADS]) {
    assert!(hi <= qy.pos + 1 && qy.q.len() >= HEADS * DIM && qy.r.len() >= HEADS * N_REL && qy.sink.len() >= HEADS && qy.proj.len() >= N_REL * qy.extent);
    let qs: Vec<f32> = qy.q[..HEADS * DIM].iter().map(|&x| x * qy.scale).collect();
    let mut s = [[0f32; HEADS]; BLOCK];
    let mut d0 = lo;
    while d0 < hi {
        let nb = (hi - d0).min(BLOCK);
        for (b, sb) in s.iter_mut().enumerate().take(nb) {
            let d = d0 + b;
            // SAFETY: the row is readable (the caller)
            let kr = unsafe { k.add((qy.pos - d) * ROW) };
            for g in 0..KV_HEADS {
                // SAFETY: as above
                let x = unsafe { dot3_f16(&qs[3 * g * DIM..(3 * g + 3) * DIM], kr.add(g * DIM)) };
                for (j, x) in x.into_iter().enumerate() {
                    let h = 3 * g + j;
                    let mut x = x;
                    if d < qy.extent {
                        for rr in 0..N_REL {
                            x += qy.r[h * N_REL + rr] * bf16_f32(qy.proj[rr * qy.extent + d]);
                        }
                    }
                    sb[h] = x - qy.sink[h];
                }
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
            let vr = unsafe { v.add((qy.pos - d0 - b) * ROW) };
            for g in 0..KV_HEADS {
                let (o0, rest) = out[3 * g..3 * g + 3].split_at_mut(1);
                let (o1, o2) = rest.split_at_mut(1);
                // SAFETY: as above
                unsafe { axpy3_f16(vr.add(g * DIM), [sb[3 * g], sb[3 * g + 1], sb[3 * g + 2]], [&mut o0[0].acc, &mut o1[0].acc, &mut o2[0].acc]) };
            }
        }
        d0 += nb;
    }
}

/// The dots of 3 queries q [3][128] with the 128 f16 values at k.
unsafe fn dot3_f16(q: &[f32], k: *const u16) -> [f32; 3] {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        return unsafe { dot3_f16_avx2(q, k) };
    }
    std::array::from_fn(|j| (0..DIM).map(|i| q[j * DIM + i] * f16_f32(unsafe { *k.add(i) })).sum())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot3_f16_avx2(q: &[f32], k: *const u16) -> [f32; 3] {
    assert!(q.len() >= 3 * DIM);
    unsafe {
        let mut a = [[_mm256_setzero_ps(); 2]; 3];
        for i in 0..DIM / 16 {
            let kv = [_mm256_cvtph_ps(_mm_loadu_si128(k.add(16 * i) as *const __m128i)), _mm256_cvtph_ps(_mm_loadu_si128(k.add(16 * i + 8) as *const __m128i))];
            for (j, a) in a.iter_mut().enumerate() {
                for (h, a) in a.iter_mut().enumerate() {
                    *a = _mm256_fmadd_ps(_mm256_loadu_ps(q.as_ptr().add(j * DIM + 16 * i + 8 * h)), kv[h], *a);
                }
            }
        }
        [
            crate::cpu::hsum(_mm256_add_ps(a[0][0], a[0][1])),
            crate::cpu::hsum(_mm256_add_ps(a[1][0], a[1][1])),
            crate::cpu::hsum(_mm256_add_ps(a[2][0], a[2][1])),
        ]
    }
}

/// acc[j] += p[j] v for the 128 f16 values v at vr, j = 0..3.
unsafe fn axpy3_f16(vr: *const u16, p: [f32; 3], acc: [&mut [f32; DIM]; 3]) {
    #[cfg(target_arch = "x86_64")]
    if simd() {
        return unsafe { axpy3_f16_avx2(vr, p, acc) };
    }
    for (a, p) in acc.into_iter().zip(p) {
        for (i, a) in a.iter_mut().enumerate() {
            *a += p * f16_f32(unsafe { *vr.add(i) });
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn axpy3_f16_avx2(vr: *const u16, p: [f32; 3], acc: [&mut [f32; DIM]; 3]) {
    unsafe {
        let pv = [_mm256_set1_ps(p[0]), _mm256_set1_ps(p[1]), _mm256_set1_ps(p[2])];
        for i in 0..DIM / 8 {
            let x = _mm256_cvtph_ps(_mm_loadu_si128(vr.add(8 * i) as *const __m128i));
            for j in 0..3 {
                let a = acc[j].as_mut_ptr().add(8 * i);
                _mm256_storeu_ps(a, _mm256_fmadd_ps(pv[j], x, _mm256_loadu_ps(a)));
            }
        }
    }
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
        // in uneven pieces, merged
        let mut all = [Partial::EMPTY; HEADS];
        for (lo, hi) in [(0, 5), (5, 70), (70, 71), (71, pos + 1)] {
            let mut p = [Partial::EMPTY; HEADS];
            unsafe { rows(&qy, k.as_ptr(), v.as_ptr(), lo, hi, &mut p) };
            for (a, b) in all.iter_mut().zip(&p) {
                a.merge(b);
            }
        }
        for (h, (p, (lse, out))) in all.iter().zip(&want).enumerate() {
            let got = p.m as f64 + (p.l as f64).ln();
            assert!((got - lse).abs() < 1e-5 * lse.abs().max(1.0), "head {h}: log-sum-exp {got} vs f64 {lse}");
            for i in 0..DIM {
                let g = p.acc[i] as f64 / p.l as f64;
                assert!((g - out[i]).abs() < 1e-5, "head {h} dim {i}: {g} vs f64 {}", out[i]);
            }
        }
    }
}
