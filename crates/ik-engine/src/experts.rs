//! The routed experts: the checkpoint's expert tensors where it is mapped, CPU compute of the misses, and the GPU cache
//! of hot experts. The cache is keyed by physical layer, so the passes of the loop share it.
//!
//! The checkpoint holds each layer's experts as two matrices, w13 (gate and up, their rows interleaved: gate i is row
//! 2i, up i row 2i + 1) [E][2 n_ff][n_embd] and w2 (down) [E][n_embd][n_ff], in one of two formats:
//! - ModelOpt NVFP4 (layers 0-19): `weight` U8 [E][rows][k/2], the E2M1 code of element 2j of a row in the low nibble
//!   of byte j and of element 2j + 1 in the high one; `weight_scale` F8_E4M3 [E][rows][k/16]; and the expert's factor,
//!   `weight_amax` F32 [E] over `weight_outer_scale_denominator` F32 [1].
//! - MXFP8 (layers 20-23): `weight` F8_E4M3 [E][rows][k]; `weight_scale` U8 [E][rows][k/32], 2^(s - 127) each.
//!
//! The matrices are views of the mapped files; their pages are read in and page-locked there (no copy), so that cache
//! uploads are DMA. A GPU cache slot holds one expert's matrix: its part of the codes tensor, then its part of the
//! scales tensor, copied as they are.

use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::checkpoint::{Checkpoint, Dtype};
use crate::cuda::{DevBuf, Result, Stream, VmBuf};
use crate::hp::Hp;

/// The E2M1 values doubled, as integers: a code's value is this times half its block's scale.
pub const FP4X2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// The magnitude of an E4M3 byte (float8 e4m3fn: sign, 4 exponent bits of bias 7, 3 mantissa bits), as the kernels
/// decode the NVFP4 scales: the sign bit is ignored, and 0x7F (NaN in e4m3fn) reads as 480. The checkpoint's scales
/// are all positive and finite.
pub fn e4m3_abs(b: u8) -> f32 {
    let (e, m) = ((b >> 3) & 15, b & 7);
    if e == 0 { m as f32 * 2f32.powi(-9) } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7) }
}

/// The value of an E4M3 byte, as the kernels decode the MXFP8 values (0x7F and 0xFF, NaN in e4m3fn, as +-480; the
/// checkpoint has none).
pub fn e4m3(b: u8) -> f32 {
    if b & 0x80 != 0 { -e4m3_abs(b) } else { e4m3_abs(b) }
}

/// A UE8M0 scale: 2^(s - 127).
pub fn ue8m0(s: u8) -> f32 {
    2f32.powi(s as i32 - 127)
}

/// The format of a layer's experts.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fmt {
    NvFp4,
    MxFp8,
}

impl Fmt {
    pub fn index(self) -> usize {
        self as usize
    }

    /// Bytes of the codes and of the scales of one expert's matrix of `rows` rows of k values.
    pub fn parts(self, rows: usize, k: usize) -> (usize, usize) {
        match self {
            Fmt::NvFp4 => (rows * k / 2, rows * k / 16),
            Fmt::MxFp8 => (rows * k, rows * k / 32),
        }
    }

    /// Bytes of a cache slot of such a matrix: its codes, then its scales, rounded up to 256.
    pub fn bytes(self, rows: usize, k: usize) -> usize {
        let (q, s) = self.parts(rows, k);
        (q + s).next_multiple_of(256)
    }
}

/// One expert's matrix: its parts of the codes and scales tensors.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub fmt: Fmt,
    /// rows of k/2 (NVFP4) or k (MXFP8) bytes
    pub q: &'a [u8],
    /// rows of k/16 E4M3 (NVFP4) or k/32 UE8M0 (MXFP8) scales
    pub s: &'a [u8],
    /// the expert's factor: amax / denominator (NVFP4), 1 (MXFP8)
    pub factor: f32,
    pub rows: usize,
    pub k: usize,
}

impl<'a> Mat<'a> {
    /// The codes of row r.
    pub fn qs(&self, r: usize) -> &'a [u8] {
        let w = self.q.len() / self.rows;
        &self.q[r * w..(r + 1) * w]
    }

    /// The scales of row r.
    pub fn sc(&self, r: usize) -> &'a [u8] {
        let w = self.s.len() / self.rows;
        &self.s[r * w..(r + 1) * w]
    }

    /// The expert's factor, which the CPU kernels leave out of a row's dot.
    pub fn scale(&self) -> f32 {
        self.factor
    }

    /// Row r as f32, the expert's factor left out (the reference of the kernels).
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let (q, s) = (self.qs(r), self.sc(r));
        match self.fmt {
            Fmt::NvFp4 => {
                for (i, o) in out[..self.k].iter_mut().enumerate() {
                    let c = (q[i / 2] >> (4 * (i % 2))) & 15;
                    *o = FP4X2[c as usize] as f32 * 0.5 * e4m3_abs(s[i / 16]);
                }
            }
            Fmt::MxFp8 => {
                for (i, o) in out[..self.k].iter_mut().enumerate() {
                    *o = e4m3(q[i]) * ue8m0(s[i / 32]);
                }
            }
        }
    }
}

/// The experts of one physical layer: where the checkpoint holds their tensors, [E][rows][..], and their factors.
pub struct HostExperts {
    pub fmt: Fmt,
    /// w13's codes and scales: (file, byte offset) of each tensor in the checkpoint
    pub gu: [(usize, usize); 2],
    /// w2's codes and scales
    pub dn: [(usize, usize); 2],
    /// the experts' factors of w13 and w2: amax / denominator (NVFP4), 1 (MXFP8)
    pub gu_f: Vec<f32>,
    pub dn_f: Vec<f32>,
    /// NVFP4: weight_amax and weight_outer_scale_denominator of w13 and w2, as stored, for the GPU kernels
    pub amax: [Vec<f32>; 2],
    pub denom: [f32; 2],
}

/// The page-aligned ranges of the checkpoint's mapped files that hold `spans` (file, byte range), merged where they
/// touch. Each is (address, length).
fn page_ranges(ck: &Checkpoint, mut spans: Vec<(usize, Range<usize>)>) -> Vec<(usize, usize)> {
    const PAGE: usize = 4096;
    spans.sort_by_key(|(f, r)| (*f, r.start));
    let mut merged: Vec<(usize, Range<usize>)> = vec![];
    for (f, r) in spans {
        // the mapping covers the file's last page whole, so rounding the end up stays inside it
        let r = r.start / PAGE * PAGE..r.end.next_multiple_of(PAGE);
        match merged.last_mut() {
            Some((mf, mr)) if *mf == f && r.start <= mr.end => mr.end = mr.end.max(r.end),
            _ => merged.push((f, r)),
        }
    }
    merged.into_iter().map(|(f, r)| (ck.file(f).as_ptr() as usize + r.start, r.len())).collect()
}

/// Reads the mapped range (address, length) in from the file where it is not cached: MADV_POPULATE_READ (Linux 5.14),
/// else a read of each page.
fn populate(p: usize, len: usize) {
    const MADV_POPULATE_READ: libc::c_int = 22;
    // SAFETY: a page-aligned part of a read-only mapping that the checkpoint holds
    if unsafe { libc::madvise(p as *mut libc::c_void, len, MADV_POPULATE_READ) } != 0 {
        let mut acc = 0u8;
        for off in (0..len).step_by(4096) {
            // SAFETY: as above
            acc ^= unsafe { std::ptr::read_volatile((p + off) as *const u8) };
        }
        std::hint::black_box(acc);
    }
}

pub struct Experts {
    /// the checkpoint, whose mapped expert tensors the matrices view
    ck: Arc<Checkpoint>,
    /// the page-locked ranges of the mapped files that hold the expert tensors (address, length), so that cache uploads
    /// are DMA and run asynchronously; empty if they could not be locked
    locked: Vec<(usize, usize)>,
    pub pinned: bool,
    pub layers: Vec<HostExperts>,
    /// the format of each physical layer's experts
    pub fmt: Vec<Fmt>,
    /// bytes of a cache slot of one expert's gate_up and down, by format
    pub gu_bytes: [usize; 2],
    pub down_bytes: [usize; 2],
    pub n_embd: usize,
    pub n_ff: usize,
    pub beta: f32,
    pub lb: f32,
}

impl Experts {
    /// The experts of the checkpoint, read where it is mapped: their tensors are read in from the file and page-locked
    /// there (unless IK_NO_PIN), not copied.
    pub fn new(ck: Arc<Checkpoint>, hp: &Hp) -> Result<Experts> {
        let (n, nff, ne) = (hp.n_embd, hp.n_ff_exp, hp.n_expert);
        // each layer's tensors, checked: (format, [(codes, scales)] of w13 and w2, factors)
        let mut spans = vec![];
        let mut layers = vec![];
        for i in 0..hp.n_phys {
            let p = |w: &str, part: &str| format!("model.llm.layers.{i}.mlp.experts.{w}_{part}");
            let fmt = match ck.tensor(&p("w13", "weight"))?.dtype {
                Dtype::U8 => Fmt::NvFp4,
                Dtype::F8E4M3 => Fmt::MxFp8,
                d => return Err(format!("layer {i}: experts of {d:?}; the engine reads ModelOpt NVFP4 (U8 codes) or MXFP8 (F8_E4M3 values)")),
            };
            let (qt, st, per_byte, block) = match fmt {
                Fmt::NvFp4 => (Dtype::U8, Dtype::F8E4M3, 2, 16),
                Fmt::MxFp8 => (Dtype::F8E4M3, Dtype::U8, 1, 32),
            };
            let mut at = [[(0usize, 0usize); 2]; 2];
            let mut fac = [vec![1f32; ne], vec![1f32; ne]];
            let mut amax = [vec![], vec![]];
            let mut denom = [0f32; 2];
            for (m, (w, rows, k)) in [("w13", 2 * nff, n), ("w2", n, nff)].into_iter().enumerate() {
                for (j, (part, dt, cols)) in [("weight", qt, k / per_byte), ("weight_scale", st, k / block)].into_iter().enumerate() {
                    let name = p(w, part);
                    ck.get(&name, dt, &[ne, rows, cols])?;
                    let (f, r) = ck.tensor(&name)?.place();
                    at[m][j] = (f, r.start);
                    spans.push((f, r));
                }
                if fmt == Fmt::NvFp4 {
                    ck.get(&p(w, "weight_amax"), Dtype::F32, &[ne])?;
                    ck.get(&p(w, "weight_outer_scale_denominator"), Dtype::F32, &[1])?;
                    amax[m] = ck.floats(&p(w, "weight_amax"))?;
                    denom[m] = ck.floats(&p(w, "weight_outer_scale_denominator"))?[0];
                    fac[m] = amax[m].iter().map(|&a| a / denom[m]).collect();
                }
            }
            let [gu_f, dn_f] = fac;
            layers.push(HostExperts { fmt, gu: at[0], dn: at[1], gu_f, dn_f, amax, denom });
        }
        let fmt: Vec<Fmt> = layers.iter().map(|l| l.fmt).collect();
        let gu_bytes = [Fmt::NvFp4.bytes(2 * nff, n), Fmt::MxFp8.bytes(2 * nff, n)];
        let down_bytes = [Fmt::NvFp4.bytes(n, nff), Fmt::MxFp8.bytes(n, nff)];
        let ranges = page_ranges(&ck, spans);
        let mut locked = vec![];
        if std::env::var("IK_NO_PIN").is_err() {
            let total: usize = ranges.iter().map(|r| r.1).sum();
            let t = std::time::Instant::now();
            ranges.par_iter().for_each(|&(p, len)| populate(p, len));
            eprintln!("experts read in {:.1}s ({} MiB, not copied)", t.elapsed().as_secs_f64(), total >> 20);
            let t = std::time::Instant::now();
            for &(p, len) in &ranges {
                if let Err(e) = crate::cuda::register_host_ro(p as *const u8, len) {
                    eprintln!("experts not page-locked ({e}): cache uploads stay synchronous");
                    unlock(&locked);
                    locked.clear();
                    break;
                }
                locked.push((p, len));
            }
            if !locked.is_empty() {
                eprintln!("experts page-locked in {:.1}s", t.elapsed().as_secs_f64());
            }
        }
        let pinned = !locked.is_empty();
        Ok(Experts { ck, locked, pinned, layers, fmt, gu_bytes, down_bytes, n_embd: n, n_ff: nff, beta: hp.situ_beta, lb: hp.situ_lb })
    }

    fn mat(&self, layer: usize, which: [(usize, usize); 2], f: &[f32], e: usize, rows: usize, k: usize) -> Mat<'_> {
        let fmt = self.fmt[layer];
        let (qb, sb) = fmt.parts(rows, k);
        let [(qf, qo), (sf, so)] = which;
        Mat { fmt, q: &self.ck.file(qf)[qo + e * qb..][..qb], s: &self.ck.file(sf)[so + e * sb..][..sb], factor: f[e], rows, k }
    }

    /// Expert e's gate_up of `layer`: 2 n_ff rows of n_embd, gate i in row 2i and up i in row 2i + 1.
    pub fn gate_up(&self, layer: usize, e: usize) -> Mat<'_> {
        let h = &self.layers[layer];
        self.mat(layer, h.gu, &h.gu_f, e, 2 * self.n_ff, self.n_embd)
    }

    /// Expert e's down of `layer`: n_embd rows of n_ff.
    pub fn down(&self, layer: usize, e: usize) -> Mat<'_> {
        let h = &self.layers[layer];
        self.mat(layer, h.dn, &h.dn_f, e, self.n_embd, self.n_ff)
    }
}

impl Drop for Experts {
    fn drop(&mut self) {
        // before the checkpoint (a field) unmaps the files
        unlock(&self.locked);
    }
}

/// Unlocks the page-locked ranges (address, length).
fn unlock(ranges: &[(usize, usize)]) {
    for &(p, _) in ranges {
        let _ = crate::cuda::unregister_host(p as *const u8);
    }
}

/// Copies matrix `m` into the cache slot at `dst` (codes, then scales), on stream `s` (asynchronous if the experts are
/// page-locked).
pub fn upload_slot(dst: crate::cuda::CUdeviceptr, m: &Mat, s: &Stream) -> Result<()> {
    crate::cuda::copy_h2d_async(dst, m.q.as_ptr(), m.q.len(), s)?;
    crate::cuda::copy_h2d_async(dst + m.q.len() as u64, m.s.as_ptr(), m.s.len(), s)
}

/// The slots of one format: two banks (gate_up, down) with addresses for `max_slots` and memory for the first
/// `n_slots`, which change as the KV cache takes VRAM or gives it back.
pub struct Bank {
    pub gu: VmBuf,
    pub down: VmBuf,
    pub gu_bytes: usize,
    pub down_bytes: usize,
    pub n_slots: usize,
    pub max_slots: usize,
    /// [slot] -> (layer, expert), usize::MAX if free
    pub owner: Vec<(usize, usize)>,
    /// slots being filled by an async upload: not cached yet, and no victims
    pub busy: Vec<bool>,
}

impl Bank {
    fn bytes_for(&self, n: usize) -> usize {
        self.gu.pages_for(n * self.gu_bytes) + self.down.pages_for(n * self.down_bytes)
    }

    /// The least used slot, free slots first; None if all are being filled.
    fn victim(&self, uses: &[Vec<f32>]) -> Option<usize> {
        let mut best = (f32::INFINITY, usize::MAX);
        for (s, &(l, e)) in self.owner[..self.n_slots].iter().enumerate() {
            if self.busy[s] {
                continue;
            }
            if l == usize::MAX {
                return Some(s);
            }
            if uses[l][e] < best.0 {
                best = (uses[l][e], s);
            }
        }
        (best.1 != usize::MAX).then_some(best.1)
    }
}

/// GPU cache of experts: a bank of slots per format, and the slot of each (physical layer, expert) in its layer's bank.
pub struct ExpertCache {
    /// [NVFP4, MXFP8]
    pub banks: [Bank; 2],
    /// the share of the cache's VRAM that the MXFP8 bank gets when both formats are there (IK_MX_SHARE, 0.15)
    pub q8_share: f64,
    /// the NVFP4 layers' weight_amax and weight_outer_scale_denominator of w13 and w2 as stored: [layer][matrix] at
    /// FACTORS floats each, amax at 0, the denominator at E
    pub factors: DevBuf,
    n_expert: usize,
    pub slot_of: Vec<Vec<i32>>, // [layer][expert] -> slot or -1
    pub uses: Vec<Vec<f32>>,    // decayed use counts
    pub hits: u64,
    pub misses: u64,
    pub uploads: u64,
    pub pending: Vec<Pending>,
}

/// An async upload of expert e of `layer` into `slot` of bank `bank`.
pub struct Pending {
    pub bank: usize,
    pub slot: usize,
    pub layer: usize,
    pub e: usize,
    pub ev: crate::cuda::Event,
}

impl ExpertCache {
    /// A cache with addresses for up to `max_slots` slots in each bank, and no memory yet (see set_slots).
    pub fn new(ex: &Experts, hp: &Hp, max_slots: usize, dev: i32) -> Result<ExpertCache> {
        let has = |f: Fmt| ex.fmt.contains(&f);
        let bank = |f: Fmt| -> Result<Bank> {
            let m = if has(f) { max_slots } else { 0 };
            let (gb, db) = (ex.gu_bytes[f.index()], ex.down_bytes[f.index()]);
            Ok(Bank { gu: VmBuf::reserve(gb * m.max(1), dev)?, down: VmBuf::reserve(db * m.max(1), dev)?, gu_bytes: gb, down_bytes: db, n_slots: 0, max_slots: m, owner: vec![(usize::MAX, usize::MAX); m], busy: vec![false; m] })
        };
        let q8_share = if has(Fmt::NvFp4) && has(Fmt::MxFp8) { std::env::var("IK_MX_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15) } else if has(Fmt::MxFp8) { 1.0 } else { 0.0 };
        let ne = hp.n_expert;
        let mut fv = vec![0f32; ex.layers.len() * 2 * (ne + 4)];
        for (l, h) in ex.layers.iter().enumerate() {
            for m in 0..2 {
                if h.fmt == Fmt::NvFp4 {
                    let o = (2 * l + m) * (ne + 4);
                    fv[o..o + ne].copy_from_slice(&h.amax[m]);
                    fv[o + ne] = h.denom[m];
                }
            }
        }
        Ok(ExpertCache {
            banks: [bank(Fmt::NvFp4)?, bank(Fmt::MxFp8)?],
            q8_share,
            factors: DevBuf::from_slice(&fv)?,
            n_expert: ne,
            slot_of: vec![vec![-1; hp.n_expert]; hp.n_phys],
            uses: vec![vec![0.0; hp.n_expert]; hp.n_phys],
            hits: 0,
            misses: 0,
            uploads: 0,
            pending: vec![],
        })
    }

    /// The device addresses of matrix m's (0 gate_up, 1 down) weight_amax and denominator of NVFP4 layer `layer`.
    pub fn factor_ptrs(&self, layer: usize, m: usize) -> (crate::cuda::CUdeviceptr, crate::cuda::CUdeviceptr) {
        let o = (2 * layer + m) * (self.n_expert + 4);
        (self.factors.at(4 * o), self.factors.at(4 * (o + self.n_expert)))
    }

    /// All the usable slots.
    pub fn n_slots(&self) -> usize {
        self.banks.iter().map(|b| b.n_slots).sum()
    }

    /// The slots of each bank that `bytes` of memory hold: the Q8_0 bank its share, the NVFP4 bank the rest.
    pub fn fit(&self, bytes: usize) -> [usize; 2] {
        let most = |b: &Bank, bytes: usize| {
            let mut n = (bytes / (b.gu_bytes + b.down_bytes)).min(b.max_slots);
            while n > 0 && b.bytes_for(n) > bytes {
                n -= 1;
            }
            n
        };
        let q8 = most(&self.banks[1], (bytes as f64 * self.q8_share) as usize);
        let nv = most(&self.banks[0], bytes - self.banks[1].bytes_for(q8));
        [nv, q8]
    }

    /// The bytes that the slots take.
    pub fn bytes_for(&self, n: [usize; 2]) -> usize {
        self.banks[0].bytes_for(n[0]) + self.banks[1].bytes_for(n[1])
    }

    /// Sets the number of usable slots of each bank: the experts in the slots that go are dropped, and new slots start
    /// free. No upload may be pending into a slot that goes, and no work in flight may read it.
    pub fn set_slots(&mut self, n: [usize; 2], s: &Stream) -> Result<()> {
        for (b, &n) in self.banks.iter_mut().zip(&n) {
            let n = n.min(b.max_slots);
            for v in n..b.n_slots {
                assert!(!b.busy[v], "slot {v} is being filled");
                let (l, e) = b.owner[v];
                if l != usize::MAX {
                    self.slot_of[l][e] = -1;
                }
                b.owner[v] = (usize::MAX, usize::MAX);
            }
            b.gu.resize(n * b.gu_bytes, s)?;
            b.down.resize(n * b.down_bytes, s)?;
            b.n_slots = n;
        }
        Ok(())
    }

    /// Uploads expert e of `layer` into `slot` of its bank (synchronous).
    pub fn fill(&mut self, ex: &Experts, slot: usize, layer: usize, e: usize) -> Result<()> {
        let b = &mut self.banks[ex.fmt[layer].index()];
        let (ol, oe) = b.owner[slot];
        if ol != usize::MAX {
            self.slot_of[ol][oe] = -1;
        }
        let (gu, dn) = (ex.gate_up(layer, e), ex.down(layer, e));
        b.gu.upload_at(slot * b.gu_bytes, gu.q)?;
        b.gu.upload_at(slot * b.gu_bytes + gu.q.len(), gu.s)?;
        b.down.upload_at(slot * b.down_bytes, dn.q)?;
        b.down.upload_at(slot * b.down_bytes + dn.q.len(), dn.s)?;
        b.owner[slot] = (layer, e);
        self.slot_of[layer][e] = slot as i32;
        self.uploads += 1;
        Ok(())
    }

    /// The least used slot of bank `bank`, free slots first; None if all are being filled.
    pub fn victim(&self, bank: usize) -> Option<usize> {
        self.banks[bank].victim(&self.uses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_decode() {
        assert_eq!((e4m3(0x38), e4m3(0x40), e4m3(0xC4), e4m3(0x7E), e4m3(0x01), e4m3(0x80)), (1.0, 2.0, -3.0, 448.0, 2f32.powi(-9), -0.0));
        assert_eq!((ue8m0(127), ue8m0(120), ue8m0(130)), (1.0, 2f32.powi(-7), 8.0));
        // NVFP4: a row of 32, bytes j = codes (2j, 2j + 1); scales 0x38 (1) and 0x40 (2)
        let q: Vec<u8> = (0..16u8).map(|j| (j & 15) | (((15 - j) & 15) << 4)).collect();
        let s = [0x38u8, 0x40];
        let m = Mat { fmt: Fmt::NvFp4, q: &q, s: &s, factor: 0.5, rows: 1, k: 32 };
        let mut out = vec![0f32; 32];
        m.dequant_row(0, &mut out);
        let e2m1 = |c: usize| FP4X2[c] as f32 / 2.0;
        for i in 0..32 {
            let c = if i % 2 == 0 { i / 2 } else { 15 - i / 2 };
            assert_eq!(out[i], e2m1(c) * if i < 16 { 1.0 } else { 2.0 }, "element {i}");
        }
        // MXFP8: a row of 64, scales 2^-3 and 2^2
        let q: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37)).collect();
        let s = [124u8, 129];
        let m = Mat { fmt: Fmt::MxFp8, q: &q, s: &s, factor: 1.0, rows: 1, k: 64 };
        let mut out = vec![0f32; 64];
        m.dequant_row(0, &mut out);
        for i in 0..64 {
            assert_eq!(out[i], e4m3(q[i]) * if i < 32 { 0.125 } else { 4.0 }, "element {i}");
        }
    }
}
