//! The routed experts: host copies in the layout of the GPU cache slots, CPU compute of the misses, and the GPU cache of
//! hot experts. The cache is keyed by physical layer, so the passes of the loop share it.
//!
//! The official GGUFs (unsloth, UD-Q4_K_XL) hold NVFP4 experts with an F32 scale each (layers 0-19) and Q8_0 experts
//! (20-23). Both are repacked at load into the layout that the GPU kernels read and the CPU kernels read too, so that
//! a cache upload is one copy. A matrix of `rows` rows of k values:
//! - NVFP4: the rows' E2M1 nibbles, k/2 bytes a row, each 64 values as 32 bytes with value j in the low nibble of
//!   byte j and value 32 + j in the high one; then the UE4M3 scales of all rows, one byte per 16 values; then the
//!   expert's scale (f32).
//! - Q8_0: the rows' int8 values, k bytes a row; then the f16 scales of all rows, one per 32 values.

use rayon::prelude::*;

use crate::cuda::{Result, Stream, VmBuf};
use crate::gguf::{GgmlType, Gguf};
use crate::hp::Hp;
use crate::quant::{FP4, dot, ue4m3};

/// The format of a layer's experts.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Fmt {
    Nv4,
    Q8,
}

impl Fmt {
    pub fn index(self) -> usize {
        self as usize
    }

    /// Bytes of a matrix of `rows` rows of k values in the slot layout, rounded up to 256.
    pub fn bytes(self, rows: usize, k: usize) -> usize {
        let b = match self {
            Fmt::Nv4 => rows * k / 2 + rows * k / 16 + 4,
            Fmt::Q8 => rows * k + rows * k / 32 * 2,
        };
        b.div_ceil(256) * 256
    }
}

/// A matrix of one expert in the slot layout.
#[derive(Clone, Copy)]
pub struct Mat<'a> {
    pub fmt: Fmt,
    pub b: &'a [u8],
    pub rows: usize,
    pub k: usize,
}

impl<'a> Mat<'a> {
    /// The quants of row r: k/2 bytes (NVFP4) or k int8 (Q8_0).
    pub fn qs(&self, r: usize) -> &'a [u8] {
        let w = match self.fmt {
            Fmt::Nv4 => self.k / 2,
            Fmt::Q8 => self.k,
        };
        &self.b[r * w..(r + 1) * w]
    }

    /// The scales of row r: k/16 UE4M3 bytes (NVFP4) or k/32 f16 (Q8_0, as bytes).
    pub fn sc(&self, r: usize) -> &'a [u8] {
        let (base, w) = match self.fmt {
            Fmt::Nv4 => (self.rows * self.k / 2, self.k / 16),
            Fmt::Q8 => (self.rows * self.k, self.k / 16),
        };
        &self.b[base + r * w..base + (r + 1) * w]
    }

    /// The expert's scale of this matrix (NVFP4), 1 for Q8_0.
    pub fn scale(&self) -> f32 {
        match self.fmt {
            Fmt::Nv4 => {
                let o = self.rows * self.k / 2 + self.rows * self.k / 16;
                f32::from_le_bytes(self.b[o..o + 4].try_into().unwrap())
            }
            Fmt::Q8 => 1.0,
        }
    }

    /// Row r as f32, the expert's scale left out (the reference of the kernels).
    pub fn dequant_row(&self, r: usize, out: &mut [f32]) {
        let (q, s) = (self.qs(r), self.sc(r));
        match self.fmt {
            Fmt::Nv4 => {
                for g in 0..self.k / 64 {
                    for j in 0..32 {
                        let b = q[g * 32 + j];
                        out[g * 64 + j] = FP4[(b & 0xF) as usize] as f32 * ue4m3(s[(g * 64 + j) / 16]);
                        out[g * 64 + 32 + j] = FP4[(b >> 4) as usize] as f32 * ue4m3(s[(g * 64 + 32 + j) / 16]);
                    }
                }
            }
            Fmt::Q8 => {
                for (i, o) in out[..self.k].iter_mut().enumerate() {
                    let d = half::f16::from_le_bytes([s[i / 32 * 2], s[i / 32 * 2 + 1]]).to_f32();
                    *o = q[i] as i8 as f32 * d;
                }
            }
        }
    }
}

/// Repacks one GGUF matrix (rows of NVFP4 or Q8_0 blocks) into the slot layout `dst`, with the expert's scale.
fn repack(fmt: Fmt, src: &[u8], rows: usize, k: usize, scale: f32, dst: &mut [u8]) {
    match fmt {
        Fmt::Nv4 => {
            let (qs_end, nb) = (rows * k / 2, k / 64);
            for r in 0..rows {
                for b in 0..nb {
                    let blk = &src[(r * nb + b) * 36..(r * nb + b + 1) * 36];
                    // ggml: value 16s + j (j < 8) in the low nibble of byte 8s + j, 16s + 8 + j in its high one
                    let mut v = [0u8; 64];
                    for s in 0..4 {
                        for j in 0..8 {
                            let q = blk[4 + s * 8 + j];
                            v[s * 16 + j] = q & 0xF;
                            v[s * 16 + 8 + j] = q >> 4;
                        }
                    }
                    let o = r * k / 2 + b * 32;
                    for j in 0..32 {
                        dst[o + j] = v[j] | (v[32 + j] << 4);
                    }
                    dst[qs_end + r * k / 16 + b * 4..qs_end + r * k / 16 + b * 4 + 4].copy_from_slice(&blk[..4]);
                }
            }
            let o = qs_end + rows * k / 16;
            dst[o..o + 4].copy_from_slice(&scale.to_le_bytes());
        }
        Fmt::Q8 => {
            let nb = k / 32;
            for (i, blk) in src.chunks_exact(34).enumerate() {
                let (r, b) = (i / nb, i % nb);
                dst[r * k + b * 32..r * k + b * 32 + 32].copy_from_slice(&blk[2..34]);
                dst[rows * k + i * 2..rows * k + i * 2 + 2].copy_from_slice(&blk[..2]);
            }
        }
    }
}

/// Byte offsets of the experts of one physical layer in the host copy: expert e's gate_up at gate_up + e * its size.
#[derive(Clone, Copy)]
pub struct HostExperts {
    pub gate_up: usize,
    pub down: usize,
}

/// Anonymous memory backed by transparent huge pages (unless IK_NO_HUGE), holding the experts.
pub struct HugeMem {
    pub ptr: *mut u8,
    pub len: usize,
}
unsafe impl Send for HugeMem {}
unsafe impl Sync for HugeMem {}

impl HugeMem {
    pub fn new(len: usize) -> Result<HugeMem> {
        // SAFETY: a fresh private anonymous mapping
        let p = unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
        if p == libc::MAP_FAILED {
            return Err(format!("mmap of {len} bytes failed"));
        }
        if std::env::var("IK_NO_HUGE").is_err() {
            unsafe { libc::madvise(p, len, libc::MADV_HUGEPAGE) };
        }
        Ok(HugeMem { ptr: p as *mut u8, len })
    }
    pub fn slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for HugeMem {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
}

pub struct Experts {
    /// the experts in the slot layout; offsets of `layers` index it
    pub mem: HugeMem,
    /// the copy is page-locked, so that cache uploads are DMA and run asynchronously
    pub pinned: bool,
    pub layers: Vec<HostExperts>,
    /// the format of each physical layer's experts
    pub fmt: Vec<Fmt>,
    /// bytes of one expert's gate_up and down, by format
    pub gu_bytes: [usize; 2],
    pub down_bytes: [usize; 2],
    pub n_embd: usize,
    pub n_ff: usize,
    pub beta: f32,
    pub lb: f32,
}

impl Experts {
    pub fn new(g: &Gguf, hp: &Hp) -> Result<Experts> {
        let (n, nff, ne) = (hp.n_embd, hp.n_ff_exp, hp.n_expert);
        let mut fmt = vec![];
        for i in 0..hp.n_phys {
            let gu = g.tensor(&format!("blk.{i}.ffn_gate_up_exps.weight"))?;
            let dn = g.tensor(&format!("blk.{i}.ffn_down_exps.weight"))?;
            let f = match (gu.ty, dn.ty) {
                (GgmlType::NVFP4, GgmlType::NVFP4) => Fmt::Nv4,
                (GgmlType::Q8_0, GgmlType::Q8_0) => Fmt::Q8,
                (a, b) => return Err(format!("layer {i}: experts are {a:?}/{b:?}; the engine takes the official GGUFs' NVFP4 or Q8_0 (unsloth Inkling-2-Mini-UD-Q4_K_XL)")),
            };
            if gu.ne != [n, 2 * nff, ne] || dn.ne != [nff, n, ne] {
                return Err(format!("layer {i}: unexpected expert shapes {:?} {:?}", gu.ne, dn.ne));
            }
            fmt.push(f);
        }
        let gu_bytes = [Fmt::Nv4.bytes(2 * nff, n), Fmt::Q8.bytes(2 * nff, n)];
        let down_bytes = [Fmt::Nv4.bytes(n, nff), Fmt::Q8.bytes(n, nff)];
        let mut layers = vec![];
        let mut at = 0;
        for &f in &fmt {
            layers.push(HostExperts { gate_up: at, down: at + ne * gu_bytes[f.index()] });
            at += ne * (gu_bytes[f.index()] + down_bytes[f.index()]);
        }
        // the expert scales of the NVFP4 layers (absent: 1)
        let scales = |name: String| -> Result<Vec<f32>> { if g.tensor(&name).is_ok() { g.f32s(&name) } else { Ok(vec![1.0; ne]) } };
        let t = std::time::Instant::now();
        let mem = HugeMem::new(at)?;
        let dst = mem.ptr as usize;
        let mut jobs = vec![];
        for (i, &f) in fmt.iter().enumerate() {
            let gu = g.tensor(&format!("blk.{i}.ffn_gate_up_exps.weight"))?;
            let dn = g.tensor(&format!("blk.{i}.ffn_down_exps.weight"))?;
            let (sg, sd) = (scales(format!("blk.{i}.ffn_gate_up_exps.scale"))?, scales(format!("blk.{i}.ffn_down_exps.scale"))?);
            let (gsrc, dsrc) = (gu.ty.row_bytes(n) * 2 * nff, dn.ty.row_bytes(nff) * n);
            for e in 0..ne {
                jobs.push((f, gu.offset + e * gsrc, gsrc, 2 * nff, n, sg[e], layers[i].gate_up + e * gu_bytes[f.index()], gu_bytes[f.index()]));
                jobs.push((f, dn.offset + e * dsrc, dsrc, n, nff, sd[e], layers[i].down + e * down_bytes[f.index()], down_bytes[f.index()]));
            }
        }
        jobs.par_iter().for_each(|&(f, src, len, rows, k, scale, off, size)| {
            // SAFETY: disjoint ranges of the fresh mapping
            let out = unsafe { std::slice::from_raw_parts_mut((dst as *mut u8).add(off), size) };
            repack(f, &g.mmap[src..src + len], rows, k, scale, out);
        });
        eprintln!("experts repacked in {:.1}s ({} MiB)", t.elapsed().as_secs_f64(), at >> 20);
        let mut pinned = false;
        if std::env::var("IK_NO_PIN").is_err() {
            let t = std::time::Instant::now();
            match crate::cuda::register_host(mem.ptr, mem.len) {
                Ok(()) => {
                    pinned = true;
                    eprintln!("experts page-locked in {:.1}s", t.elapsed().as_secs_f64());
                }
                Err(e) => eprintln!("experts not page-locked ({e}): cache uploads stay synchronous"),
            }
        }
        Ok(Experts { mem, pinned, layers, fmt, gu_bytes, down_bytes, n_embd: n, n_ff: nff, beta: hp.situ_beta, lb: hp.situ_lb })
    }

    /// Expert e's gate_up of `layer`: 2 n_ff rows of n_embd (gate rows, then up rows).
    pub fn gate_up(&self, layer: usize, e: usize) -> Mat<'_> {
        let f = self.fmt[layer];
        let (o, len) = (self.layers[layer].gate_up + e * self.gu_bytes[f.index()], self.gu_bytes[f.index()]);
        Mat { fmt: f, b: &self.mem.slice()[o..o + len], rows: 2 * self.n_ff, k: self.n_embd }
    }

    /// Expert e's down of `layer`: n_embd rows of n_ff.
    pub fn down(&self, layer: usize, e: usize) -> Mat<'_> {
        let f = self.fmt[layer];
        let (o, len) = (self.layers[layer].down + e * self.down_bytes[f.index()], self.down_bytes[f.index()]);
        Mat { fmt: f, b: &self.mem.slice()[o..o + len], rows: self.n_embd, k: self.n_ff }
    }

    /// Reference CPU path in f32: y = down(situ(gate_up x)) of expert e of `layer`.
    pub fn compute_cpu(&self, layer: usize, e: usize, x: &[f32], y: &mut [f32]) {
        let (gu, dn) = (self.gate_up(layer, e), self.down(layer, e));
        let nff = self.n_ff;
        let mut h = vec![0f32; 2 * nff];
        h.par_chunks_mut(64).enumerate().for_each(|(c, out)| {
            let mut row = vec![0f32; self.n_embd];
            for (i, o) in out.iter_mut().enumerate() {
                gu.dequant_row(c * 64 + i, &mut row);
                *o = dot(&row, x) * gu.scale();
            }
        });
        let a: Vec<f32> = (0..nff)
            .map(|i| {
                let (gv, uv) = (h[i], h[nff + i]);
                let act = self.beta * (gv / self.beta).tanh() * (1.0 / (1.0 + (-gv).exp()));
                act * self.lb * (uv / self.lb).tanh()
            })
            .collect();
        y.par_chunks_mut(64).enumerate().for_each(|(c, out)| {
            let mut row = vec![0f32; nff];
            for (i, o) in out.iter_mut().enumerate() {
                dn.dequant_row(c * 64 + i, &mut row);
                *o = dot(&row, &a) * dn.scale();
            }
        });
    }
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
    /// [NVFP4, Q8_0]
    pub banks: [Bank; 2],
    /// the share of the cache's VRAM that the Q8_0 bank gets when both formats are there (IK_Q8_SHARE, 0.15)
    pub q8_share: f64,
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
        let q8_share = if has(Fmt::Nv4) && has(Fmt::Q8) { std::env::var("IK_Q8_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15) } else if has(Fmt::Q8) { 1.0 } else { 0.0 };
        Ok(ExpertCache {
            banks: [bank(Fmt::Nv4)?, bank(Fmt::Q8)?],
            q8_share,
            slot_of: vec![vec![-1; hp.n_expert]; hp.n_phys],
            uses: vec![vec![0.0; hp.n_expert]; hp.n_phys],
            hits: 0,
            misses: 0,
            uploads: 0,
            pending: vec![],
        })
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
        b.gu.upload_at(slot * b.gu_bytes, ex.gate_up(layer, e).b)?;
        b.down.upload_at(slot * b.down_bytes, ex.down(layer, e).b)?;
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
    fn repacks_nvfp4_and_q8() {
        // one NVFP4 row of 64: scales 0x38 (1/2 halved: value = FP4 * 0.5), nibbles counting up
        let mut blk = vec![0u8; 36];
        blk[..4].copy_from_slice(&[0x38, 0x40, 0x38, 0x40]); // 0.5, 1.0, 0.5, 1.0
        for (j, b) in blk[4..].iter_mut().enumerate() {
            *b = (j % 16) as u8 | (((j + 5) % 16) as u8) << 4;
        }
        let mut want = vec![0f32; 64];
        crate::quant::dequant_nvfp4_row(&blk, &mut want);
        let mut dst = vec![0u8; Fmt::Nv4.bytes(1, 64)];
        repack(Fmt::Nv4, &blk, 1, 64, 3.0, &mut dst);
        let m = Mat { fmt: Fmt::Nv4, b: &dst, rows: 1, k: 64 };
        let mut got = vec![0f32; 64];
        m.dequant_row(0, &mut got);
        assert_eq!(got, want);
        assert_eq!(m.scale(), 3.0);
        // two Q8_0 rows of 32
        let mut src = vec![];
        for r in 0..2u8 {
            src.extend_from_slice(&half::f16::from_f32(0.5 + r as f32).to_le_bytes());
            src.extend((0..32).map(|i| (i as i8 - 16 + r as i8) as u8));
        }
        let mut dst = vec![0u8; Fmt::Q8.bytes(2, 32)];
        repack(Fmt::Q8, &src, 2, 32, 1.0, &mut dst);
        let m = Mat { fmt: Fmt::Q8, b: &dst, rows: 2, k: 32 };
        let mut got = vec![0f32; 32];
        m.dequant_row(1, &mut got);
        assert_eq!((got[0], got[31]), (-15.0 * 1.5, 16.0 * 1.5));
    }
}
