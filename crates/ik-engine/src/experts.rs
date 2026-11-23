//! The routed experts: host copies (the mapped GGUF), CPU compute of the misses, and the GPU cache of hot experts.
//! The cache is keyed by physical layer, so the passes of the loop share it.

use rayon::prelude::*;

use crate::cuda::{DevBuf, Result};
use crate::gguf::{Gguf, GgmlType};
use crate::hp::Hp;
use crate::quant::{dequant_q4k_row, dequant_q50_row, dot};

/// Byte ranges of the experts of one physical layer in the mapped file.
#[derive(Clone, Copy)]
pub struct HostExperts {
    pub gate_up: usize, // offset of expert 0
    pub down: usize,
}

/// Anonymous memory backed by transparent huge pages, holding a copy of the experts.
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
        unsafe { libc::madvise(p, len, libc::MADV_HUGEPAGE) };
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
    /// a copy of the experts in huge pages; offsets of `layers` index it
    pub mem: Option<HugeMem>,
    /// the copy is page-locked, so that cache uploads are DMA and run asynchronously
    pub pinned: bool,
    pub layers: Vec<HostExperts>,
    pub gu_bytes: usize,   // one expert's gate_up
    pub down_bytes: usize, // one expert's down
    pub gu_row: usize,
    pub down_row: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub beta: f32,
    pub lb: f32,
}

impl Experts {
    pub fn new(g: &Gguf, hp: &Hp) -> Result<Experts> {
        let mut layers = vec![];
        for i in 0..hp.n_phys {
            let gu = g.tensor(&format!("blk.{i}.ffn_gate_up_exps.weight"))?;
            let dn = g.tensor(&format!("blk.{i}.ffn_down_exps.weight"))?;
            if gu.ty != GgmlType::Q4K || dn.ty != GgmlType::Q5_0 {
                return Err(format!("layer {i}: experts are {:?}/{:?}, the kernels take Q4_K/Q5_0", gu.ty, dn.ty));
            }
            if gu.ne != [hp.n_embd, 2 * hp.n_ff_exp, hp.n_expert] || dn.ne != [hp.n_ff_exp, hp.n_embd, hp.n_expert] {
                return Err(format!("layer {i}: unexpected expert shapes {:?} {:?}", gu.ne, dn.ne));
            }
            layers.push(HostExperts { gate_up: gu.offset, down: dn.offset });
        }
        // copy the experts to huge pages: TLB misses cost the CPU experts, which stream them from RAM
        let mut mem = None;
        if std::env::var("IK_NO_HUGE").is_err() {
            let per: usize = gu_sz(g, hp) + dn_sz(g, hp);
            let m = HugeMem::new(per * hp.n_phys)?;
            let dst = m.ptr as usize;
            let jobs: Vec<(usize, usize, usize)> = (0..hp.n_phys)
                .flat_map(|i| {
                    let (gu_o, dn_o) = (layers[i].gate_up, layers[i].down);
                    [(gu_o, i * per, gu_sz(g, hp)), (dn_o, i * per + gu_sz(g, hp), dn_sz(g, hp))]
                })
                .collect();
            jobs.par_iter().for_each(|&(src, off, len)| {
                // SAFETY: disjoint ranges of the fresh mapping
                unsafe { std::ptr::copy_nonoverlapping(g.mmap.as_ptr().add(src), (dst as *mut u8).add(off), len) };
            });
            for (i, l) in layers.iter_mut().enumerate() {
                *l = HostExperts { gate_up: i * per, down: i * per + gu_sz(g, hp) };
            }
            mem = Some(m);
        }
        let mut pinned = false;
        if let Some(m) = &mem {
            if std::env::var("IK_NO_PIN").is_err() {
                let t = std::time::Instant::now();
                match crate::cuda::register_host(m.ptr, m.len) {
                    Ok(()) => {
                        pinned = true;
                        eprintln!("experts page-locked in {:.1}s", t.elapsed().as_secs_f64());
                    }
                    Err(e) => eprintln!("experts not page-locked ({e}): cache uploads stay synchronous"),
                }
            }
        }
        let gu_row = GgmlType::Q4K.row_bytes(hp.n_embd);
        let down_row = GgmlType::Q5_0.row_bytes(hp.n_ff_exp);
        Ok(Experts {
            mem,
            pinned,
            layers,
            gu_bytes: gu_row * 2 * hp.n_ff_exp,
            down_bytes: down_row * hp.n_embd,
            gu_row,
            down_row,
            n_embd: hp.n_embd,
            n_ff: hp.n_ff_exp,
            beta: hp.situ_beta,
            lb: hp.situ_lb,
        })
    }

    fn bytes<'a>(&'a self, g: &'a Gguf) -> &'a [u8] {
        match &self.mem {
            Some(m) => m.slice(),
            None => &g.mmap,
        }
    }

    pub fn gate_up<'a>(&'a self, g: &'a Gguf, layer: usize, e: usize) -> &'a [u8] {
        let o = self.layers[layer].gate_up + e * self.gu_bytes;
        &self.bytes(g)[o..o + self.gu_bytes]
    }

    pub fn down<'a>(&'a self, g: &'a Gguf, layer: usize, e: usize) -> &'a [u8] {
        let o = self.layers[layer].down + e * self.down_bytes;
        &self.bytes(g)[o..o + self.down_bytes]
    }

    /// Reference CPU path: y = down(situ(gate_up x)) of expert e of `layer`.
    pub fn compute_cpu(&self, g: &Gguf, layer: usize, e: usize, x: &[f32], y: &mut [f32]) {
        let gu = self.gate_up(g, layer, e);
        let nff = self.n_ff;
        let mut h = vec![0f32; 2 * nff];
        h.par_chunks_mut(64).enumerate().for_each(|(c, out)| {
            let mut row = vec![0f32; self.n_embd];
            for (i, o) in out.iter_mut().enumerate() {
                let r = c * 64 + i;
                dequant_q4k_row(&gu[r * self.gu_row..(r + 1) * self.gu_row], &mut row);
                *o = dot(&row, x);
            }
        });
        let a: Vec<f32> = (0..nff)
            .map(|i| {
                let (gv, uv) = (h[i], h[nff + i]);
                let act = self.beta * (gv / self.beta).tanh() * (1.0 / (1.0 + (-gv).exp()));
                act * self.lb * (uv / self.lb).tanh()
            })
            .collect();
        let dn = self.down(g, layer, e);
        y.par_chunks_mut(64).enumerate().for_each(|(c, out)| {
            let mut row = vec![0f32; nff];
            for (i, o) in out.iter_mut().enumerate() {
                let r = c * 64 + i;
                dequant_q50_row(&dn[r * self.down_row..(r + 1) * self.down_row], &mut row);
                *o = dot(&row, &a);
            }
        });
    }
}

/// GPU cache of experts: two banks of slots, and the slot of each (physical layer, expert).
pub struct ExpertCache {
    pub gu_bank: DevBuf,
    pub down_bank: DevBuf,
    pub n_slots: usize,
    pub slot_of: Vec<Vec<i32>>, // [layer][expert] -> slot or -1
    pub owner: Vec<(usize, usize)>, // [slot] -> (layer, expert), usize::MAX if free
    pub uses: Vec<Vec<f32>>, // decayed use counts
    pub hits: u64,
    pub misses: u64,
    pub uploads: u64,
    /// slots being filled by an async upload: not cached yet, and no victims
    pub busy: Vec<bool>,
    pub pending: Vec<Pending>,
}

/// An async upload of expert e of `layer` into `slot`, through staging buffer `stage`.
pub struct Pending {
    pub slot: usize,
    pub layer: usize,
    pub e: usize,
    pub stage: usize,
    pub ev: crate::cuda::Event,
}

impl ExpertCache {
    pub fn new(ex: &Experts, hp: &Hp, n_slots: usize) -> Result<ExpertCache> {
        Ok(ExpertCache {
            gu_bank: DevBuf::new(ex.gu_bytes * n_slots.max(1))?,
            down_bank: DevBuf::new(ex.down_bytes * n_slots.max(1))?,
            n_slots,
            slot_of: vec![vec![-1; hp.n_expert]; hp.n_phys],
            owner: vec![(usize::MAX, usize::MAX); n_slots],
            uses: vec![vec![0.0; hp.n_expert]; hp.n_phys],
            hits: 0,
            misses: 0,
            uploads: 0,
            busy: vec![false; n_slots],
            pending: vec![],
        })
    }

    /// Uploads expert e of `layer` into `slot` (synchronous); the down matrix goes raw to `stage_down`, for the caller
    /// to repack into the bank.
    pub fn fill(&mut self, ex: &Experts, g: &Gguf, slot: usize, layer: usize, e: usize, stage_down: &DevBuf) -> Result<()> {
        let (ol, oe) = self.owner[slot];
        if ol != usize::MAX {
            self.slot_of[ol][oe] = -1;
        }
        self.gu_bank.upload_at(slot * ex.gu_bytes, ex.gate_up(g, layer, e))?;
        stage_down.upload(ex.down(g, layer, e))?;
        self.owner[slot] = (layer, e);
        self.slot_of[layer][e] = slot as i32;
        self.uploads += 1;
        Ok(())
    }

    /// The least used slot, free slots first; None if all are being filled.
    pub fn victim(&self) -> Option<usize> {
        let mut best = (f32::INFINITY, usize::MAX);
        for (s, &(l, e)) in self.owner.iter().enumerate() {
            if self.busy[s] {
                continue;
            }
            if l == usize::MAX {
                return Some(s);
            }
            let u = self.uses[l][e];
            if u < best.0 {
                best = (u, s);
            }
        }
        (best.1 != usize::MAX).then_some(best.1)
    }
}

fn gu_sz(_g: &Gguf, hp: &Hp) -> usize {
    GgmlType::Q4K.row_bytes(hp.n_embd) * 2 * hp.n_ff_exp * hp.n_expert
}

fn dn_sz(_g: &Gguf, hp: &Hp) -> usize {
    GgmlType::Q5_0.row_bytes(hp.n_ff_exp) * hp.n_embd * hp.n_expert
}
