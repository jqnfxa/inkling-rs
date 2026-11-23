//! Quantized formats of the model: Q8_0 (repacked for the GPU), Q4_K and Q5_0 (experts), as ggml lays them out.

use half::f16;

#[inline]
fn h(b: &[u8], o: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[o], b[o + 1]])).to_f32()
}

/// Repacks Q8_0 rows ([rows][k/32] blocks of {f16 d, 32 i8}) into `qs` [rows][k] i8 and `d` [rows][k/32] f16 bits.
pub fn repack_q8(src: &[u8], rows: usize, k: usize, qs: &mut Vec<u8>, d: &mut Vec<u16>) {
    assert_eq!(k % 32, 0);
    let nb = k / 32;
    assert_eq!(src.len(), rows * nb * 34);
    qs.reserve(rows * k);
    d.reserve(rows * nb);
    for blk in src.chunks_exact(34) {
        d.push(u16::from_le_bytes([blk[0], blk[1]]));
        qs.extend_from_slice(&blk[2..34]);
    }
}

pub fn dequant_q8_row(src: &[u8], out: &mut [f32]) {
    for (blk, o) in src.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
        let d = h(blk, 0);
        for i in 0..32 {
            o[i] = blk[2 + i] as i8 as f32 * d;
        }
    }
}

fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// ggml dequantize_row_q4_K
pub fn dequant_q4k_row(src: &[u8], out: &mut [f32]) {
    for (blk, o) in src.chunks_exact(144).zip(out.chunks_exact_mut(256)) {
        let d = h(blk, 0);
        let dmin = h(blk, 2);
        let sc = &blk[4..16];
        let mut q = &blk[16..144];
        let mut is = 0;
        for j in (0..256).step_by(64) {
            let (s1, m1) = scale_min_k4(is, sc);
            let (s2, m2) = scale_min_k4(is + 1, sc);
            let (d1, mm1) = (d * s1 as f32, dmin * m1 as f32);
            let (d2, mm2) = (d * s2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                o[j + l] = d1 * (q[l] & 0xF) as f32 - mm1;
                o[j + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
            }
            q = &q[32..];
            is += 2;
        }
    }
}

/// ggml dequantize_row_q5_0
pub fn dequant_q50_row(src: &[u8], out: &mut [f32]) {
    for (blk, o) in src.chunks_exact(22).zip(out.chunks_exact_mut(32)) {
        let d = h(blk, 0);
        let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
        let qs = &blk[6..22];
        for j in 0..16 {
            let xh0 = ((qh >> j) << 4) & 0x10;
            let xh1 = (qh >> (j + 12)) & 0x10;
            let x0 = ((qs[j] as u32 & 0xF) | xh0) as i32 - 16;
            let x1 = ((qs[j] as u32 >> 4) | xh1) as i32 - 16;
            o[j] = x0 as f32 * d;
            o[j + 16] = x1 as f32 * d;
        }
    }
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 8 lanes, so that the compiler vectorizes it
    let mut acc = [0f32; 8];
    let (ca, ra) = a.split_at(a.len() / 8 * 8);
    let (cb, rb) = b.split_at(ca.len());
    for (x, y) in ca.chunks_exact(8).zip(cb.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}
