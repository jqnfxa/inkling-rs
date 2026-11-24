//! Quantized formats of the model, as ggml lays them out: Q8_0 (repacked for the GPU), NVFP4 (the experts), and the
//! float types of the small matrices.

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

/// The E2M1 values doubled, as ggml's kvalues_mxfp4: an NVFP4 nibble's value is this times its UE4M3 scale's half.
pub const FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// A UE4M3 scale, halved (ggml_ue4m3_to_fp32), so that it multiplies the doubled E2M1 values of FP4.
pub fn ue4m3(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let (e, m) = ((x >> 3) & 0xF, x & 7);
    let raw = if e == 0 { m as f32 * (2f32).powi(-9) } else { (1.0 + m as f32 / 8.0) * (2f32).powi(e as i32 - 7) };
    raw * 0.5
}

/// ggml dequantize_row_nvfp4: each block's 16-value groups hold their first 8 values in the low nibbles of 8 bytes and
/// the next 8 in the high ones.
pub fn dequant_nvfp4_row(src: &[u8], out: &mut [f32]) {
    for (blk, o) in src.chunks_exact(36).zip(out.chunks_exact_mut(64)) {
        for s in 0..4 {
            let d = ue4m3(blk[s]);
            for j in 0..8 {
                let q = blk[4 + s * 8 + j];
                o[s * 16 + j] = FP4[(q & 0xF) as usize] as f32 * d;
                o[s * 16 + 8 + j] = FP4[(q >> 4) as usize] as f32 * d;
            }
        }
    }
}

/// A row of a float tensor (F32, F16, BF16) as f32.
pub fn float_row(ty: crate::gguf::GgmlType, src: &[u8], out: &mut [f32]) {
    use crate::gguf::GgmlType;
    match ty {
        GgmlType::F32 => out.iter_mut().zip(src.chunks_exact(4)).for_each(|(o, c)| *o = f32::from_le_bytes(c.try_into().unwrap())),
        GgmlType::F16 => out.iter_mut().zip(src.chunks_exact(2)).for_each(|(o, c)| *o = h(c, 0)),
        GgmlType::BF16 => out.iter_mut().zip(src.chunks_exact(2)).for_each(|(o, c)| *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)),
        t => panic!("{t:?} is not a float type"),
    }
}

/// Quantizes float rows to Q8_0 (as ggml's quantize_row_q8_0_ref) into the repacked layout of repack_q8.
pub fn quantize_q8_rows(x: &[f32], k: usize, qs: &mut Vec<u8>, d: &mut Vec<u16>) {
    assert_eq!(k % 32, 0);
    for b in x.chunks_exact(32) {
        let amax = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        let dd = amax / 127.0;
        let id = if dd != 0.0 { 1.0 / dd } else { 0.0 };
        d.push(f16::from_f32(dd).to_bits());
        qs.extend(b.iter().map(|v| (v * id).round() as i8 as u8));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_and_ue4m3() {
        // UE4M3: 0x38 = 2^0 * 1.0 -> 0.5 halved; 0x39 = 1.125 -> 0.5625; 0x01 = 2^-9 -> 2^-10
        assert_eq!(ue4m3(0x38), 0.5);
        assert_eq!(ue4m3(0x39), 0.5625);
        assert_eq!(ue4m3(0x01), (2f32).powi(-10));
        assert_eq!(ue4m3(0x7F), 0.0);
        let mut blk = [0u8; 36];
        blk[0] = 0x38; // scale 1 (0.5 doubled)
        blk[4] = 0x7F; // element 0: 0xF -> -12, element 8: 0x7 -> 12
        let mut o = [0f32; 64];
        dequant_nvfp4_row(&blk, &mut o);
        assert_eq!((o[0], o[8], o[1]), (-6.0, 6.0, 0.0));
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
