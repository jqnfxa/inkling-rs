//! CPU expert kernels at decode: `ik-cpubw <model.gguf> [threads...]`.
//! "hot": one thread over 64 rows that stay in L2 (the kernel's own speed); "ram": the gate_up and down of 48
//! experts copied to huge pages (~230 MB, far beyond L3), read once per pass by N threads, each its own rows,
//! as the engine reads misses. Threads are unpinned unless IK_PIN lists CPUs ("0,2,4,...").
use ik_engine::cpu::{Q8_0, Q8K, dot_q4k, dot_q50, quantize_q8_0, quantize_q8k};
use ik_engine::experts::{Experts, HugeMem};
use ik_engine::gguf::Gguf;
use ik_engine::hp::Hp;
use std::time::Instant;


// the kernels before 2026-11-23, for an A/B under the same load
#[target_feature(enable = "avx2,fma")]
unsafe fn old_q4k(row: &[u8], y: &Q8K) -> f32 {
    use std::arch::x86_64::*;
    fn scales_mins(sc: &[u8]) -> ([u8; 8], [u8; 8]) {
        let mut s = [0u8; 8];
        let mut m = [0u8; 8];
        for j in 0..8 {
            if j < 4 { s[j] = sc[j] & 63; m[j] = sc[j + 4] & 63; } else { s[j] = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m[j] = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
        }
        (s, m)
    }
    unsafe {
    let m4 = _mm256_set1_epi8(0xF);
    let mut acc = _mm256_setzero_ps();
    let mut acc_m = 0f32;
    let nb = row.len() / 144;
    for b in 0..nb {
        let blk = row.as_ptr().add(b * 144);
        let d = half::f16::from_bits(u16::from_le_bytes([*blk, *blk.add(1)])).to_f32() * y.d[b];
        let dmin = half::f16::from_bits(u16::from_le_bytes([*blk.add(2), *blk.add(3)])).to_f32() * y.d[b];
        let (s, m) = scales_mins(std::slice::from_raw_parts(blk.add(4), 12));
        let bs = y.bsums.as_ptr().add(b * 16);
        let mut summ = 0i32;
        for j in 0..8 { summ += m[j] as i32 * (*bs.add(2 * j) as i32 + *bs.add(2 * j + 1) as i32); }
        acc_m += dmin * summ as f32;
        let q4 = blk.add(16);
        let q8 = y.qs.as_ptr().add(b * 256);
        let mut sumi = _mm256_setzero_si256();
        for c in 0..4 {
            let bits = _mm256_loadu_si256(q4.add(c * 32) as *const __m256i);
            let lo = _mm256_and_si256(bits, m4);
            let hi = _mm256_and_si256(_mm256_srli_epi16(bits, 4), m4);
            let y0 = _mm256_loadu_si256(q8.add(c * 64) as *const __m256i);
            let y1 = _mm256_loadu_si256(q8.add(c * 64 + 32) as *const __m256i);
            let p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, y0), _mm256_set1_epi16(s[2 * c] as i16));
            let p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, y1), _mm256_set1_epi16(s[2 * c + 1] as i16));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p1));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc);
    t.iter().sum::<f32>() - acc_m
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn old_q50(row: &[u8], y: &Q8_0) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
    let m4 = _mm_set1_epi8(0xF);
    let shuf = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101, 0x0000000000000000);
    let bitm = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe_u64 as i64);
    let ones16 = _mm256_set1_epi16(1);
    let mut acc = _mm256_setzero_ps();
    let mut corr = 0f32;
    let nb = row.len() / 22;
    for b in 0..nb {
        let blk = row.as_ptr().add(b * 22);
        let d = half::f16::from_bits(u16::from_le_bytes([*blk, *blk.add(1)])).to_f32() * y.d[b];
        let qh = u32::from_le_bytes([*blk.add(2), *blk.add(3), *blk.add(4), *blk.add(5)]);
        let raw = _mm_loadu_si128(blk.add(6) as *const __m128i);
        let lo = _mm_and_si128(raw, m4);
        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        let q = _mm256_set_m128i(hi, lo);
        let v = _mm256_shuffle_epi8(_mm256_set1_epi32(qh as i32), shuf);
        let v = _mm256_or_si256(v, bitm);
        let hb = _mm256_cmpeq_epi8(v, _mm256_set1_epi64x(-1));
        let q = _mm256_or_si256(q, _mm256_and_si256(hb, _mm256_set1_epi8(0x10)));
        let yy = _mm256_loadu_si256(y.qs.as_ptr().add(b * 32) as *const __m256i);
        let p = _mm256_madd_epi16(_mm256_maddubs_epi16(q, yy), ones16);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p), acc);
        corr += d * 16.0 * y.sums[b] as f32;
    }
    let mut t = [0f32; 8];
    _mm256_storeu_ps(t.as_mut_ptr(), acc);
    t.iter().sum::<f32>() - corr
    }
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let threads: Vec<usize> = a[2..].iter().filter_map(|s| s.parse().ok()).collect();
    let threads = if threads.is_empty() { vec![1, 2, 4, 6, 8] } else { threads };
    unsafe { std::env::set_var("IK_NO_HUGE", "1") };
    let g = Gguf::open(std::path::Path::new(&a[1]))?;
    let hp = Hp::from_gguf(&g)?;
    let ex = Experts::new(&g, &hp)?;
    let x: Vec<f32> = (0..2304).map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0).collect();
    let mut xq = Q8K { d: vec![], qs: vec![], bsums: vec![] };
    quantize_q8k(&x, &mut xq);
    let mut aq = Q8_0 { d: vec![], qs: vec![], sums: vec![] };
    quantize_q8_0(&x[..1152], &mut aq);

    let gu = ex.gate_up(&g, 3, 0);
    let dn = ex.down(&g, 3, 0);
    let (reps, rows) = (4000, 64);
    let mut s = 0f32;
    let hot = |f: &dyn Fn(&[u8]) -> f32, m: &[u8], rb: usize| -> f64 {
        let t = Instant::now();
        let mut s = 0f32;
        for _ in 0..reps {
            for r in 0..rows {
                s += f(&m[r * rb..(r + 1) * rb]);
            }
        }
        std::hint::black_box(s);
        (reps * rows * rb) as f64 / t.elapsed().as_secs_f64() / 1e9
    };
    for round in 0..3 {
        let new_gu = hot(&|r| dot_q4k(r, &xq), gu, ex.gu_row);
        let old_gu = hot(&|r| unsafe { old_q4k(r, &xq) }, gu, ex.gu_row);
        let new_dn = hot(&|r| dot_q50(r, &aq), dn, ex.down_row);
        let old_dn = hot(&|r| unsafe { old_q50(r, &aq) }, dn, ex.down_row);
        println!("hot, 1 thread, round {round}: gate_up new {new_gu:.1} old {old_gu:.1} GB/s; down new {new_dn:.1} old {old_dn:.1} GB/s");
    }
    s += dot_q4k(&gu[..ex.gu_row], &xq) - unsafe { old_q4k(&gu[..ex.gu_row], &xq) };
    s += dot_q50(&dn[..ex.down_row], &aq) - unsafe { old_q50(&dn[..ex.down_row], &aq) };
    println!("new - old on one row of each: {s}");

    let ne = 48;
    let (gb, db) = (ex.gu_bytes, ex.down_bytes);
    let mem = HugeMem::new(ne * (gb + db))?;
    let buf = unsafe { std::slice::from_raw_parts_mut(mem.ptr, mem.len) };
    for e in 0..ne {
        buf[e * gb..(e + 1) * gb].copy_from_slice(ex.gate_up(&g, 4 + e / 128, e % 128));
        buf[ne * gb + e * db..ne * gb + (e + 1) * db].copy_from_slice(ex.down(&g, 4 + e / 128, e % 128));
    }
    let buf: &[u8] = buf;
    let pins: Vec<usize> = std::env::var("IK_PIN").ok().map(|v| v.split(',').filter_map(|c| c.parse().ok()).collect()).unwrap_or_default();
    let run = |nt: usize, part: &(dyn Fn(usize) -> f32 + Sync)| -> f64 {
        let t = Instant::now();
        std::thread::scope(|sc| {
            for i in 0..nt {
                let pin = pins.get(i).copied();
                sc.spawn(move || {
                    if let Some(c) = pin {
                        unsafe {
                            let mut set: libc::cpu_set_t = std::mem::zeroed();
                            libc::CPU_SET(c, &mut set);
                            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
                        }
                    }
                    std::hint::black_box(part(i));
                });
            }
        });
        t.elapsed().as_secs_f64()
    };
    for &nt in &threads {
        let mut best = (0f64, 0f64);
        for _pass in 0..4 {
            let gu_rows = ne * 2 * ex.n_ff;
            let dt = run(nt, &|i| {
                let mut s = 0f32;
                for r in (i * gu_rows / nt)..((i + 1) * gu_rows / nt) {
                    s += dot_q4k(&buf[r * ex.gu_row..(r + 1) * ex.gu_row], &xq);
                }
                s
            });
            let dn_rows = ne * ex.n_embd;
            let base = ne * gb;
            let dt2 = run(nt, &|i| {
                let mut s = 0f32;
                for r in (i * dn_rows / nt)..((i + 1) * dn_rows / nt) {
                    s += dot_q50(&buf[base + r * ex.down_row..base + (r + 1) * ex.down_row], &aq);
                }
                s
            });
            best.0 = best.0.max((ne * gb) as f64 / dt / 1e9);
            best.1 = best.1.max((ne * db) as f64 / dt2 / 1e9);
        }
        println!("ram, {nt:2} threads: gate_up {:.1} GB/s, down {:.1} GB/s", best.0, best.1);
    }
    std::hint::black_box(s);
    Ok(())
}
