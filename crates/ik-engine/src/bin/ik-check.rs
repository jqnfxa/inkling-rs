//! Compares the engine with a llama.cpp reference dump (tools/refdump): `ik-check <checkpoint dir> <ref.bin> [cache_slots]`
//! The prompt runs one token at a time; each probed tensor is compared with the matching row of the reference.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::rc::Rc;

use ik_engine::engine::{Engine, Options};

struct Rec {
    i32: bool,
    rows: usize,
    data: Vec<f32>,
}

fn read_ref(path: &str) -> Result<HashMap<(u32, String), Rec>, String> {
    let mut b = vec![];
    std::fs::File::open(path).map_err(|e| e.to_string())?.read_to_end(&mut b).map_err(|e| e.to_string())?;
    let mut p = 0;
    let u32_at = |b: &[u8], p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
    let mut m = HashMap::new();
    while p + 8 <= b.len() {
        let step = u32_at(&b, p);
        let ln = u32_at(&b, p + 4) as usize;
        let name = String::from_utf8_lossy(&b[p + 8..p + 8 + ln]).into_owned();
        p += 8 + ln;
        let ty = u32_at(&b, p);
        p += 4;
        let ne: Vec<usize> = (0..4).map(|i| i64::from_le_bytes(b[p + 8 * i..p + 8 * i + 8].try_into().unwrap()) as usize).collect();
        p += 32;
        let cnt: usize = ne.iter().product();
        let data: Vec<f32> = (0..cnt)
            .map(|i| {
                let w = u32_at(&b, p + 4 * i);
                if ty == 1 { w as i32 as f32 } else { f32::from_bits(w) }
            })
            .collect();
        p += 4 * cnt;
        // the token dimension is the outermost one that is not 1
        let rows = if ne[3] > 1 { ne[3] } else if ne[2] > 1 && !name.starts_with("Qcur") && !name.starts_with("Kcur") && !name.starts_with("Vcur") && !name.starts_with("Rcur") { ne[2] } else { 1 };
        m.insert((step, name), Rec { i32: ty == 1, rows, data });
    }
    Ok(m)
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
    num / den.max(1e-20)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let slots: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let refs = Rc::new(read_ref(&args[2])?);
    let tokens: Vec<u32> = refs.get(&(0, "tokens".into())).ok_or("no tokens in the reference")?.data.iter().map(|&x| x as u32).collect();
    eprintln!("prompt {:?}", tokens);
    let mut eng = Engine::new(std::path::Path::new(&args[1]), Options { n_ctx: 4096, cache_slots: slots, threads: ik_engine::engine::threads_from_env(), graph: false, upload_budget: 4 })?;
    eprintln!("loaded: {} MiB of device weights on {}", eng.w.bytes >> 20, eng.ctx.name);

    // (step, row) of the current token, and the worst error per tensor kind
    let cur = Rc::new(RefCell::new((0u32, 0usize)));
    let worst: Rc<RefCell<HashMap<String, (f32, String)>>> = Rc::new(RefCell::new(HashMap::new()));
    let fails = Rc::new(RefCell::new(0usize));
    {
        let (refs, cur, worst, fails) = (refs.clone(), cur.clone(), worst.clone(), fails.clone());
        eng.probe = Some(Box::new(move |name: &str, il: usize, data: &[f32]| {
            let (step, row) = *cur.borrow();
            let key = format!("{name}-{il}");
            let Some(r) = refs.get(&(step, key.clone())) else { return };
            let len = r.data.len() / r.rows;
            if len != data.len() {
                eprintln!("{key}: length {} vs reference {len}", data.len());
                return;
            }
            let rr = &r.data[row * len..(row + 1) * len];
            if r.i32 {
                let mut a: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                let mut b: Vec<i64> = rr.iter().map(|&x| x as i64).collect();
                a.sort();
                b.sort();
                if a != b {
                    *fails.borrow_mut() += 1;
                    if *fails.borrow() <= 10 {
                        eprintln!("step {step} row {row} {key}: experts {a:?} vs reference {b:?}");
                    }
                }
                return;
            }
            let e = rel_err(data, rr);
            let mut w = worst.borrow_mut();
            let ent = w.entry(name.to_string()).or_insert((0.0, String::new()));
            if e > ent.0 {
                *ent = (e, format!("step {step} row {row} {key}"));
            }
            if e > 0.05 {
                *fails.borrow_mut() += 1;
                if *fails.borrow() <= 10 {
                    eprintln!("step {step} row {row} {key}: rel err {e:.4}");
                }
            }
        }));
    }
    let mut last = vec![];
    if std::env::var("IK_PREFILL").is_ok() {
        last = eng.prefill(&tokens)?;
    } else {
        for (i, &t) in tokens.iter().enumerate() {
            *cur.borrow_mut() = (0, i);
            last = eng.decode(t)?;
        }
    }
    let check_logits = |step: u32, logits: &[f32]| {
        if let Some(r) = refs.get(&(step, "result_output".into())) {
            let len = r.data.len() / r.rows;
            let rr = &r.data[(r.rows - 1) * len..];
            let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            eprintln!("step {step}: logits rel err {:.5}, argmax {} vs reference {}", rel_err(logits, rr), am(logits), am(rr));
        }
    };
    check_logits(0, &last);
    let t0 = std::time::Instant::now();
    let mut step = 1;
    while let Some(r) = refs.get(&(step, "input".into())) {
        *cur.borrow_mut() = (step, 0);
        let logits = eng.decode(r.data[0] as u32)?;
        check_logits(step, &logits);
        step += 1;
    }
    eprintln!("{} decode steps in {:?}", step - 1, t0.elapsed());
    let mut w: Vec<_> = worst.borrow().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    w.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, (e, at)) in w {
        eprintln!("worst {k:12}: {e:.5} at {at}");
    }
    eprintln!("cache hits {} misses {} uploads {}; failures {}", eng.cache.hits, eng.cache.misses, eng.cache.uploads, fails.borrow());
    Ok(())
}
