//! Prefill vs decode for one token, layer by layer, expert cache off: `ik-pfsmall <checkpoint dir>`
use std::cell::RefCell;
use std::rc::Rc;

use ik_engine::engine::{Engine, Options, threads_from_env};

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: 4096, cache_slots: 0, threads: threads_from_env(), graph: false, upload_budget: 0 })?;
    let rec: Rc<RefCell<Vec<(String, usize, Vec<f32>)>>> = Rc::new(RefCell::new(vec![]));
    let r2 = rec.clone();
    e.probe = Some(Box::new(move |name: &str, il: usize, d: &[f32]| r2.borrow_mut().push((name.to_string(), il, d.to_vec()))));
    let toks = [371u32, 151451, 149879];
    e.reset()?;
    for &t in &toks {
        e.decode(t)?;
    }
    let keep = |r: &(String, usize, Vec<f32>)| ["Qcur", "Kcur", "Vcur", "Rcur", "kqv_out", "ffn_inp", "ffn_norm_in", "ffn_moe_out", "l_out"].contains(&r.0.as_str());
    let dec: Vec<_> = std::mem::take(&mut *rec.borrow_mut()).into_iter().filter(keep).collect();
    e.reset()?;
    e.prefill(&toks)?;
    let pre: Vec<_> = std::mem::take(&mut *rec.borrow_mut()).into_iter().filter(keep).collect();
    // decode records: 3 tokens in order; prefill records: one per tensor with 3 rows
    let n_dec = dec.len() / 3;
    let mut shown = 0;
    for (i, (name, il, p)) in pre.iter().enumerate() {
        let len = p.len() / 3;
        for t in 0..3 {
            let Some((dn, dil, d)) = dec.get(t * n_dec + i) else { continue };
            if dn != name || dil != il {
                println!("order differs at {i}: {name}-{il} vs {dn}-{dil}");
                return Ok(());
            }
            let pr = &p[t * len..(t + 1) * len];
            let num: f32 = pr.iter().zip(d).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt();
            let den: f32 = d.iter().map(|y| y * y).sum::<f32>().sqrt();
            let rel = num / den.max(1e-20);
            if rel > 1e-3 && shown < 12 {
                println!("token {t} {name}-{il}: rel diff {rel:.5}");
                shown += 1;
            }
        }
    }
    println!("compared {} tensors", pre.len());
    Ok(())
}
