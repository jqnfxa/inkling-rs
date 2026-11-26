//! How concentrated full attention is at depth: `ik-attnmass <checkpoint dir> <text file> <n_tokens> [decode steps, 4]`.
//! Prefills n_tokens of the text, decodes a few tokens, and for each full-attention pass tells the share of 128-key
//! chunks a KV head needs (over its 3 query heads) to keep all but eps of the softmax mass.
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use ik_engine::engine::{Engine, Options, threads_from_env};

const EPS: [f64; 3] = [1e-3, 1e-4, 1e-5];

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a[3].parse().map_err(|_| "n_tokens")?;
    let steps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(4);
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: n + 64, cache_slots: usize::MAX, threads: threads_from_env(), graph: true, upload_budget: 4 })?;
    let tok = e.ck.tokenizer()?;
    let text = std::fs::read_to_string(&a[2]).map_err(|x| x.to_string())?;
    let ids = tok.encode(&text, false);
    if ids.len() < n + steps {
        return Err(format!("the text has {} tokens", ids.len()));
    }
    let t0 = std::time::Instant::now();
    e.feed(&ids[..n])?;
    eprintln!("prefill of {n} tokens: {:.1}s", t0.elapsed().as_secs_f64());
    let got: Rc<RefCell<HashMap<(String, usize), Vec<f32>>>> = Rc::new(RefCell::new(HashMap::new()));
    {
        let got = got.clone();
        e.probe = Some(Box::new(move |name: &str, il: usize, d: &[f32]| {
            if name.starts_with("attn_p") {
                got.borrow_mut().insert((name.to_string(), il), d.to_vec());
            }
        }));
    }
    // per pass: summed shares for each eps, and the count
    let mut acc: HashMap<usize, ([f64; 3], usize)> = HashMap::new();
    for i in 0..steps {
        got.borrow_mut().clear();
        e.decode(ids[n + i])?;
        let pos = e.pos; // keys = pos (the token just decoded included)
        let ns = pos.div_ceil(128);
        let g = got.borrow();
        let mut ils: Vec<usize> = g.keys().filter(|k| k.0 == "attn_pm").map(|k| k.1).collect();
        ils.sort();
        for il in ils {
            let (pm, pl) = (&g[&("attn_pm".into(), il)], &g[&("attn_pl".into(), il)]);
            let maxs = pm.len() / 24;
            let ent = acc.entry(il).or_default();
            for kvh in 0..8 {
                let mut need = [vec![false; ns], vec![false; ns], vec![false; ns]];
                for hh in 0..3 {
                    let at = |c: usize| (kvh * maxs + c) * 3 + hh;
                    let m = (0..ns).map(|c| pm[at(c)] as f64).fold(f64::NEG_INFINITY, f64::max);
                    let mut w: Vec<(f64, usize)> = (0..ns).map(|c| (pl[at(c)] as f64 * (pm[at(c)] as f64 - m).exp(), c)).collect();
                    let tot: f64 = w.iter().map(|x| x.0).sum();
                    w.sort_by(|x, y| y.0.total_cmp(&x.0));
                    for (k, eps) in EPS.iter().enumerate() {
                        let mut cum = 0.0;
                        for &(v, c) in &w {
                            if cum >= (1.0 - eps) * tot {
                                break;
                            }
                            cum += v;
                            need[k][c] = true;
                        }
                    }
                }
                for k in 0..3 {
                    ent.0[k] += need[k].iter().filter(|&&b| b).count() as f64 / ns as f64;
                }
                ent.1 += 1;
            }
        }
    }
    println!("depth {n}: share of 128-key chunks a KV head needs to keep all but eps of the attention mass");
    let mut ils: Vec<_> = acc.keys().copied().collect();
    ils.sort();
    for il in ils {
        let (s, c) = acc[&il];
        println!("  pass {il:2}: eps 1e-3 {:5.1}%   1e-4 {:5.1}%   1e-5 {:5.1}%", 100.0 * s[0] / c as f64, 100.0 * s[1] / c as f64, 100.0 * s[2] / c as f64);
    }
    Ok(())
}
