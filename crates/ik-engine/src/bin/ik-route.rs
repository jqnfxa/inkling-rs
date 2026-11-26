//! How well a layer's experts can be told before its attention: `ik-route <checkpoint dir> <text file> [n_tokens] [skip]`.
//! Decodes the text a token at a time (eager, with probes) and, at each MoE layer pass, compares the experts the router
//! picks with those picked from earlier states: the layer's input (before attention), the attention output (before the
//! residual mix), the same layer pass of the previous token, and the previous loop pass of the same layer. Prints the
//! share of the 8 experts each predictor gets right, per layer pass and overall.
use ik_engine::engine::{Engine, Options, threads_from_env};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

fn top8(router: &[f32], bias: &[f32], x: &[f32]) -> Vec<u32> {
    let n = x.len();
    let mut s: Vec<(f32, u32)> = (0..bias.len())
        .map(|e| {
            let l: f32 = router[e * n..(e + 1) * n].iter().zip(x).map(|(a, b)| a * b).sum();
            (1.0 / (1.0 + (-l).exp()) + bias[e], e as u32)
        })
        .collect();
    s.sort_by(|a, b| b.0.total_cmp(&a.0));
    s.iter().take(8).map(|p| p.1).collect()
}

fn norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, w)| v * r * w).collect()
}

fn hits(a: &[u32], b: &[u32]) -> usize {
    a.iter().filter(|e| b.contains(e)).count()
}

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().collect();
    let n_tok: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(300);
    let skip: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(32);
    let mut e = Engine::new(std::path::Path::new(&a[1]), Options { n_ctx: n_tok + 8, cache_slots: usize::MAX, threads: threads_from_env(), graph: false, upload_budget: 4 })?;
    let tok = e.ck.tokenizer()?;
    let hp = e.hp.clone();
    let n_phys = hp.n_phys;
    let sched = hp.schedule();
    let mut router = vec![];
    let mut bias = vec![];
    let mut fnorm = vec![];
    for l in 0..n_phys {
        router.push(e.ck.floats(&format!("model.llm.layers.{l}.mlp.gate.weight"))?);
        bias.push(e.ck.floats(&format!("model.llm.layers.{l}.mlp.gate.bias"))?);
        fnorm.push(e.ck.floats(&format!("model.llm.layers.{l}.mlp_norm.weight"))?);
    }
    let text = std::fs::read_to_string(&a[2]).map_err(|x| x.to_string())?;
    let ids = tok.encode(&text, false);
    let n_tok = n_tok.min(ids.len());

    type Seen = HashMap<(String, usize), Vec<f32>>;
    let seen: Rc<RefCell<Seen>> = Rc::new(RefCell::new(HashMap::new()));
    {
        let seen = seen.clone();
        e.probe = Some(Box::new(move |name: &str, il: usize, data: &[f32]| {
            if matches!(name, "l_out" | "ffn_inp" | "ffn_norm_in" | "ffn_moe_topk") {
                seen.borrow_mut().insert((name.to_string(), il), data.to_vec());
            }
        }));
    }
    let n_pass = hp.n_slots();
    // per layer pass: [checks, sanity, input, attention out, previous token, previous loop pass, input ∪ prev token (16)]
    let mut acc = vec![[0usize; 7]; n_pass];
    // over the misses (experts not in the GPU cache): [misses, input predicted, input ∪ previous token predicted]
    let mut miss = [0usize; 3];
    let mut prev_tok: Vec<Vec<u32>> = vec![vec![]; n_pass];
    for (t, &id) in ids[..n_tok].iter().enumerate() {
        seen.borrow_mut().clear();
        let cached = e.cache.slot_of.clone();
        e.decode(id)?;
        let s = seen.borrow();
        let mut actual: Vec<Vec<u32>> = vec![vec![]; n_pass];
        for il in 0..n_pass {
            if let Some(v) = s.get(&("ffn_moe_topk".into(), il)) {
                actual[il] = v.iter().map(|&x| x as u32).collect();
            }
        }
        if t >= skip {
            for k in 1..sched.len() {
                let (il, ip) = (sched[k].il, sched[k].ip);
                let act = &actual[il];
                if act.len() != 8 {
                    continue;
                }
                let r = &mut acc[il];
                r[0] += 1;
                if let Some(x) = s.get(&("ffn_norm_in".into(), il)) {
                    r[1] += hits(&top8(&router[ip], &bias[ip], x), act);
                }
                let mut p_in = vec![];
                if let Some(x) = s.get(&("l_out".into(), sched[k - 1].il)) {
                    p_in = top8(&router[ip], &bias[ip], &norm(x, &fnorm[ip], hp.eps));
                    r[2] += hits(&p_in, act);
                }
                if let Some(x) = s.get(&("ffn_inp".into(), il)) {
                    r[3] += hits(&top8(&router[ip], &bias[ip], &norm(x, &fnorm[ip], hp.eps)), act);
                }
                r[4] += hits(&prev_tok[il], act);
                // the previous pass of the same physical layer in this token
                if let Some(j) = (0..k).rev().find(|&j| sched[j].ip == ip) {
                    r[5] += hits(&actual[sched[j].il], act);
                }
                let mut u = p_in.clone();
                u.extend(prev_tok[il].iter().filter(|x| !p_in.contains(x)));
                r[6] += hits(&u, act);
                for &x in act {
                    if cached[ip][x as usize] < 0 {
                        miss[0] += 1;
                        miss[1] += p_in.contains(&x) as usize;
                        miss[2] += u.contains(&x) as usize;
                    }
                }
            }
        }
        prev_tok = actual;
    }
    println!("share of the 8 experts predicted (per layer pass; 'loop' only where an earlier pass of the layer exists)");
    println!("(in the order of evaluation)");
    println!("{:>4} {:>4} {:>7} {:>7} {:>7} {:>7} {:>7} {:>9}", "pass", "phys", "sanity", "input", "attnout", "prevtok", "loop", "in+prev");
    let mut tot = [0usize; 7];
    for st in &sched {
        let (il, r) = (st.il, &acc[st.il]);
        if r[0] == 0 {
            continue;
        }
        let d = 8.0 * r[0] as f64 / 100.0;
        println!("{il:>4} {:>4} {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>8.1}%", st.ip, r[1] as f64 / d, r[2] as f64 / d, r[3] as f64 / d, r[4] as f64 / d, r[5] as f64 / d, r[6] as f64 / d);
        for k in 0..7 {
            tot[k] += r[k];
        }
    }
    let d = 8.0 * tot[0] as f64 / 100.0;
    println!("all  {:>11.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>8.1}%", tot[1] as f64 / d, tot[2] as f64 / d, tot[3] as f64 / d, tot[4] as f64 / d, tot[5] as f64 / d, tot[6] as f64 / d);
    let m = miss[0].max(1) as f64 / 100.0;
    println!("misses {:.1} a token: input predicts {:.1}% of them, input ∪ previous token {:.1}%", miss[0] as f64 / (n_tok - skip) as f64, miss[1] as f64 / m, miss[2] as f64 / m);
    Ok(())
}
