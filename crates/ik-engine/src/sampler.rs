//! Token sampling from logits: penalties, temperature, top-k, top-p, min-p.

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

pub struct Sampler {
    pub p: SamplingParams,
    state: u64,
}

impl Sampler {
    pub fn new(p: SamplingParams) -> Sampler {
        let seed = if p.seed == 0 {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1)
        } else {
            p.seed
        };
        Sampler { p, state: seed | 1 }
    }

    fn uniform(&mut self) -> f32 {
        // splitmix64
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Picks the next token; `history` is the sequence so far (for the penalties). `banned` tokens are never picked.
    pub fn sample(&mut self, logits: &[f32], history: &[u32], banned: &[u32]) -> u32 {
        let p = &self.p;
        let mut cand: Vec<(u32, f32)> = Vec::with_capacity(256);
        let mut work: Vec<f32>;
        let src: &[f32] = if p.repeat_penalty != 1.0 || p.presence_penalty != 0.0 || p.frequency_penalty != 0.0 || !banned.is_empty() {
            work = logits.to_vec();
            let recent = &history[history.len().saturating_sub(p.repeat_last_n)..];
            let mut counts = std::collections::HashMap::new();
            for &t in recent {
                *counts.entry(t).or_insert(0u32) += 1;
            }
            for (&t, &c) in &counts {
                if let Some(l) = work.get_mut(t as usize) {
                    *l = if *l > 0.0 { *l / p.repeat_penalty } else { *l * p.repeat_penalty };
                    *l -= p.presence_penalty + p.frequency_penalty * c as f32;
                }
            }
            for &b in banned {
                if let Some(l) = work.get_mut(b as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
            &work
        } else {
            logits
        };
        if p.temperature <= 0.0 {
            return src.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
        }
        // top-k (0 = all, capped for speed)
        let k = if p.top_k == 0 { 1000 } else { p.top_k.min(1000) };
        let mut idx: Vec<u32> = (0..src.len() as u32).collect();
        if k < idx.len() {
            idx.select_nth_unstable_by(k - 1, |&a, &b| src[b as usize].total_cmp(&src[a as usize]));
            idx.truncate(k);
        }
        idx.sort_unstable_by(|&a, &b| src[b as usize].total_cmp(&src[a as usize]));
        let max = src[idx[0] as usize];
        let mut sum = 0.0;
        for &i in &idx {
            let w = ((src[i as usize] - max) / p.temperature).exp();
            cand.push((i, w));
            sum += w;
        }
        // min-p, then top-p
        let pmax = cand[0].1 / sum;
        cand.retain(|&(_, w)| w / sum >= p.min_p * pmax);
        let mut acc = 0.0;
        let total: f32 = cand.iter().map(|c| c.1).sum();
        let mut cut = cand.len();
        for (i, c) in cand.iter().enumerate() {
            acc += c.1 / total;
            if acc >= p.top_p {
                cut = i + 1;
                break;
            }
        }
        cand.truncate(cut.max(1));
        let total: f32 = cand.iter().map(|c| c.1).sum();
        let mut r = self.uniform() * total;
        for &(t, w) in &cand {
            if r < w {
                return t;
            }
            r -= w;
        }
        cand.last().map(|c| c.0).unwrap_or(0)
    }
}
