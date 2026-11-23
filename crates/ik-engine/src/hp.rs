//! Hyperparameters of Inkling-2-Mini (arch inkling2_loop), read from the GGUF.

use crate::gguf::{Gguf, Value};

#[derive(Debug, Clone)]
pub struct Hp {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff_exp: usize,
    pub n_expert: usize,
    pub n_used: usize,
    pub n_phys: usize,
    pub loop_start: usize,
    pub loop_end: usize,
    pub n_passes: usize,
    pub gate_rank: usize,
    pub block_size: usize,
    pub n_rel: usize,
    pub rel_extent: usize,
    pub n_swa: usize,
    pub n_conv: usize,
    pub is_swa: Vec<bool>,
    pub eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: f32,
    pub log_alpha: f32,
    pub situ_beta: f32,
    pub situ_lb: f32,
    pub w_scale: f32,
    pub softcap: f32,
    pub n_vocab_padded: usize,
    pub n_vocab: usize,
}

/// One evaluation of a physical layer: cache slot `il` (own KV and conv state), physical layer `ip`, loop pass.
#[derive(Debug, Clone, Copy)]
pub struct Step {
    pub il: usize,
    pub ip: usize,
    pub pass: usize,
}

impl Hp {
    pub fn from_gguf(g: &Gguf) -> Result<Hp, String> {
        let a = "inkling2_loop";
        if g.str("general.architecture")? != a {
            return Err(format!("architecture is {}, not {a}", g.str("general.architecture")?));
        }
        let k = |s: &str| format!("{a}.{s}");
        let u = |s: &str| g.u64(&k(s)).map(|v| v as usize);
        let f = |s: &str| g.f32(&k(s));
        let first = |s: &str| -> Result<usize, String> {
            match g.meta.get(&k(s)) {
                Some(Value::Arr(v)) => v.first().and_then(Value::as_u64).map(|x| x as usize).ok_or(format!("bad {s}")),
                Some(v) => v.as_u64().map(|x| x as usize).ok_or(format!("bad {s}")),
                None => Err(format!("missing {s}")),
            }
        };
        let is_swa: Vec<bool> = g
            .arr(&k("attention.sliding_window_pattern"))?
            .iter()
            .map(|v| matches!(v, Value::Bool(true)))
            .collect();
        let hp = Hp {
            n_embd: u("embedding_length")?,
            n_head: first("attention.head_count")?,
            n_head_kv: first("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            n_expert: u("expert_count")?,
            n_used: u("expert_used_count")?,
            n_phys: u("block_count")?,
            loop_start: u("loop.start")?,
            loop_end: u("loop.end")?,
            n_passes: u("loop.count")?,
            gate_rank: u("loop.gate_rank")?,
            block_size: u("attn_res.block_size")?,
            n_rel: u("attention.relative_length")?,
            rel_extent: u("attention.relative_extent")?,
            n_swa: u("attention.sliding_window")?,
            n_conv: u("shortconv.l_cache")?,
            is_swa,
            eps: f("attention.layer_norm_rms_epsilon")?,
            qk_clip: f("attention.qk_clip")?,
            rel_clip: f("attention.relative_clip")?,
            log_floor: f("attention.log_scale_floor")?,
            log_alpha: f("attention.log_scale_alpha")?,
            situ_beta: f("activation.situ_beta")?,
            situ_lb: f("activation.situ_linear_beta")?,
            w_scale: f("expert_weights_scale")?,
            softcap: f("final_logit_softcapping")?,
            n_vocab_padded: u("vocab_size")?,
            n_vocab: u("vocab_size_unpadded")?,
        };
        if hp.n_conv != 4 || hp.head_dim != 128 || hp.n_used != 8 || hp.n_rel != 16 || hp.is_swa.len() != hp.n_phys {
            return Err(format!("unexpected shape: {hp:?}"));
        }
        if hp.varying(g) {
            return Err("varying loops are not supported".into());
        }
        Ok(hp)
    }

    fn varying(&self, g: &Gguf) -> bool {
        matches!(g.meta.get("inkling2_loop.loop.varying"), Some(Value::Bool(true)))
    }

    /// The layer evaluations of one token, in order: the prelude, the passes of the loop core, the coda.
    pub fn schedule(&self) -> Vec<Step> {
        let core = self.loop_end - self.loop_start;
        let mut v = vec![];
        for ip in 0..self.loop_start {
            v.push(Step { il: ip, ip, pass: 0 });
        }
        for pass in 0..self.n_passes {
            for ip in self.loop_start..self.loop_end {
                let il = if pass == 0 { ip } else { self.n_phys + (pass - 1) * core + (ip - self.loop_start) };
                v.push(Step { il, ip, pass });
            }
        }
        for ip in self.loop_end..self.n_phys {
            v.push(Step { il: ip, ip, pass: 0 });
        }
        v
    }

    pub fn n_slots(&self) -> usize {
        self.n_phys + (self.n_passes - 1) * (self.loop_end - self.loop_start)
    }

    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// log-scaled position factor of the full-attention layers
    pub fn tau(&self, pos: usize) -> f32 {
        let r = ((pos as f32 + 1.0) / self.log_floor).max(1.0);
        1.0 + self.log_alpha * r.ln()
    }
}
