//! Hyperparameters of Inkling-2-Mini (model_type inkling2_loop), read from the checkpoint's config.json.

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
    /// The hyperparameters of the checkpoint's config.json (its text_config).
    pub fn from_config(c: &serde_json::Value) -> Result<Hp, String> {
        if c["model_type"] != "inkling2_loop" {
            return Err(format!("model_type is {}, not inkling2_loop", c["model_type"]));
        }
        let t = &c["text_config"];
        let get = |k: &str| -> Result<&serde_json::Value, String> {
            k.split('.').try_fold(t, |v, p| v.get(p)).ok_or_else(|| format!("config.json: text_config has no {k}"))
        };
        let u = |k: &str| get(k)?.as_u64().map(|v| v as usize).ok_or_else(|| format!("config.json: {k} is not a count"));
        let f = |k: &str| get(k)?.as_f64().map(|v| v as f32).ok_or_else(|| format!("config.json: {k} is not a number"));
        let flag = |k: &str| get(k).map(|v| *v == true);
        let n_phys = u("num_hidden_layers")?;
        let local: Vec<usize> = get("local_layer_ids")?.as_array().ok_or("config.json: bad local_layer_ids")?.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
        let hp = Hp {
            n_embd: u("hidden_size")?,
            n_head: u("num_attention_heads")?,
            n_head_kv: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            n_ff_exp: u("intermediate_size")?,
            n_expert: u("n_routed_experts")?,
            n_used: u("num_experts_per_tok")?,
            n_phys,
            loop_start: u("loop_config.core_start")?,
            loop_end: u("loop_config.core_end")?,
            n_passes: u("loop_config.num_passes")?,
            gate_rank: u("loop_config.gate_rank")?,
            block_size: u("n_layers_per_attn_res_block")?,
            n_rel: u("d_rel")?,
            rel_extent: u("rel_extent")?,
            n_swa: u("sliding_window_size")?,
            n_conv: u("sconv_kernel_size")?,
            is_swa: (0..n_phys).map(|i| local.contains(&i)).collect(),
            eps: f("rms_norm_eps")?,
            qk_clip: f("qk_l2_clip_threshold")?,
            rel_clip: f("rel_l2_clip_threshold")?,
            log_floor: f("log_scaling_n_floor")?,
            log_alpha: f("log_scaling_alpha")?,
            situ_beta: f("mlp_gate_softcap")?,
            situ_lb: f("mlp_up_softcap")?,
            w_scale: f("route_scale")?,
            softcap: f("final_logit_softcapping")?,
            n_vocab_padded: u("vocab_size")?,
            n_vocab: u("unpadded_vocab_size")?,
        };
        // the attention kernels read the bias profiles in rows of 1024
        if hp.n_conv != 4 || hp.head_dim != 128 || hp.n_used != 8 || hp.n_rel != 16 || hp.rel_extent != 1024 || hp.n_swa != 1024 || local.iter().any(|&i| i >= n_phys) {
            return Err(format!("unexpected shape: {hp:?}"));
        }
        // the variant the engine implements: no shared experts, values RMS-normed without a gain, the router's bias,
        // the residual mixes over normed values, the same attention shape in the sliding-window layers
        let same = u("swa_head_dim")? == hp.head_dim && u("swa_num_attention_heads")? == hp.n_head && u("swa_num_key_value_heads")? == hp.n_head_kv;
        if flag("loop_config.is_varloop")? || u("n_shared_experts")? != 0 || !flag("use_value_norm_no_gamma")? || !flag("use_gate_bias")?
            || !flag("attn_res_normalize_values")? || flag("q_bias")? || flag("o_bias")? || !same
        {
            return Err("config.json: a variant of inkling2_loop the engine does not implement".into());
        }
        Ok(hp)
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
