//! Device-resident weights: everything but the experts and the token embedding.

use crate::cuda::{DevBuf, Result};
use crate::gguf::{Gguf, GgmlType};
use crate::hp::Hp;
use crate::quant::repack_q8;

/// A repacked Q8_0 matrix: `qs` [rows][k] int8, `d` [rows][k/32] f16. Several GGUF matrices of the same k may be
/// stacked by rows, so that one GEMV computes them all.
pub struct Q8Mat {
    pub qs: DevBuf,
    pub d: DevBuf,
    pub rows: usize,
    pub k: usize,
}

impl Q8Mat {
    pub fn load(g: &Gguf, names: &[&str]) -> Result<Q8Mat> {
        let (mut qs, mut d) = (Vec::new(), Vec::new());
        let mut rows = 0;
        let mut k = 0;
        for name in names {
            let t = g.tensor(name)?;
            if t.ty != GgmlType::Q8_0 || t.ne.len() != 2 {
                return Err(format!("{name} is not a Q8_0 matrix"));
            }
            if k != 0 && k != t.ne[0] {
                return Err(format!("{name}: k {} differs from {k}", t.ne[0]));
            }
            k = t.ne[0];
            rows += t.ne[1];
            repack_q8(g.bytes(t), t.ne[1], k, &mut qs, &mut d);
        }
        Ok(Q8Mat { qs: DevBuf::from_slice(&qs)?, d: DevBuf::from_slice(&d)?, rows, k })
    }

    pub fn bytes(&self) -> usize {
        self.qs.len + self.d.len
    }
}

pub struct LoopW {
    pub msg_query: DevBuf, // [4][n_embd]
    pub down: Vec<Q8Mat>,   // [4] of [64][2 n_embd]
    pub gain: Vec<Q8Mat>,   // [4] of [n_embd][64]
    pub forget: Vec<Q8Mat>,
}

pub struct Layer {
    pub swa: bool,
    /// SWA: [attn_norm]; full attention: [q, k, v] input norms
    pub in_norms: Vec<DevBuf>,
    pub attn_res_query: Option<DevBuf>, // [nq][n_embd], nq = 1 (SWA) or 3
    pub ffn_res_query: DevBuf,
    pub ffn_norm: DevBuf,
    pub g0: Q8Mat, // q | gate | rel | sink
    pub gk: Q8Mat, // k | k_conv_proj
    pub gv: Q8Mat, // v | v_conv_proj
    pub wo: Q8Mat,
    pub q_clip: DevBuf,
    pub rel_clip: DevBuf,
    pub rel_bias: DevBuf,
    pub k_conv: DevBuf,
    pub v_conv: DevBuf,
    pub post_conv: DevBuf,
    pub ffn_post_conv: DevBuf,
    pub router: DevBuf, // [n_expert][n_embd]
    pub probs_b: DevBuf,
    pub lp: Option<LoopW>,
}

pub struct Weights {
    pub layers: Vec<Layer>,
    pub tok_norm: DevBuf,
    pub output_norm: DevBuf,
    pub output_res_query: DevBuf,
    pub output: Q8Mat,
    pub bytes: usize,
}

fn f32buf(g: &Gguf, name: &str) -> Result<DevBuf> {
    DevBuf::from_slice(&g.f32s(name)?)
}

impl Weights {
    pub fn load(g: &Gguf, hp: &Hp) -> Result<Weights> {
        let mut layers = Vec::with_capacity(hp.n_phys);
        let mut bytes = 0;
        for i in 0..hp.n_phys {
            let n = |s: &str| format!("blk.{i}.{s}");
            let swa = hp.is_swa[i];
            let in_norms = if swa {
                vec![f32buf(g, &n("attn_norm.weight"))?]
            } else {
                vec![f32buf(g, &n("attn_q_inp_norm.weight"))?, f32buf(g, &n("attn_k_inp_norm.weight"))?, f32buf(g, &n("attn_v_inp_norm.weight"))?]
            };
            let attn_res_query = if i > 0 { Some(f32buf(g, &n("attn_res_query.weight"))?) } else { None };
            let g0 = Q8Mat::load(g, &[&n("attn_q.weight"), &n("attn_gate.weight"), &n("attn_rel.weight"), &n("attn_sink_proj.weight")])?;
            let (gk, gv) = if swa {
                (Q8Mat::load(g, &[&n("attn_k.weight")])?, Q8Mat::load(g, &[&n("attn_v.weight")])?)
            } else {
                (
                    Q8Mat::load(g, &[&n("attn_k.weight"), &n("attn_k_conv_proj.weight")])?,
                    Q8Mat::load(g, &[&n("attn_v.weight"), &n("attn_v_conv_proj.weight")])?,
                )
            };
            let wo = Q8Mat::load(g, &[&n("attn_output.weight")])?;
            let lp = if i >= hp.loop_start && i < hp.loop_end {
                let ctl = ["q", "k", "v", "ffn"];
                let mut down = vec![];
                let mut gain = vec![];
                let mut forget = vec![];
                for c in ctl {
                    down.push(Q8Mat::load(g, &[&n(&format!("loop_{c}_ctrl_down.weight"))])?);
                    gain.push(Q8Mat::load(g, &[&n(&format!("loop_{c}_ctrl_gain.weight"))])?);
                    forget.push(Q8Mat::load(g, &[&n(&format!("loop_{c}_ctrl_forget.weight"))])?);
                }
                Some(LoopW { msg_query: f32buf(g, &n("loop_msg_query.weight"))?, down, gain, forget })
            } else {
                None
            };
            bytes += g0.bytes() + gk.bytes() + gv.bytes() + wo.bytes();
            layers.push(Layer {
                swa,
                in_norms,
                attn_res_query,
                ffn_res_query: f32buf(g, &n("ffn_res_query.weight"))?,
                ffn_norm: f32buf(g, &n("ffn_norm.weight"))?,
                g0,
                gk,
                gv,
                wo,
                q_clip: f32buf(g, &n("attn_q_clip.weight"))?,
                rel_clip: f32buf(g, &n("attn_rel_clip.weight"))?,
                rel_bias: f32buf(g, &n("attn_rel_bias.weight"))?,
                k_conv: f32buf(g, &n("attn_k_conv.weight"))?,
                v_conv: f32buf(g, &n("attn_v_conv.weight"))?,
                post_conv: f32buf(g, &n("attn_post_conv.weight"))?,
                ffn_post_conv: f32buf(g, &n("ffn_post_conv.weight"))?,
                router: f32buf(g, &n("ffn_gate_inp.weight"))?,
                probs_b: f32buf(g, &n("exp_probs_b.bias"))?,
                lp,
            });
        }
        let output = Q8Mat::load(g, &["output.weight"])?;
        bytes += output.bytes();
        Ok(Weights {
            layers,
            tok_norm: f32buf(g, "token_embd_norm.weight")?,
            output_norm: f32buf(g, "output_norm.weight")?,
            output_res_query: f32buf(g, "output_res_query.weight")?,
            output,
            bytes,
        })
    }
}
