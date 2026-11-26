//! Device-resident weights: everything but the experts and the token embedding. Each is a copy of a checkpoint tensor
//! as the file holds it: the BF16 matrices and vectors, the F32 queries of the residual mixes and the router's F32
//! bias. The kernels read them in these layouts.

use crate::checkpoint::{Checkpoint, Dtype};
use crate::cuda::{DevBuf, Result};
use crate::hp::Hp;
use crate::kernels::P;

/// A BF16 matrix [rows][k], row-major as the checkpoint has it.
pub struct Mat {
    pub w: DevBuf,
    pub rows: usize,
    pub k: usize,
}

impl Mat {
    fn load(ck: &Checkpoint, name: &str, rows: usize, k: usize) -> Result<Mat> {
        Self::first_rows(ck, name, rows, rows, k)
    }

    /// The first `keep` rows of BF16 [rows][k] tensor `name` (the unembedding without its padded rows).
    fn first_rows(ck: &Checkpoint, name: &str, rows: usize, keep: usize, k: usize) -> Result<Mat> {
        let b = ck.get(name, Dtype::BF16, &[rows, k])?;
        Ok(Mat { w: DevBuf::from_slice(&b[..keep * k * 2])?, rows: keep, k })
    }

    pub fn bytes(&self) -> usize {
        self.rows * self.k * 2
    }
}

/// Tensor `name`, which must be of `dtype` and `shape`, copied to the GPU.
fn tensor(ck: &Checkpoint, name: &str, dtype: Dtype, shape: &[usize]) -> Result<DevBuf> {
    DevBuf::from_slice(ck.get(name, dtype, shape)?)
}

pub struct LoopW {
    /// this loop layer's message queries (f32, two each): its q and mlp ones in Weights::msg_queries, its k and v ones
    /// in Weights::msg_kv_queries
    pub q_mlp: P,
    pub k_v: P,
    /// the controllers (message gates) of the q, k and v inputs and of the MoE input: the checkpoint's attn, k, v and
    /// mlp controllers. down [64][2 n_embd]; gain (the message gate) and forget [n_embd][64]
    pub down: Vec<Mat>,
    pub gain: Vec<DevBuf>,
    pub forget: Vec<DevBuf>,
}

impl LoopW {
    /// The message query of controller c (0 q, 1 k, 2 v, 3 the MoE input).
    pub fn query(&self, c: usize, n_embd: usize) -> P {
        match c {
            0 => self.q_mlp,
            1 => self.k_v,
            2 => self.k_v + (n_embd * 4) as u64,
            _ => self.q_mlp + (n_embd * 4) as u64,
        }
    }
}

pub struct Layer {
    pub swa: bool,
    /// SWA: [attn_norm]; full attention: [q, k, v] input norms
    pub in_norms: Vec<DevBuf>,
    pub attn_res_query: Option<DevBuf>, // [nq][n_embd] f32, nq = 1 (SWA) or 3 (q, k, v)
    pub ffn_res_query: DevBuf,          // [n_embd] f32
    pub ffn_norm: DevBuf,
    /// the projections of the q input, in their order in g0: q, the output gate, the relative states, the sinks
    pub g0: Vec<Mat>,
    /// the projections of the k (v) input in their order in gk (gv): k (v), and in full attention the dynamic taps of
    /// its convolution
    pub gk: Vec<Mat>,
    pub gv: Vec<Mat>,
    pub wo: Mat,
    /// the last rows of the output gate (gate_attn), which the CPU team takes in decode: BF16 [rows][n_embd] in host
    /// memory, until the engine gives them to the pool
    pub gate_cpu: Vec<u16>,
    pub q_clip: DevBuf,   // [n_head][head_dim]
    pub rel_clip: DevBuf, // [n_head][n_rel]
    pub rel_proj: DevBuf, // [n_rel][extent]: the bias profiles (extent n_swa or rel_extent)
    /// SWA: the conv1d weights [kv][4]; full attention: the static taps [4][kv] (lag 0 the current token)
    pub k_conv: DevBuf,
    pub v_conv: DevBuf,
    pub post_conv: DevBuf,     // [n_embd][4]
    pub ffn_post_conv: DevBuf, // [n_embd][4]
    pub router: Mat,           // [n_expert][n_embd]
    pub router_bias: DevBuf,   // [n_expert] f32
    pub lp: Option<LoopW>,
}

impl Layer {
    /// Rows of g0: q, gate, relative states and sinks.
    pub fn g0_rows(&self) -> usize {
        self.g0.iter().map(|m| m.rows).sum()
    }
}

pub struct Weights {
    pub layers: Vec<Layer>,
    pub tok_norm: DevBuf,
    pub output_norm: DevBuf,
    pub output_res_query: DevBuf,
    /// the unembedding's first n_vocab rows (the rest pad the vocabulary), less the last ones, which the CPU takes:
    /// those in host memory, BF16 [rows][n_embd]
    pub output: Mat,
    pub output_cpu: Vec<u16>,
    /// [2 loop layers][n_embd] f32 each: middle_cycle_cross_pass_queries (q, mlp) and its kv_queries (k, v)
    pub msg_queries: DevBuf,
    pub msg_kv_queries: DevBuf,
    pub bytes: usize,
}

impl Weights {
    /// The weights, with the last `cpu_rows` rows of the unembedding and the last `gate_rows` of each output gate in
    /// host memory.
    pub fn load(ck: &Checkpoint, hp: &Hp, cpu_rows: usize, gate_rows: usize) -> Result<Weights> {
        let (n, nh, kv) = (hp.n_embd, hp.n_head * hp.head_dim, hp.kv_dim());
        if gate_rows >= nh {
            return Err(format!("{gate_rows} rows of the output gate for the CPU, of {nh}"));
        }
        let (bf, f32_) = (Dtype::BF16, Dtype::F32);
        let core = hp.loop_end - hp.loop_start;
        let msg_queries = tensor(ck, "model.llm.middle_cycle_cross_pass_queries", f32_, &[2 * core, n])?;
        let msg_kv_queries = tensor(ck, "model.llm.middle_cycle_cross_pass_kv_queries", f32_, &[2 * core, n])?;
        let mut layers = Vec::with_capacity(hp.n_phys);
        let mut bytes = 0;
        for i in 0..hp.n_phys {
            let p = |s: &str| format!("model.llm.layers.{i}.{s}");
            let vec_bf = |s: &str| tensor(ck, &p(s), bf, &[n]);
            let swa = hp.is_swa[i];
            let in_norms = if swa {
                vec![vec_bf("attn_norm.weight")?]
            } else {
                vec![vec_bf("q_input_norm.weight")?, vec_bf("k_input_norm.weight")?, vec_bf("v_input_norm.weight")?]
            };
            // the query of layer i's attention inputs is attn_queries.{i - 1} (layer 0 has none)
            let attn_res_query = if i == 0 {
                None
            } else {
                let name = format!("model.llm.attn_res.attn_queries.{}", i - 1);
                Some(if swa { tensor(ck, &name, f32_, &[n])? } else { tensor(ck, &name, f32_, &[3, n])? })
            };
            let g0 = vec![
                Mat::load(ck, &p("attn.wq_du.weight"), nh, n)?,
                Mat::load(ck, &p("attn.gate_attn.weight"), nh, n)?,
                Mat::load(ck, &p("attn.wr_du.weight"), hp.n_head * hp.n_rel, n)?,
                Mat::load(ck, &p("attn.attn_sink_proj.weight"), hp.n_head, n)?,
            ];
            let mut gk = vec![Mat::load(ck, &p("attn.wk_dv.weight"), kv, n)?];
            let mut gv = vec![Mat::load(ck, &p("attn.wv_dv.weight"), kv, n)?];
            let taps = hp.n_head_kv * hp.n_conv;
            let (k_conv, v_conv) = if swa {
                (tensor(ck, &p("attn.k_sconv.weight"), bf, &[kv, 1, hp.n_conv])?, tensor(ck, &p("attn.v_sconv.weight"), bf, &[kv, 1, hp.n_conv])?)
            } else {
                gk.push(Mat::load(ck, &p("attn.k_sconv.weight_proj"), taps, n)?);
                gv.push(Mat::load(ck, &p("attn.v_sconv.weight_proj"), taps, n)?);
                (tensor(ck, &p("attn.k_sconv.static_w"), bf, &[hp.n_conv, kv])?, tensor(ck, &p("attn.v_sconv.static_w"), bf, &[hp.n_conv, kv])?)
            };
            let wo = Mat::load(ck, &p("attn.wo_ud.weight"), n, nh)?;
            let gb = ck.get(&p("attn.gate_attn.weight"), bf, &[nh, n])?;
            let gate_cpu = gb[(nh - gate_rows) * n * 2..nh * n * 2].as_chunks::<2>().0.iter().map(|&c| u16::from_le_bytes(c)).collect();
            let lp = if i >= hp.loop_start && i < hp.loop_end {
                let j = i - hp.loop_start;
                let (mut down, mut gain, mut forget) = (vec![], vec![], vec![]);
                for c in ["attn", "k", "v", "mlp"] {
                    let q = |s: &str| p(&format!("controllers.{c}.{s}"));
                    down.push(Mat::load(ck, &q("down_weight"), hp.gate_rank, 2 * n)?);
                    gain.push(tensor(ck, &q("gain_weight"), bf, &[n, hp.gate_rank])?);
                    forget.push(tensor(ck, &q("forget_weight"), bf, &[n, hp.gate_rank])?);
                }
                Some(LoopW { q_mlp: msg_queries.at(2 * j * n * 4), k_v: msg_kv_queries.at(2 * j * n * 4), down, gain, forget })
            } else {
                None
            };
            let router = Mat::load(ck, &p("mlp.gate.weight"), hp.n_expert, n)?;
            bytes += g0.iter().chain(&gk).chain(&gv).chain([&wo, &router]).map(Mat::bytes).sum::<usize>();
            bytes += lp.as_ref().map_or(0, |l| l.down.iter().map(Mat::bytes).sum::<usize>() + 2 * 4 * n * hp.gate_rank * 2);
            layers.push(Layer {
                swa,
                in_norms,
                attn_res_query,
                ffn_res_query: tensor(ck, &format!("model.llm.attn_res.mlp_queries.{i}"), f32_, &[n])?,
                ffn_norm: vec_bf("mlp_norm.weight")?,
                g0,
                gk,
                gv,
                wo,
                gate_cpu,
                q_clip: tensor(ck, &p("attn.q_l2_clip.scale"), bf, &[hp.n_head, hp.head_dim])?,
                rel_clip: tensor(ck, &p("attn.rel_l2_clip.scale"), bf, &[hp.n_head, hp.n_rel])?,
                rel_proj: tensor(ck, &p("attn.rel_logits_proj.proj"), bf, &[hp.n_rel, if swa { hp.n_swa } else { hp.rel_extent }])?,
                k_conv,
                v_conv,
                post_conv: tensor(ck, &p("attn_sconv.weight"), bf, &[n, 1, hp.n_conv])?,
                ffn_post_conv: tensor(ck, &p("mlp_sconv.weight"), bf, &[n, 1, hp.n_conv])?,
                router,
                router_bias: tensor(ck, &p("mlp.gate.bias"), f32_, &[hp.n_expert])?,
                lp,
            });
        }
        let gpu_rows = hp.n_vocab - cpu_rows.min(hp.n_vocab - 1);
        let output = Mat::first_rows(ck, "model.llm.unembed.weight", hp.n_vocab_padded, gpu_rows, n)?;
        let b = ck.get("model.llm.unembed.weight", bf, &[hp.n_vocab_padded, n])?;
        let output_cpu = b[gpu_rows * n * 2..hp.n_vocab * n * 2].as_chunks::<2>().0.iter().map(|&c| u16::from_le_bytes(c)).collect();
        bytes += output.bytes();
        Ok(Weights {
            layers,
            tok_norm: tensor(ck, "model.llm.embed_norm.weight", bf, &[n])?,
            output_norm: tensor(ck, "model.llm.norm.weight", bf, &[n])?,
            output_res_query: tensor(ck, "model.llm.attn_res.output_query", f32_, &[n])?,
            output,
            output_cpu,
            msg_queries,
            msg_kv_queries,
            bytes,
        })
    }
}
