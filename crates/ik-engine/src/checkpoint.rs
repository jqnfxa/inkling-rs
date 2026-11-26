//! The checkpoint of Inkling-2-Mini as it is published (thinkingmachines/Inkling-2-Mini on Hugging Face): the
//! safetensors files of the weights, with config.json, tokenizer.json and chat_template.jinja beside them.
//!
//! The engine uses the weights as the files hold them, in their own formats: BF16 and F32 tensors, and the routed
//! experts in ModelOpt NVFP4 (layers 0-19) and MXFP8 (20-23). Nothing is converted at load; the tensors are views of
//! the mapped files.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use serde_json::Value;

use crate::tokenizer::Tokenizer;

/// Element types of the checkpoint's tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    BF16,
    F16,
    F32,
    /// float8 e4m3fn: the MXFP8 experts' values, the NVFP4 experts' block scales
    F8E4M3,
    /// bytes: the NVFP4 experts' E2M1 pairs, the MXFP8 experts' UE8M0 block scales
    U8,
    /// a type the engine does not read (of tensors it does not use)
    Other,
}

impl Dtype {
    fn parse(s: &str) -> Dtype {
        match s {
            "BF16" => Dtype::BF16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "F8_E4M3" => Dtype::F8E4M3,
            "U8" => Dtype::U8,
            _ => Dtype::Other,
        }
    }

    /// Bytes of an element (None for Other).
    pub fn size(self) -> Option<usize> {
        match self {
            Dtype::BF16 | Dtype::F16 => Some(2),
            Dtype::F32 => Some(4),
            Dtype::F8E4M3 | Dtype::U8 => Some(1),
            Dtype::Other => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub dtype: Dtype,
    /// dims, outermost first (as PyTorch has them)
    pub shape: Vec<usize>,
    file: usize,
    /// byte range in the file
    start: usize,
    end: usize,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    pub fn nbytes(&self) -> usize {
        self.end - self.start
    }
    /// The checkpoint file that holds the tensor (see Checkpoint::file), and its byte range in it.
    pub fn place(&self) -> (usize, std::ops::Range<usize>) {
        (self.file, self.start..self.end)
    }
}

pub struct Checkpoint {
    pub dir: PathBuf,
    files: Vec<Mmap>,
    pub tensors: HashMap<String, Tensor>,
    /// config.json
    pub config: Value,
}

impl Checkpoint {
    /// Opens the checkpoint in directory `path`, or in the directory of the .safetensors file `path`.
    pub fn open(path: &Path) -> Result<Checkpoint, String> {
        let dir = if path.is_dir() { path.to_path_buf() } else { path.parent().map(Path::to_path_buf).unwrap_or_default() };
        let config = read_json(&dir.join("config.json"))?;
        // the weight files: those the index names (one or more shards), else model.safetensors
        let index = dir.join("model.safetensors.index.json");
        let mut names: Vec<String> = if index.is_file() {
            let ix = read_json(&index)?;
            let map = ix["weight_map"].as_object().ok_or_else(|| format!("{}: no weight_map", index.display()))?;
            map.values().filter_map(Value::as_str).map(str::to_string).collect()
        } else {
            vec!["model.safetensors".to_string()]
        };
        names.sort();
        names.dedup();
        let mut files = vec![];
        let mut tensors = HashMap::new();
        for name in &names {
            if name.contains('/') || name.contains("..") {
                return Err(format!("{}: weight file {name} is not in the checkpoint's directory", index.display()));
            }
            let p = dir.join(name);
            let f = File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            // SAFETY: the checkpoint is not modified while it is mapped
            let m = unsafe { Mmap::map(&f) }.map_err(|e| format!("mmap {}: {e}", p.display()))?;
            parse_header(&m, files.len(), &p, &mut tensors)?;
            files.push(m);
        }
        Ok(Checkpoint { dir, files, tensors, config })
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors.get(name).ok_or_else(|| format!("the checkpoint has no tensor {name}"))
    }

    pub fn bytes(&self, t: &Tensor) -> &[u8] {
        &self.files[t.file][t.start..t.end]
    }

    /// Weight file i as mapped (the files in the order of their names).
    pub fn file(&self, i: usize) -> &[u8] {
        &self.files[i]
    }

    /// The bytes of tensor `name`, which must be of `dtype` and `shape`.
    pub fn get(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&[u8], String> {
        let t = self.tensor(name)?;
        if t.dtype != dtype || t.shape != shape {
            return Err(format!("{name} is {:?} {:?}, not {dtype:?} {shape:?}", t.dtype, t.shape));
        }
        Ok(self.bytes(t))
    }

    /// A float tensor's values as f32 (for the host side: tools and checks; the GPU reads the tensors as they are).
    pub fn floats(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?;
        let b = self.bytes(t);
        Ok(match t.dtype {
            Dtype::F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Dtype::BF16 => b.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
            Dtype::F16 => b.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            d => return Err(format!("{name} is {d:?}, not a float tensor")),
        })
    }

    /// The text of a file of the checkpoint, such as chat_template.jinja.
    pub fn text(&self, file: &str) -> Result<String, String> {
        let p = self.dir.join(file);
        std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))
    }

    /// The model's name: the checkpoint directory's.
    pub fn name(&self) -> String {
        self.dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "inkling".into())
    }

    /// The ids that end a reply: config.json's eos_token_id (a number or a list).
    pub fn eos(&self) -> Vec<u32> {
        match &self.config["eos_token_id"] {
            Value::Array(a) => a.iter().filter_map(Value::as_u64).map(|v| v as u32).collect(),
            v => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
        }
    }

    /// The tokenizer of tokenizer.json.
    pub fn tokenizer(&self) -> Result<Tokenizer, String> {
        Tokenizer::from_hf(&self.text("tokenizer.json")?, &self.eos())
    }
}

fn read_json(p: &Path) -> Result<Value, String> {
    let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", p.display()))
}

/// Reads the header of safetensors file `m` (number `file`): a u64 of its length, then the JSON of the tensors, each
/// with its dtype, shape and byte range after the header.
fn parse_header(m: &[u8], file: usize, p: &Path, tensors: &mut HashMap<String, Tensor>) -> Result<(), String> {
    let bad = |what: String| format!("{}: {what}", p.display());
    let n = u64::from_le_bytes(m.get(..8).ok_or_else(|| bad("too short for a safetensors file".into()))?.try_into().unwrap());
    let base = usize::try_from(n).ok().and_then(|n| n.checked_add(8)).filter(|&b| b <= m.len()).ok_or_else(|| bad(format!("header of {n} bytes past the end")))?;
    let h: serde_json::Map<String, Value> = serde_json::from_slice(&m[8..base]).map_err(|e| bad(format!("header: {e}")))?;
    for (name, v) in h {
        if name == "__metadata__" {
            continue;
        }
        let dtype = Dtype::parse(v["dtype"].as_str().ok_or_else(|| bad(format!("{name}: no dtype")))?);
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .ok_or_else(|| bad(format!("{name}: no shape")))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| bad(format!("{name}: bad shape")))?;
        let off: Vec<usize> = v["data_offsets"].as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|o| o as usize).collect()).unwrap_or_default();
        let [a, b] = off[..] else { return Err(bad(format!("{name}: bad data_offsets"))) };
        let (start, end) = (base.checked_add(a), base.checked_add(b));
        let (Some(start), Some(end)) = (start, end) else { return Err(bad(format!("{name}: bad data_offsets"))) };
        if b < a || end > m.len() {
            return Err(bad(format!("{name}: bytes {a}..{b} are not in the file")));
        }
        let numel = shape.iter().try_fold(1usize, |p, &d| p.checked_mul(d));
        if let (Some(sz), Some(numel)) = (dtype.size(), numel)
            && numel.checked_mul(sz) != Some(b - a)
        {
            return Err(bad(format!("{name}: {} bytes for {dtype:?} {shape:?}", b - a)));
        }
        tensors.insert(name, Tensor { dtype, shape, file, start, end });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header.as_bytes());
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn reads_a_header() {
        let h = r#"{"__metadata__":{"format":"pt"},"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]},"b":{"dtype":"F32","shape":[1],"data_offsets":[12,16]},"c":{"dtype":"I64","shape":[1],"data_offsets":[16,24]}}"#;
        let data: Vec<u8> = (0..24).collect();
        let f = file(h, &data);
        let mut t = HashMap::new();
        parse_header(&f, 0, Path::new("x"), &mut t).unwrap();
        let a = &t["a"];
        assert_eq!((a.dtype, a.shape.clone(), a.nbytes()), (Dtype::BF16, vec![2, 3], 12));
        assert_eq!(&f[a.start..a.end], &data[..12]);
        assert_eq!(&f[t["b"].start..t["b"].end], &data[12..16]);
        assert_eq!(t["c"].dtype, Dtype::Other);
    }

    #[test]
    fn rejects_bad_ranges() {
        for h in [
            r#"{"a":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,10]}}"#,
            r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#,
        ] {
            let f = file(h, &[0u8; 12]);
            assert!(parse_header(&f, 0, Path::new("x"), &mut HashMap::new()).is_err(), "{h}");
        }
        let mut f = file(r#"{}"#, &[]);
        f[0] = 200;
        assert!(parse_header(&f, 0, Path::new("x"), &mut HashMap::new()).is_err());
    }
}
