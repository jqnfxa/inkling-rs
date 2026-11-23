//! GGUF reader: metadata and tensor views over a memory-mapped file.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => self.as_u64().map(|v| v as f32),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

/// ggml tensor types that this model uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Q5_0,
    Q8_0,
    Q4K,
    Other(u32),
}

impl GgmlType {
    fn from_u32(t: u32) -> GgmlType {
        match t {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            6 => GgmlType::Q5_0,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            o => GgmlType::Other(o),
        }
    }
    /// (elements per block, bytes per block)
    pub fn block(self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 => (1, 2),
            GgmlType::Q5_0 => (32, 22),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q4K => (256, 144),
            GgmlType::Other(t) => panic!("unsupported ggml type {t}"),
        }
    }
    pub fn row_bytes(self, n: usize) -> usize {
        let (be, bb) = self.block();
        assert!(n % be == 0, "row of {n} elements is not a whole number of {self:?} blocks");
        n / be * bb
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// dims, innermost first (ggml order)
    pub ne: Vec<usize>,
    pub ty: GgmlType,
    /// absolute byte offset in the file
    pub offset: usize,
    pub nbytes: usize,
}

pub struct Gguf {
    pub mmap: Mmap,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub order: Vec<String>,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.p..self.p + n).ok_or("GGUF truncated")?;
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value, String> {
        Ok(match ty {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n);
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            t => return Err(format!("unknown GGUF value type {t}")),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: the model file is not modified while it is mapped.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;
        let mut r = Reader { b: &mmap, p: 0 };
        if r.take(4)? != b"GGUF" {
            return Err("not a GGUF file".into());
        }
        let version = r.u32()?;
        if version < 2 {
            return Err(format!("GGUF version {version} is too old"));
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let k = r.string()?;
            let ty = r.u32()?;
            let v = r.value(ty)?;
            meta.insert(k, v);
        }
        let mut infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            let mut ne = Vec::with_capacity(nd);
            for _ in 0..nd {
                ne.push(r.u64()? as usize);
            }
            let ty = GgmlType::from_u32(r.u32()?);
            let off = r.u64()? as usize;
            infos.push((name, ne, ty, off));
        }
        let align = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
        let data = r.p.div_ceil(align) * align;
        let mut tensors = HashMap::new();
        let mut order = Vec::new();
        for (name, ne, ty, off) in infos {
            let nbytes = match ty {
                GgmlType::Other(_) => 0,
                _ => ty.row_bytes(ne[0]) * ne[1..].iter().product::<usize>(),
            };
            let offset = data + off;
            if offset + nbytes > mmap.len() {
                return Err(format!("tensor {name} is past the end of the file"));
            }
            order.push(name.clone());
            tensors.insert(name.clone(), TensorInfo { name, ne, ty, offset, nbytes });
        }
        Ok(Gguf { mmap, meta, tensors, order })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors.get(name).ok_or_else(|| format!("missing tensor {name}"))
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.nbytes]
    }

    pub fn f32s(&self, name: &str) -> Result<Vec<f32>, String> {
        let t = self.tensor(name)?;
        if t.ty != GgmlType::F32 {
            return Err(format!("{name} is {:?}, not F32", t.ty));
        }
        Ok(self.bytes(t).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    pub fn u64(&self, key: &str) -> Result<u64, String> {
        self.meta.get(key).and_then(Value::as_u64).ok_or_else(|| format!("missing or bad key {key}"))
    }
    pub fn f32(&self, key: &str) -> Result<f32, String> {
        self.meta.get(key).and_then(Value::as_f32).ok_or_else(|| format!("missing or bad key {key}"))
    }
    pub fn str(&self, key: &str) -> Result<&str, String> {
        self.meta.get(key).and_then(Value::as_str).ok_or_else(|| format!("missing or bad key {key}"))
    }
    pub fn arr(&self, key: &str) -> Result<&[Value], String> {
        self.meta.get(key).and_then(Value::as_arr).ok_or_else(|| format!("missing or bad key {key}"))
    }
}
