//! Byte-level BPE tokenizer of Inkling-2 (pre-tokenizer "inkling2": the GPT-4o split, merges ignored for whole-word tokens, no BOS).

use std::collections::HashMap;

use fancy_regex::Regex;

use crate::gguf::{Gguf, Value};

const SPLIT: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const TYPE_CONTROL: i32 = 3;
const TYPE_USER_DEFINED: i32 = 4;

pub struct Tokenizer {
    pieces: Vec<String>,
    types: Vec<i32>,
    vocab: HashMap<String, u32>,
    merges: HashMap<(u32, u32), (u32, u32)>, // (left, right) -> (rank, merged)
    split: Regex,
    byte_char: [char; 256],
    char_byte: HashMap<char, u8>,
    /// special tokens, longest first: (text, id, is_control)
    specials: Vec<(String, u32, bool)>,
    pub eos: Vec<u32>,
}

/// GPT-2 bytes_to_unicode
fn byte_chars() -> [char; 256] {
    let mut res = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        res[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    res
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer, String> {
        if g.str("tokenizer.ggml.model")? != "gpt2" || g.str("tokenizer.ggml.pre")? != "inkling2" {
            return Err("the tokenizer is not the inkling2 BPE".into());
        }
        let pieces: Vec<String> = g.arr("tokenizer.ggml.tokens")?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let types: Vec<i32> = match g.meta.get("tokenizer.ggml.token_type") {
            Some(Value::Arr(a)) => a.iter().map(|v| v.as_u64().map_or(1, |x| x as i32)).collect(),
            _ => vec![1; pieces.len()],
        };
        let vocab: HashMap<String, u32> = pieces.iter().enumerate().map(|(i, p)| (p.clone(), i as u32)).collect();
        let mut merges = HashMap::new();
        for (rank, m) in g.arr("tokenizer.ggml.merges")?.iter().enumerate() {
            let m = m.as_str().unwrap_or("");
            let Some((a, b)) = m.split_once(' ') else { continue };
            let (Some(&ia), Some(&ib), Some(&im)) = (vocab.get(a), vocab.get(b), vocab.get(&format!("{a}{b}"))) else { continue };
            merges.entry((ia, ib)).or_insert((rank as u32, im));
        }
        let byte_char = byte_chars();
        let char_byte = byte_char.iter().enumerate().map(|(b, &c)| (c, b as u8)).collect();
        let mut specials: Vec<(String, u32, bool)> = types
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t == TYPE_CONTROL || t == TYPE_USER_DEFINED)
            .map(|(i, &t)| (pieces[i].clone(), i as u32, t == TYPE_CONTROL))
            .filter(|(s, _, _)| !s.is_empty())
            .collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
        let mut eos = vec![];
        for k in ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id"] {
            if let Ok(id) = g.u64(k) {
                if !eos.contains(&(id as u32)) {
                    eos.push(id as u32);
                }
            }
        }
        Ok(Tokenizer { pieces, types, vocab, merges, split: Regex::new(SPLIT).map_err(|e| e.to_string())?, byte_char, char_byte, specials, eos })
    }

    pub fn n_vocab(&self) -> usize {
        self.pieces.len()
    }

    pub fn token_id(&self, piece: &str) -> Option<u32> {
        self.vocab.get(piece).copied()
    }

    pub fn is_control(&self, id: u32) -> bool {
        self.types.get(id as usize) == Some(&TYPE_CONTROL)
    }

    /// Tokenizes `text`; with `parse_special`, control tokens written in the text are tokens too.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // the earliest special token, the longest at that position
            let mut hit: Option<(usize, &(String, u32, bool))> = None;
            for sp in &self.specials {
                if sp.2 && !parse_special {
                    continue;
                }
                if let Some(pos) = rest.find(sp.0.as_str()) {
                    if hit.is_none_or(|(p, h)| pos < p || (pos == p && sp.0.len() > h.0.len())) {
                        hit = Some((pos, sp));
                    }
                }
            }
            match hit {
                Some((pos, sp)) => {
                    self.encode_plain(&rest[..pos], &mut out);
                    out.push(sp.1);
                    rest = &rest[pos + sp.0.len()..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        let mut start = 0;
        while start < text.len() {
            let Ok(Some(m)) = self.split.find_from_pos(text, start) else { break };
            if m.start() > start {
                self.encode_word(&text[start..m.start()], out);
            }
            if m.end() == m.start() {
                // an empty match cannot advance, take one char
                let c = text[m.start()..].chars().next().unwrap();
                self.encode_word(&text[m.start()..m.start() + c.len_utf8()], out);
                start = m.start() + c.len_utf8();
                continue;
            }
            self.encode_word(m.as_str(), out);
            start = m.end();
        }
    }

    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        let enc: String = word.bytes().map(|b| self.byte_char[b as usize]).collect();
        // ignore_merges: a word that is a token is that token
        if let Some(&id) = self.vocab.get(&enc) {
            out.push(id);
            return;
        }
        let mut syms: Vec<u32> = Vec::with_capacity(enc.len());
        for c in enc.chars() {
            let mut buf = [0u8; 4];
            match self.vocab.get(c.encode_utf8(&mut buf) as &str) {
                Some(&id) => syms.push(id),
                None => syms.push(u32::MAX), // no byte-level token, cannot happen with a full byte vocabulary
            }
        }
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.merges.get(&(syms[i], syms[i + 1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, merged));
                    }
                }
            }
            let Some((_, i, merged)) = best else { break };
            syms[i] = merged;
            syms.remove(i + 1);
        }
        out.extend(syms.into_iter().filter(|&s| s != u32::MAX));
    }

    /// The bytes of a token; control tokens give their text only with `special`.
    pub fn piece_bytes(&self, id: u32, special: bool) -> Vec<u8> {
        let Some(p) = self.pieces.get(id as usize) else { return vec![] };
        match self.types.get(id as usize) {
            Some(&TYPE_CONTROL) => {
                if special {
                    p.as_bytes().to_vec()
                } else {
                    vec![]
                }
            }
            Some(&TYPE_USER_DEFINED) => p.as_bytes().to_vec(),
            _ => p.chars().map(|c| self.char_byte.get(&c).copied().unwrap_or(b'?')).collect(),
        }
    }

    pub fn decode(&self, ids: &[u32], special: bool) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|&id| self.piece_bytes(id, special)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}
