//! The Inkling-2 chat format: the GGUF template rendered natively (it uses Hugging Face extensions of Jinja),
//! and an incremental parser of the generated messages (thinking, text, tool calls).

use serde_json::{Map, Value};

pub const MESSAGE_USER: &str = "<|message_user|>";
pub const MESSAGE_MODEL: &str = "<|message_model|>";
pub const MESSAGE_SYSTEM: &str = "<|message_system|>";
pub const MESSAGE_TOOL: &str = "<|message_tool|>";
pub const CONTENT_TEXT: &str = "<|content_text|>";
pub const CONTENT_THINKING: &str = "<|content_thinking|>";
pub const CONTENT_TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub const CONTENT_XML: &str = "<|content_xml|>";
pub const END_MESSAGE: &str = "<|end_message|>";
pub const END_SAMPLING: &str = "<|content_model_end_sampling|>";

fn effort_value(effort: Option<&Value>) -> Result<f64, String> {
    let num = match effort {
        None | Some(Value::Null) => 0.9,
        Some(Value::String(s)) => match s.trim() {
            "" | "default" | "auto" => 0.9,
            "none" => 0.0,
            "minimal" => 0.1,
            "low" => 0.2,
            "medium" => 0.7,
            "high" => 0.9,
            "xhigh" | "max" => 0.99,
            o => return Err(format!("Unknown reasoning_effort: {o}")),
        },
        Some(v) => v.as_f64().ok_or("reasoning_effort must be a number or a level")?,
    };
    if !(0.0..=0.99).contains(&num) {
        return Err("reasoning_effort must be in [0.0, 0.99]".into());
    }
    Ok(num)
}

fn effort_message(num: f64) -> String {
    let n = if num == 0.0 { "0".to_string() } else { format!("{num}") };
    format!("{MESSAGE_SYSTEM}{CONTENT_TEXT}Thinking effort level: {n}{END_MESSAGE}")
}

/// JSON with sorted keys and no spaces, as the template's tojson(sort_keys=true, separators=(",", ":")).
pub fn json_sorted(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let parts: Vec<String> = keys.iter().map(|k| format!("{}:{}", Value::String((*k).clone()), json_sorted(&m[*k]))).collect();
            format!("{{{}}}", parts.join(","))
        }
        Value::Array(a) => format!("[{}]", a.iter().map(json_sorted).collect::<Vec<_>>().join(",")),
        o => o.to_string(),
    }
}

fn s<'a>(m: &'a Map<String, Value>, k: &str) -> Option<&'a str> {
    m.get(k).and_then(Value::as_str).filter(|x| !x.is_empty())
}

/// Renders OpenAI-style `messages` (and `tools`) as the model's prompt.
pub fn render(messages: &[Value], tools: Option<&Value>, effort: Option<&Value>) -> Result<String, String> {
    render_from(messages, 0, tools, effort)
}

/// The tool JSON section of a call: `{"name":...,"args":...}`. Arguments that parse as a JSON object go in as they are
/// written (the model's own bytes, which it wrote with the template's sorted keys), other JSON as the template
/// formats it, and text that does not parse verbatim, so that any reply the server streamed renders back: a raw
/// section with its own `{"name"` wrapper (the parser's fallback) as it is.
pub fn tool_json(name: &str, args: Option<&Value>) -> String {
    let n = Value::String(name.into());
    match args {
        None | Some(Value::Null) => format!("{{\"name\":{n},\"args\":{{}}}}"),
        Some(Value::String(a)) => match serde_json::from_str::<Value>(a) {
            Ok(Value::Object(_)) => format!("{{\"name\":{n},\"args\":{a}}}"),
            Ok(v) => format!("{{\"name\":{n},\"args\":{}}}", json_sorted(&v)),
            Err(_) if a.trim_start().starts_with("{\"name\"") => a.clone(),
            Err(_) => format!("{{\"name\":{n},\"args\":{a}}}"),
        },
        Some(v) => format!("{{\"name\":{n},\"args\":{}}}", json_sorted(v)),
    }
}

/// Renders `messages[start..]` as the continuation of the rendering of `messages[..start]` (the tools and the effort
/// message, which come before the first message that is not a system one, are already there when start > 0).
pub fn render_from(messages: &[Value], start: usize, tools: Option<&Value>, effort: Option<&Value>) -> Result<String, String> {
    let mut out = String::new();
    if let Some(Value::Array(tools)) = tools.filter(|_| start == 0) {
        if !tools.is_empty() {
            let specs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    let f = t.get("function").unwrap_or(t);
                    serde_json::json!({
                        "description": f.get("description").cloned().filter(|d| !d.is_null()).unwrap_or(Value::String(String::new())),
                        "name": f.get("name").cloned().unwrap_or(Value::Null),
                        "parameters": f.get("parameters").cloned().filter(|p| !p.is_null()).unwrap_or(Value::Object(Map::new())),
                        "type": t.get("type").cloned().filter(|p| !p.is_null()).unwrap_or(Value::String("function".into())),
                    })
                })
                .collect();
            out += &format!("{MESSAGE_SYSTEM}tool_declare{CONTENT_XML}{}{END_MESSAGE}", json_sorted(&Value::Array(specs)));
        }
    }
    let effort = effort_value(effort)?;
    let mut effort_done = messages[..start.min(messages.len())].iter().any(|m| m.get("role").and_then(Value::as_str) != Some("system"));
    for (i, msg) in messages.iter().enumerate().skip(start) {
        let m = msg.as_object().ok_or("a message must be an object")?;
        let role = s(m, "role").ok_or("a message needs a role")?;
        let mut header = match role {
            "user" => MESSAGE_USER,
            "assistant" => MESSAGE_MODEL,
            "system" => MESSAGE_SYSTEM,
            "tool" => MESSAGE_TOOL,
            r => return Err(format!("Unknown message role: {r}")),
        }
        .to_string();
        if !effort_done && role != "system" {
            out += &effort_message(effort);
            effort_done = true;
        }
        if role == "tool" {
            if let Some(n) = s(m, "name") {
                header += n;
            } else if let Some(id) = s(m, "tool_call_id") {
                // the call it answers: the nearest one before it with that id (ids may repeat across replies)
                let call = messages[..i]
                    .iter()
                    .rev()
                    .filter(|p| p.get("role").and_then(Value::as_str) == Some("assistant"))
                    .find_map(|p| p.get("tool_calls").and_then(Value::as_array).and_then(|tcs| tcs.iter().find(|tc| tc.get("id").and_then(Value::as_str) == Some(id))));
                if let Some(n) = call.and_then(|tc| tc.pointer("/function/name")).and_then(Value::as_str) {
                    header += n;
                }
            }
        } else if role == "assistant" {
            if let Some(n) = s(m, "name") {
                header += n;
            }
        }
        let tool_calls = m.get("tool_calls").and_then(Value::as_array).filter(|a| !a.is_empty());
        let reasoning = if role == "assistant" { s(m, "reasoning_content") } else { None };
        if let Some(r) = reasoning {
            out += &format!("{header}{CONTENT_THINKING}{r}{END_MESSAGE}");
        }
        match m.get("content") {
            Some(Value::String(c)) => {
                if !c.is_empty() || !(role == "assistant" && (tool_calls.is_some() || reasoning.is_some())) {
                    out += &format!("{header}{CONTENT_TEXT}{c}{END_MESSAGE}");
                }
            }
            Some(Value::Array(parts)) => {
                for p in parts {
                    match p {
                        Value::String(t) => out += &format!("{header}{CONTENT_TEXT}{t}{END_MESSAGE}"),
                        Value::Object(po) => {
                            let ty = po.get("type").and_then(Value::as_str).unwrap_or("text");
                            match ty {
                                "text" | "input_text" => {
                                    let t = po.get("text").and_then(Value::as_str).unwrap_or("");
                                    out += &format!("{header}{CONTENT_TEXT}{t}{END_MESSAGE}");
                                }
                                "image" | "input_image" | "image_url" => return Err("image input is not supported by this engine yet".into()),
                                "audio" | "input_audio" | "audio_url" => return Err("Inkling 2 does not accept audio input".into()),
                                t => return Err(format!("Unsupported content part type: {t}")),
                            }
                        }
                        _ => return Err("bad content part".into()),
                    }
                }
            }
            None | Some(Value::Null) => {
                if !(role == "assistant" && (tool_calls.is_some() || reasoning.is_some())) {
                    out += &format!("{header}{CONTENT_TEXT}{END_MESSAGE}");
                }
            }
            _ => return Err("bad message content".into()),
        }
        for tc in tool_calls.into_iter().flatten() {
            let f = tc.get("function").ok_or("tool call without function")?;
            let name = f.get("name").and_then(Value::as_str).ok_or("tool call function name must be a string")?;
            out += &format!("{MESSAGE_MODEL}{name}{CONTENT_TOOL_JSON}{}{END_MESSAGE}", tool_json(name, f.get("arguments")));
        }
        if role == "assistant" {
            out += END_SAMPLING;
        }
    }
    if !effort_done {
        out += &effort_message(effort);
    }
    Ok(out)
}

/// A piece of the model's reply.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
    /// a finished tool call: name, arguments (JSON text)
    ToolCall(String, String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Section {
    Header, // after <|message_model|>, before the content tag: the name of a tool call, if any
    Thinking,
    Text,
    ToolJson,
    Other,
}

/// Turns the generated text (with special tokens) into deltas. Feed it the text of each token as it comes.
pub struct Parser {
    sec: Section,
    header: String,
    tool_buf: String,
    pending: String, // a possible special token, cut between pieces
}

impl Default for Parser {
    fn default() -> Self {
        Parser { sec: Section::Text, header: String::new(), tool_buf: String::new(), pending: String::new() }
    }
}

impl Parser {
    /// Whether the model is writing its thinking.
    pub fn thinking(&self) -> bool {
        self.sec == Section::Thinking
    }

    pub fn push(&mut self, piece: &str) -> Vec<Delta> {
        let mut out = vec![];
        self.pending += piece;
        let text = std::mem::take(&mut self.pending);
        let mut rest = text.as_str();
        while !rest.is_empty() {
            if let Some(i) = rest.find("<|") {
                if i > 0 {
                    self.emit(&rest[..i], &mut out);
                    rest = &rest[i..];
                }
                match rest.find("|>") {
                    Some(j) => {
                        let tag = &rest[..j + 2];
                        self.tag(tag, &mut out);
                        rest = &rest[j + 2..];
                    }
                    None => {
                        self.pending = rest.to_string();
                        break;
                    }
                }
            } else {
                // keep a trailing '<' that may start a tag
                if rest.ends_with('<') {
                    self.emit(&rest[..rest.len() - 1], &mut out);
                    self.pending = "<".into();
                } else {
                    self.emit(rest, &mut out);
                }
                break;
            }
        }
        out
    }

    fn emit(&mut self, t: &str, out: &mut Vec<Delta>) {
        if t.is_empty() {
            return;
        }
        match self.sec {
            Section::Header => self.header += t,
            Section::Thinking => out.push(Delta::Reasoning(t.to_string())),
            Section::Text => out.push(Delta::Content(t.to_string())),
            Section::ToolJson => self.tool_buf += t,
            Section::Other => {}
        }
    }

    fn tag(&mut self, tag: &str, out: &mut Vec<Delta>) {
        match tag {
            MESSAGE_MODEL => {
                self.sec = Section::Header;
                self.header.clear();
            }
            CONTENT_THINKING => self.sec = Section::Thinking,
            CONTENT_TEXT => self.sec = if self.sec == Section::Header || self.sec == Section::Text { Section::Text } else { Section::Text },
            CONTENT_TOOL_JSON => {
                self.sec = Section::ToolJson;
                self.tool_buf.clear();
            }
            END_MESSAGE => {
                if self.sec == Section::ToolJson {
                    let name = self.header.trim().to_string();
                    out.push(Delta::ToolCall(name, tool_args(&self.tool_buf)));
                }
                self.sec = Section::Other;
            }
            _ => {}
        }
    }
}

/// The arguments of a tool call from its JSON section: the model's own bytes when the section has the template's shape
/// `{"name":...,"args":...}` (tool_json renders them back as they are), the parsed `args` re-serialized when its shape
/// differs, and the whole section verbatim when it does not parse.
fn tool_args(buf: &str) -> String {
    let Ok(v) = serde_json::from_str::<Value>(buf) else { return buf.to_string() };
    let (Some(name), Some(args)) = (v.get("name"), v.get("args")) else { return buf.to_string() };
    let raw = buf.strip_prefix(&format!("{{\"name\":{name},\"args\":")).and_then(|r| r.strip_suffix('}'));
    match raw {
        Some(r) if serde_json::from_str::<Value>(r).ok().as_ref() == Some(args) && matches!(args, Value::Object(_)) => r.to_string(),
        _ => args.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_and_parses() {
        let msgs: Vec<Value> = serde_json::from_str(r#"[{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]"#).unwrap();
        let p = render(&msgs, None, None).unwrap();
        assert_eq!(p, "<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>");
        let mut ps = Parser::default();
        let mut d = vec![];
        for piece in ["<|message_model|>", "<|content_thinking|>", "Hmm", "<|end_message|>", "<|message_model|><|content_text|>", "Hel", "lo<", "|end_message|>"] {
            d.extend(ps.push(piece));
        }
        assert_eq!(d, vec![Delta::Reasoning("Hmm".into()), Delta::Content("Hel".into()), Delta::Content("lo".into())]);
        let mut ps = Parser::default();
        let d = ps.push("<|message_model|>get_time<|content_invoke_tool_json|>{\"name\":\"get_time\",\"args\":{\"tz\":\"UTC\"}}<|end_message|>");
        assert_eq!(d, vec![Delta::ToolCall("get_time".into(), "{\"tz\":\"UTC\"}".into())]);
    }

    fn msgs(j: &str) -> Vec<Value> {
        serde_json::from_str(j).unwrap()
    }

    #[test]
    fn tool_results_find_their_own_call() {
        // two replies whose calls share an id (as every reply's first call was call_1): each result names its own
        let m = msgs(r#"[{"role":"user","content":"go"},
            {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"compile","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call_1","content":"ok"},
            {"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"run_cpp","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call_1","content":"42"}]"#);
        let p = render(&m, None, None).unwrap();
        assert!(p.contains("<|message_tool|>compile<|content_text|>ok"));
        assert!(p.contains("<|message_tool|>run_cpp<|content_text|>42"));
        assert!(!p.contains("compilerun_cpp") && !p.contains("run_cpprun_cpp"));
        assert_eq!(p, render(&m, None, None).unwrap());
        // the rendering from a message on continues the rendering before it
        assert_eq!(render(&m[..3], None, None).unwrap() + &render_from(&m, 3, None, None).unwrap(), p);
    }

    #[test]
    fn unparsable_arguments_render_verbatim() {
        let m = msgs(r#"[{"role":"user","content":"go"},
            {"role":"assistant","content":"","tool_calls":[{"id":"a","type":"function","function":{"name":"run_cpp","arguments":"{\"code\": \"int main(){"}}]},
            {"role":"tool","tool_call_id":"a","content":"bad arguments"},
            {"role":"assistant","content":"","tool_calls":[{"id":"b","type":"function","function":{"name":"run_cpp","arguments":"{\"name\":\"run_cpp\",\"args\":{\"code\":"}}]}]"#);
        let p = render(&m, None, None).unwrap();
        assert!(p.contains("run_cpp<|content_invoke_tool_json|>{\"name\":\"run_cpp\",\"args\":{\"code\": \"int main(){}<|end_message|>"));
        assert!(p.contains("run_cpp<|content_invoke_tool_json|>{\"name\":\"run_cpp\",\"args\":{\"code\":<|end_message|>"));
        assert_eq!(p, render(&m, None, None).unwrap());
    }

    /// What the model generated comes back from parse -> deltas -> message -> render as it was.
    #[test]
    fn generated_replies_render_back_exactly() {
        let prefix = msgs(r#"[{"role":"system","content":"Solve it."},{"role":"user","content":"Problem"}]"#);
        let calls = [
            // unsorted keys, newlines, quotes, <, >, &, unicode, a nested object
            r##"{"name":"run_cpp","args":{"source":"#include <bits/stdc++.h>\nint main(){ puts(\"a&b<c>\"); }","input":"é ü 日本\t1 2","opts":{"z":1,"a":[1.5,"x"]}}}"##,
            // not JSON at all: the model's bytes stay
            r#"{"name":"run_cpp","args":{"source":"int main(){"#,
        ];
        // (a section of another shape, {"args":...,"name":...}, gives the client its arguments re-serialized: the
        // server's stored sequence, not render, keeps such a reply exact)
        for call in calls {
            let generated = format!("<|message_model|><|content_thinking|>Let me test it.\n<|end_message|><|message_model|><|content_text|>Running.<|end_message|><|message_model|>run_cpp<|content_invoke_tool_json|>{call}<|end_message|><|content_model_end_sampling|>");
            let mut ps = Parser::default();
            let mut deltas = vec![];
            // in pieces as tokens come, some cut inside a special token
            let chars: Vec<char> = generated.chars().collect();
            for piece in chars.chunks(3) {
                deltas.extend(ps.push(&piece.iter().collect::<String>()));
            }
            let (mut reasoning, mut content, mut tool_calls) = (String::new(), String::new(), vec![]);
            for d in deltas {
                match d {
                    Delta::Reasoning(r) => reasoning += &r,
                    Delta::Content(c) => content += &c,
                    Delta::ToolCall(name, args) => tool_calls.push(serde_json::json!({"id": format!("call_{}", tool_calls.len()), "type": "function", "function": {"name": name, "arguments": args}})),
                }
            }
            let mut all = prefix.clone();
            all.push(serde_json::json!({"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": tool_calls}));
            let before = render(&prefix, None, None).unwrap();
            assert_eq!(render(&all, None, None).unwrap(), before + &generated, "{call}");
        }
    }
}
