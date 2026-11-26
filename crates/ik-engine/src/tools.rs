//! The chat UI's server tools, on with `inkling-server --tools`: the model reads, searches and writes files on this
//! machine, as the user the server runs as. The llama.cpp chat UI lists them (GET /tools), offers them to the model,
//! asks the user before each call unless told to always allow it, and runs them (POST /tools {"tool", "params"}).
//! A relative path is taken from the conversation's working directory, which the UI sends (x-tool-cwd).
//! A result has a shape the UI reads: {"plain_text_response"} is text for the model, {"error"} a failure, and another
//! object goes to the model as JSON.

use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

/// read_file's text in one call, at most
const READ_LINES: usize = 2000;
const READ_BYTES: usize = 64 << 10;
/// read_file's bytes for the UI's read_media (x-resp-type: base64), at most
const MEDIA_BYTES: u64 = 32 << 20;
/// a walk (file_glob_search, grep_search) ends after this many entries or this long
const WALK_ENTRIES: usize = 500_000;
const WALK_TIME: Duration = Duration::from_secs(10);
/// grep_search skips larger files, and shows this much of a long line
const GREP_FILE_BYTES: u64 = 32 << 20;
const GREP_LINE_CHARS: usize = 400;
/// the keys a model gives a path under (the UI's renderers read all three)
const PATH: &[&str] = &["path", "file_path", "filePath"];

/// GET /tools: the tools in the form the UI reads, {tool, definition (what the model sees), uses_cwd}.
pub fn list() -> Value {
    let path = |what: &str| json!({"type": "string", "description": format!("{what}: absolute, ~/..., or relative to the working directory")});
    let int = |what: &str| json!({"type": "integer", "description": what});
    let string = |what: &str| json!({"type": "string", "description": what});
    let boolean = |what: &str| json!({"type": "boolean", "description": what});
    Value::Array(vec![
        def(
            "read_file",
            "Read a text file: all of it, or lines start_line to end_line (counted from 1). At most 2000 lines or 64 KiB come \
             at once; a last line in [brackets] then says where to read on.",
            json!({"path": path("the file"), "start_line": int("the first line to read (from 1)"), "end_line": int("the last line to read")}),
            &["path"],
        ),
        def(
            "write_file",
            "Write a text file: create it, or replace all of it. Missing directories are made.",
            json!({"path": path("the file"), "content": string("the whole text of the file")}),
            &["path", "content"],
        ),
        def(
            "edit_file",
            "Edit a text file in place. Each edit replaces old_text, which must occur exactly once in the file, with \
             new_text. The edits apply in order, all or none.",
            json!({"path": path("the file"), "edits": {"type": "array", "description": "the replacements", "items": {
                "type": "object", "required": ["old_text", "new_text"], "properties": {
                    "old_text": string("text in the file, with enough of its surroundings to occur once"),
                    "new_text": string("what replaces it")}}}}),
            &["path", "edits"],
        ),
        def(
            "file_glob_search",
            "Find files and directories under a directory, nearest first. Patterns: * and ? within a name, ** across \
             directories, [abc], {a,b}. A pattern without / matches a name at any depth (*.rs); one with / matches the \
             path from the directory (src/**/*.rs). Hidden directories, node_modules and build caches (like cargo's \
             target) are entered only when the search starts in one or the pattern names a hidden one.",
            json!({"path": path("the directory (default: the working directory)"),
                   "include": string("the pattern to find (default **, everything)"), "exclude": string("a pattern to leave out"),
                   "type": {"type": "string", "enum": ["all", "file", "dir"], "description": "what to list (default all)"},
                   "max_depth": int("how many levels down to look (default 64)"), "limit": int("the most results to give (default 100)")}),
            &[],
        ),
        def(
            "grep_search",
            "Search the lines of text files for a regular expression (Rust regex syntax, with look-around). Gives \
             file:line:text for each matching line, the files nearest the start first. Skips what file_glob_search skips.",
            json!({"pattern": string("the regular expression"),
                   "path": path("a directory to search under, or one file (default: the working directory)"),
                   "include": string("a glob of the files to search, like *.rs (as in file_glob_search)"),
                   "exclude": string("a glob of files to leave out"), "ignore_case": boolean("match either case (default false)"),
                   "return_line_numbers": boolean("give line numbers (default true)"), "limit": int("the most lines to give (default 200)")}),
            &["pattern"],
        ),
        def("get_info", "Get this machine's OS, the working directory and the home directory.", json!({}), &[]),
    ])
}

fn def(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({"tool": name, "uses_cwd": true, "definition": {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}})
}

/// POST /tools runs as the server's user, so it answers only this machine (a loopback `peer`), only under one of this
/// server's names in `host` (a DNS-rebinding page sends its own), and no other site's page (`origin`, which browsers
/// send with a POST). `port` is the one the request came to.
pub fn caller_ok(peer: IpAddr, port: u16, host: Option<&str>, origin: Option<&str>) -> Result<(), String> {
    if !peer.to_canonical().is_loopback() {
        return Err(format!("the tools answer only this machine, not {peer}"));
    }
    let host = host.unwrap_or("");
    let (name, p) = match host.rsplit_once(':') {
        Some((n, p)) if !p.contains(']') => (n, p.parse().ok()),
        _ => (host, Some(80)),
    };
    let ip = name.trim_start_matches('[').trim_end_matches(']').parse::<IpAddr>();
    if p != Some(port) || !(name.eq_ignore_ascii_case("localhost") || ip.is_ok_and(|ip| ip.to_canonical().is_loopback())) {
        return Err(format!("Host {host:?} is not this server"));
    }
    match origin {
        Some(o) if !o.eq_ignore_ascii_case(&format!("http://{host}")) => Err(format!("a page from {o} may not use the tools")),
        _ => Ok(()),
    }
}

/// POST /tools: runs `tool` with `params`, None if there's no such tool. `cwd` is the conversation's working
/// directory (x-tool-cwd); `raw` is the UI's x-resp-type: base64, for which read_file gives the file's bytes.
pub fn call(tool: &str, params: &Value, cwd: Option<&str>, raw: bool) -> Option<Value> {
    let (p, cwd) = (params, workdir(cwd));
    let r = match tool {
        "read_file" if raw => read_bytes(p, &cwd),
        "read_file" => read_file(p, &cwd).map(text),
        "write_file" => write_file(p, &cwd),
        "edit_file" => edit_file(p, &cwd),
        "file_glob_search" => glob_search(p, &cwd),
        "grep_search" => grep_search(p, &cwd).map(text),
        "get_info" => Ok(get_info(&cwd)),
        _ => return None,
    };
    Some(r.unwrap_or_else(|e| json!({"error": e})))
}

fn text(s: String) -> Value {
    json!({"plain_text_response": s})
}

fn home() -> PathBuf {
    std::env::var_os("HOME").filter(|h| !h.is_empty()).map_or_else(|| PathBuf::from("/"), PathBuf::from)
}

/// The conversation's working directory (x-tool-cwd), else the server's.
fn workdir(cwd: Option<&str>) -> PathBuf {
    let here = std::env::current_dir().unwrap_or_else(|_| home());
    match cwd.map(str::trim).filter(|c| !c.is_empty()) {
        Some(c) => resolve(c, &here),
        None => here,
    }
}

/// A path as the model gave it: `~` is the home directory, a relative path is under `cwd`.
fn resolve(p: &str, cwd: &Path) -> PathBuf {
    if p == "~" {
        home()
    } else if let Some(r) = p.strip_prefix("~/") {
        home().join(r)
    } else {
        cwd.join(p)
    }
}

/// `p` with its `.` and `..` worked out in the text, to show.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    out
}

/// The first of `keys` that `p` has as a string.
fn s<'a>(p: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| p.get(*k).and_then(Value::as_str))
}

/// The first of `keys` as a whole number (models send "10" as often as 10).
fn n(p: &Value, keys: &[&str]) -> Option<i64> {
    keys.iter().find_map(|k| match p.get(*k)? {
        Value::Number(x) => x.as_i64().or_else(|| x.as_f64().map(|f| f as i64)),
        Value::String(t) => t.trim().parse().ok(),
        _ => None,
    })
}

fn flag(p: &Value, k: &str) -> Option<bool> {
    match p.get(k)? {
        Value::Bool(b) => Some(*b),
        Value::String(t) => t.trim().parse().ok(),
        _ => None,
    }
}

/// The path a tool works on, as given and resolved.
fn path_arg<'a>(tool: &str, p: &'a Value, cwd: &Path) -> Result<(&'a str, PathBuf), String> {
    let given = s(p, PATH).filter(|g| !g.is_empty()).ok_or_else(|| format!("{tool} needs a path"))?;
    Ok((given, resolve(given, cwd)))
}

fn read_file(p: &Value, cwd: &Path) -> Result<String, String> {
    let (given, path) = path_arg("read_file", p, cwd)?;
    let start = n(p, &["start_line", "line_start", "startLine", "from_line"]).unwrap_or(1).max(1) as usize;
    let end = n(p, &["end_line", "line_end", "endLine", "to_line"])
        .or_else(|| n(p, &["line_count", "count", "num_lines"]).map(|c| start as i64 + c.max(1) - 1))
        .map(|e| e.max(0) as usize);
    if end.is_some_and(|e| e < start) {
        return Err(format!("end_line comes before start_line {start}"));
    }
    let m = fs::metadata(&path).map_err(|e| format!("{given}: {e}"))?;
    if m.is_dir() {
        return Err(format!("{given} is a directory (file_glob_search lists it)"));
    } else if !m.is_file() {
        return Err(format!("{given} is not a regular file"));
    }
    let mut r = BufReader::new(File::open(&path).map_err(|e| format!("{given}: {e}"))?);
    if r.fill_buf().map_err(|e| format!("{given}: {e}"))?.contains(&0) {
        return Err(format!("{given} is a binary file"));
    }
    let (mut out, mut line, mut total, mut last, mut cut) = (String::new(), Vec::new(), 0usize, 0usize, false);
    loop {
        line.clear();
        if r.read_until(b'\n', &mut line).map_err(|e| format!("{given}: {e}"))? == 0 {
            break;
        }
        total += 1;
        if total < start || end.is_some_and(|e| total > e) || cut {
            continue;
        }
        if total - start >= READ_LINES || out.len() + line.len() > READ_BYTES {
            if out.is_empty() {
                // one line longer than READ_BYTES: its start
                let t = String::from_utf8_lossy(&line);
                let k = (0..=READ_BYTES).rev().find(|&k| t.is_char_boundary(k)).unwrap_or(0);
                out.push_str(&t[..k]);
                out.push_str(&format!("\n[line {total} is longer than 64 KiB: this is its start]"));
                last = total;
            }
            cut = true;
            continue;
        }
        out.push_str(&String::from_utf8_lossy(&line));
        last = total;
    }
    if total == 0 {
        return Ok("(the file is empty)".into());
    }
    if start > total {
        return Err(format!("{given} has {total} lines"));
    }
    if cut && last < total {
        if !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&format!("[lines {start}-{last} of {total}; read on with start_line={}]", last + 1));
    }
    Ok(out)
}

/// read_file for the UI's read_media: the file's bytes, in base64.
fn read_bytes(p: &Value, cwd: &Path) -> Result<Value, String> {
    let (given, path) = path_arg("read_file", p, cwd)?;
    let m = fs::metadata(&path).map_err(|e| format!("{given}: {e}"))?;
    if !m.is_file() || m.len() > MEDIA_BYTES {
        return Err(format!("{given}: not a file of {} MiB at most", MEDIA_BYTES >> 20));
    }
    let b = fs::read(&path).map_err(|e| format!("{given}: {e}"))?;
    Ok(json!({"base64": base64(&b), "size_bytes": b.len()}))
}

fn base64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let v = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            s.push(if k <= c.len() { T[(v >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

fn write_file(p: &Value, cwd: &Path) -> Result<Value, String> {
    let (given, path) = path_arg("write_file", p, cwd)?;
    let content = s(p, &["content", "contents", "text"]).ok_or("write_file needs the content")?;
    if let Some(d) = path.parent() {
        fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    fs::write(&path, content).map_err(|e| format!("{given}: {e}"))?;
    Ok(json!({"bytes": content.len(), "result": format!("wrote {} bytes to {}", content.len(), path.display())}))
}

fn edit_file(p: &Value, cwd: &Path) -> Result<Value, String> {
    let (given, path) = path_arg("edit_file", p, cwd)?;
    let edits: Vec<Value> = match p.get("edits") {
        Some(Value::Array(a)) => a.clone(),
        // the list as a JSON string, or one edit given at the top level
        Some(Value::String(t)) => serde_json::from_str(t).map_err(|_| "edits: a list of {old_text, new_text}")?,
        _ if p.get("old_text").is_some() => vec![p.clone()],
        _ => return Err("edit_file needs edits: [{old_text, new_text}]".into()),
    };
    if edits.is_empty() {
        return Err("edit_file needs edits: [{old_text, new_text}]".into());
    }
    let mut t = fs::read_to_string(&path).map_err(|e| format!("{given}: {e}"))?;
    let crlf = t.contains("\r\n");
    let none = if edits.len() > 1 { " (no edit was made)" } else { "" };
    for (i, e) in edits.iter().enumerate() {
        let i = i + 1;
        let old = s(e, &["old_text", "old_string", "oldText"]).filter(|o| !o.is_empty()).ok_or_else(|| format!("edit {i}: no old_text{none}"))?;
        let new = s(e, &["new_text", "new_string", "newText"]).unwrap_or("");
        // a model writes \n for the file's \r\n
        let (old, new) = if crlf && !t.contains(old) && old.contains('\n') {
            let w = |x: &str| x.replace("\r\n", "\n").replace('\n', "\r\n");
            (w(old), w(new))
        } else {
            (old.to_string(), new.to_string())
        };
        match t.matches(old.as_str()).count() {
            1 => t = t.replacen(old.as_str(), &new, 1),
            0 => return Err(format!("edit {i}: old_text is not in {given}{none}")),
            k => return Err(format!("edit {i}: old_text occurs {k} times in {given}; give more of the text around it{none}")),
        }
    }
    fs::write(&path, &t).map_err(|e| format!("{given}: {e}"))?;
    let k = edits.len();
    Ok(json!({"edits_applied": k, "result": format!("made {k} edit{} in {}", if k == 1 { "" } else { "s" }, path.display())}))
}

/// A glob: `*` and `?` within a name, `**` across directories, `[abc]` and `[!abc]`, `{a,b}`. A pattern without `/`
/// matches a name at any depth (`*.rs`); one with `/` matches the path from the walk's start (`src/**/*.rs`).
struct Glob {
    /// each alternative, and whether it matches the path (else the name)
    pats: Vec<(Vec<char>, bool)>,
    /// it names a hidden entry (a part that starts with "."): the walk enters hidden directories
    hidden: bool,
}

impl Glob {
    fn new(p: &str) -> Glob {
        let mut alts = vec![];
        expand(p.trim(), &mut alts);
        let pats = alts
            .iter()
            .map(|a| {
                let a = a.trim_start_matches("./").trim_start_matches('/').trim_end_matches('/');
                (a.chars().collect(), a.contains('/'))
            })
            .collect();
        let hidden = alts.iter().any(|a| a.split('/').any(|c| c.starts_with('.') && c != "." && c != ".."));
        Glob { pats, hidden }
    }

    fn matches(&self, rel: &str, name: &str) -> bool {
        let (rel, name): (Vec<char>, Vec<char>) = (rel.chars().collect(), name.chars().collect());
        self.pats.iter().any(|(p, path)| glob_match(p, if *path { &rel } else { &name }))
    }
}

/// `{a,b}` alternatives, expanded (64 patterns at most).
fn expand(p: &str, out: &mut Vec<String>) {
    let b = p.as_bytes();
    let (mut i, mut depth, mut open, mut commas) = (0, 0, 0, vec![]);
    let mut close = None;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 1,
            b'{' => {
                if depth == 0 {
                    open = i;
                }
                depth += 1;
            }
            b',' if depth == 1 => commas.push(i),
            b'}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    close = Some(i);
                    break;
                }
            }
            _ => {}
        }
        i += 1;
    }
    let Some(c) = close else {
        out.push(p.to_string());
        return;
    };
    let mut from = open + 1;
    for k in commas.into_iter().chain([c]) {
        if out.len() >= 64 {
            return;
        }
        expand(&format!("{}{}{}", &p[..open], &p[from..k], &p[c + 1..]), out);
        from = k + 1;
    }
}

fn glob_match(p: &[char], s: &[char]) -> bool {
    // 0: not known yet, 1: no, 2: yes, for (pattern at i, text at j)
    let mut memo = vec![0u8; (p.len() + 1) * (s.len() + 1)];
    glob_at(p, s, 0, 0, &mut memo)
}

fn glob_at(p: &[char], s: &[char], i: usize, j: usize, memo: &mut [u8]) -> bool {
    let key = i * (s.len() + 1) + j;
    if memo[key] != 0 {
        return memo[key] == 2;
    }
    let r = if i == p.len() {
        j == s.len()
    } else {
        match p[i] {
            '*' => {
                let mut e = i;
                while e < p.len() && p[e] == '*' {
                    e += 1;
                }
                if e - i >= 2 {
                    // ** takes any characters, / too; "**/" may also be no directory at all
                    (e < p.len() && p[e] == '/' && glob_at(p, s, e + 1, j, memo)) || (j..=s.len()).any(|t| glob_at(p, s, e, t, memo))
                } else {
                    // * takes any characters but /
                    let mut t = j;
                    loop {
                        if glob_at(p, s, e, t, memo) {
                            break true;
                        }
                        if t == s.len() || s[t] == '/' {
                            break false;
                        }
                        t += 1;
                    }
                }
            }
            '?' => j < s.len() && s[j] != '/' && glob_at(p, s, i + 1, j + 1, memo),
            '[' if class(p, i, '\0').is_some() => match s.get(j) {
                Some(&c) if c != '/' => {
                    let (next, hit) = class(p, i, c).unwrap();
                    hit && glob_at(p, s, next, j + 1, memo)
                }
                _ => false,
            },
            '\\' if i + 1 < p.len() => s.get(j) == Some(&p[i + 1]) && glob_at(p, s, i + 2, j + 1, memo),
            c => s.get(j) == Some(&c) && glob_at(p, s, i + 1, j + 1, memo),
        }
    };
    memo[key] = if r { 2 } else { 1 };
    r
}

/// The class `[...]` at `p[i]`: where the pattern goes on after it, and whether `c` is in it. None if it doesn't
/// close (then `[` is a character).
fn class(p: &[char], i: usize, c: char) -> Option<(usize, bool)> {
    let mut k = i + 1;
    let neg = matches!(p.get(k), Some('!' | '^'));
    if neg {
        k += 1;
    }
    let (mut hit, first) = (false, k);
    while k < p.len() {
        if p[k] == ']' && k > first {
            return Some((k + 1, hit != neg));
        }
        if k + 2 < p.len() && p[k + 1] == '-' && p[k + 2] != ']' {
            hit |= p[k] <= c && c <= p[k + 2];
            k += 3;
        } else {
            hit |= p[k] == c;
            k += 1;
        }
    }
    None
}

enum Stop {
    /// all of it was seen
    Done,
    /// the caller had enough
    Limit,
    /// WALK_ENTRIES or WALK_TIME ran out
    Budget,
}

/// Visits what's under `root`, nearest first and by name within a directory, `depth` levels down, until `visit`
/// (relative path, name, is a directory, path) says enough. It doesn't enter excluded directories, hidden ones
/// (unless `hidden`), node_modules, build caches (a directory with a CACHEDIR.TAG, as cargo's target has) or a link
/// to a directory; `root` itself is walked whatever it is.
fn walk(root: &Path, depth: usize, hidden: bool, exclude: Option<&Glob>, mut visit: impl FnMut(&str, &str, bool, &Path) -> bool) -> Stop {
    let t0 = Instant::now();
    let mut seen = 0usize;
    let mut queue = VecDeque::from([(root.to_path_buf(), String::new(), 0usize)]);
    while let Some((dir, rel, d)) = queue.pop_front() {
        let Ok(rd) = fs::read_dir(&dir) else { continue };
        let mut ents: Vec<(String, fs::FileType)> = rd.filter_map(Result::ok).filter_map(|e| Some((e.file_name().into_string().ok()?, e.file_type().ok()?))).collect();
        if d > 0 && ents.iter().any(|(n, _)| n == "CACHEDIR.TAG") {
            continue;
        }
        ents.sort_by(|a, b| a.0.cmp(&b.0));
        for (name, ft) in ents {
            seen += 1;
            if seen > WALK_ENTRIES || (seen.is_multiple_of(1024) && t0.elapsed() > WALK_TIME) {
                return Stop::Budget;
            }
            let r = if rel.is_empty() { name.clone() } else { format!("{rel}/{name}") };
            if exclude.is_some_and(|x| x.matches(&r, &name)) {
                continue;
            }
            let path = dir.join(&name);
            let is_dir = ft.is_dir() || (ft.is_symlink() && path.is_dir());
            if !visit(&r, &name, is_dir, &path) {
                return Stop::Limit;
            }
            if ft.is_dir() && d + 1 < depth && (hidden || !name.starts_with('.')) && name != "node_modules" {
                queue.push_back((path, r, d + 1));
            }
        }
    }
    Stop::Done
}

/// A walk's entry as the model would name it: under the path it gave.
fn shown(given: &str, rel: &str) -> String {
    match given.trim_end_matches('/') {
        _ if given.is_empty() => rel.to_string(),
        "." => rel.to_string(),
        "" => format!("/{rel}"),
        g => format!("{g}/{rel}"),
    }
}

/// Adds a line of a list the UI shows (it ends the list at the first "---\n", so a line can't end with "---").
fn push_line(out: &mut String, l: &str) {
    out.push_str(l);
    if l.ends_with("---") {
        out.push(' ');
    }
    out.push('\n');
}

fn glob_search(p: &Value, cwd: &Path) -> Result<Value, String> {
    let given = s(p, PATH).unwrap_or("");
    let root = if given.is_empty() { cwd.to_path_buf() } else { resolve(given, cwd) };
    if !root.is_dir() {
        return Err(format!("{}: not a directory", root.display()));
    }
    let include = Glob::new(s(p, &["include", "pattern", "glob"]).filter(|x| !x.trim().is_empty()).unwrap_or("**"));
    let exclude = s(p, &["exclude"]).filter(|x| !x.trim().is_empty()).map(Glob::new);
    let kind = s(p, &["type"]).unwrap_or("all");
    let limit = n(p, &["limit", "max_results"]).map_or(100, |l| l.clamp(1, 10_000) as usize);
    let depth = n(p, &["max_depth"]).map_or(64, |d| d.clamp(1, 256) as usize);
    let (mut entries, mut lines) = (vec![], String::new());
    let stop = walk(&root, depth, include.hidden, exclude.as_ref(), |rel, name, is_dir, _| {
        if match kind {
            "file" => is_dir,
            "dir" => !is_dir,
            _ => false,
        } || !include.matches(rel, name)
        {
            return true;
        }
        entries.push(json!({"path": rel, "type": if is_dir { "dir" } else { "file" }}));
        push_line(&mut lines, &(shown(given, rel) + if is_dir { "/" } else { "" }));
        entries.len() < limit
    });
    lines.push_str(&format!("---\nTotal matches: {}", entries.len()));
    lines.push_str(match stop {
        Stop::Done => "",
        Stop::Limit => "\n(stopped at the limit: narrow the pattern or raise limit)",
        Stop::Budget => "\n(stopped early, too much to look through: search a smaller directory)",
    });
    Ok(json!({"base": normalize(&root).display().to_string(), "entries": entries, "plain_text_response": lines}))
}

fn grep_search(p: &Value, cwd: &Path) -> Result<String, String> {
    let pattern = s(p, &["pattern", "query", "regex"]).filter(|x| !x.is_empty()).ok_or("grep_search needs a pattern")?;
    let given = s(p, PATH).unwrap_or("");
    let root = if given.is_empty() { cwd.to_path_buf() } else { resolve(given, cwd) };
    let case = if flag(p, "ignore_case").or(flag(p, "case_insensitive")).unwrap_or(false) { "(?i)" } else { "" };
    let numbers = flag(p, "return_line_numbers").unwrap_or(true);
    let limit = n(p, &["limit", "max_results"]).map_or(200, |l| l.clamp(1, 10_000) as usize);
    let include = s(p, &["include"]).filter(|x| !x.trim().is_empty()).map(Glob::new);
    let exclude = s(p, &["exclude"]).filter(|x| !x.trim().is_empty()).map(Glob::new);
    // not a regular expression (an unclosed "(" say): the text itself
    let literal = fancy_regex::Regex::new(&format!("{case}{pattern}")).is_err();
    let pat = if literal { fancy_regex::escape(pattern).into_owned() } else { pattern.to_string() };
    let re = fancy_regex::Regex::new(&format!("{case}{pat}")).map_err(|e| e.to_string())?;
    // One look at all of a file first, ^ and $ at its lines' ends, which most files fail. A line's match is the file's
    // too, unless the pattern looks around, anchors at the text (\A \z \G) or turns flags off (?-m); or has a $ and the
    // file a \r ((?m)$ is before \n, not before \r\n).
    let off = pat.match_indices("(?").any(|(i, _)| pat[i + 2..].chars().take_while(|c| c.is_ascii_alphabetic() || *c == '-').any(|c| c == '-'));
    let whole = !off && ["(?=", "(?!", "(?<=", "(?<!", "\\A", "\\z", "\\G"].iter().all(|x| !pat.contains(x));
    let whole = whole.then(|| fancy_regex::Regex::new(&format!("(?m){case}{pat}")).ok()).flatten();
    let dollar = pat.contains('$');
    let (mut out, mut found) = (String::new(), 0usize);
    let mut search = |path: &Path, name: &str| -> bool {
        // a FIFO or device, a large file, or a binary one (a NUL in its first 8 KiB) is left out, the last one unread
        if !fs::metadata(path).is_ok_and(|m| m.is_file() && m.len() <= GREP_FILE_BYTES) {
            return true;
        }
        let Ok(mut f) = File::open(path) else { return true };
        let mut b = Vec::new();
        if Read::by_ref(&mut f).take(8192).read_to_end(&mut b).is_err() || b.contains(&0) || f.read_to_end(&mut b).is_err() {
            return true;
        }
        let t = String::from_utf8_lossy(&b);
        if whole.as_ref().is_some_and(|w| !(dollar && t.contains('\r')) && !w.is_match(&t).unwrap_or(true)) {
            return true;
        }
        for (k, line) in t.lines().enumerate() {
            if !re.is_match(line).unwrap_or(false) {
                continue;
            }
            let line = match line.char_indices().nth(GREP_LINE_CHARS) {
                Some((cut, _)) => format!("{}…", &line[..cut]),
                None => line.to_string(),
            };
            push_line(&mut out, &if numbers { format!("{name}:{}:{line}", k + 1) } else { format!("{name}:{line}") });
            found += 1;
            if found >= limit {
                return false;
            }
        }
        true
    };
    let stop = if root.is_file() {
        if search(&root, given) { Stop::Done } else { Stop::Limit }
    } else if root.is_dir() {
        let hidden = include.as_ref().is_some_and(|g| g.hidden);
        walk(&root, 256, hidden, exclude.as_ref(), |rel, name, is_dir, path| {
            is_dir || !include.as_ref().is_none_or(|g| g.matches(rel, name)) || search(path, &shown(given, rel))
        })
    } else {
        return Err(format!("{}: no such file or directory", root.display()));
    };
    out.push_str(&format!("---\nTotal matches: {found}"));
    if literal {
        out.push_str("\n(not a regular expression, so searched for as plain text)");
    }
    out.push_str(match stop {
        Stop::Done => "",
        Stop::Limit => "\n(stopped at the limit: narrow the search or raise limit)",
        Stop::Budget => "\n(stopped early, too much to look through: search a smaller directory)",
    });
    Ok(out)
}

/// get_info: what the model needs to know before it names a path.
fn get_info(cwd: &Path) -> Value {
    let name = fs::read_to_string("/etc/os-release").ok().and_then(|t| t.lines().find_map(|l| Some(l.strip_prefix("PRETTY_NAME=")?.trim_matches('"').to_string())));
    let os = match name {
        Some(name) => format!("{name} ({})", std::env::consts::OS),
        None => std::env::consts::OS.to_string(),
    };
    json!({"os": os, "arch": std::env::consts::ARCH, "cwd": cwd.display().to_string(), "home": home().display().to_string()})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(p: &str, rel: &str) -> bool {
        Glob::new(p).matches(rel, rel.rsplit('/').next().unwrap())
    }

    #[test]
    fn globs() {
        assert!(m("*.rs", "src/main.rs") && !m("*.rs", "src/main.rsx")); // no "/": a name at any depth
        assert!(m("src/**/*.rs", "src/a/b/c.rs") && m("src/**/*.rs", "src/c.rs")); // ** may be no directory
        assert!(!m("src/*.rs", "src/a/c.rs")); // * stays within a name
        assert!(m("**", "a/b/c") && m("**/c", "c") && m("./src/*", "src/x"));
        assert!(m("*.{rs,toml}", "Cargo.toml") && m("{a,b{c,d}}", "bd") && !m("*.{rs,toml}", "x.md"));
        assert!(m("*[sS][rR][cC]*", "my-Src-dir")); // the UI's case-blind pattern
        assert!(m("[!a]?", "bc") && !m("[!a]?", "ac") && m("[a-c]x", "bx") && !m("[a-c]x", "dx"));
        assert!(m("[]]", "]") && m("a[", "a[") && m("\\*", "*") && !m("\\*", "x"));
        assert!(Glob::new(".github/**").hidden && Glob::new("**/.env").hidden && !Glob::new("**/*.rs").hidden);
    }

    #[test]
    fn callers() {
        let lo: IpAddr = "127.0.0.1".parse().unwrap();
        assert!(caller_ok(lo, 8080, Some("127.0.0.1:8080"), Some("http://127.0.0.1:8080")).is_ok());
        assert!(caller_ok(lo, 8080, Some("localhost:8080"), None).is_ok()); // curl
        assert!(caller_ok("::1".parse().unwrap(), 8080, Some("[::1]:8080"), Some("http://[::1]:8080")).is_ok());
        assert!(caller_ok(lo, 80, Some("localhost"), Some("http://localhost")).is_ok());
        assert!(caller_ok(lo, 8080, Some("127.0.0.1:8080"), Some("http://evil.example")).is_err()); // another site's page
        assert!(caller_ok(lo, 8080, Some("evil.example:8080"), Some("http://evil.example:8080")).is_err()); // DNS rebinding
        assert!(caller_ok(lo, 8080, Some("127.0.0.1:8080"), Some("null")).is_err());
        assert!(caller_ok(lo, 8080, None, None).is_err());
        assert!(caller_ok(lo, 8080, Some("127.0.0.1:9090"), None).is_err());
        assert!(caller_ok("192.168.1.5".parse().unwrap(), 8080, Some("192.168.1.5:8080"), None).is_err()); // another machine
    }

    #[test]
    fn files() {
        let d = std::env::temp_dir().join(format!("ik-tools-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        let cwd = d.to_str().unwrap().to_string();
        let call = |t: &str, p: Value| super::call(t, &p, Some(&cwd), false).unwrap();
        let five: String = (1..=5).map(|i| format!("line {i}\n")).collect();
        assert_eq!(call("write_file", json!({"path": "a/b.txt", "content": five}))["bytes"], 35);
        assert_eq!(call("read_file", json!({"path": "a/b.txt", "start_line": 2, "end_line": 3}))["plain_text_response"], "line 2\nline 3\n");
        assert_eq!(call("read_file", json!({"path": "a/b.txt", "start_line": 9}))["error"], "a/b.txt has 5 lines");
        // all edits or none: the second occurs 4 times once the first is made
        let r = call("edit_file", json!({"path": "a/b.txt", "edits": [{"old_text": "line 2", "new_text": "two"}, {"old_text": "line", "new_text": "x"}]}));
        assert!(r["error"].as_str().unwrap().starts_with("edit 2: old_text occurs 4 times"), "{r}");
        assert_eq!(fs::read_to_string(d.join("a/b.txt")).unwrap(), five);
        let r = call("edit_file", json!({"path": "a/b.txt", "edits": [{"old_text": "line 2\nline 3", "new_text": "two\nthree"}]}));
        assert_eq!(r["edits_applied"], 1, "{r}");
        // a file with \r\n, edited with \n
        call("write_file", json!({"path": "crlf.txt", "content": "a\r\nb\r\n"}));
        assert_eq!(call("edit_file", json!({"path": "crlf.txt", "edits": [{"old_text": "a\nb", "new_text": "c\nd"}]}))["edits_applied"], 1);
        assert_eq!(fs::read_to_string(d.join("crlf.txt")).unwrap(), "c\r\nd\r\n");
        // the walk leaves out hidden directories and build caches, unless the pattern names a hidden one
        call("write_file", json!({"path": ".hid/c.txt", "content": "line 4\n"}));
        call("write_file", json!({"path": "target/CACHEDIR.TAG", "content": ""}));
        call("write_file", json!({"path": "target/d.txt", "content": "line 4\n"}));
        // a FIFO is neither read nor searched (opening one waits for a writer)
        let fifo = std::ffi::CString::new(d.join("fifo").to_str().unwrap()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        assert_eq!(call("read_file", json!({"path": "fifo"}))["error"], "fifo is not a regular file");
        let r = call("file_glob_search", json!({"include": "*.txt"}));
        assert_eq!(r["entries"], json!([{"path": "crlf.txt", "type": "file"}, {"path": "a/b.txt", "type": "file"}]), "{r}");
        assert_eq!(r["plain_text_response"], "crlf.txt\na/b.txt\n---\nTotal matches: 2");
        assert_eq!(r["base"], cwd);
        let r = call("file_glob_search", json!({"path": d.join("a").to_str().unwrap(), "type": "dir", "include": "**"}));
        assert_eq!(r["entries"], json!([]));
        assert_eq!(call("file_glob_search", json!({"include": ".hid/*"}))["entries"], json!([{"path": ".hid/c.txt", "type": "file"}]));
        assert_eq!(call("grep_search", json!({"pattern": "line [45]", "path": "."}))["plain_text_response"], "a/b.txt:4:line 4\na/b.txt:5:line 5\n---\nTotal matches: 2");
        // the look at a whole file first, (?m); a look past a line's end, a text anchor, (?-m), $ and \r\n: lines only
        for pat in ["^line 5$", "5(?!\\s)", "\\Aline 5\\z", "(?-m)5$"] {
            assert_eq!(call("grep_search", json!({"pattern": pat, "path": "a/b.txt"}))["plain_text_response"], "a/b.txt:5:line 5\n---\nTotal matches: 1", "{pat}");
        }
        for pat in ["d$", "^d"] {
            assert_eq!(call("grep_search", json!({"pattern": pat, "path": "crlf.txt"}))["plain_text_response"], "crlf.txt:2:d\n---\nTotal matches: 1", "{pat}");
        }
        let r = call("grep_search", json!({"pattern": "LINE (4", "ignore_case": true, "include": "b.*"}));
        assert_eq!(r["plain_text_response"], "---\nTotal matches: 0\n(not a regular expression, so searched for as plain text)");
        assert_eq!(super::call("read_file", &json!({"path": "crlf.txt"}), Some(&cwd), true).unwrap(), json!({"base64": "Yw0KZA0K", "size_bytes": 6}));
        fs::remove_dir_all(&d).unwrap();
    }
}
