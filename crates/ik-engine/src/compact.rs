//! Summaries of the thinking. Compaction inside one answer, after Pi (docs.pi.dev, compaction): when the thinking fills
//! the context, the model summarizes its reasoning on the cache it has, and the sequence is rebuilt as the prompt, the
//! start of the thinking, the summary, and the last part of the thinking kept verbatim. Progress notes, as chat apps
//! show while a model thinks: now and then the model writes a few lines on what it is doing, on a branch of the cache
//! that is dropped after. Compaction between turns, after Pi too: when a chat no longer fits, its older messages are
//! summarized (serialized as plain text, without the thinking) and replaced by one system message. The pieces that
//! need no engine live here.

use crate::chat::{CONTENT_TEXT, END_MESSAGE, MESSAGE_MODEL, MESSAGE_SYSTEM, MESSAGE_USER};

#[derive(Debug, Clone)]
pub struct Settings {
    pub enabled: bool,
    /// compact when the sequence comes within this many tokens of the context size
    pub reserve: usize,
    /// tokens of the latest thinking kept verbatim, at most
    pub keep: usize,
    /// compactions in one answer, at most; then the thinking is closed when it reaches the reserve
    pub max: usize,
    /// tokens of a summary, at most
    pub summary_tokens: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { enabled: true, reserve: 16384, keep: 20000, max: 8, summary_tokens: 3072 }
    }
}

/// A message as the summarizer of a conversation reads it: `[User]: ...`, `[Assistant]: ...`, `[Assistant tool calls]:
/// ...`, `[Tool result]: ...` (cut to 2000 characters); the thinking is left out.
pub fn serialize(m: &serde_json::Value) -> String {
    use serde_json::Value;
    let role = m.get("role").and_then(Value::as_str).unwrap_or("");
    let text = match m.get("content") {
        Some(Value::String(c)) => c.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.as_str().map(String::from).or_else(|| p.get("text").and_then(Value::as_str).map(String::from)))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    let mut out = vec![];
    match role {
        "tool" => {
            let t: String = text.chars().take(2000).collect();
            let more = if text.chars().count() > 2000 { " [...]" } else { "" };
            out.push(format!("[Tool result]: {t}{more}"));
        }
        "user" => out.push(format!("[User]: {text}")),
        "system" => out.push(format!("[System]: {text}")),
        _ => {
            if !text.is_empty() {
                out.push(format!("[Assistant]: {text}"));
            }
            let calls: Vec<String> = m
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|c| {
                    let f = c.get("function").unwrap_or(c);
                    let args = match f.get("arguments") {
                        Some(Value::String(a)) => a.clone(),
                        Some(a) => a.to_string(),
                        None => "{}".into(),
                    };
                    format!("{}({args})", f.get("name").and_then(Value::as_str).unwrap_or("?"))
                })
                .collect();
            if !calls.is_empty() {
                out.push(format!("[Assistant tool calls]: {}", calls.join("; ")));
            }
        }
    }
    out.join("\n")
}

/// What identifies a message for the cache of summaries: what a chat UI sends again unchanged (the thinking, which
/// some send and some do not, is left out).
pub fn message_key(m: &serde_json::Value) -> String {
    let g = |k: &str| m.get(k).map(|v| v.to_string()).unwrap_or_default();
    format!("{}\0{}\0{}\0{}\0{}", g("role"), g("content"), g("tool_calls"), g("name"), g("tool_call_id"))
}

/// Where the kept messages start: the last ones that fit in `keep` tokens (the last message always), from `start` on,
/// never with a tool result first (its call goes with it). None if nothing before it would be summarized.
pub fn cut_point(sizes: &[usize], roles: &[&str], start: usize, keep: usize) -> Option<usize> {
    let n = sizes.len();
    if n == 0 {
        return None;
    }
    let (mut cut, mut acc) = (n, 0);
    for i in (start..n).rev() {
        if acc + sizes[i] > keep && i < n - 1 {
            break;
        }
        acc += sizes[i];
        cut = i;
    }
    while cut > start && cut < n && roles[cut] == "tool" {
        cut -= 1;
    }
    (cut > start).then_some(cut)
}

/// The summarizer's request for the conversation `conversation` (serialized), updating `previous` if there is one:
/// system and user messages.
pub fn history_request(conversation: &str, previous: Option<&str>, words: usize) -> Vec<serde_json::Value> {
    let system = "You summarize a conversation between a user and an AI assistant, so that the assistant can continue it from your summary and the latest messages alone: the older messages will be gone. You do not take part in the conversation: do not answer or continue anything in it.";
    let ask = match previous {
        Some(p) => format!("<previous-summary>\n{p}\n</previous-summary>\n\nThe conversation above comes after what the previous summary covers. Write an updated summary that replaces it: keep what still holds, add what is new, drop what was settled otherwise."),
        None => "Write the summary of the conversation above.".into(),
    };
    let user = format!(
        "<conversation>\n{conversation}\n</conversation>\n\n{ask}\n\nUse these sections:\n\n## Goal\nWhat the user is trying to do.\n\n## Constraints and preferences\nRequirements and preferences the user stated.\n\n## Progress\n### Done\n### In progress\n### Blocked\n\n## Key decisions\nWhat was decided, and why.\n\n## Next steps\nWhat comes next, in order.\n\n## Critical context\nFacts, names, numbers, file paths, code and errors the assistant needs exactly as they were.\n\nKeep names, numbers and code exact. {words} words at most."
    );
    vec![serde_json::json!({"role": "system", "content": system}), serde_json::json!({"role": "user", "content": user})]
}

/// The system message that stands for the summarized messages.
pub fn history_message(summary: &str) -> serde_json::Value {
    serde_json::json!({"role": "system", "content": format!("Summary of the earlier conversation (its older messages were compacted to save context):\n\n{summary}")})
}

/// Progress notes on the thinking: off, between the raw thinking (inline), or in its place (only).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoteMode {
    Off,
    Inline,
    Only,
}

#[derive(Debug, Clone)]
pub struct Notes {
    pub mode: NoteMode,
    /// a note every this many tokens of thinking (the first after a quarter of it)
    pub every: usize,
}

impl Default for Notes {
    fn default() -> Self {
        Notes { mode: NoteMode::Off, every: 1024 }
    }
}

impl NoteMode {
    /// `reasoning_summary` as a request or a flag gives it: off/none, inline, only (auto, concise and detailed, as in
    /// OpenAI's API, are only), or a boolean.
    pub fn parse(v: &serde_json::Value) -> Option<NoteMode> {
        match v {
            serde_json::Value::Bool(b) => Some(if *b { NoteMode::Only } else { NoteMode::Off }),
            serde_json::Value::String(s) => match s.as_str() {
                "off" | "none" | "" => Some(NoteMode::Off),
                "inline" => Some(NoteMode::Inline),
                "only" | "auto" | "concise" | "detailed" => Some(NoteMode::Only),
                _ => None,
            },
            _ => None,
        }
    }
}

/// How a note starts: its answer is begun with it, so that it opens with a bold title.
pub const NOTE_START: &str = "**";

/// The request for a progress note, after the thinking so far. It does not show the last note (shown one, or its
/// title, the model copied it), nor call it a note (the model made "Progress note" its title).
pub fn note_request(system: bool) -> String {
    let role = if system { MESSAGE_SYSTEM } else { MESSAGE_USER };
    format!(
        "{END_MESSAGE}{role}{CONTENT_TEXT}Pause for a moment. For the user, write a bold title of three to six words that names what you are doing in the latest part of your reasoning, then on the next line one or two plain sentences about it in the first person, 50 words at most. No formulas or code; do not solve anything here.{END_MESSAGE}{MESSAGE_MODEL}{CONTENT_TEXT}{NOTE_START}"
    )
}

/// A note as the model wrote it, without a label it may put first ("Progress note:").
pub fn clean_note(text: &str) -> String {
    let t = text.trim();
    let marker: String = t.chars().take_while(|&c| c == '*' || c == '_').collect();
    let bare = &t[marker.len()..];
    if bare.get(..13).is_some_and(|p| p.eq_ignore_ascii_case("progress note")) {
        let mut rest = &bare[13..];
        rest = rest.strip_prefix(':').unwrap_or(rest);
        rest = rest.strip_prefix(marker.as_str()).unwrap_or(rest);
        rest = rest.strip_prefix(':').unwrap_or(rest);
        return rest.trim_start().to_string();
    }
    t.to_string()
}

/// A note shown between the raw thinking: a quote block.
pub fn quote(note: &str) -> String {
    note.lines().map(|l| if l.trim().is_empty() { ">".to_string() } else { format!("> {}", l.trim_end()) }).collect::<Vec<_>>().join("\n")
}

/// What the summary is put between, in the rebuilt thinking.
pub const SUMMARY_HEAD: &str = "[Summary of my earlier reasoning, which was compacted to save context]\n";
pub const SUMMARY_TAIL: &str = "\n[End of summary. My latest reasoning follows verbatim.]\n";

const SECTIONS: &str = "Write it for yourself, dense and exact, in these sections:

## Task
The problem restated in full: what is given and what to find, the exact input and output format, all limits (sizes, value ranges, time and memory), and the samples with their expected outputs.

## Constraints
Every condition a solution must meet, including any the user stated.

## Established facts
Observations, lemmas and results already proven or checked, with their formulas, numbers and sample traces exactly as found.

## Approaches tried
Each approach that was dropped, and exactly why it fails (a counterexample, its complexity, a wrong answer on a sample).

## Current approach
The algorithm being pursued: the idea, the steps, the data structures, the complexity, and how far it has been checked.

## Open questions and next steps
What is still unknown or unverified, and what to do next, in order.

## Code
The code drafted so far; its key parts verbatim. Leave this section out if there is none.

Keep every formula, number, index convention, edge case and sample trace exactly: do not round, paraphrase or drop them. Do not continue solving; write only the summary.";

/// The request for a summary of at most about `words` words. `again`: the thinking already begins with an earlier
/// summary, to be updated.
pub fn instruction(again: bool, words: usize) -> String {
    let lead = "The context is almost full. Your reasoning so far will be replaced by a summary that you write now: you will continue from that summary and from the last part of your reasoning only, and everything else in your reasoning will be lost.";
    let update = if again {
        "\n\nYour reasoning above begins with a summary of still earlier reasoning. Write an updated summary that replaces it: keep what is still true and relevant from it, add what was found since, and drop what was disproved. Under Approaches tried, add every idea explored since that summary, even one that went nowhere, with where it got stuck, so that you do not explore it again."
    } else {
        ""
    };
    format!("{lead}{update}\n\n{SECTIONS} Be brief where you can: {words} words at most.")
}

/// The text appended after the thinking to ask for the summary, and to start the model's answer as plain text.
/// `system`: ask in a system message rather than a user message.
pub fn request(again: bool, system: bool, words: usize) -> String {
    let role = if system { MESSAGE_SYSTEM } else { MESSAGE_USER };
    format!("{END_MESSAGE}{role}{CONTENT_TEXT}{}{END_MESSAGE}{MESSAGE_MODEL}{CONTENT_TEXT}", instruction(again, words))
}

/// The lines of a summary as it is written, to drop a line that repeats an earlier one (the model looping).
#[derive(Default)]
pub struct Lines {
    seen: std::collections::HashSet<String>,
    line: String,
    start: usize,
    /// repeated lines dropped so far
    pub repeats: usize,
}

impl Lines {
    /// Takes the text of the summary's token number `at`. When it ends a line of 24 characters or more (list markers
    /// aside) that was written before, returns where that line's tokens begin: the summary is to be cut there.
    pub fn push(&mut self, at: usize, text: &str) -> Option<usize> {
        let Some(nl) = text.find('\n') else {
            self.line += text;
            return None;
        };
        self.line += &text[..nl];
        let line = std::mem::replace(&mut self.line, text[text.rfind('\n').unwrap() + 1..].to_string());
        let key = line.trim().trim_start_matches(|c: char| c.is_ascii_digit() || ".)-*# ".contains(c)).trim_end();
        if key.chars().count() >= 24 && !self.seen.insert(key.to_string()) {
            self.repeats += 1;
            self.line.clear();
            return Some(self.start);
        }
        self.start = at + 1;
        None
    }
}

/// Where to start the kept tail of a thinking whose tokens have the texts `pieces`: at most `keep` tokens from the end,
/// at a paragraph start if one is in the first quarter of that window, else at a line start there, else exactly
/// `keep` tokens from the end. Returns the index of the first kept token.
pub fn cut(pieces: &[Vec<u8>], keep: usize) -> usize {
    let n = pieces.len();
    if keep >= n {
        return 0;
    }
    if keep == 0 {
        return n;
    }
    let lo = n - keep;
    let hi = lo + keep / 4;
    // the last two bytes of the text before each token in lo..=hi
    let mut tail: [u8; 2] = [0, 0];
    let mut ends: Vec<[u8; 2]> = Vec::with_capacity(hi - lo + 1);
    for (i, p) in pieces.iter().enumerate().take(hi + 1) {
        if i >= lo {
            ends.push(tail);
        }
        match p.len() {
            0 => {}
            1 => tail = [tail[1], p[0]],
            l => tail = [p[l - 2], p[l - 1]],
        }
    }
    if let Some(i) = ends.iter().position(|e| e == b"\n\n") {
        return lo + i;
    }
    if let Some(i) = ends.iter().position(|e| e[1] == b'\n') {
        return lo + i;
    }
    lo
}

/// The tokens after the prompt once compacted: `open` (the model's message up to its thinking tag), `head`, the
/// summary, `tail`, then the kept end of `body` (the thinking since the last summary, with token texts `pieces`), at
/// most `keep` tokens. Returns them and how many tokens of `body` were kept.
pub fn rebuild(open: &[u32], head: &[u32], summary: &[u32], tail: &[u32], body: &[u32], pieces: &[Vec<u8>], keep: usize) -> (Vec<u32>, usize) {
    let from = cut(pieces, keep);
    let kept = &body[from..];
    let mut out = Vec::with_capacity(open.len() + head.len() + summary.len() + tail.len() + kept.len());
    for part in [open, head, summary, tail, kept] {
        out.extend_from_slice(part);
    }
    (out, kept.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(s: &[&str]) -> Vec<Vec<u8>> {
        s.iter().map(|p| p.as_bytes().to_vec()).collect()
    }

    #[test]
    fn cuts_at_paragraph_then_line() {
        // 12 tokens, keep 8: window starts at 4, first quarter 4..=6
        let p = pieces(&["a", "b", "c", "d", "e\n", "f", "g\n", "\n", "h", "i", "j", "k"]);
        assert_eq!(cut(&p, 8), 5); // a line starts at 5 (after "e\n"); no paragraph in 4..=6
        let p = pieces(&["a", "b", "c", "d", "e\n", "\n", "g", "h", "i", "j", "k", "l"]);
        assert_eq!(cut(&p, 8), 6); // the paragraph start at 6 wins over the line start at 5
        let p = pieces(&["a", "b", ".\n\n", "d", "e", "f", "g", "h", "i", "j", "k", "l"]);
        assert_eq!(cut(&p, 8), 4); // a newline before the window does not count
        let p = pieces(&["a", "b", "c", "\n", "e", "f", "g", "h", "i", "j", "k", "l"]);
        assert_eq!(cut(&p, 8), 4); // the line start right at the window's start: the text before 4 ends in "\n"
        // a "\n\n" split between tokens
        let p = pieces(&["a", "b", "c", "d", "x\n", "y", "\n", "\n", "i", "j", "k", "l"]);
        assert_eq!(cut(&p, 8), 5);
        let p = pieces(&["a", "b", "c", "d", "e", "\n", "\n", "h", "i", "j", "k", "l"]);
        assert_eq!(cut(&p, 8), 6); // after "e","\n": a line start at 6; "\n","\n" makes a paragraph at 7, past the quarter
        assert_eq!(cut(&p, 20), 0);
        assert_eq!(cut(&p, 0), 12);
    }

    #[test]
    fn drops_repeated_lines() {
        let mut l = Lines::default();
        let toks = ["  5. Better: keep a Fenwick", " tree over f", " for the sums\n", "  6. Better: keep a Fenwick", " tree over f", " for the sums\n", "## Next\n"];
        let mut summary: Vec<usize> = vec![];
        for t in toks {
            summary.push(summary.len());
            if let Some(cut) = l.push(summary.len() - 1, t) {
                summary.truncate(cut);
            }
        }
        // the second line went; "## Next" is short, and kept
        assert_eq!(summary, vec![0, 1, 2, 3]);
        assert_eq!(l.repeats, 1);
        let mut l = Lines::default();
        assert_eq!(l.push(0, "ok\nok\n"), None);
        assert_eq!(l.push(1, "ok\n"), None); // short lines repeat freely
    }

    #[test]
    fn history() {
        use serde_json::json;
        // sizes of: user, assistant (tool call), tool, assistant, user
        let roles = ["user", "assistant", "tool", "assistant", "user"];
        assert_eq!(cut_point(&[100, 50, 300, 80, 40], &roles, 0, 200), Some(3)); // 40 + 80 fit, 300 does not
        assert_eq!(cut_point(&[100, 50, 300, 80, 40], &roles, 0, 500), Some(1)); // the tool result would start: its call too
        assert_eq!(cut_point(&[100, 50, 300, 80, 40], &roles, 0, 10), Some(4)); // the last message always
        assert_eq!(cut_point(&[100, 50, 300, 80, 40], &roles, 0, 1000), None); // all fits: nothing to summarize
        assert_eq!(cut_point(&[100, 50, 300, 80, 40], &roles, 3, 10), Some(4));
        assert_eq!(cut_point(&[100, 50, 300, 80, 40], &roles, 4, 10), None);
        let a = json!({"role": "assistant", "content": "", "reasoning_content": "hmm", "tool_calls": [{"function": {"name": "ls", "arguments": "{\"d\":\"/\"}"}}]});
        assert_eq!(serialize(&a), "[Assistant tool calls]: ls({\"d\":\"/\"})");
        let t = json!({"role": "tool", "content": "x".repeat(2500)});
        assert_eq!(serialize(&t).len(), "[Tool result]: ".len() + 2000 + " [...]".len());
        assert_eq!(serialize(&json!({"role": "user", "content": [{"type": "text", "text": "hi"}]})), "[User]: hi");
        assert_eq!(message_key(&a), message_key(&json!({"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "ls", "arguments": "{\"d\":\"/\"}"}}]})));
        let r = history_request("[User]: hi", Some("old"), 300);
        assert!(r[1]["content"].as_str().unwrap().contains("<previous-summary>\nold\n</previous-summary>"));
    }

    #[test]
    fn notes() {
        let r = note_request(false);
        assert!(r.starts_with("<|end_message|><|message_user|><|content_text|>Pause"));
        assert!(r.ends_with("<|end_message|><|message_model|><|content_text|>**"));
        assert_eq!(clean_note("**Progress note:** I'm analyzing."), "I'm analyzing.");
        assert_eq!(clean_note("Progress note: **Analyzing it**\n\nText"), "**Analyzing it**\n\nText");
        assert_eq!(clean_note("**Analyzing it**"), "**Analyzing it**");
        assert_eq!(clean_note("**Progress note**: x"), "x");
        assert_eq!(clean_note("Прогресс: да"), "Прогресс: да");
        assert!(note_request(true).contains("<|message_system|>"));
        assert_eq!(quote("**T**\n\nText."), "> **T**\n>\n> Text.");
        assert_eq!(NoteMode::parse(&serde_json::json!("auto")), Some(NoteMode::Only));
        assert_eq!(NoteMode::parse(&serde_json::json!(false)), Some(NoteMode::Off));
        assert_eq!(NoteMode::parse(&serde_json::json!("bogus")), None);
    }

    #[test]
    fn rebuilds_in_order() {
        let body: Vec<u32> = (100..112).collect();
        let p = pieces(&["a", "b", "c", "d", "e\n", "f", "g", "h", "i", "j", "k", "l"]);
        let (out, kept) = rebuild(&[1, 2], &[3], &[4, 5, 6], &[7], &body, &p, 8);
        assert_eq!(kept, 7);
        assert_eq!(out, vec![1, 2, 3, 4, 5, 6, 7, 105, 106, 107, 108, 109, 110, 111]);
        let (out, kept) = rebuild(&[1], &[3], &[4], &[7], &body, &p, 0);
        assert_eq!((out, kept), (vec![1, 3, 4, 7], 0));
    }

    #[test]
    fn request_closes_the_thinking_and_opens_a_text_answer() {
        let r = request(false, false, 500);
        assert!(r.starts_with("<|end_message|><|message_user|><|content_text|>The context is almost full."));
        assert!(r.ends_with("only the summary. Be brief where you can: 500 words at most.<|end_message|><|message_model|><|content_text|>"));
        assert!(request(true, true, 500).starts_with("<|end_message|><|message_system|><|content_text|>"));
        assert!(instruction(true, 500).contains("updated summary"));
    }
}
