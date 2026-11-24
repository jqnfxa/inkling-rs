//! Context compaction inside one answer, after Pi (docs.pi.dev, compaction): when the thinking fills the context, the
//! model summarizes its reasoning on the cache it has, and the sequence is rebuilt as the prompt, the start of the
//! thinking, the summary, and the last part of the thinking kept verbatim. The pieces that need no engine live here.

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
