#!/usr/bin/env python3
"""End-to-end check of compaction and prompt caching in an agentic turn, against a running inkling-server.

    python3 tools/agentic-check.py [--url http://127.0.0.1:8080] [--until 90000] [--max-calls 80] [--threshold 81920]

One turn as an agentic client drives it: system, user (a hard problem), then model calls that each end in a tool
call, the tool's short JSON result, and the whole history sent again (with each reply's reasoning_content, tools,
reasoning_effort "max", stream, timings_per_token), until the replies' tokens add up to --until (past the compaction
threshold) or the model answers. It reports, and checks:

  (a) at least one compaction (timings.compactions), and no thinking closed at the reserve (compaction stage "closed");
  (b) on every call after the first, timings.cache_n >= 95% of the prompt; and once, when the sequence is within 16k
      tokens of --threshold, the tool returns a long one-line result (stdout as escaped JSON) that pushes the prompt
      over it: that call must reuse the whole cache (cache_n >= 95% of the previous prompt and reply), not feed it again;
  (c) a history with a malformed tool call in it is accepted (no HTTP 400).

Standard library only. Takes long: about 80k tokens of generation at ~45 t/s is half an hour or more.
"""
import argparse, json, sys, time, urllib.request, urllib.error

SYSTEM = (
    "You are an expert competitive programmer working on a hard problem. You cannot run code yourself; the tool "
    "check_idea tells you whether an idea survives the judge's hidden tests. Think each idea through in depth "
    "(derive, prove, trace the samples by hand, work out the complexity), then call check_idea with it. Keep "
    "refining and calling check_idea until it answers verdict \"accepted\"; only then write the final solution."
)
PROBLEM = (
    "Problem. You are given n <= 200000 points on a circle, labelled 1..n clockwise, and m <= 200000 chords between "
    "them, no two sharing an endpoint. A set of chords is good if no two of them cross and every point of the circle "
    "is covered by the arc under at most k chords of the set (the arc under chord (a, b) is the clockwise arc from "
    "a to b). For each k from 1 to n, output the maximum size of a good set. Prove the correctness of your algorithm "
    "and its O((n + m) log n) bound before you submit it."
)
TOOLS = [{
    "type": "function",
    "function": {
        "name": "check_idea",
        "description": "Checks a solution idea against the hidden tests. Returns a short JSON verdict.",
        "parameters": {
            "type": "object",
            "properties": {
                "idea": {"type": "string", "description": "the algorithm, in a few sentences"},
                "complexity": {"type": "string", "description": "its time and memory bound"},
            },
            "required": ["idea", "complexity"],
        },
    },
}]


def check_idea(call_no, args):
    # never accepted: the turn goes on as long as the check runs
    return json.dumps({"verdict": "wrong answer", "test": 7 + call_no, "note": "fails on a hidden test; find the flaw and refine"})


def long_result(call_no, tokens):
    # a program's stdout of about `tokens` tokens or more (digits tokenize at under 4 characters a token), one JSON line
    lines, n = [], 0
    while n < 4 * tokens:
        i = len(lines)
        lines.append(f"case {i}: n={(i * 7919) % 100003} m={(i * 104729) % 65537} got {(i * 31) % 97} expected {(i * 37) % 97}")
        n += len(lines[-1]) + 1
    return json.dumps({"verdict": "wrong answer", "test": 7 + call_no, "stdout": "\n".join(lines)})


def stream(url, body):
    """POSTs a streaming chat request; returns (reply message, last timings, compaction stats, seconds)."""
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    reasoning, content, calls, timings, closed, finish = [], [], {}, {}, 0, None
    with urllib.request.urlopen(req, timeout=24 * 3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            c = json.loads(data)
            if "error" in c:
                raise RuntimeError(c["error"])
            if c.get("compaction", {}).get("stage") == "closed":
                closed += 1
            if c.get("timings"):
                timings = c["timings"]
            for ch in c.get("choices", []):
                d = ch.get("delta", {})
                finish = ch.get("finish_reason") or finish
                if d.get("reasoning_content"):
                    reasoning.append(d["reasoning_content"])
                if d.get("content"):
                    content.append(d["content"])
                for tc in d.get("tool_calls", []) or []:
                    e = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    e["id"] = tc.get("id") or e["id"]
                    f = tc.get("function", {})
                    e["name"] += f.get("name") or ""
                    e["arguments"] += f.get("arguments") or ""
    msg = {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}
    if calls:
        msg["tool_calls"] = [{"id": e["id"], "type": "function", "function": {"name": e["name"], "arguments": e["arguments"]}} for _, e in sorted(calls.items())]
    return msg, timings, closed, finish, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--until", type=int, default=90000, help="stop once the replies' tokens add up to this")
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--threshold", type=int, default=81920, help="the server's n_ctx - compaction reserve")
    a = ap.parse_args()

    ok = True
    # (c) first, as it is quick: a malformed tool call (and repeated ids) in the history must not fail the request
    hist = [
        {"role": "user", "content": "Say hi."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "check_idea", "arguments": "{\"name\":\"check_idea\",\"args\":{\"idea\": \"unterminated"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"error\": \"bad arguments\"}"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "check_idea", "arguments": "{\"idea\": \"x"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"error\": \"bad arguments\"}"},
    ]
    try:
        stream(a.url, {"messages": hist, "tools": TOOLS, "stream": True, "max_tokens": 8, "reasoning_effort": "none"})
        print("(c) malformed tool call in the history: accepted")
    except urllib.error.HTTPError as e:
        print(f"(c) malformed tool call in the history: HTTP {e.code} {e.read()[:200]!r}")
        ok = False

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROBLEM}]
    total_gen, compactions, closed_all, low_cache = 0, 0, 0, []
    # the crossing: the call whose prompt a long tool result pushes over the threshold, and the sequence before it
    crossing, crossed, seq = None, None, 0
    print(f"{'call':>4} {'prompt':>7} {'cache_n':>7} {'cached':>7} {'gen':>6} {'total gen':>9} {'compact':>7} {'s':>6}")
    for call in range(1, a.max_calls + 1):
        body = {"messages": msgs, "tools": TOOLS, "stream": True, "reasoning_effort": "max", "timings_per_token": True}
        try:
            msg, t, closed, finish, secs = stream(a.url, body)
        except urllib.error.HTTPError as e:
            print(f"call {call}: HTTP {e.code} {e.read()[:300]!r}")
            ok = False
            break
        prompt = t.get("cache_n", 0) + t.get("prompt_n", 0)
        cached = t.get("cache_n", 0) / prompt if prompt else 0.0
        gen = t.get("predicted_n", 0)
        total_gen += gen
        compactions += t.get("compactions", 0)
        closed_all += closed
        if call == crossing:
            # the prompt is the cache plus the long result: the whole cache reused, the result fed
            crossed = t.get("cache_n", 0) / seq if seq else 0.0
            if crossed < 0.95:
                low_cache.append((call, crossed))
        elif call > 1 and cached < 0.95:
            low_cache.append((call, cached))
        print(f"{call:>4} {prompt:>7} {t.get('cache_n', 0):>7} {cached:>6.1%} {gen:>6} {total_gen:>9} {t.get('compactions', 0):>7} {secs:>6.0f}"
              + (f"  closed at the reserve x{closed}" if closed else "")
              + (f"  over the threshold by a tool result: {crossed:.1%} of the cache ({seq}) reused" if call == crossing else ""))
        seq = prompt + gen if not t.get("compactions") else 0
        msgs.append(msg)
        if not msg.get("tool_calls"):
            print(f"the model answered (finish {finish}) after {call} calls")
            break
        need = a.threshold - seq + 1000
        long = crossing is None and seq and need <= 16000
        for i, tc in enumerate(msg["tool_calls"]):
            content = long_result(call, need) if long and i == 0 else check_idea(call, tc["function"]["arguments"])
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
        if long:
            crossing = call + 1
        if total_gen >= a.until:
            break

    print()
    a_ok = compactions >= 1 and closed_all == 0
    print(f"(a) compactions {compactions}, thinking closed at the reserve {closed_all}: {'ok' if a_ok else 'FAIL'}"
          + ("" if total_gen >= a.until else f" (the turn ended at {total_gen} tokens, before --until {a.until})"))
    print(f"(b) calls after the first with cache_n < 95% of the prompt: {low_cache or 'none'}: {'ok' if not low_cache else 'FAIL'}"
          + ("" if crossed is not None else " (no call crossed the threshold by a tool result: FAIL)"))
    ok = ok and a_ok and not low_cache and crossed is not None
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
