# inkling-rs

An inference engine for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), as the official
GGUF ([unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)) holds it, on one laptop:
RTX 3070 Ti Laptop (8 GB) + i7-12700H + 64 GB DDR4-3200. Written from scratch in Rust; the GPU kernels are written in
[kernite](https://github.com/fxeqxmulfx/kernite). It serves an OpenAI-compatible API and llama.cpp's chat UI.

The model is a mixture of experts (128 experts, 8 per token) whose middle layers run three times per token; at
UD-Q4_K_XL it is 17.6 GB, so it does not fit in 8 GB of VRAM. The speeds below were measured before the engine moved
to the official GGUF and are to be measured again.

## Requirements

- Linux on x86-64 with AVX2, FMA and F16C.
- An NVIDIA GPU with 8 GB and a driver for CUDA 12 (tested: sm_86, driver 550). The kernels are PTX for sm_70 and
  later, built with the crate; no CUDA toolkit is needed.
- 32 GB of RAM or more: the experts (15.8 GB) stay in page-locked host memory.
- Rust through rustup: `rust-toolchain.toml` pins 1.100.0, which kernite needs.
- The model file `Inkling-2-Mini-UD-Q4_K_XL.gguf` (17.6 GB) from unsloth's `Mainline_Support` folder: NVFP4 experts
  with a scale each in layers 0-19, Q8_0 experts in 20-23, Q8_0 and float small matrices. Other quantizations are not
  supported.

## Run

```sh
hf download unsloth/Inkling-2-Mini-GGUF Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file --local-dir /tmp/inkling
mv /tmp/inkling/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file ~/models/Inkling-2-Mini-UD-Q4_K_XL.gguf
tools/fetch-ui.sh        # llama.cpp's chat UI, into ./ui (optional)
./run-inkling-rs.sh      # builds, then serves on http://127.0.0.1:8080
```

Open http://127.0.0.1:8080 to chat. `run-inkling-rs.sh` reads `MODEL`, `CTX` (131072), `PORT`, `HOST` and `IK_THREADS`
(the CPU expert threads: a count, or a list of CPUs to pin).

## API

`/v1/chat/completions` (streaming, `reasoning_content`, tool calls), `/v1/completions`, `/v1/models`, `/health`, and the
llama-server endpoints the chat UI reads (`/props`, `/tools`, and the resumable streams below).

- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high` (the default), `xhigh`, `max`.
  `chat_template_kwargs.enable_thinking: false` is `none`.
- As in llama-server, no `max_tokens` (or 0, -1) generates until the context is full.
- The server ends a thinking block (it writes the tokens that close it, then the model answers) at
  `thinking_budget_tokens`, when a finite `max_tokens` has a quarter (at most 1024) left for the answer, or on
  `POST /v1/chat/completions/control {"action": "reasoning_end", "id": <completion id>}`.
- Compaction, after [Pi](https://docs.pi.dev/latest/compaction), inside an answer: when the thinking comes within
  `reserve_tokens` (16384, or an eighth of a smaller `--ctx`) of the context size, the model writes a summary of its
  reasoning on the cache it has (task, constraints, established facts, approaches tried, current approach, next steps,
  code), and the sequence is rebuilt as the prompt, the summary and the last `keep_recent_tokens` (20000) of the
  thinking, cut at a line start; it takes at most half the context before the reserve. The reply stays one message: the
  summary and a marker `[context compacted: ...]` go to `reasoning_content`, and stream chunks carry
  `compaction: {stage, done, total}` while it runs. At most `max_compactions` (8) per answer; then, or when a summary
  cannot fit, the thinking is closed at the reserve and the model answers.
- Compaction between turns: when a chat's prompt and the reserve do not fit, its older messages are serialized
  (`[User]: ...`, `[Assistant]: ...`, tool calls and results, without the thinking), summarized in a separate
  generation (in chunks, each updating the summary, when they are longer than the context), and replaced by one system
  message; the last messages up to `keep_recent_tokens` stay, never split from a tool call. Summaries are kept by a hash
  of the messages they cover, so the next turn of the same chat reuses the summary and the KV cache.
- Compaction settings: request field `compaction`: `false`, or `{enabled, reserve_tokens, keep_recent_tokens,
  max_compactions, summary_tokens}`; server flags `--compact-reserve N`, `--compact-keep N`, `--compact-max N`,
  `--no-compact`. `POST /v1/chat/completions/control {"action": "compact", "id": ...}` compacts the thinking now (also
  with automatic compaction off). `timings` leave compaction out of the decode time and report `compactions` and
  `compact_ms`. Summaries are asked for in a system message (`IK_COMPACT_ROLE=user`: a user message).
- Progress notes, as chat apps show while a model thinks: `reasoning_summary`: `only` (also `auto`, `concise`,
  `detailed`, as in OpenAI's API) sends notes in place of the raw thinking, `inline` between it, `off` (the default;
  server flag `--reasoning-summary`). Every `reasoning_summary_interval` (1024; `--summary-every`) tokens of thinking,
  the first after a quarter of it, the model writes a bold title and one or two sentences on what it is doing, on a
  branch of the cache that is dropped after (its own context keeps the raw thinking); about 2.4 s each. Stream chunks
  carry each note in `delta.reasoning_summary` too, and `timings` report `notes` and `notes_ms`.
- `timings_per_token` puts `timings` on every chunk of a stream; `return_progress` adds `prompt_progress`.
- Streams send an SSE comment (`: keepalive`) after a second without data (thinking hidden behind notes, a compaction):
  the chat UI takes 3 s of silence, when its tab comes back, for a lost connection.
- Resumable streams, as llama-server's: a streaming request with an `X-Conversation-Id` header (the chat UI sends one)
  runs on when its connection drops or the page reloads. `GET /v1/stream?conv_id=ID&from=N` sends its bytes from byte N
  and follows, `DELETE /v1/stream?conv_id=ID` stops it (the UI's Stop button), `POST /v1/streams/lookup
  {"conversation_ids": [...]}` lists them; a new request in the same conversation stops the one before. Without the
  header, a client that leaves stops its request at the next token.
- The KV cache is reused when a prompt extends the previous one. When a chat UI sends the history again, the cache is
  rolled back to the end of the last prompt.

## Design

- Attention, router and output head stay on the GPU. The 3072 routed experts live in host RAM (huge pages,
  page-locked), repacked at load into the layout the GPU kernels read, which the CPU kernels read too: NVFP4 as rows
  of E2M1 nibbles (value j of each 64 in the low nibble of byte j, 32 + j in the high one), the UE4M3 scales and the
  expert's scale; Q8_0 as int8 rows and f16 scales. A VRAM cache holds the hot experts, a bank of slots per format
  (the Q8_0 bank takes `IK_Q8_SHARE`, 0.15, of it), keyed by physical layer so the three loop passes share them,
  with LFU admission and uploads (one copy each) on a copy stream.
- The small BF16 and F32 matrices (projections, loop gates) are quantized to Q8_0 at load, to join the Q8_0 GEMVs
  they are stacked with.
- Decode is one CUDA graph per token. At each MoE layer the GPU posts the routing to the CPU through pinned memory
  (`router_dispatch`), computes the cached experts while the CPU threads compute the misses (AVX2, int8), and waits on
  a flag (`moe_wait`). No host round trip per layer.
- The CPU threads take work from shared counters with no barrier; each gate/up chunk also applies the activation and
  quantizes its block for the down projection.
- Prefill runs ubatches of 256 tokens: dp4a GEMMs, flash-style attention, and experts grouped so each is read once per
  ubatch (grouped GEMMs on the GPU, batched int8 kernels on the CPU).
- The KV cache is int8 with an f16 scale per 32 values, as q8_0: 1088 bytes per position and layer pass. Perplexity over
  1984 tokens is 6.4237, against 6.4402 with an f16 cache (`IK_KV_F16=1`).
- Kernels follow Marlin where it applies at these batch sizes: weights repacked into aligned layouts, 16-byte loads,
  int8 activations, one pass over each weight byte.
- Serving follows mini-SGLang: HTTP front end, chat template and tokenizer, a scheduler thread that owns the engine, an
  incremental detokenizer and a reply parser (thinking, text, tool calls), SSE.

## Context

The model has 8 full-attention layer passes per token; the other 40 use a sliding window of 1024 tokens. The full
passes' KV cache takes 17 KiB per token. It has addresses for all of `--ctx` from the start but memory only for the
tokens in use, mapped in 2 MiB pages (CUDA virtual memory) as the context grows: the expert cache gets the rest, so a
short chat runs with the whole budget (about 1260 slots) whatever `--ctx` is, and each 2048 tokens of context take
about 7 slots back. A new chat that is shorter than the last gives the pages back to the expert cache.

Decode slows as the context fills, from fewer expert slots and from attention over more keys (about 0.4 ms per token
for each 8k tokens):

| Context in use | Expert slots | Decode |
|---|---|---|
| short | 1260 | 50 t/s |
| 64k | ~1036 | ~38 t/s |
| 128k | ~800 | ~33 t/s |

Measured (before the move to the official GGUF) with `ik-bench` at a short context with the slots capped, plus the
attention time from `ik-kbench`; the server takes `--ctx` up to what the VRAM holds (about 350k tokens, with no expert
cache left).

## Tools

| Binary | What it does |
|---|---|
| `ik-bench <model> [n_gen]` | decode speed and expert cache hit rate; `IK_PROF=1` adds the GPU time of each kernel kind, `IK_PREFILL=1` prefills the prompt |
| `ik-ctx <model> <text> <n_ctx> [n_gen] [max depth]` | prefill and decode speed as the context grows |
| `ik-ppl <model> <text> [n]` | perplexity |
| `ik-check <model> <ref.bin> [slots]` | compares every tensor with a llama.cpp dump from `tools/refdump.cpp` |
| `ik-pfcheck <model> <text> [n]` | prefill against decode |
| `ik-rollback <model> [repeats]` | KV cache rollback against a fresh prefill |
| `ik-branch <model> [tokens]` | a branch taken at a checkpoint and dropped, against no branch |
| `ik-kbench`, `ik-kcmp` | GPU kernel timings; new kernels against the ones they replace |
| `ik-cpubench <model>` | CPU expert kernels on real rows, one token and batched |
| `ik-moecheck` | the GPU expert kernels (decode and prefill) against the CPU ones, on random banks |
| `ik-attnmass <model> <text> <n>` | how much of the context full attention needs at depth n |

`IK_FAKE_CPU=1` skips the CPU experts and so measures the GPU path alone; `IK_NO_GRAPH=1` (`ik-bench`) launches each
kernel instead of one CUDA graph per token.

## Limits

One sequence at a time: requests are served in order. No image input. The context size is fixed at start.

## License

MIT. The chat UI that `tools/fetch-ui.sh` downloads is llama.cpp's (MIT). The model is Apache-2.0.
