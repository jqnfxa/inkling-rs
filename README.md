# inkling-rs

An inference engine for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) that runs the
checkpoint as Thinking Machines publish it, from its original weights, on one laptop: RTX 3070 Ti Laptop (8 GB) +
i7-12700H + 64 GB DDR4-3200. Written from scratch in Rust; the GPU kernels are written in
[kernite](https://github.com/fxeqxmulfx/kernite). It serves an OpenAI-compatible API and llama.cpp's chat UI.

The model is a mixture of experts (128 experts, 8 per token) whose middle layers run three times per token; its
checkpoint is 19 GB, so it does not fit in 8 GB of VRAM. The speeds below were measured before the engine moved to the
original checkpoint and are to be measured again.

## Requirements

- Linux on x86-64 with AVX2, FMA and F16C.
- An NVIDIA GPU with 8 GB and a driver for CUDA 12 (tested: sm_86, driver 550). The kernels are PTX for sm_70 and
  later, built with the crate; no CUDA toolkit is needed.
- RAM for the experts. Their copy takes 14.6 GiB, and the whole engine runs within a 20 GB memory limit (tested on
  30 GB). `IK_MAP_EXPERTS=1` needs less: it leaves the experts in the page cache, but the CPU experts run slower.
- Rust through rustup: `rust-toolchain.toml` pins 1.100.0, which kernite needs.
- The checkpoint [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): the
  safetensors weights (BF16 and F32 dense tensors, ModelOpt NVFP4 experts in layers 0-19, MXFP8 experts in 20-23),
  `config.json`, `tokenizer.json` and `chat_template.jinja`. The engine reads the tensors as the files store them;
  nothing is converted at load.

## Run

```sh
tools/fetch-safetensors.sh  # the checkpoint, into ~/models/thinkingmachines/Inkling-2-Mini (resumes, checks hashes)
tools/fetch-ui.sh           # llama.cpp's chat UI, into ./ui (optional)
./run-inkling-rs.sh         # builds, then serves on http://127.0.0.1:8080
```

Open http://127.0.0.1:8080 to chat. `run-inkling-rs.sh` reads `MODEL` (the checkpoint's directory), `CTX` (131072),
`PORT`, `HOST`, `IK_THREADS` (the CPU expert threads: a count, or a list of CPUs to pin) and `MEM_MAX` (20G). The
server runs in a systemd scope of its own, with that memory limit and no swap. If memory runs out, the server stops,
not the terminal it was started from (systemd-oomd stops a whole cgroup). `MEM_MAX=` turns the limit off.

## API

`/v1/chat/completions` (streaming, `reasoning_content`, tool calls), `/v1/completions`, `/v1/models`, `/health`, and the
llama-server endpoints the chat UI reads (`/props`, `/tools`, and the resumable streams below).

- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high` (the default), `xhigh`, `max`.
  `chat_template_kwargs.enable_thinking: false` is `none`.
- As in llama-server, no `max_tokens` (or 0, -1) generates until the context is full.
- Sampling: a request's `temperature`, `top_k` (0 = all), `top_p`, `min_p`, `repeat_penalty`, `presence_penalty`,
  `frequency_penalty`, `repeat_last_n`, `seed`; otherwise the server's, set by `--temp`, `--top-k`, `--top-p`, `--min-p`,
  `--repeat-penalty`, `--presence-penalty`, `--frequency-penalty`, `--repeat-last-n` (llama-server's defaults: 1.0,
  40, 0.95, 0.05, 1.0, 0, 0, 64). The checkpoint's `generation_config.json` sets temperature 1.0 and no other sampler.
- The server ends a thinking block (it writes the tokens that close it, then the model answers) at
  `thinking_budget_tokens`, when a finite `max_tokens` has a quarter (at most 1024) left for the answer, or on
  `POST /v1/chat/completions/control {"action": "reasoning_end", "id": <completion id>}`.
- Compaction, after [Pi](https://docs.pi.dev/latest/compaction), inside an answer: when the thinking comes within
  `reserve_tokens` (16384, or an eighth of a smaller `--ctx`) of the context size, the model writes a summary of its
  reasoning on the cache it has (task, constraints, established facts, approaches tried, current approach, next steps,
  code), and the sequence is rebuilt as the prompt, the summary and the last `keep_recent_tokens` (20000) of the
  thinking, cut at a line start; it takes at most half the context before the reserve. The reply stays one message: the
  summary and a marker `[context compacted: ...]` go to `reasoning_content`, and stream chunks carry
  `compaction: {stage, done, total}` while it runs (`summary`, then `rebuild`). At most `max_compactions` (8) per
  answer; then, or when a summary cannot fit, the thinking is closed at the reserve (a chunk with stage `closed`) and
  the model answers.
- In a tool-using turn the compaction covers the whole turn, everything after the last user message: the model's
  earlier messages of the turn with their thinking, its tool calls and their results, and the thinking under way. The
  sequence is rebuilt as the prompt up to the turn, then one model message whose thinking is the summary and the
  turn's last `keep_recent_tokens` kept verbatim (tool calls and results in it as they were); a later compaction of
  the turn updates that summary. When a request's prompt and the reserve do not fit and the overflow is in its current
  turn, the prompt is fed up to the threshold (cut at a line end or between messages), the turn compacted the same
  way, and the rest fed after it, as often as it takes (stages `summary` and `rebuild` too, before the reply's text).
- Replies the server ended (stop or tool calls) are kept as the token sequence the cache holds, by a hash of the
  messages they stand for: the request's and the reply as a client sends it back (tool-call ids, content and
  arguments; the thinking, which may hold summaries, does not count). When the next request is that history plus
  messages that are not the user's (tool results), its prompt is the kept sequence and the new messages rendered after
  it: the cache is reused to the token (the model's own tokenization, which a new rendering of its text need not
  give), and a compacted turn stays compacted. The latest 16 are kept. A history that matches none (other ids, an
  edited message, a restarted server) is rendered as it is.
- Tool-call ids are unique (`call_<completion>_<n>`); a tool result's name is its nearest call with that id. Tool-call
  arguments come back as they were written: JSON objects as their text, arguments that do not parse verbatim (a
  malformed call the model wrote renders back as it was, instead of failing the request).
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

- Attention, router and output head stay on the GPU. The 3072 routed experts live in host RAM, on huge pages.
  - They are copied at load from the checkpoint as it stores them, and the file's pages are dropped from the page
    cache as they are copied.
  - The GPU and CPU kernels read the stored formats:
    - ModelOpt NVFP4: E2M1 pairs (element 2j in the low nibble of byte j), an E4M3 scale per 16, and a factor per
      expert (`weight_amax` over the denominator).
    - MXFP8: E4M3 values with a UE8M0 scale per 32.
  - A VRAM cache holds the hot experts, with LFU admission.
    - It has a bank of slots per format; the MXFP8 bank takes `IK_MX_SHARE` (0.15) of the VRAM.
    - It is keyed by physical layer, so the three loop passes share it.
    - A slot holds an expert's codes, then its scales, copied as they are.
  - Uploads copy between tokens. `IK_PIN=1` page-locks the copy, so that uploads run on a copy stream, during prefill
    too. On a Ryzen 5 5600H laptop with its AMD IOMMU on, locking 15 GB took minutes; the plain copy decodes as fast.
- The dense weights stay as the checkpoint stores them: BF16, and F32 for the residual-mix queries and the router
  bias. The GEMVs read BF16. The embedding stays in the mapped file; a token's row is converted on the host.
- Decode is one CUDA graph per token. At each MoE layer the GPU posts the routing to the CPU through pinned memory
  (`router_dispatch`), computes the cached experts while the CPU threads compute the misses (AVX2, int8), and waits on
  a flag (`moe_wait`). No host round trip per layer.
- The CPU threads take work from shared counters with no barrier; each gate/up chunk also applies the activation and
  quantizes its block for the down projection.
- Prefill runs ubatches of up to 256 tokens. The BF16 GEMVs over the ubatch's tokens and flash-style attention run
  on the GPU. The experts run on the CPU, grouped so each is read once per ubatch. Prefill is slow until tensor-core
  BF16 GEMMs and grouped NVFP4/MXFP8 GEMMs move it to the GPU.
- The KV caches: the sliding-window passes' are int8 with an f16 scale per 32 values, as q8_0 (1088 bytes per position
  and layer pass); the full-attention passes' are int4 (symmetric, stored plus 8, the same scales: 576 bytes).
  Perplexity over 16k tokens of the model's own reasoning: 1.6984 with int8 full-attention caches (`IK_KV=8`), 1.7058
  with int4 (+0.44%); over 1984 tokens, int8 against f16 (`IK_KV_F16=1`): 6.4237 against 6.4402.
- Decode attention splits the keys into chunks of 256, a block per (kv head, chunk): the query quantized to int8 per 32
  for dp4a scores, the values converted without `cvt` (each byte into the low byte of 2^23), 16-byte loads in the V
  pass. Full attention over 131k keys takes 507 us per layer pass with int4 caches, 746 with int8 (831 before).
- The GPU expert kernels decode E2M1 by table lookups in `prmt` (8 values in 13 instructions) for dp4a. In decode a
  warp reads 128 contiguous bytes of each of 4 rows, a thread keeps 4 rows, and x comes through L1 (no shared-memory
  staging, no barrier): the gate_up of 8 NVFP4 experts (2304 x 2304) takes 62 us (388 GB/s), the down 34 us.
- Kernels follow Marlin where it applies at these batch sizes: 16-byte loads and one pass over each weight byte. Here
  they read the weights in the checkpoint's own layouts, with int8 activations for the NVFP4 experts' dp4a.
- Serving follows mini-SGLang: HTTP front end, chat template and tokenizer, a scheduler thread that owns the engine, an
  incremental detokenizer and a reply parser (thinking, text, tool calls), SSE.

## Context

The model has 8 full-attention layer passes per token; the other 40 use a sliding window of 1024 tokens. The full
passes' KV cache takes 9 KiB per token (int4; 17 KiB with `IK_KV=8`). It has addresses for all of `--ctx` from the start
but memory only for the tokens in use, mapped in 2 MiB pages (CUDA virtual memory) as the context grows: the expert
cache gets the rest, so a short chat runs with the whole budget (about 1260 slots) whatever `--ctx` is, and each 2048
tokens of context take about 4 slots back (7 with int8). A new chat that is shorter than the last gives the pages back
to the expert cache.

Decode slows as the context fills, from fewer expert slots and from attention over more keys (about 0.22 ms per token
for each 8k tokens):

| Context in use | Expert slots | Decode | Prefill |
|---|---|---|---|
| 1k | 1267 | 51 t/s | |
| 16k | 1238 | 44 t/s | 94 t/s |
| 32k | 1203 | 38 t/s | 113 t/s |
| 98k | 1079 | 40 t/s | 82 t/s |

Measured with `ik-ctx` on one long text (code and the model's reasoning), 512 tokens decoded at each depth, prefill
the average up to it; the hit rate changes with the text, and runs vary 10-20% with load and heat. The server takes
`--ctx` up to what the VRAM holds (by the cache size, about 650k tokens with no expert cache left).

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
| `ik-wcheck <model>` | the GPU kernels that read the checkpoint's weights, on its real tensors, against f64 references on the CPU |
| `ik-attnmass <model> <text> <n>` | how much of the context full attention needs at depth n |

`tools/agentic-check.py [--url U] [--until 90000]` drives one long tool-using turn against a running server and
checks that it compacts without closing the thinking, that each call after the first reuses 95% of its prompt from the
cache, and that a malformed tool call in the history is accepted.

`IK_FAKE_CPU=1` skips the CPU experts and so measures the GPU path alone; `IK_NO_GRAPH=1` (`ik-bench`) launches each
kernel instead of one CUDA graph per token.

## Limits

One sequence at a time: requests are served in order. No image input. The context size is fixed at start.

## License

MIT. The chat UI that `tools/fetch-ui.sh` downloads is llama.cpp's (MIT). The model is Apache-2.0.
