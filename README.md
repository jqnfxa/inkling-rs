# inkling-rs

An inference engine for [Inkling-2-Mini](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF) on one laptop:
RTX 3070 Ti Laptop (8 GB) + i7-12700H + 64 GB DDR4-3200. Written from scratch in Rust; the GPU kernels are written in
[kernite](https://github.com/fxeqxmulfx/kernite). It serves an OpenAI-compatible API and llama.cpp's chat UI.

The model is 15.4 GiB, a mixture of experts (128 experts, 8 per token) whose middle layers run three times per token,
so it does not fit in 8 GB of VRAM. Decode runs at about 50 tokens/s on this laptop (llama.cpp with `--n-cpu-moe`:
11-13 t/s).

## Requirements

- Linux on x86-64 with AVX2, FMA and F16C.
- An NVIDIA GPU with 8 GB and a driver for CUDA 12 (tested: sm_86, driver 550). The kernels are PTX for sm_70 and
  later, built with the crate; no CUDA toolkit is needed.
- 32 GB of RAM or more: the experts (14.8 GB) stay in page-locked host memory.
- Rust through rustup: `rust-toolchain.toml` pins 1.100.0, which kernite needs.
- The model file `Inkling-2-Mini-AD-Q4_K.gguf` (16.5 GB). Other quantizations are not supported.

## Run

```sh
curl -L -o ~/models/Inkling-2-Mini-AD-Q4_K.gguf \
  https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-AD-Q4_K.gguf
tools/fetch-ui.sh        # llama.cpp's chat UI, into ./ui (optional)
./run-inkling-rs.sh      # builds, then serves on http://127.0.0.1:8080
```

Open http://127.0.0.1:8080 to chat. `run-inkling-rs.sh` reads `MODEL`, `CTX` (131072), `PORT`, `HOST` and `IK_THREADS`
(the CPU expert threads: a count, or a list of CPUs to pin).

## API

`/v1/chat/completions` (streaming, `reasoning_content`, tool calls), `/v1/completions`, `/v1/models`, `/health`, and the
llama-server endpoints the chat UI reads (`/props`, `/tools`, `/v1/streams/lookup`).

- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high` (the default), `xhigh`, `max`.
  `chat_template_kwargs.enable_thinking: false` is `none`.
- As in llama-server, no `max_tokens` (or 0, -1) generates until the context is full.
- The server ends a thinking block (it writes the tokens that close it, then the model answers) at
  `thinking_budget_tokens`, when a finite `max_tokens` has a quarter (at most 1024) left for the answer, or on
  `POST /v1/chat/completions/control {"action": "reasoning_end", "id": <completion id>}`.
- `timings_per_token` puts `timings` on every chunk of a stream; `return_progress` adds `prompt_progress`.
- The KV cache is reused when a prompt extends the previous one. When a chat UI sends the history again, the cache is
  rolled back to the end of the last prompt.

## Design

- Attention, router and output head stay on the GPU. The 3072 routed experts live in host RAM (huge pages,
  page-locked), and a VRAM cache holds the hot ones: up to about 1260 slots, keyed by physical layer so the three loop
  passes share them, with LFU admission and uploads on a copy stream.
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

Measured with `ik-bench` at a short context with the slots capped, plus the attention time from `ik-kbench`; the
server takes `--ctx` up to what the VRAM holds (about 350k tokens, with no expert cache left).

## Tools

| Binary | What it does |
|---|---|
| `ik-bench <model> [n_gen]` | decode speed; `IK_PROF=1` adds the GPU time of each kernel kind |
| `ik-ctx <model> <text> <n_ctx> [n_gen] [max depth]` | prefill and decode speed as the context grows |
| `ik-ppl <model> <text> [n]` | perplexity |
| `ik-check <model> <ref.bin> [slots]` | compares every tensor with a llama.cpp dump from `tools/refdump.cpp` |
| `ik-pfcheck <model> <text> [n]` | prefill against decode |
| `ik-rollback <model> [repeats]` | KV cache rollback against a fresh prefill |
| `ik-kbench`, `ik-kcmp` | GPU kernel timings; new kernels against the ones they replace |
| `ik-cpubw <model> [threads..]` | CPU expert kernels, hot in cache and streaming from RAM |

`IK_FAKE_CPU=1` skips the CPU experts and so measures the GPU path alone.

## Limits

One sequence at a time: requests are served in order. No image input. The context size is fixed at start.

## License

MIT. The chat UI that `tools/fetch-ui.sh` downloads is llama.cpp's (MIT). The model is Apache-2.0.
