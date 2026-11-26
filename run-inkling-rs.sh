#!/usr/bin/env bash
# Run the Inkling-2-Mini engine (OpenAI-compatible server) on the checkpoint as Thinking Machines publish it.
#
# Usage:  ./run-inkling-rs.sh [extra args...]
# Env:    MODEL (~/models/thinkingmachines/Inkling-2-Mini: the safetensors checkpoint, tools/fetch-safetensors.sh)
#         CTX (131072)  PORT (8080)  HOST (127.0.0.1)
#         IK_THREADS  expert CPU threads: "8" (default, unpinned) or a CPU list "0,2,4,6,8,10"
#         MEM_MAX (20G)  the server's memory limit: it runs in a systemd scope of its own with this MemoryMax and no
#                        swap, so that running out of memory stops the server, not the terminal it was started from.
#                        MEM_MAX= runs it without a limit.
# Endpoints: /v1/chat/completions (stream, reasoning_content, tools), /v1/completions, /v1/models, /props, /health;
#            the chat UI in a browser at http://HOST:PORT/
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-$HOME/models/thinkingmachines/Inkling-2-Mini}"
BIN=target/release/inkling-server
if [[ ! -f "$MODEL/config.json" ]]; then
  echo "no checkpoint at $MODEL; download it (20.2 GB), or set MODEL=dir:" >&2
  echo "  tools/fetch-safetensors.sh" >&2
  exit 1
fi
# rebuild if the sources changed (a no-op otherwise); CPU features are detected at run time
cargo build --release -q -p ik-engine --bin inkling-server
# the browser UI: llama.cpp's chat app in ./ui (tools/fetch-ui.sh downloads it); API only without it
UI_ARGS=()
if [[ -f ui/index.html ]]; then UI_ARGS=(--ui "$PWD/ui"); fi
# systemd-oomd stops the whole cgroup under memory pressure: with the server in its own, a terminal's other programs
# survive it
CAP=()
MEM_MAX="${MEM_MAX-20G}"
if [[ -n $MEM_MAX ]]; then
  if systemd-run --user --scope --quiet true 2>/dev/null; then
    CAP=(systemd-run --user --scope --quiet -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 --)
  else
    echo "no systemd user manager: running without the $MEM_MAX memory limit" >&2
  fi
fi
exec "${CAP[@]}" "$BIN" "$MODEL" --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" --ctx "${CTX:-131072}" "${UI_ARGS[@]}" "$@"
