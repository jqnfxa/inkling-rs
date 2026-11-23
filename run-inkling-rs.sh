#!/usr/bin/env bash
# Run the Inkling-2-Mini engine (OpenAI-compatible server) on an 8 GB GPU.
#
# Usage:  ./run-inkling-rs.sh [extra args...]
# Env:    MODEL (~/models/Inkling-2-Mini-AD-Q4_K.gguf)  CTX (65536)  PORT (8080)  HOST (127.0.0.1)
#         IK_THREADS  expert CPU threads: "8" (default, unpinned) or a CPU list "0,2,4,6,8,10"
# Endpoints: /v1/chat/completions (stream, reasoning_content, tools), /v1/completions, /v1/models, /props, /health;
#            the chat UI in a browser at http://HOST:PORT/
set -euo pipefail
cd "$(dirname "$0")"
MODEL="${MODEL:-$HOME/models/Inkling-2-Mini-AD-Q4_K.gguf}"
BIN=target/release/inkling-server
# rebuild if the sources changed (a no-op otherwise); CPU features are detected at run time
cargo build --release -q -p ik-engine --bin inkling-server
# the browser UI: llama.cpp's chat app in ./ui (tools/fetch-ui.sh downloads it); API only without it
UI_ARGS=()
if [[ -f ui/index.html ]]; then UI_ARGS=(--ui "$PWD/ui"); fi
exec "$BIN" "$MODEL" --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" --ctx "${CTX:-65536}" "${UI_ARGS[@]}" "$@"
