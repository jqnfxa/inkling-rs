#!/usr/bin/env bash
# Download llama.cpp's built chat UI into ./ui, where run-inkling-rs.sh serves it from.
# The server answers the endpoints of the version below; others may ask for more.
#
# Usage:  tools/fetch-ui.sh [version]     (default b12648; "latest" also works)
set -euo pipefail
cd "$(dirname "$0")/.."
VER="${1:-b12648}"
BASE="https://huggingface.co/buckets/ggml-org/llama-ui/resolve/$VER"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fsSL "$BASE/dist.tar.gz.sha256?download=true" -o "$TMP/dist.tar.gz.sha256"
curl -fL --progress-bar "$BASE/dist.tar.gz?download=true" -o "$TMP/dist.tar.gz"
want=$(grep -oE '^[0-9a-fA-F]{64}' "$TMP/dist.tar.gz.sha256" | tr A-F a-f)
have=$(sha256sum "$TMP/dist.tar.gz" | cut -d' ' -f1)
[[ -n "$want" && "$want" == "$have" ]] || { echo "checksum mismatch: $have, expected $want" >&2; exit 1; }
mkdir -p "$TMP/ui"
tar xzf "$TMP/dist.tar.gz" -C "$TMP/ui"
[[ -f "$TMP/ui/index.html" ]] || { echo "the archive has no index.html" >&2; exit 1; }
rm -rf ui
mv "$TMP/ui" ui
echo "ui: llama.cpp chat UI $VER"
