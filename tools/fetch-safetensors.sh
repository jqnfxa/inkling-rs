#!/usr/bin/env bash
# Download Inkling-2-Mini as Thinking Machines publish it: huggingface.co/thinkingmachines/Inkling-2-Mini at a pinned
# revision (the weights in safetensors, the MTP head, the tokenizer and configs: 20.2 GB). It checks the free space
# before it starts, resumes after a dropped connection (and when run again), and gives a file its name only once it
# matches the hash the Hub lists for it (sha256 for LFS files, the git blob id for the rest). inkling-rs runs this
# checkpoint as it is (run-inkling-rs.sh reads it from the default dir).
#
# Usage:  tools/fetch-safetensors.sh [dir [glob...]]
#           dir   where the files go: ~/models/thinkingmachines/Inkling-2-Mini by default
#           glob  only the files whose path matches, e.g. '*.json' 'tiktoken/*'
# Env:    REV (c9d0283, main on 2026-11-24; "main" for the latest)  REPO  HF_TOKEN (a gated or private repo)
#         HF_ENDPOINT (a mirror of huggingface.co)
set -euo pipefail
REPO="${REPO:-thinkingmachines/Inkling-2-Mini}"
REV="${REV:-c9d028302cd0f035fdf8e7e91f089a4ce94d2195}"
DIR="${1:-$HOME/models/$REPO}"
(($# == 0)) || shift
HUB="${HF_ENDPOINT:-https://huggingface.co}"
for c in curl jq sha256sum sha1sum flock numfmt; do
  command -v "$c" >/dev/null || { echo "needs $c" >&2; exit 1; }
done
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
AUTH=()
if [[ -n "${HF_TOKEN:-}" ]]; then # from a file: on the command line, ps would show it
  printf 'Authorization: Bearer %s\n' "$HF_TOKEN" >"$TMP/auth"
  AUTH=(-H "@$TMP/auth")
fi
size() { if (($1 < 1000)); then echo "$1B"; else numfmt --to=si --round=nearest --format=%.1f --suffix=B "$1"; fi; }

# the revision as a commit, so that the listing and the downloads are of the same files when REV is a branch
sha=$(curl -fsS "${AUTH[@]}" "$HUB/api/models/$REPO/revision/$(jq -rn --arg r "$REV" '$r | @uri')" | jq -r .sha) &&
  [[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "no revision $REV of $REPO (gated or private? set HF_TOKEN)" >&2; exit 1; }
# path, size, hash kind, hash, the path in a URL; a page at a time, the Link header names the next
url="$HUB/api/models/$REPO/tree/$sha?recursive=true"
while [[ -n "$url" ]]; do
  curl -fsS "${AUTH[@]}" -D "$TMP/hdr" "$url" | jq -r '.[] | select(.type == "file")
    | [.path, .size, (if .lfs then "sha256", .lfs.oid else "blob", .oid end), (.path | split("/") | map(@uri) | join("/"))]
    | @tsv' >>"$TMP/list"
  url=$(grep -i '^link:' "$TMP/hdr" | grep -oE '<[^>]+>; *rel="next"' | sed -E 's/^<([^>]+)>.*/\1/' || true)
done
if (($#)); then
  while IFS=$'\t' read -r path rest; do
    # shellcheck disable=SC2053 # $g unquoted: a glob
    for g in "$@"; do if [[ $path == $g ]]; then printf '%s\t%s\n' "$path" "$rest"; break; fi; done
  done <"$TMP/list" >"$TMP/sel"
  mv "$TMP/sel" "$TMP/list"
fi
[[ -s $TMP/list ]] || { echo "no file of $REPO@${sha:0:7} matches $*" >&2; exit 1; }
sort -t $'\t' -k2,2n -o "$TMP/list" "$TMP/list" # the small files first

mkdir -p "$DIR"
exec 9>"$DIR/.lock"
flock -n 9 || { echo "another download into $DIR is running" >&2; exit 1; }
MAN="$DIR/.verified" # "hash  path" of each file found to match the hash
check() {            # file size kind hash: the file is the one the Hub lists
  local got
  [[ $(stat -c %s "$1") == "$2" ]] || return 1
  if [[ $3 == sha256 ]]; then got=$(sha256sum <"$1"); else got=$({ printf 'blob %s\0' "$2"; cat "$1"; } | sha1sum); fi
  [[ ${got%% *} == "$4" ]]
}

# what is there already: files checked before, files of the right size to check now, partial downloads
need=0
: >"$TMP/todo"
while IFS=$'\t' read -r path bytes kind hash enc; do
  f="$DIR/$path"
  if [[ -f $f ]]; then
    grep -qxF "$hash  $path" "$MAN" 2>/dev/null && continue
    if [[ $(stat -c %s "$f") == "$bytes" ]]; then
      echo "checking $path, there from before"
      if check "$f" "$bytes" "$kind" "$hash"; then echo "$hash  $path" >>"$MAN"; continue; fi
    fi
  fi
  have=0
  [[ -f $f.part ]] && have=$(stat -c %s "$f.part")
  ((have <= bytes)) || { rm -f "$f.part"; have=0; }
  need=$((need + bytes - have))
  printf '%s\t%s\t%s\t%s\t%s\n' "$path" "$bytes" "$kind" "$hash" "$enc" >>"$TMP/todo"
done <"$TMP/list"
[[ -s $TMP/todo ]] || { echo "$DIR: $REPO@${sha:0:7}, every file there and checked"; exit 0; }
avail=$(df --output=avail -B1 "$DIR" | tail -n1)
if ((need + 1000000000 > avail)); then
  echo "$(size "$need") to download and $(size "$avail") free on the disk of $DIR (1 GB is to stay free):" >&2
  echo "free $(size $((need + 1000000000 - avail))) there, or give a directory on another disk: $0 DIR" >&2
  exit 1
fi

fetch() { # url file kind: the file, resumed from what is there, again after a dropped or stalled connection
  local meter=(-sS) try code rc
  [[ $3 == sha256 ]] && meter=()
  for ((try = 1; ; try++)); do
    code=$(curl -fL -C - "${AUTH[@]}" "${meter[@]}" --connect-timeout 30 --speed-limit 4096 --speed-time 60 \
      -w '%{http_code}' -o "$2" "$1") && return
    rc=$?
    case $code in 401 | 403 | 404) echo "HTTP $code for $1: gated, private or gone (HF_TOKEN?)" >&2; exit 1 ;; esac
    ((try < 30)) || { echo "$1: gave up after $try tries" >&2; exit 1; }
    echo "stopped (curl exit $rc${code:+, HTTP $code}); resuming in 10 s" >&2
    sleep 10
  done
}
echo "$REPO@${sha:0:7}: $(wc -l <"$TMP/todo") files, $(size "$need") to download into $DIR"
while IFS=$'\t' read -r path bytes kind hash enc; do
  f="$DIR/$path"
  mkdir -p "$(dirname "$f")"
  have=0
  [[ -f $f.part ]] && have=$(stat -c %s "$f.part")
  if ((have)); then echo "$path: $(size "$have") of $(size "$bytes") there, resuming"; else echo "$path ($(size "$bytes"))"; fi
  if ((have < bytes)); then fetch "$HUB/$REPO/resolve/$sha/$enc" "$f.part" "$kind" </dev/null; else : >>"$f.part"; fi
  [[ $kind == blob ]] || echo "checking the sha256 of $path"
  if ! check "$f.part" "$bytes" "$kind" "$hash"; then
    rm -f "$f.part"
    echo "$path: not the file the Hub lists (its $kind differs); deleted, run again" >&2
    exit 1
  fi
  mv -f "$f.part" "$f"
  echo "$hash  $path" >>"$MAN"
done <"$TMP/todo"
echo "done: $DIR ($REPO@${sha:0:7})"
