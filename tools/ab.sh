#!/usr/bin/env bash
# A/B of engine switches with ik-bench, interleaved to spread background load and heat over the variants:
#   tools/ab.sh [-n rounds] [-g tokens] [-m model] 'ENV=a' 'ENV=b' ...
# Each argument is a variant: environment assignments (space-separated, '' for none). Prints each run's t/s, then
# the median and spread of each variant. A laptop's results move 5-15% from run to run: use 3 rounds or more.
set -euo pipefail
cd "$(dirname "$0")/.."
ROUNDS=3 GEN=256 MODEL="$HOME/models/unsloth/Inkling-2-Mini-UD-Q4_K_XL.gguf"
while getopts n:g:m: o; do
  case $o in n) ROUNDS=$OPTARG ;; g) GEN=$OPTARG ;; m) MODEL=$OPTARG ;; *) exit 1 ;; esac
done
shift $((OPTIND - 1))
(($# > 0)) || { echo "usage: tools/ab.sh [-n rounds] [-g tokens] 'ENV=a' 'ENV=b' ..." >&2; exit 1; }
declare -A res
for ((r = 1; r <= ROUNDS; r++)); do
  for v in "$@"; do
    out=$(env $v ./target/release/ik-bench "$MODEL" "$GEN" 2>&1 | grep -E '^decode:') || { echo "$v: no result" >&2; continue; }
    tps=$(sed -E 's/.*= ([0-9.]+) t\/s.*/\1/' <<<"$out")
    res["k:$v"]="${res["k:$v"]:-} $tps"
    printf 'round %d  %-40s %s t/s\n' "$r" "${v:-(default)}" "$tps"
  done
done
echo
for v in "$@"; do
  printf '%-40s median %s  (%s)\n' "${v:-(default)}" "$(tr ' ' '\n' <<<"${res["k:$v"]}" | grep . | sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')" "${res["k:$v"]# }"
done
