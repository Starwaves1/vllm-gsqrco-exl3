#!/bin/bash
# llama.cpp b11211 vs vLLM+plugin logit parity on the same GGUF (HANDOFF §2 "Correct").
#   GSQ_ALLOW_GPU=1 bench/parity/run.sh [--kv-cache-dtype auto|fp8] [--only seq_000,...]
# Steps (one engine on the GPU at a time): prompts.py -> llama_logits -> vllm_logprobs.py
# -> compare.py. Results in $GSQ_RUNS/<ts>-parity/. Disk: ~1 MB per dumped position per
# engine (~3 GB each with the default set); the raw dumps are deleted unless KEEP=1.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=$GSQ_VENV/bin/python
gsq_require_gpu
gsq_refuse_if_prod_live GSQ_ALLOW_BESIDE_PROD
[ -f "$GSQ_GGUF" ] || gsq_die "GGUF not found: $GSQ_GGUF"
BIN=${LLAMA_LOGITS:-$GSQ_RUNS/parity-bin/llama_logits}
[ -x "$BIN" ] || gsq_die "build it first: bench/parity/build.sh"

OUT=${OUT:-$GSQ_RUNS/$(date +%Y%m%d-%H%M%S)-parity}
mkdir -p "$OUT"/{prompts,llama,vllm}
"$PY" "$HERE/prompts.py" --out "$OUT/prompts" | tee "$OUT/prompts.txt"
"$BIN" -m "$GSQ_GGUF" -d "$OUT/prompts" -o "$OUT/llama" --kv "${LLAMA_KV:-f16}" 2>&1 | tee "$OUT/llama.log"
"$PY" "$HERE/vllm_logprobs.py" -d "$OUT/prompts" -o "$OUT/vllm" --gguf "$GSQ_GGUF" "$@" 2>&1 | tee "$OUT/vllm.log"
rc=0
"$PY" "$HERE/compare.py" -d "$OUT/prompts" -l "$OUT/llama" -v "$OUT/vllm" --json "$OUT/parity.json" | tee "$OUT/compare.txt" || rc=$?
[ "${KEEP:-0}" = 1 ] || rm -rf "$OUT/llama" "$OUT/vllm"
echo "parity: $([ $rc = 0 ] && echo PASS || echo FAIL), $OUT/parity.json"
exit $rc
