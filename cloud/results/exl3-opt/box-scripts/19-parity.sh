#!/bin/bash
# EXL3 opt, job 19 (~5 min per short sequence, ~60 min for all 11 at one KV dtype): phase 1's 05 (vLLM
# logits vs the exllamav3 reference of 04, /workspace/runs/exl3/parity) for one mode of this worktree,
# into this job's own run dir. Args: MODE [SEQS] (e.g. 2h seq_000,seq_001; default all sequences).
# PARITY_KV (default auto = bf16 KV). KEEP=1 keeps the vLLM dumps (for position-level analysis).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 19-parity
require_idle_gpu
require_mr_build
mode=${MODES_ARGS[0]:-2h}; only=${MODES_ARGS[1]:-}
P=/workspace/runs/exl3/parity
[ -f "$P/exl3/exl3_logits.json" ] || die "no exllamav3 reference in $P (phase 1's 04)"
[ -n "$only" ] || only=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(",".join(n for n,s in r["sequences"].items() if s["status"]=="ok"))' "$P/exl3/exl3_logits.json")
O=$R/19-parity-$mode${MODES_ARGS[1]:+-sub}; rm -rf "$O"; mkdir -p "$O/ref"
for f in "$P"/exl3/*.exl3.f32; do ln -sf "$f" "$O/ref/$(basename "$f" .exl3.f32).llama.f32"; done
mode_env "$mode"
rc=0
for kv in ${PARITY_KV:-auto}; do
  V=$O/vllm-$kv
  "$GSQ_VENV/bin/python" bench/parity/vllm_logprobs.py -d "$P/prompts" -o "$V" --model "$EXL3_MODEL" \
    --kv-cache-dtype "$kv" --only "$only" > "$O/vllm-$kv.log" 2>&1 || { echo "vllm_logprobs (kv $kv) failed"; rc=1; }
  python3 bench/parity/compare.py -d "$P/prompts" -l "$O/ref" -v "$V" --json "$O/parity-$kv.json" | tee "$O/compare-$kv.txt" || true
  "$GSQ_VENV/bin/python" "$S/parity_glitch.py" "$O/parity-$kv.json" "$P/exl3" "$P/prompts" > "$O/candidates-$kv.txt" 2>&1 || true
  [ "${KEEP:-0}" = 1 ] || rm -rf "$V"
done
gzip -kf "$O"/vllm-*.log
keep "$O" "19-parity-$mode${MODES_ARGS[1]:+-sub}" "$O"/compare-*.txt "$O"/candidates-*.txt "$O"/parity-*.json "$O"/vllm-*.log.gz
exit $rc
