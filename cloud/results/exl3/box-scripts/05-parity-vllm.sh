#!/bin/bash
# EXL3 phase 1, job 05 (~40-80 min per KV dtype): the vLLM side of logit parity against job 04's
# exllamav3 dumps. bench/parity/vllm_logprobs.py --model <EXL3 dir> (plugins gguf + exl3 loaded,
# production's model flags minus MTP and KV offload, prefix-cached probes), then compare.py with
# the exllamav3 logits as the reference. PARITY_KV (default "auto fp8"): bf16 KV (the reference
# setting) then production's fp8. compare.py's fixed thresholds are llama.cpp's; the EXL3 gate is
# relative to exllamav3's own spread (04's spread.json), judged when reading the results.
# The raw vLLM dumps are deleted after scoring (KEEP=1 keeps them).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 05-parity-vllm
require_idle_gpu
P=$R/parity; O=$R/05-parity-vllm; rm -rf "$O"; mkdir -p "$O" "$P/ref"
[ -f "$P/exl3/exl3_logits.json" ] || die "run 04-parity-ref first"
for f in "$P"/exl3/*.exl3.f32; do ln -sf "$f" "$P/ref/$(basename "$f" .exl3.f32).llama.f32"; done
only=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(",".join(n for n,s in r["sequences"].items() if s["status"]=="ok"))' "$P/exl3/exl3_logits.json")
for kv in ${PARITY_KV:-auto fp8}; do
  V=$P/vllm-$kv; rm -rf "$V"
  "$GSQ_VENV/bin/python" bench/parity/vllm_logprobs.py -d "$P/prompts" -o "$V" --model "$EXL3_MODEL" \
    --kv-cache-dtype "$kv" --only "$only" > "$O/vllm-$kv.log" 2>&1 || echo "vllm_logprobs (kv $kv) failed"
  grep -E "^seq_|sequences, |Error|error" "$O/vllm-$kv.log" | tail -20 || true
  python3 bench/parity/compare.py -d "$P/prompts" -l "$P/ref" -v "$V" --json "$O/parity-$kv.json" | tee "$O/compare-$kv.txt" || true
  [ "${KEEP:-0}" = 1 ] || rm -rf "$V"
done
{ echo "05-parity-vllm $(date -u +%FT%TZ) sequences: $only"
  for kv in ${PARITY_KV:-auto fp8}; do echo "== kv $kv"; grep -E "tok  KLD|kld_mean|top1\"|long_" "$O/compare-$kv.txt"; done
  echo "== exllamav3 spread (f16acc vs fp32acc, 04):"; grep -E "kld_mean|\"top1\"" "$R/04-parity-ref/spread.txt" 2>/dev/null || echo "none"
} | tee "$O/summary.txt"
gzip -kf "$O"/vllm-*.log
keep "$O" 05-parity-vllm "$O/summary.txt" "$O"/compare-*.txt "$O"/parity-*.json "$O"/vllm-*.log.gz
