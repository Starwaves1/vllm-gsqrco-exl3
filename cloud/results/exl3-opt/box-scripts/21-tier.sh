#!/bin/bash
# EXL3 opt, job 21 (~60 min per tier): one erlidev Swift-1.5 tier end to end within the box's small disk:
# download (phase 1's dl.sh, the HF branch TIER), draft head (tools/exl3_draft_head.py, production's ids),
# short logit parity vs exllamav3 (seq_002 code 1.5k, seq_003 code 4k, seq_005 prose 8k: exl3_logits.py in
# the reference venv, vllm_logprobs.py with MODE's switches, compare.py; bf16 KV, no MTP), then production's
# ladder c=1/2/4/8 at fixed k (lib.sh EXL3_SPEC_K) with an 8k prefill, then the checkpoint is deleted
# (KEEP_MODEL=1 keeps it; the shipped 3.50 tier is never deleted). If the tier does not fit max-model-len
# 196608, the server's own estimate of the largest fitting length (rounded down to 1024) is used.
# Args: TIER [MODE (default 2hf)]. Results: 21-tier-<TIER>/ (summary.txt, parity, ladder, load).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
tier=${MODES_ARGS[0]:?tier, e.g. SC_3.00bpw_H4_V4}; mode=${MODES_ARGS[1]:-2hf}
job_log "21-tier-$tier"
export GSQ_EXL3_MODEL=/workspace/models/Swift-1.5-Qwen3.8-27B-exl3-$tier EXL3_CACHE_SUFFIX=-$tier
export EXL3_MODEL=$GSQ_EXL3_MODEL
O=$R/21-tier-$tier$KS; rm -rf "$O"; mkdir -p "$O"
SEQS=seq_002,seq_003,seq_005
P=/workspace/runs/exl3/parity
require_mr_build
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); [ -z "$apps" ] || die "GPU busy: $apps"

# 1. download (resumable; every file checked against the Hub)
size=$(curl -sSfL "https://huggingface.co/api/models/erlidev/Swift-1.5-Qwen3.8-27B-EXL3/tree/$tier?recursive=true" \
  | python3 -c 'import json,sys; print(sum((f.get("lfs") or {}).get("size", f["size"]) for f in json.load(sys.stdin) if f["type"] == "file"))')
free=$(df --output=avail -B1 /workspace | tail -1)
[ -f "$EXL3_MODEL/config.json" ] || [ "$free" -gt $((size + 8 * 1073741824)) ] || die "disk: $free free, tier $size + 8 GiB margin"
t0=$(date +%s)
DL_REV=$tier "$WT/cloud/results/exl3/box-scripts/dl.sh" repo erlidev/Swift-1.5-Qwen3.8-27B-EXL3 "$EXL3_MODEL" > "$O/dl.log" 2>&1 \
  || { tail -20 "$O/dl.log"; die "download failed"; }
echo "download $size bytes in $(( $(date +%s) - t0 )) s" | tee "$O/summary.txt"
grep -o '"bits": *[0-9.]*\|"head_bits": *[0-9]*\|"calibration[^,]*' "$EXL3_MODEL/config.json" "$EXL3_MODEL/quantization_config.json" 2>/dev/null | tee -a "$O/summary.txt" || true

# 2. draft head
[ -f "$EXL3_MODEL/mtp_draft_head.safetensors" ] || "$GSQ_VENV/bin/python" tools/exl3_draft_head.py "$EXL3_MODEL" --ids "$DRAFT_IDS" > "$O/draft-head.log" 2>&1 \
  || { tail -20 "$O/draft-head.log"; die "draft head"; }

# 3. short parity vs exllamav3 (the salted prompt set is regenerated when the box lost it)
rc=0
[ -f "$P/prompts/manifest.json" ] || "$GSQ_VENV/bin/python" bench/parity/prompts.py --out "$P/prompts" > "$O/prompts.txt" 2>&1 \
  || { tail -20 "$O/prompts.txt"; die "prompts"; }
"$GSQ_EXL3_VENV/bin/python" bench/parity/exl3_logits.py -m "$EXL3_MODEL" -d "$P/prompts" -o "$O/ref" --only "$SEQS" > "$O/exl3-ref.log" 2>&1 \
  || { tail -20 "$O/exl3-ref.log"; rc=1; }
mkdir -p "$O/ref-l"; for f in "$O"/ref/*.exl3.f32; do ln -sf "$f" "$O/ref-l/$(basename "$f" .exl3.f32).llama.f32"; done
( mode_env "$mode"; "$GSQ_VENV/bin/python" bench/parity/vllm_logprobs.py -d "$P/prompts" -o "$O/vllm" --model "$EXL3_MODEL" \
    --only "$SEQS" > "$O/vllm-parity.log" 2>&1 ) || { tail -20 "$O/vllm-parity.log"; rc=1; }
python3 bench/parity/compare.py -d "$P/prompts" -l "$O/ref-l" -v "$O/vllm" --json "$O/parity.json" | tee "$O/parity.txt" | tee -a "$O/summary.txt" || true
rm -rf "$O/ref" "$O/ref-l" "$O/vllm"

# 4. ladder (fits 196608 or the server's own estimate)
D=$O/ladder
if ! serve_mr "$mode" "$D"; then
  fit=$(grep -o 'estimated maximum model length is [0-9]*' "$D/server.log" | grep -o '[0-9]*$' | tail -1 || true)
  stopall
  [ -n "$fit" ] || die "server failed (not a fit error)"
  export GSQ_MAX_MODEL_LEN=$(( fit / 1024 * 1024 ))
  echo "max-model-len 196608 does not fit; server estimate $fit -> $GSQ_MAX_MODEL_LEN" | tee -a "$O/summary.txt"
  serve_mr "$mode" "$D" || die "server failed at $GSQ_MAX_MODEL_LEN"
fi
GSQ_PREFILL="8192:1:2" OUT=$D bench/speed/run.sh exl3 > "$D/speed.log" 2>&1 || { echo "speed rc=$?"; rc=1; }
stopall
gzip -kf "$D/server.log"
"$GSQ_VENV/bin/python" "$S/ladder_summ.py" "$D" | tee -a "$O/summary.txt" || rc=1
grep -E "GPU KV cache size|Model loading took|VRAM after" "$D/load.txt" | tee -a "$O/summary.txt" || true

# 5. delete the checkpoint (never the shipped tier)
if [ "${KEEP_MODEL:-0}" != 1 ] && [ "$tier" != SC_3.50bpw_H4_V6 ]; then rm -rf "$EXL3_MODEL"; echo "deleted $EXL3_MODEL"; fi
keep "$O" "21-tier-$tier$KS" "$O/summary.txt" "$O/parity.txt" "$O/parity.json" "$O/dl.log" "$D/summary.txt" "$D/load.txt" "$D/argv.txt"
exit $rc
