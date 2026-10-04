#!/bin/bash
# EXL3 phase 1, job 04 (~20-40 min): the parity reference. prompts.py builds the 11 salted
# sequences (1k..120k tokens; corpus from venv-main's vLLM source, tokenizer identical to the EXL3
# checkpoint's), then exllamav3 (reference venv) dumps full-vocab logits at every probed position
# (bench/parity/exl3_logits.py, EXL3_INT8_GEMV=0, default EXL3_HGEMM_F16ACC). A sequence that
# does not fit exllamav3's cache on 24 GB is recorded as oom in exl3_logits.json (the context limit
# to document), not fatal. EXL3_SPREAD=1 (default): a second pass with EXL3_HGEMM_F16ACC=0 scored
# against the first = exllamav3's own spread, the scale for the vLLM gate; its dumps are deleted.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 04-parity-ref
require_idle_gpu
P=$R/parity; O=$R/04-parity-ref; rm -rf "$O" "$P/exl3" "$P/exl3-fp32acc"; mkdir -p "$O" "$P"
[ -f "$P/prompts/manifest.json" ] || "$GSQ_VENV/bin/python" bench/parity/prompts.py --out "$P/prompts" | tee "$O/prompts.txt"
"$GSQ_EXL3_VENV/bin/python" bench/parity/exl3_logits.py --dry-run -d "$P/prompts" | tee "$O/plan.txt"
"$GSQ_EXL3_VENV/bin/python" bench/parity/exl3_logits.py -m "$EXL3_MODEL" -d "$P/prompts" -o "$P/exl3" 2>&1 | tee "$O/exl3.log"
cp "$P/exl3/exl3_logits.json" "$O/"
if [ "${EXL3_SPREAD:-1}" = 1 ] && [ "$(df --output=avail -B1G /workspace | tail -1)" -ge 9 ]; then
  EXL3_HGEMM_F16ACC=0 "$GSQ_EXL3_VENV/bin/python" bench/parity/exl3_logits.py -m "$EXL3_MODEL" -d "$P/prompts" \
    -o "$P/exl3-fp32acc" 2>&1 | tee "$O/exl3-fp32acc.log"
  cp "$P/exl3-fp32acc/exl3_logits.json" "$O/exl3_logits-fp32acc.json"
  mkdir -p "$P/spread-ref" "$P/spread-other"
  for f in "$P"/exl3/*.exl3.f32; do n=$(basename "$f" .exl3.f32)
    ln -sf "$f" "$P/spread-ref/$n.llama.f32"
    [ -f "$P/exl3-fp32acc/$n.exl3.f32" ] && ln -sf "$P/exl3-fp32acc/$n.exl3.f32" "$P/spread-other/$n.vllm.f32"; done
  python3 bench/parity/compare.py -d "$P/prompts" -l "$P/spread-ref" -v "$P/spread-other" --json "$O/spread.json" \
    | tee "$O/spread.txt" || true
  rm -rf "$P/exl3-fp32acc" "$P/spread-ref" "$P/spread-other"
else
  echo "spread pass skipped (EXL3_SPREAD=${EXL3_SPREAD:-1}, $(df --output=avail -B1G /workspace | tail -1) GB free)" | tee "$O/spread.txt"
fi
python3 - "$O/exl3_logits.json" <<'PY' | tee "$O/summary.txt"
import json, sys
r = json.load(open(sys.argv[1]))
print("04-parity-ref: exllamav3", r["exllamav3"], "cache", r["cache_tokens"], "tokens, load", r["load_s"], "s,",
      r["gpu_mib_after_load"], "MiB after load")
for n, s in r["sequences"].items():
    print(f"  {n}: {s['n_tokens']} tok {s['status']} {s['seconds']} s peak {s['peak_gpu_mib']} MiB")
bad = [n for n, s in r["sequences"].items() if s["status"] != "ok"]
print("context limit hit:" if bad else "all 11 sequences fit", " ".join(bad))
PY
keep "$O" 04-parity-ref "$O/summary.txt" "$O/plan.txt" "$O/exl3_logits.json" "$O/exl3_logits-fp32acc.json" \
  "$O/spread.txt" "$O/spread.json" "$O/prompts.txt"
