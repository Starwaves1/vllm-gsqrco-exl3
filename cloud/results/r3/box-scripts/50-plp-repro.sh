#!/bin/bash
# R3-50 reproduce the torture-smoke failures on main 32ae6ec under production's main argv, GSQ vs the
# stock W4A16 path (same argv, same box), with a NaN/row-count probe on every LogitsProcessor call
# (pyhook R3_NANCHECK):
#   (1) prompt_logprobs=1 on 512 / 1024 / 2048 / 2600 / 3936-token prompts (smoke: EngineCore OOM,
#       a 1.57 GiB allocation in the GGUF lm_head path)
#   (2) completions echo + logprobs on "The capital of Denmark is" (smoke: HTTP 400 "nan" 7/7), plus
#       echo without logprobs, prompt_logprobs on the same prompt, and echo on 40 / 200-token prompts
# Plugin build: $R3_PLUGIN_WT (default the 32ae6ec build); R3_TAG for a rerun on a fixed build.
# Output: /workspace/logs/r3/50-plp-repro[-TAG]/{summary.txt, gsq/, w4a16/ (results.jsonl, nan.jsonl)}
# GPU time: ~15 min.
#   bash 50-plp-repro.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 50-plp-repro "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,13p' "$0"; exit 0; fi
export R3_EXTRA_PYTHONPATH=$R3_S/pyhook
r3_env
r3_preflight
[ -d "$GSQ_BASELINE_MODEL" ] || r3_die "baseline model missing: $GSQ_BASELINE_MODEL"
run() {  # tag kind
  export R3_MODEL_KIND=$2
  export R3_NANCHECK=$L/$1/nan.jsonl
  mkdir -p "$L/$1"
  r3_serve "$1"
  "$PY" "$R3_S/r3plp.py" --out "$R/results.jsonl" || r3_die "r3plp"
  tail -5 "$R/server.log" | cut -c1-200
}
v_gsq() { run gsq gsq; }
v_w4a16() { run w4a16 baseline; }
r3_summary "R3-50 prompt-logprobs / echo repro (box, $(date -u +%F)), plugin $(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD) ${R3_TAG:-}"
for v in gsq w4a16; do
  r3_step "$v"; r3_variant "$v" "v_$v"
  r3_summary "--- $v ---"
  [ -f "$L/$v/results.jsonl" ] && r3_summary "$(cut -c1-400 "$L/$v/results.jsonl")"
  [ -f "$L/$v/nan.jsonl" ] && r3_summary "nan/large-n logits calls:" "$(head -30 "$L/$v/nan.jsonl" | cut -c1-400)"
  grep -h -E "out of memory|Traceback" "$L/$v/server.log" 2>/dev/null | head -3 | cut -c1-300 | sed 's/^/server: /' >> "$L/summary.txt"
done
cat "$L/summary.txt"
r3_finish
