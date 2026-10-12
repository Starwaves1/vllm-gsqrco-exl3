#!/bin/bash
# R3-56 direct unit test of the drafter's products at 9..16 rows (and the lm_head at the verify
# rows) with garbage in the CUDA-graph padding rows: tests/gpu/test_drafter_rows.py on production's
# plugin build ($R3_PLUGIN_WT, 32ae6ec), VLLM_GGUF_LCPP=1. No server. GPU time ~5 min.
# Output: /workspace/logs/r3/56-drafter-unit/{summary.txt, test.log, junit.xml}
#   bash 56-drafter-unit.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 56-drafter-unit "$@"
if [ "$R3_PLAN" = 1 ]; then sed -n '2,6p' "$0"; exit 0; fi
r3_env
r3_preflight
[ -d "$R3_WT/build/pytest" ] || cp -a "$R3_PLUGIN_WT/build/pytest" "$R3_WT/build/pytest"
cd "$R3_WT" || r3_die "no $R3_WT"
export VLLM_GGUF_LCPP=1 GSQ_RUNS=$L/runs
tools/pytest tests/gpu/test_drafter_rows.py -q -rs -s -p no:cacheprovider --junitxml="$L/junit.xml" > "$L/test.log" 2>&1
rc=$?
r3_summary "R3-56 drafter rows unit test (box, $(date -u +%F)), plugin $(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD): rc=$rc" \
  "$(grep -E "^(draft_head|nextn|attn_|ffn_|lm_head) " "$L/test.log")" "" "$(tail -15 "$L/test.log" | grep -E "FAILED|passed|failed|Error" )"
cat "$L/summary.txt"
