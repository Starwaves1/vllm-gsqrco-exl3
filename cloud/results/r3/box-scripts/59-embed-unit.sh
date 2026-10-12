#!/bin/bash
# R3-59 the token-embedding hypothesis (IQ2_S token_embd -> ggml_dequantize into an uninitialised
# output): tests/gpu/test_drafter_rows.py::test_embedding_rows at n = 1..64 (poisoned vs zeroed free
# memory, vs gguf-py, graph replay) on production's plugin build. GPU ~3 min.
# Output: /workspace/logs/r3/59-embed-unit/{summary.txt, test.log}
#   bash 59-embed-unit.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 59-embed-unit "$@"
if [ "$R3_PLAN" = 1 ]; then sed -n '2,6p' "$0"; exit 0; fi
r3_env
r3_preflight
[ -d "$R3_WT/build/pytest" ] || cp -a "$R3_PLUGIN_WT/build/pytest" "$R3_WT/build/pytest"
cd "$R3_WT" || r3_die "no $R3_WT"
export GSQ_RUNS=$L/runs
tools/pytest tests/gpu/test_drafter_rows.py -k embedding_rows -q -rs -p no:cacheprovider > "$L/test.log" 2>&1
rc=$?
r3_summary "R3-59 embedding rows 1..64 (box, $(date -u +%F)), plugin $(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD): rc=$rc $(tail -1 "$L/test.log")" "$(grep -E "^E  |FAILED" "$L/test.log" | head -12)"
cat "$L/summary.txt"
