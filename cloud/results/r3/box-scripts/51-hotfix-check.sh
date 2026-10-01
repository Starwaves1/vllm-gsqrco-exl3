#!/bin/bash
# R3-51 hotfix check for the torture-smoke prompt_logprobs / echo failures. Plugin under test:
# this checkout ($R3_WT, branch r3 or hotfix-prompt-logprobs), csrc identical to the 32ae6ec build,
# so its _C_gguf .so is copied from $R3_BUILT_WT instead of rebuilt (the copy is refused if csrc
# or setup.py differ).
#   1. tests/gpu test_lm_head_prompt_rows (lm_head 248,320 rows at 1..3,936 rows: finite, vs
#      dequantized fp32, peak memory bound) on the OLD build (expected to fail the bound) and the
#      fixed one (must pass)
#   2. the whole tests/gpu/test_kernel_parity.py with VLLM_GGUF_LCPP=1 on the fixed plugin (parity rerun)
#   3. 50-plp-repro on the fixed plugin (servers: GSQ and stock W4A16), logs in 50-plp-repro-fixed
# Output: /workspace/logs/r3/51-hotfix-check/{summary.txt, *.log}
# GPU time: ~35 min.
#   bash 51-hotfix-check.sh [--plan]
source "$(dirname "$0")/lib.sh"
R3_BUILT_WT=${R3_BUILT_WT:-/workspace/wt-gsq-32ae6ec}
r3_init 51-hotfix-check "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,15p' "$0"; exit 0; fi
r3_step "copy the built .so"
base=$(git -C "$R3_BUILT_WT" rev-parse HEAD)
git -C "$R3_WT" diff --quiet "$base" HEAD -- plugin/vllm_gguf_plugin/csrc plugin/setup.py \
  || r3_die "csrc/setup.py differ from $R3_BUILT_WT ($base): rebuild instead"
cp "$R3_BUILT_WT/plugin/vllm_gguf_plugin/_C_gguf.abi3.so" "$R3_WT/plugin/vllm_gguf_plugin/_C_gguf.abi3.so"
[ -d "$R3_WT/build/pytest" ] || cp -a "$R3_BUILT_WT/build/pytest" "$R3_WT/build/pytest"
cd "$R3_WT" || r3_die "no $R3_WT"
export GSQ_RUNS=$L/runs
t() {  # label plugin-wt pytest-args...
  local label=$1 wt=$2; shift 2
  ( export PYTHONPATH=$wt/plugin:$wt/tools VLLM_GGUF_LCPP=1
    "$R3_WT/tools/pytest" "$@" -q -rs -s --junitxml="$L/$label.xml" ) > "$L/$label.log" 2>&1
  local rc=$?
  r3_summary "$label (plugin $(git -C "$wt" rev-parse --short HEAD)): rc=$rc $(tail -1 "$L/$label.log")"
  grep -E "^lm_head .* n=" "$L/$label.log" | sed 's/^/  /' >> "$L/summary.txt"
  return $rc
}
r3_env
r3_preflight
r3_summary "R3-51 hotfix check (box, $(date -u +%F)); fixed plugin = $R3_WT ($(git -C "$R3_WT" rev-parse --short HEAD))"
r3_step "lm_head test, old build"
t lmhead-old "$R3_BUILT_WT" tests/gpu/test_kernel_parity.py -k lm_head_prompt_rows || true
r3_step "lm_head test, fixed"
t lmhead-fixed "$R3_WT" tests/gpu/test_kernel_parity.py -k lm_head_prompt_rows || r3_die "lm_head test fails on the fixed plugin"
r3_step "parity, fixed"
t parity-fixed "$R3_WT" tests/gpu/test_kernel_parity.py || r3_die "kernel parity fails on the fixed plugin"
r3_step "repro on the fixed plugin"
( R3_TAG=fixed R3_PLUGIN_WT=$R3_WT bash "$R3_S/50-plp-repro.sh" ) > "$L/repro-fixed.log" 2>&1
r3_summary "repro on fixed plugin: rc=$? (see $R3_LOGS/50-plp-repro-fixed/summary.txt)" \
  "$(cut -c1-300 "$R3_LOGS/50-plp-repro-fixed/summary.txt" 2>/dev/null | head -40)"
cat "$L/summary.txt"
