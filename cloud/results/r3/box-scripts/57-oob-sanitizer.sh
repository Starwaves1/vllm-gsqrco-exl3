#!/bin/bash
# R3-57 out-of-bounds hunt for the >= 9-running corruption: the Route L paths at the row counts the
# k=2 tier makes (draft loop 9..16, verify / padded first draft pass 27..48, odd counts included),
# on the drafter's tensors, the lm_heads and one target tensor per quant type, plus a mixed-type
# fused qkv with apply()'s shared x_q8 (tests/gpu/test_drafter_rows.py).
#   1. the whole file, normally (finite, padding-independent, graph replay, vs dequant)
#   2. compute-sanitizer memcheck, torch caching allocator off (every tensor its own allocation):
#      test_rows_exact + test_rows_exact_mixed_qkv at n = 9, 13, 16, 27, 29, 31, 33, 41, 47, 48
#   3. initcheck on the same (uninitialised reads, e.g. a q8 tail or scratch)
# Plugin: this checkout's Python + the 32ae6ec .so (csrc unchanged). Output: /workspace/logs/r3/57-oob-sanitizer/
# GPU time ~40 min.
#   bash 57-oob-sanitizer.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 57-oob-sanitizer "$@"
if [ "$R3_PLAN" = 1 ]; then sed -n '2,13p' "$0"; exit 0; fi
base=$(git -C "$R3_PLUGIN_WT" rev-parse HEAD)
git -C "$R3_WT" diff --quiet "$base" HEAD -- plugin/vllm_gguf_plugin/csrc plugin/setup.py || r3_die "csrc differs from $R3_PLUGIN_WT"
cp "$R3_PLUGIN_WT/plugin/vllm_gguf_plugin/_C_gguf.abi3.so" "$R3_WT/plugin/vllm_gguf_plugin/"
[ -d "$R3_WT/build/pytest" ] || cp -a "$R3_PLUGIN_WT/build/pytest" "$R3_WT/build/pytest"
export R3_PLUGIN_WT=$R3_WT
r3_env
r3_preflight
cd "$R3_WT" || r3_die "no $R3_WT"
export VLLM_GGUF_LCPP=1 GSQ_RUNS=$L/runs
T=tests/gpu/test_drafter_rows.py
SAN=$(command -v compute-sanitizer || echo /usr/local/cuda/bin/compute-sanitizer)
[ -x "$SAN" ] || r3_die "no compute-sanitizer"
r3_step "plain"
tools/pytest $T -q -rs -s -p no:cacheprovider > "$L/plain.log" 2>&1; rc0=$?
ids=$(tools/pytest $T --collect-only -q 2>/dev/null | grep -E "test_rows_exact(_mixed_qkv)?\[" | grep -E "[-\[](9|13|16|27|29|31|33|41|47|48)[\]-]")
echo "$ids" > "$L/sanitizer-ids.txt"; echo "sanitizer cases: $(echo "$ids" | wc -l)"
r=()
for tool in memcheck initcheck; do
  r3_step "$tool"
  # shellcheck disable=SC2086
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 "$SAN" --tool $tool --error-exitcode 99 --print-limit 30 \
    tools/pytest -q -p no:cacheprovider $ids > "$L/$tool.log" 2>&1
  r+=("$tool rc=$? $(grep -E "passed|failed" "$L/$tool.log" | tail -1) | $(grep -m1 -E "ERROR SUMMARY|Invalid __global__|Uninitialized" "$L/$tool.log")")
done
r3_summary "R3-57 OOB hunt (box, $(date -u +%F)), plugin $(git -C "$R3_WT" rev-parse --short HEAD)" \
  "plain: rc=$rc0 $(grep -E "passed|failed" "$L/plain.log" | tail -1)" "$(grep -E "FAILED|AssertionError" "$L/plain.log" | head -10)" "${r[@]}" \
  "first sanitizer errors:" "$(grep -m6 -E -A4 "Invalid __global__|Uninitialized __global__" "$L/memcheck.log" "$L/initcheck.log" | cut -c1-200)"
cat "$L/summary.txt"
