#!/bin/bash
# R3-31 the IQ3 repack GPU tests on main 32ae6ec (62f27c0 changed pack_'s chunking) WITH Route L on:
# tests/gpu/test_kernel_parity.py -k "pack or packed" under VLLM_GGUF_LCPP=1, so the layer-level
# tests (test_lcpp_packed_layer, test_routing_packed_whole_tensor: the load path's pack_) run too;
# the earlier gsq-pack-32ae6ec job ran without it and skipped those 30.
# Output: /workspace/logs/r3/31-pack-tests-lcpp/{summary.txt, test.log, junit.xml}. GPU time ~3 min.
#   bash 31-pack-tests-lcpp.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 31-pack-tests-lcpp "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,7p' "$0"; exit 0; fi
r3_env
r3_preflight
r3_step tests
cd "$R3_PLUGIN_WT" || r3_die "no $R3_PLUGIN_WT"
[ "$(git rev-parse --short HEAD)" = 32ae6ec ] || r3_die "plugin worktree is not 32ae6ec"
export GSQ_RUNS=$L/runs
tools/pytest tests/gpu/test_kernel_parity.py -k "pack or packed" -q -rs -p no:cacheprovider \
  --junitxml="$L/junit.xml" > "$L/test.log" 2>&1
rc=$?
r3_summary "R3-31 pack tests with VLLM_GGUF_LCPP=1 on $(git rev-parse --short HEAD) (box, $(date -u +%F)): rc=$rc" "$(tail -4 "$L/test.log")"
cat "$L/summary.txt"
[ $rc = 0 ] || r3_die "pack tests failed (rc=$rc)"
