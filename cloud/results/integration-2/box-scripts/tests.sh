#!/bin/bash
# integration-2, one gpuq job (after the in-place build): vendored sha256s, CPU guards + routing
# table, kernel parity with Route L on and off, and off on Integration 1's worktree (same
# session, for the stock-path count), GPU guards on the Route L ops.
source /workspace/wt-int2/cloud/results/integration-2/box-scripts/lib.sh
date -u +"start %FT%TZ"; echo "build rc=$(cat $L/build.rc)"; tail -1 $L/build.log
( cd plugin/vllm_gguf_plugin/csrc/lcpp && sed -n '/^```$/,/^```$/p' VENDORED.md | grep -E '^[0-9a-f]{64} ' | sha256sum -c --quiet && echo "vendored sha256 OK" )
tools/pytest tests/cpu/test_lcpp_guards.py tests/cpu/test_lcpp_routing.py -q > $L/cpu-guards.log 2>&1; echo "cpu rc=$?: $(tail -1 $L/cpu-guards.log)"
T=tests/gpu/test_kernel_parity.py
tools/pytest $T -q -rs > $L/parity-lcpp.log 2>&1; echo "parity lcpp rc=$?: $(tail -1 $L/parity-lcpp.log)"
env -u VLLM_GGUF_LCPP tools/pytest $T -q -rs > $L/parity-stock.log 2>&1; echo "parity stock rc=$?: $(tail -1 $L/parity-stock.log)"
( cd /workspace/wt-int; git log --oneline -1
  env -u VLLM_GGUF_LCPP PYTHONPATH=/workspace/wt-int/plugin:/workspace/wt-int/tools tools/pytest $T -q -rs ) > $L/parity-stock-int1.log 2>&1
echo "parity stock (Integration 1 worktree): $(tail -1 $L/parity-stock-int1.log)"
tools/pytest tests/gpu/test_kernel_guards.py -q -rs -k "lcpp or first_call" > $L/guards.log 2>&1; echo "guards rc=$?: $(tail -1 $L/guards.log)"
date -u +"end %FT%TZ"
