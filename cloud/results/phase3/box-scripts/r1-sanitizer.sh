#!/bin/bash
# R1 sanitizer rerun (fixed test ids): full kernel parity (Route L on), GPU guards, CPU guards, compute-sanitizer
# memcheck + initcheck on packed-kernel / unpack cases (exact allocations, no capture).
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 PYTHONPATH=/workspace/wt-r1/plugin:/workspace/wt-r1/tools
L=/workspace/logs/opt/r1; cd /workspace/wt-r1; PT=/workspace/gsq-vllm/tools/pytest
ids=$($PT tests/gpu/test_kernel_parity.py --collect-only -q 2>/dev/null | grep -E "test_lcpp_iq3_packed\[" | grep -E "float32" | grep -E "\-(1|8|9|16|32)-" | grep -E "real|k_min|few_rows|k_tail")
echo "sanitizer cases: $(echo $ids | wc -w)"
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 20 \
    $PT -q $ids "tests/gpu/test_kernel_parity.py::test_lcpp_packed_layer[8-z_iq3]" "tests/gpu/test_kernel_parity.py::test_lcpp_packed_layer[8-z_other]" "tests/gpu/test_kernel_parity.py::test_lcpp_packed_layer[128-z_iq3]" "tests/gpu/test_kernel_parity.py::test_lcpp_packed_layer[128-z_other]" > $L/final-sanitizer-$tool.log 2>&1
  echo "$tool rc=$?"; grep -E "passed|failed" $L/final-sanitizer-$tool.log | tail -1; tail -1 $L/final-sanitizer-$tool.log
done
