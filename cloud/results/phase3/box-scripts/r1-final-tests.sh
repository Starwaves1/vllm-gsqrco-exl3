#!/bin/bash
# R1 final: full kernel parity (Route L on), GPU guards, CPU guards, compute-sanitizer
# memcheck + initcheck on packed-kernel / unpack cases (exact allocations, no capture).
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 PYTHONPATH=/workspace/wt-r1/plugin:/workspace/wt-r1/tools
L=/workspace/logs/opt/r1; cd /workspace/wt-r1; PT=/workspace/gsq-vllm/tools/pytest
$PT tests/cpu/test_lcpp_guards.py tests/cpu/test_iq3_pack.py -q > $L/final-cpu.log 2>&1; echo "cpu guards + pack rc=$?"; tail -1 $L/final-cpu.log
( cd plugin && CUDA_VISIBLE_DEVICES= PYTHONPATH=/workspace/wt-r1/plugin $PT tests/test_plugin.py tests/diffusion tests/test_gemma4_adapter.py tests/test_gguf_utils.py tests/test_ggml_common_tables.py -q ) > $L/final-plugin-cpu.log 2>&1; echo "plugin cpu rc=$?"; tail -1 $L/final-plugin-cpu.log
$PT tests/gpu/test_kernel_parity.py -q > $L/final-parity.log 2>&1; echo "parity rc=$?"; tail -3 $L/final-parity.log
$PT tests/gpu/test_kernel_guards.py -q -k "iq3" > $L/final-guards.log 2>&1; echo "guards rc=$?"; tail -3 $L/final-guards.log
ids=$($PT tests/gpu/test_kernel_parity.py --collect-only -q 2>/dev/null | grep -E "test_lcpp_iq3_packed\[" | grep -E "float32" | grep -E "\-(1|8|9|16|32)-" | grep -E "real|k_min|few_rows|k_tail")
echo "sanitizer cases: $(echo $ids | wc -w)"
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 20 \
    $PT -q $ids "tests/gpu/test_kernel_parity.py::test_lcpp_packed_layer[8-z_iq3]" "tests/gpu/test_kernel_parity.py::test_lcpp_packed_layer[8-z_other]" "tests/gpu/test_kernel_parity.py::test_lcpp_packed_layer[128-z_iq3]" "tests/gpu/test_kernel_parity.py::test_lcpp_packed_layer[128-z_other]" > $L/final-sanitizer-$tool.log 2>&1
  echo "$tool rc=$?"; grep -E "passed|failed" $L/final-sanitizer-$tool.log | tail -1; tail -1 $L/final-sanitizer-$tool.log
done
