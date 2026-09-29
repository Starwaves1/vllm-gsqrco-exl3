#!/bin/bash
# integration-1, one gpuq job: build in place, CPU guards, kernel parity with Route L on and off
# (off also on the box's main checkout, for the stock-path count), compute-sanitizer memcheck +
# initcheck on the owned-kernel cases, GPU guards on the Route L ops (stock kernels untouched).
source /workspace/wt-int/cloud/results/integration-1/box-scripts/lib.sh
date -u +"start %FT%TZ"
( source tools/cuda-env.sh; export PATH=$GSQ_VENV/bin:$PATH; cd plugin
  VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=24 python setup.py build_ext --inplace ) > $L/build.log 2>&1
echo "build rc=$?"; tail -2 $L/build.log
tools/pytest tests/cpu/test_lcpp_guards.py tests/cpu/test_lcpp_routing.py -q > $L/cpu-guards.log 2>&1; tail -1 $L/cpu-guards.log
T=tests/gpu/test_kernel_parity.py
tools/pytest $T -q -rs > $L/parity-lcpp.log 2>&1; echo "lcpp rc=$?" >> $L/parity-lcpp.log; tail -2 $L/parity-lcpp.log
env -u VLLM_GGUF_LCPP tools/pytest $T -q -rs > $L/parity-stock.log 2>&1; echo "stock rc=$?" >> $L/parity-stock.log; tail -2 $L/parity-stock.log
( cd /workspace/gsq-vllm; git log --oneline -1
  env -u VLLM_GGUF_LCPP PYTHONPATH=/workspace/gsq-vllm/plugin:/workspace/gsq-vllm/tools tools/pytest $T -q ) > $L/parity-stock-main.log 2>&1
tail -1 $L/parity-stock-main.log
ids=$(tools/pytest $T --collect-only -q 2>/dev/null | grep -E "^$T::(test_lcpp_iq3\[.*-(1-bfloat16-real|4-bfloat16-row_tail|8-float16-k_tail|6-bfloat16-odd_rows|7-bfloat16-k_min|3-float16-few_rows|5-float32-row_tail|8-float32-real)\]|test_lcpp_x_q8\[(IQ3_S|IQ3_XXS|Q4_K|IQ2_S)-(4|8)\]|test_quantize_x_q8_1_mixed_route|test_lcpp_iq3\[.*own-(3|8)-bfloat16-many_tiles\]|test_lcpp_mixed_shard_layer\[(4|6|8)-(IQ4_XS\+Q4_K|IQ3_XXS\+IQ2_S)\]|test_lcpp_same_type_run\[(4|6|8)-qkv_widest-(IQ3_XXS\+Q4_K|Q4_K\+IQ3_S)\])")
echo "$ids" > $L/sanitizer-ids.txt; echo "sanitizer cases: $(echo "$ids" | wc -l)"
for tool in memcheck initcheck; do
  PYTORCH_NO_CUDA_MEMORY_CACHING=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 20 \
    tools/pytest -q $ids > $L/sanitizer-$tool.log 2>&1
  echo "$tool rc=$?"; grep -E "passed|failed" $L/sanitizer-$tool.log | tail -1; tail -1 $L/sanitizer-$tool.log
done
[ -n "${SKIP_GUARDS:-}" ] || { tools/pytest tests/gpu/test_kernel_guards.py -q -rs -k "lcpp or first_call" > $L/guards.log 2>&1; echo "guards rc=$?"; tail -1 $L/guards.log; }
date -u +"end %FT%TZ"
