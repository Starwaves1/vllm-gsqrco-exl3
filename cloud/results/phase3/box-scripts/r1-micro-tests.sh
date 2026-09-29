#!/bin/bash
# R1 it3: new packed tests + microbench 1..32 rows (packed vs dp4a / mma / MMQ)
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 PYTHONPATH=/workspace/wt-r1/plugin:/workspace/wt-r1/tools
L=/workspace/logs/opt/r1; cd /workspace/wt-r1; PY=/workspace/gsq-vllm/.venv/bin/python
/workspace/gsq-vllm/tools/pytest tests/gpu/test_kernel_parity.py -q -k "packed or pack_roundtrip" > $L/it3-parity.log 2>&1; echo "parity rc=$?"; tail -3 $L/it3-parity.log
/workspace/gsq-vllm/tools/pytest tests/gpu/test_kernel_guards.py -q -k "packed" > $L/it3-guards.log 2>&1; echo "guards rc=$?"; tail -3 $L/it3-guards.log
$PY bench/micro/gemm.py --types IQ3_S,IQ3_XXS --tokens 1,2,4,8,16,32 --variants lcpp_iq3,lcpp_iq3_mma,lcpp_iq3_mma_packed,lcpp_mmq --out $L/it3-micro.tsv > $L/it3-micro.log 2>&1; echo "micro rc=$?"; grep -E "n=|^IQ" $L/it3-micro.log
