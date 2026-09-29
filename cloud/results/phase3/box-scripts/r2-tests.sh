#!/bin/bash
# R2 tests (one gpuq job): packed parity (R1's + the tiled op, routing, layers), GPU guards -k packed,
# CPU guards.   [WT=/workspace/wt-r2] r2-tests.sh TAG [pytest -k expr]
source /workspace/box-env.sh
WT=${WT:-/workspace/wt-r2}
export VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 PYTHONPATH=$WT/plugin:$WT/tools
L=/workspace/logs/opt/r2; cd $WT; PT=/workspace/gsq-vllm/tools/pytest; TAG=$1; K=${2:-"packed or pack_roundtrip or tiled"}
$PT tests/cpu/test_lcpp_guards.py -q > $L/$TAG-cpu.log 2>&1; echo "cpu guards rc=$?"; tail -1 $L/$TAG-cpu.log
$PT tests/gpu/test_kernel_parity.py -q -k "$K" > $L/$TAG-parity.log 2>&1; echo "parity rc=$?"; tail -3 $L/$TAG-parity.log
$PT tests/gpu/test_kernel_guards.py -q -k "packed" > $L/$TAG-guards.log 2>&1; echo "guards rc=$?"; tail -3 $L/$TAG-guards.log
