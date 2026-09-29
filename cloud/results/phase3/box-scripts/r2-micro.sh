#!/bin/bash
# R2: microbench / quick parity of lcpp_mul_mat_iq3_packed (one gpuq job).  r2-micro.sh TAG [r2_bench args]
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 WT=${WT:-/workspace/wt-r2}
export PYTHONPATH=$WT/plugin:$WT/tools:$WT/tests/gpu
L=/workspace/logs/opt/r2; TAG=$1; shift
cd $WT
/workspace/gsq-vllm/.venv/bin/python cloud/results/phase3/box-scripts/r2_bench.py --out $L/$TAG-micro.tsv "$@" > $L/$TAG-micro.log 2>&1
echo "rc=$?" >> $L/$TAG-micro.log
