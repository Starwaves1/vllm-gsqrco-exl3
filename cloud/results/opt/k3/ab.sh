#!/bin/bash
# ab.sh TAG ROOT TYPES TOKENS CFGS [SHAPES]: interleaved A/B (k3_bench.py) on worktree ROOT
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 PYTHONPATH=$2/plugin:$2/tools K3_ROOT=$2
L=/workspace/logs/opt/k3; cd $2
/workspace/gsq-vllm/.venv/bin/python $L/k3_bench.py $L/$1-ab.tsv $3 $4 "$5" ${6:-17408x5120,5120x17408,10240x5120} > $L/$1-ab.log 2>&1; rc=$?; cat $L/$1-ab.log; exit $rc
