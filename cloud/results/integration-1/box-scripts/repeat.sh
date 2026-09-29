#!/bin/bash
# integration-1: repeat the vLLM side of logit parity on seq_009/seq_010 (one process start per
# job), against the phase-1b llama.cpp dumps; spike.py lists the largest-KLD positions.
#   repeat.sh TAG
source /workspace/wt-int/cloud/results/integration-1/box-scripts/lib.sh
P=/workspace/runs/p1b-parity O=/workspace/runs/int1-repeat-$1; rm -rf $O; mkdir -p $O/vllm
date -u +"start %FT%TZ"; stopall
$GSQ_VENV/bin/python bench/parity/vllm_logprobs.py -d $P/prompts -o $O/vllm --gguf $GSQ_GGUF \
  --only seq_009,seq_010 > $O/vllm.log 2>&1; echo "vllm rc=$?"
$GSQ_VENV/bin/python $S/spike.py $P/prompts $P/llama $O/vllm $GSQ_HF_CONFIG $O/kld.json seq_009 seq_010 \
  > $O/spike.txt 2>&1; echo "spike rc=$?"; cat $O/spike.txt
rm -rf $O/vllm
date -u +"end %FT%TZ"
