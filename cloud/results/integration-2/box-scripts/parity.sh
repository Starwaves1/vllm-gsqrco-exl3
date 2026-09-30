#!/bin/bash
# integration-2: vLLM-side logit parity against the phase-1b llama.cpp CUDA dumps (bf16 KV, not
# regenerated; same 11 prompts; per-position KLD / top-1 kept in parity.json, the vLLM dumps kept
# on the box), then MTP greedy acceptance on a production-argv server against the phase-1b
# llama-server readout (bench/speed/mtp_acceptance.py). One gpuq job.
source /workspace/wt-int2/cloud/results/integration-2/box-scripts/lib.sh
P=/workspace/runs/p1b-parity O=/workspace/runs/int2-parity; rm -rf $O; mkdir -p $O/vllm
date -u +"start %FT%TZ"; stopall
$GSQ_VENV/bin/python bench/parity/vllm_logprobs.py -d $P/prompts -o $O/vllm --gguf $GSQ_GGUF > $O/vllm.log 2>&1; echo "vllm rc=$?"
$GSQ_VENV/bin/python bench/parity/compare.py -d $P/prompts -l $P/llama -v $O/vllm --json $O/parity.json > $O/compare.txt 2>&1; echo "compare rc=$?"
cat $O/compare.txt; box_clean_shm
R=/workspace/runs/int2-mtp; rm -rf $R; mkdir -p $R
serve
$GSQ_VENV/bin/python bench/speed/mtp_acceptance.py --engine vllm --url $GSQ_URL --out $R/vllm.json \
  --reference /workspace/runs/p1b-mtp/llama.json > $R/vllm.txt 2>&1; echo "mtp rc=$?"; cat $R/vllm.txt
stopall
date -u +"end %FT%TZ"
