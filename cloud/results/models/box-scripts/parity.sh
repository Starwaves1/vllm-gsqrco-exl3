#!/bin/bash
# Base GGUF logit parity on the 6 short prompts (seq_000-005, phase-1b prompt set): llama.cpp b11211
# reference dumps (llama_logits, f16 KV) then the vLLM side (bench/parity/vllm_logprobs.py, bf16
# KV, Route L on, hf-config/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp), compare.py. One gpuq job.
source /workspace/wt-models/cloud/results/models/box-scripts/lib.sh
export GSQ_MODEL_NAME=Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp GSQ_GGUF=/workspace/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
cd $WT; source scripts/env.sh
P=/workspace/runs/p1b-parity/prompts O=/workspace/runs/models/base-parity; rm -rf $O; mkdir -p $O/prompts $O/llama $O/vllm
for n in seq_000 seq_001 seq_002 seq_003 seq_004 seq_005; do ln -s $P/$n.ids $P/$n.pos $O/prompts/; echo $n >> $O/prompts/manifest.txt; done
python3 -c "import json; m=json.load(open('$P/manifest.json')); m['sequences']=[s for s in m['sequences'] if s['name']<='seq_005']; json.dump(m, open('$O/prompts/manifest.json','w'), indent=1)"
date -u +"start %FT%TZ"; stopall
/workspace/ref/runs/parity-bin/llama_logits -m $GSQ_GGUF -d $O/prompts -o $O/llama --kv f16 > $O/llama.log 2>&1; echo "llama rc=$?"
date -u +"llama done %FT%TZ"
$GSQ_VENV/bin/python bench/parity/vllm_logprobs.py -d $O/prompts -o $O/vllm --gguf $GSQ_GGUF > $O/vllm.log 2>&1; echo "vllm rc=$?"
grep -m1 -o "hf_config_path=[^,]*" $O/vllm.log; grep -m1 "GPU KV cache size" $O/vllm.log | cut -c1-200
$GSQ_VENV/bin/python bench/parity/compare.py -d $O/prompts -l $O/llama -v $O/vllm --json $O/parity.json > $O/compare.txt 2>&1; echo "compare rc=$? (exit 1 expected: no >=100k prompt in this set)"
cat $O/compare.txt; stopall
date -u +"end %FT%TZ"
