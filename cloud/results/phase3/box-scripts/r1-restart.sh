#!/bin/bash
# R1: warm restart of the $WT server (compile cache already populated): startup time and memory only.
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_VENV=/workspace/gsq-vllm/.venv
WT=${WT:-/workspace/wt-r1}
export PYTHONPATH=$WT/plugin:$WT/tools
cd $WT; source scripts/env.sh
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
TAG=$1; R=/workspace/runs/opt-r1-$TAG-restart; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ $WT"; stopall
gsq_load_prod_argv
gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG"
export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
mkdir -p "$GSQ_KV_TIER_ROOT"
for i in 1 2; do  # the first start after a code change may recompile; the second shows the warm start
T0=$(date +%s)
setsid "${GSQ_ARGV[@]}" > $R/server$i.log 2>&1 &
gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server$i.log; stopall; exit 1; }
echo "start $i: server up after $(( $(date +%s) - T0 )) s"
grep -E "Model loading took|GPU KV cache size|Available KV cache memory|init engine" $R/server$i.log | cut -c1-240
nvidia-smi --query-gpu=memory.used --format=csv,noheader
stopall
done
date -u +"end %FT%TZ"
