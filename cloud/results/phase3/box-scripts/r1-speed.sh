#!/bin/bash
# R1: one gpuq job = one server: production run_benchmarks.sh single (2 passes, decode only),
# then stop the server and clear /dev/shm. Plugin from $WT (wt-r1, or wt-k2 for the base)
# via PYTHONPATH, shared venv.   speed.sh TAG
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_VENV=/workspace/gsq-vllm/.venv
WT=${WT:-/workspace/wt-r1}
export PYTHONPATH=$WT/plugin:$WT/tools
P=/workspace/logs/opt/r1
cd $WT
source scripts/env.sh
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
TAG=$1; R=/workspace/runs/opt-r1-$TAG-speed; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ $WT"; stopall
gsq_load_prod_argv
gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG"
export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
mkdir -p "$GSQ_KV_TIER_ROOT"
T0=$(date +%s)
setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server.log; stopall; exit 1; }
echo "server up after $(( $(date +%s) - T0 )) s"
grep -E "Loading weights took|Model loading took|GPU KV cache size|Available KV cache memory|weights memory|init engine" $R/server.log | cut -c1-240 | head -8
nvidia-smi --query-gpu=memory.used --format=csv,noheader
OUT=$R GSQ_PREFILL="8192:1:4" bench/speed/run.sh gsq > $P/$TAG-speed.log 2>&1; echo "speed rc=$?"
stopall
grep -E "T=0|MTP|prefill" $R/summary.txt | tail -12
date -u +"end %FT%TZ"
