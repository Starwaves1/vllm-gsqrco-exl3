#!/bin/bash
# integration-2 box helpers: source me. Worktree /workspace/wt-int2 (branch integrate2), its plugin
# via PYTHONPATH, shared venv. Route L on unless a caller unsets VLLM_GGUF_LCPP.
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_VENV=/workspace/gsq-vllm/.venv
WT=/workspace/wt-int2
export PYTHONPATH=$WT/plugin:$WT/tools
L=/workspace/logs/int2; mkdir -p $L
S=$WT/cloud/results/integration-2/box-scripts
cd $WT
source scripts/env.sh
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
# production's argv with this worktree's hf-config; extra args appended. Prints the served
# config: draft head rows, load time, KV cache tokens, VRAM after load.
serve() { gsq_load_prod_argv
  gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG" "$@"
  export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
  mkdir -p "$GSQ_KV_TIER_ROOT"
  setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
  gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server.log; stopall; exit 1; }
  grep -E "draft head|speculative|Loading weights took|Model loading took|init engine|GPU KV cache size" $R/server.log | cut -c1-220 | head -8
  echo "VRAM after load: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"; }
