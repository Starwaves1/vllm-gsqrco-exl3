#!/bin/bash
# final-phase box helpers: source me. Worktree $WT (default /workspace/wt-final = main), its plugin
# via PYTHONPATH, shared venv, Route L on.
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_VENV=/workspace/gsq-vllm/.venv
WT=${WT:-/workspace/wt-final}
export PYTHONPATH=$WT/plugin:$WT/tools
L=/workspace/logs/final; mkdir -p $L
S=/workspace/wt-final/cloud/results/final/box-scripts
cd $WT
source scripts/env.sh
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
serve() { gsq_load_prod_argv
  gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG" "$@"
  export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
  mkdir -p "$GSQ_KV_TIER_ROOT"
  setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
  gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server.log; stopall; exit 1; }
  grep -E "Loading weights took|Model loading took|init engine|GPU KV cache size" $R/server.log | cut -c1-220 | head -6
  echo "VRAM after load: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"; }
