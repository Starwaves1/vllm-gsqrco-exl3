#!/bin/bash
# opt-p box helpers: source me. Route L on, the plugin of worktree $WT (default wt-p) via
# PYTHONPATH, shared venv.
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_VENV=/workspace/gsq-vllm/.venv
WT=${WT:-/workspace/wt-p}
export PYTHONPATH=$WT/plugin:$WT/tools
P=/workspace/logs/opt/p; mkdir -p $P
cd $WT
source scripts/env.sh
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
