#!/bin/bash
# model-matrix box helpers: source me. Worktree /workspace/wt-models = main + the model-matrix
# commits; its plugin .so is wt-final's build (b9cdfa5; plugin C++/CUDA sources identical, sha256
# 56a6a390...). Route L on for the GGUF rows. Shared venv unless the caller sets GSQ_VENV.
source /workspace/box-env.sh
WT=/workspace/wt-models
export VLLM_GGUF_LCPP=1 GSQ_VENV=${GSQ_VENV:-/workspace/gsq-vllm/.venv} PYTHONPATH=$WT/plugin:$WT/tools
L=/workspace/logs/models; mkdir -p $L
S=$WT/cloud/results/models/box-scripts
LB=/workspace/ref/llama.cpp-b11211/build/bin
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
