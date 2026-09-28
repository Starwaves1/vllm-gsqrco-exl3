#!/bin/bash
# phase 2 step 4b: torch-profiler capture of c=1 decode steps (MTP k=3, CUDA graphs) under Route L.
# Production argv via env.sh's own rewrite, plus --profiler-config (6 engine steps after 60).
set -uo pipefail
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1
cd /workspace/gsq-vllm; source scripts/env.sh
R=/workspace/runs/p2-profile; rm -rf $R; mkdir -p $R
box_clean_shm; rm -rf /workspace/kvtier
gsq_load_prod_argv
gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG" \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$R/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":60,\"max_iterations\":6}"
export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
mkdir -p "$GSQ_KV_TIER_ROOT"
setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
SP=$!
gsq_wait_health 2400 $SP || { echo SERVER_FAILED; exit 1; }
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
req() { curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of the printing press.\"}],\"max_tokens\":$1,\"temperature\":0}"; }
req 64 > /dev/null
curl -s -X POST "$GSQ_URL/start_profile" "${H[@]}"; echo " start_profile"
req 400 > $R/response.json
curl -s -X POST "$GSQ_URL/stop_profile" "${H[@]}"; echo " stop_profile"
sleep 20
curl -s "$GSQ_URL/metrics" "${H[@]}" | grep -E '^vllm:spec_decode' > $R/spec.prom
pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier
find $R/trace -type f | head; echo PROFILE_DONE
