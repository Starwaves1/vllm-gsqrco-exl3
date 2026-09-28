#!/bin/bash
# phase 3: torch-profiler capture of c=1 decode steps (MTP k=3, CUDA graphs), production argv.
#   p3-profile.sh TAG gsq|baseline   -> /workspace/runs/p3-TAG-profile (trace, server.log, spec.prom)
set -uo pipefail
source /workspace/p3/p3-lib.sh
TAG=$1 KIND=${2:-gsq}
R=/workspace/runs/p3-$TAG-profile; rm -rf $R; mkdir -p $R
stopall; source scripts/env.sh
gsq_load_prod_argv
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$R/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":60,\"max_iterations\":6}"
if [ $KIND = gsq ]; then
  gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG" --profiler-config "$PROF"
  export VLLM_PLUGINS=$GSQ_PLUGINS_ON
else
  gsq_rewrite_argv "$GSQ_BASELINE_MODEL" --profiler-config "$PROF"
  export VLLM_PLUGINS=$GSQ_PLUGINS_OFF
fi
export VLLM_API_KEY=$GSQ_API_KEY
mkdir -p "$GSQ_KV_TIER_ROOT"
setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
SP=$!
gsq_wait_health 2400 $SP || { echo SERVER_FAILED; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader > $R/vram-after-load.txt
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
req() { curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of the printing press.\"}],\"max_tokens\":$1,\"temperature\":0}"; }
req 64 > /dev/null
curl -s -X POST "$GSQ_URL/start_profile" "${H[@]}"; echo " start_profile"
req 400 > $R/response.json
curl -s -X POST "$GSQ_URL/stop_profile" "${H[@]}"; echo " stop_profile"
sleep 20
curl -s "$GSQ_URL/metrics" "${H[@]}" | grep -E '^vllm:spec_decode' > $R/spec.prom
stopall
find $R/trace -type f | head -3; echo PROFILE_DONE
