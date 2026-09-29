#!/bin/bash
# c8prof.sh TAG: one server (worktree $WT), 8 concurrent decode requests, torch profile of 6 steps, then stop
source /workspace/logs/opt/k3/lib.sh
TAG=$1; R=/workspace/runs/opt-k3-$TAG-c8prof; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ"; stopall
gsq_load_prod_argv
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$R/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":20,\"max_iterations\":6}"
gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG" --profiler-config "$PROF"
export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
mkdir -p "$GSQ_KV_TIER_ROOT"
setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server.log; stopall; exit 1; }
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
req() { curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about topic $2 in the history of science.\"}],\"max_tokens\":$1,\"temperature\":0}"; }
req 64 x > /dev/null
pids=()
for i in 1 2 3 4 5 6 7 8; do req 700 $i > $R/resp-$i.json & pids+=($!); done
sleep 6
curl -s -X POST "$GSQ_URL/start_profile" "${H[@]}"; sleep 8; curl -s -X POST "$GSQ_URL/stop_profile" "${H[@]}"
wait "${pids[@]}"; sleep 15; stopall
T=$(find $R/trace -name "*.json" | head -1)
[ -n "$T" ] && $GSQ_VENV/bin/python $P/k3step.py "$T" > $P/$TAG-c8prof.txt && cat $P/$TAG-c8prof.txt
date -u +"end %FT%TZ"
