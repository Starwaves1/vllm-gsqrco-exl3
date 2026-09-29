#!/bin/bash
# opt-p: torch-profiler captures of 6 decode steps at c=4 and at c=8 (4 / 8 concurrent greedy requests,
# 16 / 32 rows per target pass with MTP k=3) on one server; per-type GEMM breakdown (pstep2.py).
#   profile48.sh TAG     (WT=<stage worktree>)
source /workspace/logs/opt/p/lib.sh
TAG=$1; R=/workspace/runs/opt-p-$TAG-prof48; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ"; stopall
gsq_load_prod_argv
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$R/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":60,\"max_iterations\":6}"
gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG" --profiler-config "$PROF"
export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
mkdir -p "$GSQ_KV_TIER_ROOT"
setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server.log; stopall; exit 1; }
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
req() { curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of the printing press, part $2.\"}],\"max_tokens\":$1,\"temperature\":0}" > /dev/null; }
req 64 0
for C in ${CS:-4 8}; do
  curl -s -X POST "$GSQ_URL/start_profile" "${H[@]}"
  pids=(); for i in $(seq $C); do req 400 $i & pids+=($!); done; wait "${pids[@]}"  # not the server
  curl -s -X POST "$GSQ_URL/stop_profile" "${H[@]}"; sleep 20
  T=$(ls -t $(find $R/trace -name "*.json") | head -1); mv "$T" $R/c$C.trace.json
done
stopall
for C in ${CS:-4 8}; do echo "== c=$C"; $GSQ_VENV/bin/python $P/pstep2.py $R/c$C.trace.json; done > $P/$TAG-prof48.txt; cat $P/$TAG-prof48.txt
date -u +"end %FT%TZ"
