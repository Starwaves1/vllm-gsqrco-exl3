#!/bin/bash
# opt-p: one gpuq job = one server: production run_benchmarks.sh single (2 passes, decode only,
# no prefill ladder), optional AFTER command, then a torch-profiler capture of 6 c=1 decode steps
# (phase 3's p3-profile.sh request; the profiler is configured at start and idle until
# /start_profile), then stop the server and clear /dev/shm.
#   speed.sh TAG     env: GSQ_PROD_ARGV=<argv file> (k sweep), MTP_DRAFT_VOCAB=0, AFTER=<command>
source /workspace/logs/opt/p/lib.sh
TAG=$1; R=/workspace/runs/opt-p-$TAG-speed; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ"; stopall
gsq_load_prod_argv
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$R/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":60,\"max_iterations\":6}"
gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG" --profiler-config "$PROF"
export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
mkdir -p "$GSQ_KV_TIER_ROOT"
setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server.log; stopall; exit 1; }
grep -E "draft head|speculative|Loading weights took|GPU KV cache size" $R/server.log | cut -c1-220 | head -6
OUT=$R GSQ_PREFILL=" " bench/speed/run.sh gsq > $P/$TAG-speed.log 2>&1; echo "speed rc=$?"
[ -n "${AFTER:-}" ] && { eval "$AFTER"; echo "after rc=$?"; }
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
req() { curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of the printing press.\"}],\"max_tokens\":$1,\"temperature\":0}"; }
req 64 > /dev/null
curl -s -X POST "$GSQ_URL/start_profile" "${H[@]}"; req 400 > $R/profile-response.json
curl -s -X POST "$GSQ_URL/stop_profile" "${H[@]}"; sleep 20
stopall
grep -E "T=0|MTP|clocks" $R/summary.txt | tail -9
T=$(find $R/trace -name "*.json" | head -1)
[ -n "$T" ] && $GSQ_VENV/bin/python $P/pstep.py "$T" > $P/$TAG-profile.txt && cat $P/$TAG-profile.txt
date -u +"end %FT%TZ"
