#!/bin/bash
# opt-p2, one gpuq job = one stage: build in place, kernel parity (Route L on and off), one
# server: production run_benchmarks.sh single twice, decode only (bench/speed/run.sh gsq,
# unmodified, GSQ_PREFILL=" " skips the prefill ladder; clocks logged), then torch-profiler
# captures of 6 decode steps at each concurrency in PROF_C (default "1 4"; the profiler is
# configured at start and idle until /start_profile), stop, clear /dev/shm.
#   job.sh TAG      env: SKIP_PARITY=1, SKIP_BENCH=1, PROF_C="1 4 8"
TAG=$1; WT=/workspace/wt-p2-$TAG
source /workspace/wt-p2-$TAG/cloud/results/opt-p2/box-scripts/lib.sh
R=/workspace/runs/opt-p2-$TAG; rm -rf $R; mkdir -p $R
exec > >(tee $L/$TAG-job.log) 2>&1
date -u +"start %FT%TZ"; stopall
build
[ -z "${SKIP_PARITY:-}" ] && parity
serve --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$R/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":60,\"max_iterations\":6}"
if [ -z "${SKIP_BENCH:-}" ]; then
  OUT=$R GSQ_PREFILL=" " bench/speed/run.sh gsq > $L/$TAG-speed.log 2>&1; echo "speed rc=$?"
fi
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
req() { curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of the printing press, part $2.\"}],\"max_tokens\":$1,\"temperature\":0}" > /dev/null; }
req 64 0
for C in ${PROF_C:-1 4}; do
  curl -s -X POST "$GSQ_URL/start_profile" "${H[@]}"
  pids=(); for i in $(seq $C); do req 400 $i & pids+=($!); done; wait "${pids[@]}"
  curl -s -X POST "$GSQ_URL/stop_profile" "${H[@]}"; sleep 20
  T=$(ls -t $(find $R/trace -name "*.json") | head -1); mv "$T" $R/c$C.trace.json
done
stopall
grep -E "ROW cohort C. real prompts T=0|MTP whole|clocks.*decode" $R/summary.txt
for C in ${PROF_C:-1 4}; do echo "== c=$C"; $GSQ_VENV/bin/python $S/pstep2.py $R/c$C.trace.json; done > $L/$TAG-profile.txt
grep -E "^== |complete steps|copy/cast|q8_1 quantize|memset|iq1_m|IQ1_M" $L/$TAG-profile.txt
date -u +"end %FT%TZ"
