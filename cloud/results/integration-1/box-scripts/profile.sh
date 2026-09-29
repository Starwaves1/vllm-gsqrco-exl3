#!/bin/bash
# integration-1, one gpuq job = one server: torch-profiler captures of 6 decode steps at c=1 and
# c=4 (1 / 4 concurrent greedy requests; 4 / 16 rows per target pass with MTP k=3), per-class
# and per-type GEMM breakdown (pstep.py / pstep2.py, with the owned kernels classified).
source /workspace/wt-int/cloud/results/integration-1/box-scripts/lib.sh
R=/workspace/runs/int1-profile; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ"; stopall
serve --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$R/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":60,\"max_iterations\":6}"
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
req() { curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of the printing press, part $2.\"}],\"max_tokens\":$1,\"temperature\":0}" > /dev/null; }
req 64 0
for C in 1 4; do
  curl -s -X POST "$GSQ_URL/start_profile" "${H[@]}"
  pids=(); for i in $(seq $C); do req 400 $i & pids+=($!); done; wait "${pids[@]}"
  curl -s -X POST "$GSQ_URL/stop_profile" "${H[@]}"; sleep 20
  T=$(ls -t $(find $R/trace -name "*.json") | head -1); mv "$T" $R/c$C.trace.json
done
stopall
for C in 1 4; do echo "== c=$C"; $GSQ_VENV/bin/python $S/pstep2.py $R/c$C.trace.json; done > $L/profile.txt; cat $L/profile.txt
date -u +"end %FT%TZ"
