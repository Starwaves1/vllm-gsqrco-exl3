#!/bin/bash
# p1-ab.sh gsq|baseline full|clock : serve-<kind>.sh on production's argv, sample GPU clocks,
# decode cohorts with production run_benchmarks.sh's command (real prompts, 8 x 1024 out, T=default,
# c=1,2) and bench/speed/run.sh's salted prefill ladder. 'clock' = one short decode cohort only.
set -uo pipefail
KIND=$1 MODE=$2
source /workspace/box-env.sh
cd /workspace/gsq-vllm; source scripts/env.sh
OUT=/workspace/runs/p1-ab-$KIND-$MODE; mkdir -p $OUT
box_clean_shm || exit 1
[ $KIND = gsq ] && TOK=$GSQ_HF_CONFIG || TOK=$GSQ_BASELINE_MODEL
GSQ_LOG=$OUT/server.log "scripts/serve-$KIND.sh" > /dev/null 2>&1 &
SERVER_PID=$!
gsq_wait_health 2400 "$SERVER_PID" || { echo SERVER_FAILED; exit 1; }
export VLLM_API_KEY=$GSQ_API_KEY OPENAI_API_KEY=$GSQ_API_KEY
M() { curl -s "$GSQ_URL/metrics" -H "Authorization: Bearer $GSQ_API_KEY"; }
B=("$GSQ_VENV/bin/vllm" bench serve --host 127.0.0.1 --port "$GSQ_PORT" --model "$TOK" --served-model-name qwen3.8-27b)
num() { awk "/$1/ {print \$$2}" "$3"; }
clk() { nohup nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv,noheader -l 1 > $OUT/clocks-$1.csv 2>&1 & CLK=$!; }
cohort() { local c=$1 n=$2 log=$OUT/cohort_c$1.log
  M | grep -E '^vllm:spec_decode' > $OUT/spec_c$c.before
  clk decode-c$c
  "${B[@]}" --dataset-name custom --dataset-path /workspace/deploy/bench/prompts_real.jsonl --custom-output-len 1024 \
    --num-prompts $n --max-concurrency $c > $log 2>&1
  kill $CLK; M | grep -E '^vllm:spec_decode' > $OUT/spec_c$c.after
  echo "ROW $KIND cohort C$c real prompts T=default n=$n | e2e=$(num "Output token throughput" 5 $log) tok/s | decode(C/meanTPOT)=$(python3 -c "print(f'{$c*1000/$(num "Mean TPOT" 4 $log):.1f}')") | meanTTFT=$(num "Mean TTFT" 4 $log) ms"
}
pf() { local len=$1 c=$2 n=$3 seed=$((RANDOM * 32768 + RANDOM)) log=$OUT/prefill_${1}_c$2.log
  clk prefill-$1-c$2
  "${B[@]}" --dataset-name random --random-input-len "$len" --random-output-len 1 \
    --num-prompts "$n" --max-concurrency "$c" --seed "$seed" > "$log" 2>&1
  kill $CLK
  local in dur; in=$(num "Total input tokens" 4 "$log"); dur=$(num "Benchmark duration" 4 "$log")
  echo "ROW $KIND prefill len=$len conc=$c seed=$seed | $(python3 -c "print(f'{$in/$dur:.0f}')") tok/s | meanTTFT=$(num "Mean TTFT" 4 "$log") ms"
}
{
  "${B[@]}" --dataset-name random --random-input-len 256 --random-output-len 64 --num-prompts 4 --max-concurrency 2 > $OUT/warmup.log 2>&1
  if [ $MODE = clock ]; then cohort 1 2
  elif [ $MODE = ladderclock ]; then cohort 1 2; pf 8192 1 8; pf 8192 2 8; pf 65536 1 2; pf 65536 2 4; pf 180000 1 2
  elif [ $MODE = tail ]; then pf 65536 1 1; pf 180000 1 1
  else cohort 1 8; cohort 2 8; pf 8192 1 8; pf 8192 2 8; pf 65536 1 1; pf 180000 1 1; fi
} | tee $OUT/summary.txt
kill -INT $SERVER_PID; for i in $(seq 60); do kill -0 $SERVER_PID 2>/dev/null || break; sleep 2; done
pkill -9 -f "VLLM::EngineCore"; sleep 3; box_clean_shm
echo "AB_DONE $KIND $MODE"
