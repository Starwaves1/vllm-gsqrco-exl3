#!/bin/bash
# Step 5 trimmed: after prod bench pass-1 cohorts T=default c=1,2 finish, stop run.sh and run
# run.sh's salted prefill ladder (same commands) + MTP acceptance on a fresh serve-gsq.sh.
set -uo pipefail
source /workspace/box-env.sh
cd /workspace/gsq-vllm; source scripts/env.sh
OUT=/workspace/runs/p1-speed-gsq; P1=$OUT/prod-pass1
until grep -q "Output token throughput" $P1/cohort_Tdefault_c2.log 2>/dev/null; do sleep 10; done
RP=$(pgrep -f "^bash /workspace/gsq-vllm/bench/speed/run.sh gsq|bench/speed/run.sh gsq --start" | head -1)
pkill -f "run_benchmarks.sh single"; kill $RP 2>/dev/null
for i in $(seq 60); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
pkill -9 -f "VLLM::EngineCore"; sleep 3; box_clean_shm || exit 1
S=$OUT/decode.txt
for C in 1 2; do F=$P1/cohort_Tdefault_c$C.log
  echo "ROW cohort C$C real prompts T=default (prod bench pass 1) | e2e=$(awk '/Output token throughput/ {print $5}' $F) tok/s | decode(C/meanTPOT)=$(python3 -c "print(f'{$C*1000/$(awk '/Mean TPOT/ {print $4}' $F):.1f}')") | meanTTFT=$(awk '/Mean TTFT/ {print $4}' $F) ms"
done | tee $S
GSQ_LOG=$OUT/server-ladder.log "$GSQ_ROOT/scripts/serve-gsq.sh" > /dev/null 2>&1 &
SERVER_PID=$!
gsq_wait_health 2400 "$SERVER_PID" || { echo SERVER_FAILED; exit 1; }
export VLLM_API_KEY=$GSQ_API_KEY
M() { curl -s "$GSQ_URL/metrics" -H "Authorization: Bearer $GSQ_API_KEY"; }
B=("$GSQ_VENV/bin/vllm" bench serve --host 127.0.0.1 --port "$GSQ_PORT" --model "$GSQ_HF_CONFIG" --served-model-name qwen3.8-27b)
num() { awk "/$1/ {print \$$2}" "$3"; }
pf() { local len=$1 c=$2 n=$3 seed=$((RANDOM * 32768 + RANDOM)) log=$OUT/prefill_${1}_c$2.log
  "${B[@]}" --dataset-name random --random-input-len "$len" --random-output-len 1 \
    --num-prompts "$n" --max-concurrency "$c" --seed "$seed" > "$log" 2>&1
  local in dur; in=$(num "Total input tokens" 4 "$log"); dur=$(num "Benchmark duration" 4 "$log")
  echo "ROW prefill len=$len conc=$c seed=$seed | $(python3 -c "print(f'{$in/$dur:.0f}')") tok/s | meanTTFT=$(num "Mean TTFT" 4 "$log") ms"
}
{ echo "# salted prefill ladder (bench/speed/run.sh commands)"; pf 8192 1 8; pf 8192 2 8; pf 65536 1 2; pf 65536 2 4; pf 180000 1 2; } | tee $OUT/ladder.txt
M | grep -E '^vllm:spec_decode' > $OUT/spec_after_ladder.prom
kill -INT $SERVER_PID; for i in $(seq 60); do kill -0 $SERVER_PID 2>/dev/null || break; sleep 2; done
pkill -9 -f "VLLM::EngineCore"; sleep 3; box_clean_shm
echo LADDER_DONE
