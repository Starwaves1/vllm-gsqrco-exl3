#!/bin/bash
# R2: one gpuq job = one server: production run_benchmarks.sh single (2 passes, decode) unless
# DECODE=0, then the prefill specs in GSQ_PREFILL; stop the server, clear /dev/shm.
#   [WT=/workspace/wt-r2] [DECODE=0] GSQ_PREFILL="8192:1:4" r2-speed.sh TAG
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_VENV=/workspace/gsq-vllm/.venv
WT=${WT:-/workspace/wt-r2}
export PYTHONPATH=$WT/plugin:$WT/tools
P=/workspace/logs/opt/r2
cd $WT
source scripts/env.sh
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
TAG=$1; R=/workspace/runs/opt-r2-$TAG-speed; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ $WT"; stopall
gsq_load_prod_argv
gsq_rewrite_argv "$GSQ_GGUF" --hf-config-path "$GSQ_HF_CONFIG" --tokenizer "$GSQ_HF_CONFIG"
export VLLM_PLUGINS=$GSQ_PLUGINS_ON VLLM_API_KEY=$GSQ_API_KEY
mkdir -p "$GSQ_KV_TIER_ROOT"
T0=$(date +%s)
setsid "${GSQ_ARGV[@]}" > $R/server.log 2>&1 &
gsq_wait_health 2400 $! || { echo SERVER_FAILED; tail -30 $R/server.log; stopall; exit 1; }
echo "server up after $(( $(date +%s) - T0 )) s"
grep -E "Loading weights took|Model loading took|GPU KV cache size|Available KV cache memory|weights memory|init engine" $R/server.log | cut -c1-240 | head -8
nvidia-smi --query-gpu=memory.used --format=csv,noheader
if [ "${DECODE:-1}" = 0 ]; then  # prefill only: bench/speed/run.sh's salted prefill rows, no decode passes
  B=("$GSQ_VENV/bin/vllm" bench serve --host 127.0.0.1 --port "$GSQ_PORT" --model "$GSQ_HF_CONFIG" --served-model-name qwen3.8-27b)
  export OPENAI_API_KEY=$GSQ_API_KEY
  for spec in $GSQ_PREFILL; do
    IFS=: read len c n <<< "$spec"; log=$R/prefill_${len}_c$c.log
    "${B[@]}" --dataset-name random --random-input-len $len --random-output-len 1 --num-prompts $n \
      --max-concurrency $c --seed $((RANDOM * 32768 + RANDOM)) > $log 2>&1
    in=$(awk '/Total input tokens/ {print $4}' $log); dur=$(awk '/Benchmark duration/ {print $4}' $log)
    echo "ROW prefill len=$len conc=$c | $(python3 -c "print(f'{$in/$dur:.0f}')") tok/s | meanTTFT=$(awk '/Mean TTFT/ {print $4}' $log) ms" | tee -a $R/summary.txt
  done > $P/$TAG-speed.log 2>&1; echo "prefill rc=$?"
else
  OUT=$R bench/speed/run.sh gsq > $P/$TAG-speed.log 2>&1; echo "speed rc=$?"
fi
stopall
grep -E "T=0|MTP|prefill" $R/summary.txt | tail -14
date -u +"end %FT%TZ"
