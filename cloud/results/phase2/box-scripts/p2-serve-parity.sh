#!/bin/bash
# phase 2 step 3: serve-gsq.sh with Route L (smoke, load, VRAM), then vLLM-side logit parity
# against the phase-1b llama.cpp CUDA dumps (not regenerated), then the per-seq floor table.
set -uo pipefail
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1
cd /workspace/gsq-vllm; source scripts/env.sh
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
stopall
R=/workspace/runs/p2-smoke; mkdir -p $R
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $R/vram-before.csv
T0=$(date +%s)
GSQ_LOG=$R/server.log setsid nohup scripts/serve-gsq.sh > /dev/null 2>&1 &
SP=$!
if gsq_wait_health 2400 $SP; then
  echo "health after $(( $(date +%s) - T0 )) s" | tee $R/load-time.txt
  nvidia-smi --query-gpu=memory.used,memory.total,power.draw,power.limit --format=csv | tee $R/vram-after-load.csv
  .venv/bin/python /workspace/smoke_chat.py > $R/smoke.json 2>&1; echo "smoke rc=$?" >> $R/smoke.json
  nvidia-smi --query-gpu=memory.used --format=csv > $R/vram-after-smoke.csv
  grep -E "Model loading took|GPU KV cache size|Maximum concurrency|lcpp|LCPP" $R/server.log | head -20 > $R/load-lines.txt
else
  echo SERVER_FAILED
fi
stopall
date -u +"parity start %FT%TZ"
P=/workspace/runs/p1b-parity O=/workspace/runs/p2-parity; mkdir -p $O/vllm
.venv/bin/python bench/parity/vllm_logprobs.py -d $P/prompts -o $O/vllm --gguf $GSQ_GGUF > $O/vllm.log 2>&1; echo "vllm rc=$?"
.venv/bin/python bench/parity/compare.py -d $P/prompts -l $P/llama -v $O/vllm --json $O/parity.json > $O/compare.txt 2>&1; echo "compare rc=$?"
.venv/bin/python /workspace/p2floor.py > $O/floor.txt 2>&1; echo "floor rc=$?"
date -u +"parity end %FT%TZ"
echo P2_SERVE_PARITY_DONE
