#!/bin/bash
# phase-1 step 2: start serve-gsq.sh (left running for the fit test), time load, VRAM, smoke chat/tool
set -uo pipefail
source /workspace/box-env.sh
R=/workspace/runs/p1-smoke; mkdir -p $R
box_clean_shm || exit 1
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $R/vram-before.csv
T0=$(date +%s)
GSQ_LOG=$R/server.log setsid nohup /workspace/gsq-vllm/scripts/serve-gsq.sh > /dev/null 2>&1 &
echo $! > $R/serve.pid
source /workspace/gsq-vllm/scripts/env.sh
if gsq_wait_health 2400 $(cat $R/serve.pid); then
  T1=$(date +%s); echo "health after $((T1-T0)) s" | tee $R/load-time.txt
  nvidia-smi --query-gpu=memory.used,memory.total,power.draw,power.limit --format=csv | tee $R/vram-after-load.csv
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv >> $R/vram-after-load.csv
  /workspace/gsq-vllm/.venv/bin/python /workspace/smoke_chat.py > $R/smoke.json 2>&1; echo "smoke rc=$?" >> $R/smoke.json
  nvidia-smi --query-gpu=memory.used --format=csv > $R/vram-after-smoke.csv
  echo SMOKE_STEP_DONE
else
  echo SERVER_FAILED
fi
