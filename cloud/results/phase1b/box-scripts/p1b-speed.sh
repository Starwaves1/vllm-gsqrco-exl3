#!/bin/bash
# phase 1b: bench/speed/run.sh gsq then baseline, identical settings, fresh shm + fs tier each run
set -uo pipefail
source /workspace/box-env.sh
cd /workspace/gsq-vllm
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
stopall
for k in gsq baseline; do
  date -u +"$k start %FT%TZ"
  OUT=/workspace/runs/p1b-speed-$k bench/speed/run.sh $k --start > /workspace/logs/p1b-speed-$k.log 2>&1; echo "$k rc=$?"
  stopall; nvidia-smi --query-gpu=memory.used --format=csv,noheader
  date -u +"$k end %FT%TZ"
done
echo P1B_SPEED_DONE
