#!/bin/bash
# phase 2 step 5: short speed pass, Route L (VLLM_GGUF_LCPP=1): production decode script (c=1..8,
# report c=1/c=2) + prefill 8k and 180k at c=1, clocks logged by run.sh.
set -uo pipefail
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1 GSQ_PREFILL="8192:1:8 180000:1:2"
cd /workspace/gsq-vllm
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
stopall
date -u +"speed start %FT%TZ"
OUT=/workspace/runs/p2-speed-gsq-lcpp bench/speed/run.sh gsq --start > /workspace/logs/p2/speed-gsq-lcpp.log 2>&1; echo "rc=$?"
stopall
date -u +"speed end %FT%TZ"
echo P2_SPEED_DONE
