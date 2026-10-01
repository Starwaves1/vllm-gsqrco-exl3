#!/bin/bash
# one gpuq job: the diagnostics (diag.py), then the 20-minute torture smoke without the two request
# kinds that crashed or failed smoke 1 (--skip plog,echo), on one server. GSQ-RCO GGUF, main, under
# production's vLLM-main argv. A wrapper because gpuq stores its command as "$*".
source /workspace/box-env.sh
export GSQ_VENV=/workspace/venv-main VLLM_USE_V2_MODEL_RUNNER=0 GSQ_PROD_ARGV=/workspace/wt-torture/env/prod-main-serve-argv.txt
export PYTHONPATH=/workspace/wt-torture/plugin:/workspace/wt-torture/tools VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1
export GSQ_KV_TIER_MAX_BYTES=8000000000   # /workspace has ~16 GB free
PY=/workspace/venv-main/bin/python
cd /workspace/wt-torture
R=/workspace/runs/torture/$(date +%Y%m%d-%H%M%S)-smoke2; mkdir -p $R
echo "start $(date -u +%FT%TZ) $(cat REV 2>/dev/null)"
box_clean_shm; rm -rf /workspace/kvtier
setsid scripts/serve-gsq.sh > $R/server.log 2>&1 &
SPID=$!
source scripts/env.sh
gsq_wait_health 2400 $SPID || { echo SERVER_FAILED; tail -30 $R/server.log; exit 1; }
$PY cloud/results/torture-smoke/box-scripts/diag.py http://127.0.0.1:18090/v1 "$GSQ_API_KEY" 2>&1 | tee $R/diag.txt
$PY bench/torture/torture run --base-url http://127.0.0.1:18090/v1 --api-key "$GSQ_API_KEY" --minutes 20 \
  --server-pid $SPID --server-log $R/server.log --skip plog,echo --out $R/torture; rc=$?
kill -INT -- -$SPID 2>/dev/null; for i in $(seq 60); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
pkill -9 -f "VLLM::EngineCore|bin/vllm serve" 2>/dev/null; sleep 3; box_clean_shm; rm -rf /workspace/kvtier
echo "end $(date -u +%FT%TZ) torture rc=$rc"
