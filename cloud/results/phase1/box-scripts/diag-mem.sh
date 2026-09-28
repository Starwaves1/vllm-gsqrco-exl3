#!/bin/bash
source /workspace/box-env.sh
D=/workspace/runs/p1-diag-mem; mkdir -p $D
cd /workspace/gsq-vllm
export GSQ_MEMDIAG=1 PYTHONPATH=/tmp/p1diag/site
GSQ_LOG=$D/server.log setsid nohup scripts/serve-gsq.sh >/dev/null 2>&1 &
echo $! > $D/serve.pid
