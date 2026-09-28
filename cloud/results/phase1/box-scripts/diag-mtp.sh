#!/bin/bash
# serve-gsq.sh (prod argv, MTP k=3) with VRAM sampling + py-spy stacks during the drafter load
source /workspace/box-env.sh
D=/workspace/runs/p1-diag-mtp; mkdir -p $D
cd /workspace/gsq-vllm
GSQ_LOG=$D/server.log setsid nohup scripts/serve-gsq.sh >/dev/null 2>&1 &
echo $! > $D/serve.pid
until grep -q "Loading drafter model" $D/server.log 2>/dev/null; do sleep 2; kill -0 $(cat $D/serve.pid) 2>/dev/null || exit 1; done
EP=$(pgrep -f "VLLM::EngineCore" | head -1); echo "engine pid $EP" > $D/pyspy.txt
for i in $(seq 1 40); do
  echo "=== t=$((i*3))s vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)" >> $D/pyspy.txt
  /venv/main/bin/py-spy dump --pid $EP >> $D/pyspy.txt 2>&1 || break
  sleep 3
done
echo DIAG_DONE >> $D/pyspy.txt
