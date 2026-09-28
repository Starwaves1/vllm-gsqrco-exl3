#!/bin/bash
# wait for gsq 8k c=2, stop that run, then gsq 64k/180k (n=1) and the baseline with the same counts
L=/workspace/logs/p1-ab-gsq.log
until grep -q "len=8192 conc=2" $L; do sleep 10; done
kill $(pgrep -f "^/bin/bash /workspace/p1-after-ladder.sh") $(pgrep -f "^/bin/bash /workspace/p1-ab.sh gsq ladderclock") 2>/dev/null
pkill -f "bench serve --host 127.0.0.1"; pkill -f "clocks.sm,clocks.mem"
P=$(pgrep -f "bin/vllm serve" | head -1); kill -INT $P
for i in $(seq 60); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
pkill -9 -f "VLLM::EngineCore"; sleep 3
/workspace/p1-ab.sh gsq tail > /workspace/logs/p1-ab-gsq-tail.log 2>&1
/workspace/p1-ab.sh baseline full > /workspace/logs/p1-ab-baseline.log 2>&1
echo QUEUE2_DONE
