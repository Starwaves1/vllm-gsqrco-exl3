#!/bin/bash
kill $(pgrep -f "^/bin/bash /workspace/p1-queue2.sh") $(pgrep -f "^/bin/bash /workspace/p1-ab.sh") 2>/dev/null
pkill -f "bench serve --host 127.0.0.1"; pkill -f "clocks.sm,clocks.mem"
P=$(pgrep -f "bin/vllm serve" | head -1); [ -n "$P" ] && kill -INT $P
for i in $(seq 60); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
pkill -9 -f "VLLM::EngineCore"; sleep 3
ps -eo pid,cmd | grep -E "vllm|VLLM|p1-|nvidia-smi" | grep -v -E "grep|p1-stop"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo STOPPED
