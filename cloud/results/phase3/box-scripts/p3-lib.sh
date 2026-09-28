#!/bin/bash
# phase 3 box helpers: source me. Route L on (VLLM_GGUF_LCPP=1) for every gsq run.
source /workspace/box-env.sh
export VLLM_GGUF_LCPP=1
L=/workspace/logs/p3; mkdir -p $L
cd /workspace/gsq-vllm
stopall() { pkill -INT -f "bin/vllm serve"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
parity() {  # $1 tag: full kernel parity suite with Route L on
  tools/pytest tests/gpu/test_kernel_parity.py -q > $L/$1-parity.log 2>&1; echo "rc=$?" >> $L/$1-parity.log
  tail -2 $L/$1-parity.log; }
speed() {  # $1 tag: decode only (production run_benchmarks.sh single, 2 passes), no prefill ladder
  stopall; OUT=/workspace/runs/p3-$1-speed GSQ_PREFILL=" " bench/speed/run.sh gsq --start > $L/$1-speed.log 2>&1
  echo "speed rc=$?"; stopall; grep -E "T=0|MTP|clocks" /workspace/runs/p3-$1-speed/summary.txt | tail -8; }
