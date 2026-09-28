#!/bin/bash
# phase 1b: logit parity on the stock kernels (bf16 KV), then MTP acceptance llama.cpp vs vLLM
set -uo pipefail
source /workspace/box-env.sh
cd /workspace/gsq-vllm; source scripts/env.sh
LB=/workspace/ref/llama.cpp-b11211/build/bin
stopall() { pkill -INT -f "bin/vllm serve"; pkill -INT -f "$LB/llama-server"; for i in $(seq 90); do pgrep -f "VLLM::EngineCore|bin/vllm serve|$LB/llama-server" >/dev/null || break; sleep 2; done
  pkill -9 -f "VLLM::EngineCore"; pkill -9 -f "bin/vllm serve"; pkill -9 -f "$LB/llama-server"; sleep 5; box_clean_shm; rm -rf /workspace/kvtier; }
stopall
date -u +"parity start %FT%TZ"
LLAMA_LOGITS=/workspace/ref/runs/parity-bin/llama_logits OUT=/workspace/runs/p1b-parity KEEP=1 bench/parity/run.sh > /workspace/logs/p1b-parity.log 2>&1; echo "parity rc=$?"
date -u +"parity end %FT%TZ"
stopall
R=/workspace/runs/p1b-mtp; mkdir -p $R
CTX=32768 LLAMA_SERVER=$LB/llama-server bench/speed/serve-llamacpp.sh > $R/llama-server.log 2>&1 &
for i in $(seq 120); do curl -sf -o /dev/null http://127.0.0.1:18091/health && break; sleep 5; done
.venv/bin/python bench/speed/mtp_acceptance.py --engine llama --url http://127.0.0.1:18091 --out $R/llama.json > $R/llama.txt 2>&1; echo "mtp llama rc=$?"
stopall
GSQ_LOG=$R/server.log scripts/serve-gsq.sh > /dev/null 2>&1 &
gsq_wait_health 2400 $! && { .venv/bin/python bench/speed/mtp_acceptance.py --engine vllm --out $R/vllm.json --reference $R/llama.json > $R/vllm.txt 2>&1; echo "mtp vllm rc=$?"; }
stopall
date -u +"mtp end %FT%TZ"
echo P1B_PARITY_DONE
