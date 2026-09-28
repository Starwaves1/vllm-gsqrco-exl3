#!/bin/bash
# phase 1b parity, resumed: vLLM side for seq_002..010 (seq_000/001 done in the first pass, same config),
# compare, then MTP acceptance llama.cpp vs vLLM. llama.cpp CPU noise-floor dumps run in parallel (CPU only).
set -uo pipefail
source /workspace/box-env.sh
cd /workspace/gsq-vllm; source scripts/env.sh
P=/workspace/runs/p1b-parity D=/workspace/runs/p1b-parity-diag LB=/workspace/ref/llama.cpp-b11211/build/bin
mkdir -p $D/pdcpu $D/llamacpu
cp $P/prompts/seq_00[1-5].* $D/pdcpu/; printf 'seq_001\nseq_002\nseq_003\nseq_004\nseq_005\n' > $D/pdcpu/manifest.txt
CUDA_VISIBLE_DEVICES= setsid /workspace/ref/runs/parity-bin/llama_logits -m $GSQ_GGUF -d $D/pdcpu -o $D/llamacpu --ngl 0 > $D/llamacpu-2.log 2>&1 < /dev/null &
date -u +"parity-vllm start %FT%TZ"
.venv/bin/python bench/parity/vllm_logprobs.py -d $P/prompts -o $P/vllm --gguf $GSQ_GGUF \
  --only seq_002,seq_003,seq_004,seq_005,seq_006,seq_007,seq_008,seq_009,seq_010 >> $P/vllm.log 2>&1; echo "vllm rc=$?"
.venv/bin/python bench/parity/compare.py -d $P/prompts -l $P/llama -v $P/vllm --json $P/parity.json > $P/compare.txt 2>&1; echo "compare rc=$?"
date -u +"parity-vllm end %FT%TZ"
R=/workspace/runs/p1b-mtp; mkdir -p $R
CTX=32768 LLAMA_SERVER=$LB/llama-server bench/speed/serve-llamacpp.sh > $R/llama-server.log 2>&1 &
LP=$!
for i in $(seq 120); do curl -sf -o /dev/null http://127.0.0.1:18091/health && break; sleep 5; done
.venv/bin/python bench/speed/mtp_acceptance.py --engine llama --url http://127.0.0.1:18091 --out $R/llama.json > $R/llama.txt 2>&1; echo "mtp llama rc=$?"
kill -INT $LP; wait $LP
box_clean_shm; rm -rf /workspace/kvtier
GSQ_LOG=$R/server.log scripts/serve-gsq.sh > /dev/null 2>&1 &
SP=$!
gsq_wait_health 2400 $SP && { .venv/bin/python bench/speed/mtp_acceptance.py --engine vllm --out $R/vllm.json --reference $R/llama.json > $R/vllm.txt 2>&1; echo "mtp vllm rc=$?"; }
kill -INT $SP; for i in $(seq 60); do pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null || break; sleep 2; done
E=$(pgrep -f "VLLM::EngineCore"); [ -n "$E" ] && kill -9 $E; sleep 3; box_clean_shm; rm -rf /workspace/kvtier
date -u +"mtp end %FT%TZ"
echo P1B_PARITY_DONE
