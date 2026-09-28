#!/bin/bash
# phase 3: VRAM of a gsq server start (production argv): peak memory.used while loading
# (200 ms samples), after load, "Model loading took", KV cache tokens; one chat smoke.
#   p3-vram.sh TAG   (extra env, e.g. MTP_DRAFT_VOCAB=0, passes through to vLLM)
set -uo pipefail
source /workspace/p3/p3-lib.sh
TAG=$1; R=/workspace/runs/p3-$TAG-vram; rm -rf $R; mkdir -p $R
stopall; source scripts/env.sh
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > $R/vram-load.csv &
S=$!
GSQ_LOG=$R/server.log scripts/serve-gsq.sh > /dev/null 2>&1 &
SP=$!
gsq_wait_health 2400 $SP || { kill $S; echo SERVER_FAILED; exit 1; }
kill $S
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Name three primes."}],"max_tokens":200,"temperature":0}' > $R/smoke.json
{
  python3 - $R/vram-load.csv <<'PY'
import sys
v = [int(x) for x in open(sys.argv[1]) if x.strip()]
i = next(i for i in range(1, len(v)) if v[i] < v[i - 1] - 3000)  # weights freed after the draft load
print(f"MiB: load peak (target + draft weights) {max(v[:i])}, after load {v[i]}, peak with KV cache {max(v)}")
PY
  echo "MiB after load+smoke: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
  grep -E "Model loading took|GPU KV cache size|Maximum concurrency|draft head|Loading Qwen3.5 MTP" $R/server.log | sed 's/^.*\] //'
  python3 -c "import json;print('smoke:', json.load(open('$R/smoke.json'))['choices'][0]['message'].get('content','')[:120].replace(chr(10),' '))"
} | tee $R/summary.txt
stopall; echo VRAM_DONE
