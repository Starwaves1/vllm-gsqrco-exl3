#!/bin/bash
# model matrix, one gpuq job = one server: $1 = {swift,base,w4a16,official}-{mtp,nomtp}.
#   swift/base: scripts/serve-gsq.sh (GGUF + plugin, Route L), hf-config/<model>
#   w4a16:      scripts/serve-baseline.sh, production's W4A16 on the patched venv
#   official:   scripts/serve-baseline.sh with GSQ_VENV=/workspace/venv-stock (vLLM 0.27.1 wheel,
#               no overlay, no plugin) and RedHatAI/Qwen3.8-27B-INT4
# nomtp = GSQ_NO_MTP=1 (production's argv minus --speculative-config, nothing else).
# Then: served config (dtype, weights, KV tokens, max len, VRAM), smoke (base/official),
# bench/speed/run.sh unmodified with GSQ_PREFILL = 8k c=1 (+ 180k c=1 if the max len allows).
CFG=$1; MODEL=${CFG%-*}; SPEC=${CFG##*-}
source /workspace/wt-models/cloud/results/models/box-scripts/lib.sh   # box-env: Swift GGUF, prod W4A16
case $MODEL in
  swift) KIND=gsq ;;
  base) KIND=gsq; export GSQ_MODEL_NAME=Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp GSQ_GGUF=/workspace/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf ;;
  w4a16) KIND=baseline ;;
  official) KIND=baseline; export GSQ_VENV=/workspace/venv-stock GSQ_BASELINE_MODEL=/workspace/models/RedHatAI-Qwen3.8-27B-INT4 ;;
  *) echo "bad config $CFG"; exit 2 ;;
esac
case $SPEC in mtp) ;; nomtp) export GSQ_NO_MTP=1 ;; *) echo "bad config $CFG"; exit 2 ;; esac
cd $WT; source scripts/env.sh
R=/workspace/runs/models/$CFG; rm -rf $R; mkdir -p $R
date -u +"start %FT%TZ"; git log --oneline -1; echo "cfg=$CFG kind=$KIND venv=$GSQ_VENV model=$([ $KIND = gsq ] && echo $GSQ_GGUF || echo $GSQ_BASELINE_MODEL) no_mtp=${GSQ_NO_MTP:-0}"
stopall
start() { GSQ_LOG=$R/server.log setsid scripts/serve-$KIND.sh > /dev/null 2>&1 & gsq_wait_health 2400 $!; }
if ! start; then
  if [ $MODEL = official ] && grep -q -i "maximum model length" $R/server.log; then
    grep -i -E "KV cache is needed|maximum model length" $R/server.log | tail -2 | cut -c1-400
    echo "production's --max-model-len 200000 does not fit; retrying with --max-model-len -1 (fit to memory)"
    stopall; mv $R/server.log $R/server-200k.log; export GSQ_MAX_MODEL_LEN=-1
    start || { echo SERVER_FAILED; tail -40 $R/server.log; stopall; exit 1; }
  else echo SERVER_FAILED; tail -40 $R/server.log; stopall; exit 1; fi
fi
echo "VRAM after load: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
grep -E "argv vs production|^[<>] |speculative_config=|dtype=|Loading weights took|Model loading took|draft head|GPU KV cache size|Maximum concurrency|init engine|max_model_len" $R/server.log | grep -v -E "^\(APIServer.*non-default args" | cut -c1-300 | head -40
MML=$(curl -s -H "Authorization: Bearer $GSQ_API_KEY" $GSQ_URL/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["max_model_len"])')
echo "max_model_len=$MML; spec_decode metric lines: $(curl -s -H "Authorization: Bearer $GSQ_API_KEY" $GSQ_URL/metrics | grep -c '^vllm:spec_decode')"
case $MODEL in base|official) python3 $S/smoke_chat.py $GSQ_URL > $R/smoke.json 2>&1; tail -1 $R/smoke.json ;; esac
PF="8192:1:8"; [ "$MML" -ge 180001 ] && PF="$PF 180000:1:2" || echo "max_model_len $MML < 180001: 180k prefill skipped"
GSQ_PREFILL="$PF" OUT=$R/speed bench/speed/run.sh $KIND > $R/speed.log 2>&1; echo "speed rc=$?"
stopall
grep -E "^ROW" $R/speed/summary.txt | grep -v -E "T=def|C[1248] .*default" | tail -30
date -u +"end %FT%TZ"
