#!/bin/bash
M=/workspace/models
for i in 1 2 3 4 5 6; do
  HF_HUB_ENABLE_HF_TRANSFER=0 timeout 3000 /venv/main/bin/hf download ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --local-dir $M >> /workspace/logs/dl-gguf.log 2>&1 && break
  echo "attempt $i failed rc=$?"
done
date -u +"gguf done %FT%TZ"; sha256sum $M/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf; echo GGUF_FINISHED
