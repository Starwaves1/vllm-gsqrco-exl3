#!/bin/bash
B=/workspace/models/Qwen3.8-27B-W4A16-AutoRound-fast
for i in 1 2 3 4 5 6 7 8; do
  timeout 2400 /venv/main/bin/hf download Starw1/Qwen3.8-27B-absolute-heresy-W4A16 --local-dir $B --max-workers 16 >> /workspace/logs/dl-baseline.log 2>&1 && break
  echo "attempt $i rc=$?"
done
date -u +"baseline done %FT%TZ"; cd $B && sha256sum config.json quantization_config.json model.safetensors.index.json; echo BASELINE_FINISHED
