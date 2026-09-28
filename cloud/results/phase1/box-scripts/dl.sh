#!/bin/bash
set -uo pipefail
M=/workspace/models
HF=/venv/main/bin/hf
export HF_HUB_ENABLE_HF_TRANSFER=0
date -u +"start %FT%TZ"
( $HF download ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf --local-dir $M > /workspace/logs/dl-gguf.log 2>&1; echo "gguf rc=$?"; date -u +"gguf done %FT%TZ"; sha256sum $M/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf ) &
( $HF download Starw1/Qwen3.8-27B-absolute-heresy-W4A16 --local-dir $M/Qwen3.8-27B-W4A16-AutoRound-fast > /workspace/logs/dl-baseline.log 2>&1; echo "baseline rc=$?"; date -u +"baseline done %FT%TZ"; cd $M/Qwen3.8-27B-W4A16-AutoRound-fast && sha256sum config.json quantization_config.json model.safetensors.index.json ) &
wait
echo ALLDONE
