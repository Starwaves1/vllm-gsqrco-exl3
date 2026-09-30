#!/bin/bash
# RedHatAI/Qwen3.8-27B-INT4 (llm-compressor W4A16 g128 sym of Qwen/Qwen3.8-27B, + model_mtp.safetensors)
set -eu
REV=91bd022d5b49442a868bc35008f6c21e1860edfa
D=/workspace/models/RedHatAI-Qwen3.8-27B-INT4
date -u +"start %FT%TZ"
HF_HUB_ENABLE_HF_TRANSFER=0 /workspace/gsq-vllm/.venv/bin/hf download RedHatAI/Qwen3.8-27B-INT4 --revision $REV --local-dir $D --max-workers 16
ls -la $D; du -sh $D; date -u +"end %FT%TZ"
