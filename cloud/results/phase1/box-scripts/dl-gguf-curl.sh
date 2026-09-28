#!/bin/bash
U=https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/resolve/main/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
P=/workspace/models/gguf.part
for i in $(seq 1 30); do
  [ "$(stat -c %s $P)" = 12120016896 ] && break
  curl -sSL -C - --retry 3 --speed-limit 1000000 --speed-time 60 -o $P $U; echo "curl rc=$? size=$(stat -c %s $P)"
done
date -u +"gguf done %FT%TZ"
S=$(sha256sum $P | cut -d" " -f1); echo "sha256 $S"
[ "$S" = 9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5 ] && mv $P /workspace/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf && echo GGUF_VERIFIED || echo GGUF_BAD
