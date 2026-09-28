#!/bin/bash
# per-file resumable download of the public W4A16 baseline (hf/xet stalled on this box)
R=Starw1/Qwen3.8-27B-absolute-heresy-W4A16
B=/workspace/models/Qwen3.8-27B-W4A16-AutoRound-fast
cd $B || exit 1
curl -s https://huggingface.co/api/models/$R/tree/main | python3 -c 'import json,sys; [print(f["path"], f.get("lfs",{}).get("size", f["size"])) for f in json.load(sys.stdin) if f["type"]=="file"]' > /workspace/logs/baseline-files.txt
one() { f=$1 sz=$2
  for t in $(seq 1 100); do
    have=$( [ -f "$f" ] && stat -c %s "$f" || echo 0 ); [ "$have" = "$sz" ] && { echo "ok $f"; return 0; }
    curl -sSL --connect-timeout 20 --speed-limit 100000 --speed-time 60 -C - -o "$f" "https://huggingface.co/$R/resolve/main/$f"
  done; echo "FAILED $f"; }
export -f one; export R
xargs -P 8 -L 1 bash -c 'one "$0" "$1"' < /workspace/logs/baseline-files.txt
date -u +"baseline done %FT%TZ"; sha256sum config.json quantization_config.json model.safetensors.index.json; echo BASELINE_FINISHED
