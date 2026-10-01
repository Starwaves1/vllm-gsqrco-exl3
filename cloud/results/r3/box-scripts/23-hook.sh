#!/bin/bash
# shellcheck disable=SC2011,SC2012  # file names here are ours (no spaces)
# Called by r3load.py steady --hook from 23-attn-share.sh: one torch-profiler capture (the server's
# --profiler-config stops it after 25 iterations), saved as $R/trace-LABEL.pt.trace.json.gz.
set -uo pipefail
label=$1
before=$(ls "$L"/attn/trace/*.pt.trace.json* 2>/dev/null | wc -l)
"$GSQ_VENV/bin/python" "$(dirname "$0")/r3load.py" profile --seconds 4 || exit 3
for _ in $(seq 60); do
  now=$(ls "$L"/attn/trace/*.pt.trace.json* 2>/dev/null | wc -l)
  [ "$now" -gt "$before" ] && break
  sleep 2
done
f=$(ls -t "$L"/attn/trace/*.pt.trace.json* 2>/dev/null | head -1)
[ -n "$f" ] && [ "$now" -gt "$before" ] || { echo "no new trace for $label"; exit 4; }
sleep 3   # let the writer finish
mv "$f" "$R/trace-$label.pt.trace.json.gz"
echo "trace $label: $(du -h "$R/trace-$label.pt.trace.json.gz" | cut -f1)"
