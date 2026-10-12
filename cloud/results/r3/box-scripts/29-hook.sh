#!/bin/bash
# Called by r3load.py steady --hook from 29-pyspy-frames.sh: the production sampling recipe
# (prod-sample-20261001.md: 250 Hz, --nonblocking, no native frames, 30 s), then 100 Hz --idle 20 s.
set -uo pipefail
tag=$1
"$PYSPY" record --pid "$ENGINE_PID" --duration 30 --rate 250 --nonblocking --threads --format raw \
  -o "$R/pyspy-$tag.raw" > "$R/pyspy-$tag.log" 2>&1 || { echo "py-spy $tag failed: $(tail -1 "$R/pyspy-$tag.log")"; exit 2; }
"$PYSPY" record --pid "$ENGINE_PID" --duration 20 --rate 100 --nonblocking --idle --threads --format raw \
  -o "$R/pyspy-$tag-idle.raw" > "$R/pyspy-$tag-idle.log" 2>&1 || { echo "py-spy idle $tag failed"; exit 3; }
