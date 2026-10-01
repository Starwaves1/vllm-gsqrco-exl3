#!/bin/bash
# R3-30 GEMM sweep at production's row counts (no server): every quantized type of the GGUF x the
# four big weight shapes x n in {6,12,18,20,24,27,28,32,36,48,128,136,160}: the routed op vs MMQ,
# mma_k (and 64-row chunked mma_k above 64), packed tiled / decode IQ3, MMVQ / owned decode.
# Picks the kernel work: which (type, n) sit furthest above the DRAM floor at the rows k=5/k=3/k=2
# produce, and whether an owned kernel beats MMQ on 128-160-row prefill chunks.
# Output: /workspace/logs/r3/30-gemm-rows/{summary.txt, sweep.tsv, sweep.log}
# GPU time: ~15 min.
#   bash 30-gemm-rows.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 30-gemm-rows "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,8p' "$0"; exit 0; fi
r3_env
r3_preflight
r3_step sweep
"$PY" "$R3_S/r3gemm.py" --out "$L/sweep.tsv" > "$L/sweep.log" 2>&1 || { tail -20 "$L/sweep.log"; r3_die "sweep"; }
"$PY" - "$L/sweep.tsv" > "$L/table.txt" <<'PYEOF' || r3_die "table"
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
best = collections.defaultdict(dict)
for r in rows:
    key = (r["type"], r["rows"], r["K"], int(r["n"]))
    best[key][r["variant"]] = (float(r["us"]), r["is_route"] == "1", float(r["floor_us"]))
print("type      shape         n   route(us)  x floor   best alternative (us)       gain")
tot = collections.defaultdict(float)
for (t, R, K, n), v in sorted(best.items()):
    route = [(k, x) for k, x in v.items() if x[1]]
    if not route:
        continue
    rn, (ru, _, fl) = route[0]
    alt = min(((k, x) for k, x in v.items() if not x[1]), key=lambda kv: kv[1][0], default=None)
    gain = f"{ru - alt[1][0]:+.1f}" if alt and alt[1][0] < ru else ""
    print(f"{t:8s} {R:>5s}x{K:<6s} {n:4d}  {rn:18s} {ru:7.1f}  {ru / fl:5.2f}   "
          + (f"{alt[0]:18s} {alt[1][0]:7.1f}" if alt else "") + f"   {gain}")
PYEOF
r3_summary "R3-30 GEMM sweep (box, $(date -u +%F)), plugin $(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD)" "$(cat "$L/table.txt")"
cat "$L/summary.txt" | head -80
