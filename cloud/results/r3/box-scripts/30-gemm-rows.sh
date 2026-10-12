#!/bin/bash
# R3-30 GEMM sweep at production's row counts (no server): every (type, rows, K) linear of the GGUF
# (with its count) x n in {6,12,18,20,24,27,28,32,36,48,128,136,160}: the routed op vs MMQ,
# mma_k (and 64-row chunked mma_k above 64), packed tiled / decode IQ3, MMVQ / owned decode.
# Picks the kernel work: which (type, n) sit furthest above the DRAM floor at the rows k=5/k=3/k=2
# produce, and whether an owned kernel beats MMQ on 128-160-row prefill chunks.
# Output: /workspace/logs/r3/30-gemm-rows/{summary.txt, sweep.tsv, sweep.log}
# GPU time: ~25 min. R3_TOKENS=1,2,... overrides the row counts (with R3_TAG for a second run).
#   bash 30-gemm-rows.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 30-gemm-rows "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,8p' "$0"; exit 0; fi
r3_env
r3_preflight
r3_step sweep
"$PY" "$R3_S/r3gemm.py" --out "$L/sweep.tsv" ${R3_TOKENS:+--tokens "$R3_TOKENS"} > "$L/sweep.log" 2>&1 || { tail -20 "$L/sweep.log"; r3_die "sweep"; }
"$PY" - "$L/sweep.tsv" > "$L/table.txt" <<'PYEOF' || r3_die "table"
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
cases = collections.defaultdict(dict)
for r in rows:
    key = (r["type"], int(r["rows"]), int(r["K"]), int(r["n"]))
    cases[key][r["variant"]] = (float(r["us"]), r["is_route"] == "1", float(r["floor_us"]), int(r["count"]))
model = collections.defaultdict(lambda: [0.0, 0.0, 0.0])   # n -> route, best, floor (ms per forward)
bytype = collections.defaultdict(lambda: [0.0, 0.0, 0.0])  # (n, type)
print("type      rows x K       n  cnt  route              us   x floor | best other          us    gain/call")
for (t, R, K, n), v in sorted(cases.items()):
    route = [(k, x) for k, x in v.items() if x[1]]
    if not route:
        continue
    rn, (ru, _, fl, cnt) = route[0]
    alt = min(((k, x) for k, x in v.items() if not x[1]), key=lambda kv: kv[1][0], default=None)
    best = min(ru, alt[1][0]) if alt else ru
    for acc in (model[n], bytype[(n, t)]):
        acc[0] += cnt * ru / 1e3; acc[1] += cnt * best / 1e3; acc[2] += cnt * fl / 1e3
    gain = f"{ru - alt[1][0]:+7.1f}" if alt and alt[1][0] < ru else ""
    print(f"{t:8s} {R:>6d}x{K:<6d} {n:4d} {cnt:4d}  {rn:18s} {ru:7.1f} {ru / fl:5.2f}  | "
          + (f"{alt[0]:18s} {alt[1][0]:7.1f}" if alt else " " * 26) + f"  {gain}")
print("\nmodel-level GEMM time per forward (all linears of the GGUF incl. lm_head and the MTP layer, ms):")
print("   n    route    best-of-variants   DRAM floor   route/floor")
for n in sorted(model):
    r, b, f = model[n]
    print(f"{n:4d}  {r:7.2f}   {b:7.2f}            {f:7.2f}      {r / f:5.2f}")
print("\nper type and n (route ms, floor ms):")
for n in sorted(model):
    print(f"  n={n}: " + "  ".join(f"{t} {v[0]:.2f}/{v[2]:.2f}" for (m, t), v in sorted(bytype.items()) if m == n))
PYEOF
r3_summary "R3-30 GEMM sweep (box, $(date -u +%F)), plugin $(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD)" "$(cat "$L/table.txt")"
head -80 "$L/summary.txt"  # (cat | head under pipefail: rc 141, job 30)
