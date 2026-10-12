#!/bin/bash
# R3-23 attention / GDN share of the decode step vs context, to size the attention term and
# separate it from host idle. Production's main argv + VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 and a
# torch --profiler-config (25 iterations per capture, CUPTI). One server; for each point a steady
# decode window (unprofiled ms/step from /metrics, 10 s) then one capture:
#   c1-8k     c=1,   8k context   (k=5)
#   c1-100k   c=1, 100k           (k=5; prompt A)
#   c1-195k   c=1, 195k = A + 95k fresh tail (prefix hit on A)   (k=5)
#   c2-96k    c=2, 2 x 96k (as 20/22's c2)                        (k=5)
# Kernel classes by name (attention = FlashInfer, gdn = FLA/Triton GDN + conv, gemm_plugin = Route L,
# ...), split by launching scope (target forward vs MTP draft passes) and compared with the HBM floor
# (fp8 KV: 32 KiB/token target + 2 KiB/token per draft pass, 936 GB/s).
# Output: /workspace/logs/r3/23-attn-share/{summary.txt, attn/{trace-*.pt.trace.json.gz, attn.txt}}
# GPU time: ~20 min (load 4; prefill 8k + 100k + 95k + 2 x 96k ~ 9 min; 4 captures ~ 4 min).
#   bash 23-attn-share.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 23-attn-share "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,17p' "$0"; exit 0; fi
r3_env
r3_preflight
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
PROF="{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$L/attn/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":true,\"torch_profiler_dump_cuda_time_total\":false,\"ignore_frontend\":true,\"delay_iterations\":0,\"max_iterations\":25}"
R3_MUT=("add|--profiler-config|$PROF")
r3_serve attn
"${LOAD[@]}" warm || r3_die warm
export R L
hook() { echo "bash $R3_S/23-hook.sh $1"; }
r3_step c1-8k;   "${LOAD[@]}" steady --conc 1 --tokens 8000 --max-tokens 6000 --window 10 --k 5 --tag c1-8k --out "$R" --hook "$(hook c1-8k)" || r3_die c1-8k
r3_step c1-100k; "${LOAD[@]}" steady --conc 1 --tokens 100000 --max-tokens 5000 --window 10 --k 5 --tag c1-100k --out "$R" --reuse attnA:100000 --hook "$(hook c1-100k)" || r3_die c1-100k
r3_step c1-195k; "${LOAD[@]}" steady --conc 1 --tokens 195000 --max-tokens 4000 --window 10 --k 5 --tag c1-195k --out "$R" --prefix attnA:chat:100000 --hook "$(hook c1-195k)" || r3_die c1-195k
r3_step c2-96k;  "${LOAD[@]}" steady --conc 2 --tokens 96000 --max-tokens 6000 --window 10 --k 5 --tag c2-96k --out "$R" --hook "$(hook c2-96k)" || r3_die c2-96k
r3_stop
r3_step analyze
args=()
for t in c1-8k c1-100k c1-195k c2-96k; do
  [ -f "$R/trace-$t.pt.trace.json.gz" ] || r3_die "missing trace for $t"
  args+=("$t=$R/trace-$t.pt.trace.json.gz")
done
"$PY" "$R3_S/r3analyze.py" attn "${args[@]}" --json "$R/attn.json" > "$R/attn.txt" || r3_die "attn analysis"
"$PY" - "$R/attn.json" > "$R/floor.txt" <<'EOF' || r3_die "floor table"
import json, sys
rows = json.load(open(sys.argv[1]))
ctx = {"c1-8k": 8000, "c1-100k": 100000, "c1-195k": 195000, "c2-96k": 192000}
print("attention vs HBM floor (k=5: 32 KiB/token target + 5 x 2 KiB/token draft, 936 GB/s):")
for k, r in rows.items():
    a = r["kernel_ms_by_class"].get("attention", 0.0)
    fl = ctx[k] * (32768 + 5 * 2048) / 936e9 * 1e3
    print(f"  {k:8s} ctx {ctx[k]:>7d}: attention {a:6.2f} ms/step, floor {fl:5.2f} ms -> {100 * fl / a if a else 0:4.0f}% of bandwidth;"
          f"  gdn {r['kernel_ms_by_class'].get('gdn', 0.0):5.2f}  GPU idle {r['idle_ms']:5.2f}  span {r['span_ms']:6.2f}")
EOF
r3_summary "R3-23 attention share (box, $(date -u +%F)), production's main argv + profiler" "$(cat "$R/lines.txt")" "" \
  "$(cat "$R/attn.txt")" "" "$(cat "$R/floor.txt")" "" \
  "production (section 6): long-context GPU work +17.8 ms at n=2/204k, 0.076-0.095 ms per 1k ctx; floor 0.046 ms per 1k at k=5"
cat "$L/summary.txt"
