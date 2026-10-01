#!/bin/bash
# EXL3 opt, job 10 (~20-40 min): exl3_gemm_mr (trellis-serve Marlin-EXL3) parity on the
# checkpoint's real tensors, tests/gpu/test_exl3_mr.py: decoded weights == exl3_dequant (every
# row and column; K3/K5 as stored, K4 repacked), the op at 17/24/32/48/64/96/144 rows (+1/8/16
# for repacked K4) vs fp64 inside exl3_gemm's error x1.5 (bf16 and fp16 model outputs; the
# dequant + fp16 GEMM printed as a third reference), EXL3_MR=1/2 routing (K2 stays on
# exl3_gemm bit for bit; repacked K4 above 144 rows == stored), determinism, graph replay after
# exl3_mr_warmup, capture before warmup refused (fresh process); then compute-sanitizer memcheck
# + initcheck on the decode, gemm and routing cases.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 10-mr-parity
require_idle_gpu
require_mr_build
O=$R/10-mr-parity; rm -rf "$O"; mkdir -p "$O"
rc=0
T=tests/gpu/test_exl3_mr.py
tools/pytest $T -q -rsX -s --junitxml="$O/mr.xml" > "$O/mr.log" 2>&1 || rc=1
echo "mr parity: $(junit_line "$O/mr.xml")"; tail -3 "$O/mr.log"
grep -E "^(FAILED|ERROR)" "$O/mr.log" | head -40 || true
# error table: one line per (tensor, rows, dtype): mr vs exl3_gemm vs dequant+GEMM, rel_rms / max_rel
"$GSQ_VENV/bin/python" "$S/errtab.py" "$O/mr.log" > "$O/errors.txt" || true
tail -1 "$O/errors.txt"
RE="^$T::(test_decode_exact\[(K4-kproj|K5-kproj)\]|test_gemm_vs_fp64\[(K3-down|K4-kproj|K5-kproj|K4-oproj)-(1|17|48|64|144)-bf16model\]|test_routing_mr1\[(K2-up|K3-down|K5-kproj)-.*\]|test_routing_repacked\[K4-kproj-.*\]|test_deterministic\[(K3-down|K4-kproj|K5-kproj)\])$"  # no lm_head under the sanitizers (0.6 GB per pass)
ids=$(tools/pytest $T --collect-only -q 2>/dev/null | grep -E "$RE" || true)
printf '%s\n' "$ids" > "$O/sanitizer-ids.txt"; echo "sanitizer cases: $(printf '%s\n' "$ids" | grep -c .)"
for tool in memcheck initcheck; do
  # shellcheck disable=SC2086
  r=0; PYTORCH_NO_CUDA_MEMORY_CACHING=1 "$SAN" --tool $tool --error-exitcode 99 --print-limit 20 \
    tools/pytest -q -p no:cacheprovider --junitxml="$O/sanitizer-$tool.xml" $ids > "$O/sanitizer-$tool.log" 2>&1 || r=$?
  [ $r = 0 ] || rc=1
  echo "$tool rc=$r: $(junit_line "$O/sanitizer-$tool.xml" 2>/dev/null) | $(grep -E 'ERROR SUMMARY' "$O/sanitizer-$tool.log" | tail -1)"
done
{ echo "10-mr-parity $(date -u +%FT%TZ) rc=$rc wt $(git -C "$WT" rev-parse --short HEAD)"
  echo "tests: $(junit_line "$O/mr.xml")"
  for t in memcheck initcheck; do echo "$t: $(junit_line "$O/sanitizer-$t.xml" 2>/dev/null) $(grep -E 'ERROR SUMMARY' "$O/sanitizer-$t.log" | tail -1)"; done
} | tee "$O/summary.txt"
gzip -kf "$O/mr.log" "$O/sanitizer-memcheck.log" "$O/sanitizer-initcheck.log"
keep "$O" 10-mr-parity "$O/summary.txt" "$O/errors.txt" "$O/mr.xml" "$O/mr.log.gz" "$O/sanitizer-memcheck.log.gz" \
  "$O/sanitizer-initcheck.log.gz" "$O/sanitizer-ids.txt"
exit $rc
