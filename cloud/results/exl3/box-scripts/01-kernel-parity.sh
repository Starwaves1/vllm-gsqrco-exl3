#!/bin/bash
# EXL3 phase 1, job 01 (~1-1.5 h): kernel parity and plumbing for torch.ops._C_exl3.
#  1. exllamav3 itself on the case tensors (tests/gpu/exl3_ref_dump.py in the reference venv):
#     dequant hashes, gemm error stats; fills the shared autotune cache first
#  2. tests/gpu/test_exl3_kernels.py (vLLM venv): dequant bit-exact vs exllamav3, routed gemm vs
#     fp64 inside exllamav3's own error, route agreement, graph capture/replay, fresh-process cases
#     (capture before warmup refused, first call inside a capture, guards on CUDA inputs)
#  3. compute-sanitizer memcheck + initcheck on a subset (no graph tests: capture cannot
#     cudaMalloc with the caching allocator off)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 01-kernel-parity
require_idle_gpu
O=$R/01-kernel-parity; rm -rf "$O"; mkdir -p "$O"
rc=0
"$GSQ_EXL3_VENV/bin/python" tests/gpu/exl3_ref_dump.py 2>&1 | tee "$O/ref-dump.log" || die "exllamav3 reference dump failed"
cp -r "$EXL3_REF_DIR" "$O/kernel-ref"
T=tests/gpu/test_exl3_kernels.py
tools/pytest $T -q -rsX --junitxml="$O/kernels.xml" > "$O/kernels.log" 2>&1 || rc=1
echo "kernels: $(junit_line "$O/kernels.xml")"; tail -3 "$O/kernels.log"
grep -E "^(FAILED|ERROR)" "$O/kernels.log" | head -40 || true
RE="^$T::(test_dequant_bitexact\[(K2-up|K3-down|K4-kproj|K5-kproj|K5-oproj)-(had|rot)\]|test_gemm_vs_fp64\[(K2-up|K3-down|K4-kproj|K5-kproj|K5-oproj|K4-mtp-up)-(1|2|3|8|9|16|17|48|145|1024)-fp32\]|test_routes_agree\[.*\]|test_fp32_fp16_outputs_agree\[(K3-down|K4-kproj)-.*\]|test_subprocess_case\[(x_|k_|suh_|trellis_|dequant_).*\])$"   # no capture cases: without the caching allocator a capture must cudaMalloc
ids=$(tools/pytest $T --collect-only -q 2>/dev/null | grep -E "$RE" || true)
printf '%s\n' "$ids" > "$O/sanitizer-ids.txt"; echo "sanitizer cases: $(printf '%s\n' "$ids" | grep -c .)"
for tool in memcheck initcheck; do
  # shellcheck disable=SC2086
  r=0; PYTORCH_NO_CUDA_MEMORY_CACHING=1 "$SAN" --tool $tool --error-exitcode 99 --print-limit 20 \
    tools/pytest -q -p no:cacheprovider --junitxml="$O/sanitizer-$tool.xml" $ids > "$O/sanitizer-$tool.log" 2>&1 || r=$?
  [ $r = 0 ] || rc=1
  echo "$tool rc=$r: $(junit_line "$O/sanitizer-$tool.xml" 2>/dev/null) | $(grep -E 'ERROR SUMMARY' "$O/sanitizer-$tool.log" | tail -1)"
done
{ echo "01-kernel-parity $(date -u +%FT%TZ) rc=$rc"; echo "reference: $(tail -6 "$O/ref-dump.log" | tr '\n' ' ')"
  echo "kernels: $(junit_line "$O/kernels.xml")"
  for t in memcheck initcheck; do echo "$t: $(junit_line "$O/sanitizer-$t.xml" 2>/dev/null) $(grep -E 'ERROR SUMMARY' "$O/sanitizer-$t.log" | tail -1)"; done
} | tee "$O/summary.txt"
gzip -kf "$O/kernels.log" "$O/sanitizer-memcheck.log" "$O/sanitizer-initcheck.log"
keep "$O" 01-kernel-parity "$O/summary.txt" "$O/kernels.xml" "$O/kernels.log.gz" "$O/sanitizer-memcheck.log.gz" \
  "$O/sanitizer-initcheck.log.gz" "$O/sanitizer-ids.txt" "$O/kernel-ref" "$O/ref-dump.log"
exit $rc
