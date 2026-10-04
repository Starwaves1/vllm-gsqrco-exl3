#!/bin/bash
# EXL3 opt, job 13 (~15 min per mode): where one decode step goes, so the next levers are named
# by data. Per EXL3_MR mode (EXL3_OPT_PROFILE_MODES, default "0 2h"): one server on production's
# main argv with vLLM's torch profiler (6 iterations after 60), greedy chat requests at c=1 and
# c=4 (6 / 24 rows per verify pass at MTP k=5), then per complete step (exl3_pstep.py): GPU
# activities and ms per class (exl3_gemm, exl3_gemm_mr, its input Hadamard, dequant, hgemm,
# activation casts and part concatenation, attention, GDN, sampling, other), the top non-GEMM
# kernels, idle; host gaps per op (opt-p2's gaps.py); and the load/warmup time from the server
# log (Model loading took ...: exl3_warmup and exl3_mr_warmup run inside it).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 13-profile
require_idle_gpu
require_mr_build
[ -f "$EXL3_MODEL/mtp_draft_head.safetensors" ] || die "no draft head (phase 1's 02-draft-head)"
O=$R/13-profile; mkdir -p "$O"
H=(-H "Authorization: Bearer $GSQ_API_KEY" -H "Content-Type: application/json")
req() { curl -s "$GSQ_URL/v1/chat/completions" "${H[@]}" -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a detailed essay about the history of the printing press, part $2.\"}],\"max_tokens\":$1,\"temperature\":0}" > /dev/null; }
rc=0
modes=("${MODES_ARGS[@]}"); [ ${#modes[@]} = 0 ] && modes=(${EXL3_OPT_PROFILE_MODES:-0 2h})
for mr in "${modes[@]}"; do
  D=$O/mr$mr; rm -rf "$D"; mkdir -p "$D/trace"
  echo "=== EXL3_MR=$mr $(date -u +%FT%TZ)"
  serve_mr "$mr" "$D" --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$D/trace\",\"torch_profiler_with_stack\":false,\"torch_profiler_use_gzip\":false,\"ignore_frontend\":true,\"delay_iterations\":60,\"max_iterations\":6}" \
    || { rc=1; stopall; continue; }
  req 64 0
  for C in 1 4; do
    curl -s -X POST "$GSQ_URL/start_profile" "${H[@]}"
    pids=(); for i in $(seq $C); do req 400 "$i" & pids+=($!); done; wait "${pids[@]}"
    curl -s -X POST "$GSQ_URL/stop_profile" "${H[@]}"; sleep 20
    T=$(find "$D/trace" -name "*.json" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
    [ -n "$T" ] && mv "$T" "$D/c$C.trace.json" || { echo "no trace for c=$C"; rc=1; }
  done
  stopall
  { grep -E "Model loading took|Loading weights took|init engine" "$D/server.log" | cut -c1-200 || true
    for C in 1 4; do
      [ -f "$D/c$C.trace.json" ] || continue
      echo "== EXL3_MR=$mr c=$C"
      "$GSQ_VENV/bin/python" "$S/exl3_pstep.py" "$D/c$C.trace.json"
      "$GSQ_VENV/bin/python" "$WT/cloud/results/opt-p2/box-scripts/gaps.py" "$D/c$C.trace.json" 5 | head -20 || true
    done; } > "$D/profile.txt" 2>&1
  cat "$D/profile.txt"
  gzip -kf "$D/server.log"; for f in "$D"/c*.trace.json; do [ -f "$f" ] && gzip -kf "$f"; done
  keep "$D" "13-profile/mr$mr" "$D/profile.txt" "$D/load.txt" "$D/server.log.gz"
done
exit $rc
