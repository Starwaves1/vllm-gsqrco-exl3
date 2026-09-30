#!/bin/bash
# EXL3 phase 1, job 03 (~15-20 min): serve the EXL3 checkpoint on production's main argv
# (scripts/serve-exl3.sh: MTP k=5 schedule, fp8 KV, 200k, 16 seqs, graphs to 48, prefix caching,
# KV offload; quantization auto-detected; port 18090), then chat / reasoning / tool-call smoke,
# the draft head row count, MTP drafts and acceptance counters, VRAM after load, KV tokens.
# Needs job 02 (draft head) first.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
job_log 03-smoke
require_idle_gpu
O=$R/03-smoke; rm -rf "$O"; mkdir -p "$O"
[ -f "$EXL3_MODEL/mtp_draft_head.safetensors" ] || die "run 02-draft-head first"
serve "$O" || die "server did not come up (see $O/server.log)"
rows=$(sed -n 's/.*MTP drafter uses a \([0-9]*\)-token draft head.*/\1/p' "$O/server.log" | head -1)
echo "draft head rows: ${rows:-none} (want 40960)" | tee "$O/draft.txt"
"$GSQ_VENV/bin/python" cloud/results/phase1/box-scripts/smoke_chat.py "$GSQ_URL" > "$O/smoke.json" 2>&1 || true
tail -1 "$O/smoke.json"
curl -s "$GSQ_URL/metrics" -H "Authorization: Bearer $GSQ_API_KEY" | grep -E '^vllm:(spec_decode|kv_cache_usage|prefix_cache)' > "$O/metrics.prom" || true
drafts=$(awk '/^vllm:spec_decode_num_drafts/ {s += $NF} END {print s + 0}' "$O/metrics.prom")
acc=$(awk '/^vllm:spec_decode_num_accepted_tokens_total/ {a += $NF} /^vllm:spec_decode_num_draft_tokens_total/ {d += $NF} END {if (d) printf "%.3f", a / d; else print "n/a"}' "$O/metrics.prom")
{ echo "03-smoke $(date -u +%FT%TZ)"; cat "$O/load.txt"; cat "$O/draft.txt"
  echo "smoke: $(tail -1 "$O/smoke.json")"; echo "MTP drafts $drafts, draft-token acceptance $acc"; } > "$O/summary.txt"
stopall; trap - EXIT
gzip -kf "$O/server.log"
keep "$O" 03-smoke "$O/summary.txt" "$O/argv.txt" "$O/smoke.json" "$O/metrics.prom" "$O/server.log.gz"
cat "$O/summary.txt"
grep -q SMOKE_OK "$O/smoke.json" && [ "${drafts%.*}" -gt 0 ] && [ "$rows" = 40960 ] || die "smoke failed (see $O/summary.txt)"
