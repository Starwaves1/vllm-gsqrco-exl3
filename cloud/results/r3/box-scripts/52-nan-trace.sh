#!/bin/bash
# R3-52 where the echo / short-prompt prompt_logprobs NaN enters (50: GSQ's final hidden states hold
# NaN at some prompt positions, W4A16 does not). Production's main argv + --enforce-eager (so the
# Python hook runs per call), pyhook R3_NANTRACE: every GGUF linear call while armed -> layer, rows,
# types, input/output absmax and NaN/inf counts; the first call that turns finite input into
# NaN/inf saves x and the weight. Requests (armed one at a time): echo + logprobs on "The capital of
# Denmark is", then echo + logprobs on the 40-token prompt.
# Output: /workspace/logs/r3/52-nan-trace/{summary.txt, srv/nantrace-*.jsonl, *.first.pt}
# GPU time: ~10 min.
#   bash 52-nan-trace.sh [--plan]
source "$(dirname "$0")/lib.sh"
r3_init 52-nan-trace "$@"
if [ $R3_PLAN = 1 ]; then sed -n '2,10p' "$0"; exit 0; fi
export R3_EXTRA_PYTHONPATH=$R3_S/pyhook
r3_env
r3_preflight
export R3_NANCHECK=$L/nancheck.jsonl R3_NANTRACE=$L/nantrace.jsonl
R3_MUT=("flag|--enforce-eager")
r3_serve srv
send() {  # tag json-body
  touch "$R3_NANTRACE.arm"
  curl -s -o "$R/$1.json" -w "%{http_code}" "$GSQ_URL/v1/completions" -H "Authorization: Bearer $GSQ_API_KEY" \
    -H "Content-Type: application/json" -d "$2" > "$R/$1.status"
  rm -f "$R3_NANTRACE.arm"
  sleep 1
  mv "$R3_NANTRACE" "$R/nantrace-$1.jsonl" 2>/dev/null
  [ -f "$R3_NANTRACE.first.pt" ] && mv "$R3_NANTRACE.first.pt" "$R/$1.first.pt"
  echo "$1: HTTP $(cat "$R/$1.status") $(head -c 200 "$R/$1.json")"
}
r3_step requests
send echo5 '{"model":"qwen3.8-27b","prompt":"The capital of Denmark is","echo":true,"logprobs":1,"max_tokens":4,"temperature":0}'
P40=$("$PY" -c "import sys,json; sys.path.insert(0,'$R3_S'); import r3load; print(json.dumps(r3load.prompt_ids('plp-echo40','chat',40)))")
send echo40 "{\"model\":\"qwen3.8-27b\",\"prompt\":$P40,\"echo\":true,\"logprobs\":1,\"max_tokens\":4,\"temperature\":0}"
r3_stop
r3_step analyze
for t in echo5 echo40; do
  "$PY" - "$R/nantrace-$t.jsonl" > "$R/first-$t.txt" <<'PYEOF'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
print(f"{len(recs)} linear calls traced")
first = next((i for i, r in enumerate(recs) if r.get("out_bad") and not r.get("x_bad")), None)
if first is None:
    print("no call turned finite input into NaN/inf");
    bad = [r for r in recs if r.get("out_bad") or r.get("x_bad")]
    print(f"calls with any non-finite: {len(bad)}")
    for r in bad[:5]: print(json.dumps(r))
else:
    for r in recs[max(0, first - 6): first + 3]:
        print(("FIRST " if r is recs[first] else "      ") + json.dumps(r))
big = sorted(recs, key=lambda r: -r.get("x_absmax", 0))[:8]
print("largest inputs:"); [print("  " + json.dumps({k: r.get(k) for k in ("layer", "n", "wtype", "x_absmax", "out_absmax", "out_bad")})) for r in big]
PYEOF
done
r3_summary "R3-52 NaN trace (box, $(date -u +%F)), eager server, plugin $(git -C "$R3_PLUGIN_WT" rev-parse --short HEAD)" \
  "echo5: HTTP $(cat "$R/echo5.status")" "$(cat "$R/first-echo5.txt")" "" "echo40: HTTP $(cat "$R/echo40.status")" "$(cat "$R/first-echo40.txt")" \
  "" "logits-processor NaN probe:" "$(head -6 "$R3_NANCHECK" 2>/dev/null | cut -c1-300)"
cat "$L/summary.txt"
