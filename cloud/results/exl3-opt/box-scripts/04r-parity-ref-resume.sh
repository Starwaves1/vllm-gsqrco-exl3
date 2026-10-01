#!/bin/bash
# Phase 1's 04-parity-ref, resumed (exllamav3 reference logits, independent of the plugin): dumps
# only the sequences missing from /workspace/runs/exl3/parity/exl3 (04 was stopped twice to let the
# exl3-opt jobs run; 8 of 11 survived), merges them and their exl3_logits.json records, then 04's
# spread pass (EXL3_HGEMM_F16ACC=0 vs the default, the gate scale for 05).
#   gpuq submit exl3-04r -- env WT=/workspace/wt-exl3-opt bash .../exl3-opt/box-scripts/04r-parity-ref-resume.sh
export WT=${WT:-/workspace/wt-exl3-opt}
source "$WT/cloud/results/exl3/box-scripts/lib.sh"
job_log 04r-parity-ref-resume
require_idle_gpu
P=$R/parity; O=$R/04-parity-ref; mkdir -p "$O"
[ -f "$P/prompts/manifest.json" ] || die "no prompts in $P/prompts (run 04 first)"
# missing: no record in exl3_logits.json (an "oom" record stays as it is: the context limit)
missing=$(python3 - "$P" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
j = p / "exl3" / "exl3_logits.json"
done = json.loads(j.read_text())["sequences"] if j.exists() else {}
names = sorted(f.stem for f in (p / "prompts").glob("seq_*.ids"))
print(",".join(n for n in names if n not in done or (done[n]["status"] == "ok" and not (p / "exl3" / f"{n}.exl3.f32").exists())))
PY
)
echo "missing reference dumps: ${missing:-none}"
if [ -n "$missing" ]; then
  rm -rf "$P/exl3-resume"
  "$GSQ_EXL3_VENV/bin/python" bench/parity/exl3_logits.py -m "$EXL3_MODEL" -d "$P/prompts" -o "$P/exl3-resume" \
    --only "$missing" 2>&1 | tee "$O/exl3-resume.log"
  mkdir -p "$P/exl3"; mv "$P"/exl3-resume/*.exl3.f32 "$P/exl3/" 2>/dev/null || true
  python3 - "$P/exl3/exl3_logits.json" "$P/exl3-resume/exl3_logits.json" <<'PY'
import json, os, sys
base = json.load(open(sys.argv[1])) if os.path.exists(sys.argv[1]) else {"sequences": {}}
new = json.load(open(sys.argv[2]))
base = {**new, "sequences": base["sequences"]}
base["sequences"].update(new["sequences"])
base["sequences"] = dict(sorted(base["sequences"].items()))
json.dump(base, open(sys.argv[1], "w"), indent=1)
print({n: s["status"] for n, s in base["sequences"].items()})
PY
  rm -rf "$P/exl3-resume"
fi
cp "$P/exl3/exl3_logits.json" "$O/"
# nothing was missing and a spread from after the dumps exists (e.g. a full 04 ran first): keep it
if [ -z "$missing" ] && [ -s "$O/spread.json" ] && [ "$O/spread.json" -nt "$P/exl3/exl3_logits.json" ]; then
  echo "spread kept: $O/spread.json is newer than the dumps"
elif [ "${EXL3_SPREAD:-1}" = 1 ] && [ "$(df --output=avail -B1G /workspace | tail -1)" -ge 9 ]; then
  EXL3_HGEMM_F16ACC=0 "$GSQ_EXL3_VENV/bin/python" bench/parity/exl3_logits.py -m "$EXL3_MODEL" -d "$P/prompts" \
    -o "$P/exl3-fp32acc" 2>&1 | tee "$O/exl3-fp32acc.log"
  cp "$P/exl3-fp32acc/exl3_logits.json" "$O/exl3_logits-fp32acc.json"
  mkdir -p "$P/spread-ref" "$P/spread-other"
  for f in "$P"/exl3/*.exl3.f32; do n=$(basename "$f" .exl3.f32)
    ln -sf "$f" "$P/spread-ref/$n.llama.f32"
    [ -f "$P/exl3-fp32acc/$n.exl3.f32" ] && ln -sf "$P/exl3-fp32acc/$n.exl3.f32" "$P/spread-other/$n.vllm.f32"; done
  python3 bench/parity/compare.py -d "$P/prompts" -l "$P/spread-ref" -v "$P/spread-other" --json "$O/spread.json" \
    | tee "$O/spread.txt" || true
  rm -rf "$P/exl3-fp32acc" "$P/spread-ref" "$P/spread-other"
else
  echo "spread pass skipped (EXL3_SPREAD=${EXL3_SPREAD:-1}, $(df --output=avail -B1G /workspace | tail -1) GB free)" | tee "$O/spread.txt"
fi
keep "$O" 04-parity-ref "$O/exl3_logits.json" "$O/exl3_logits-fp32acc.json" "$O/spread.txt" "$O/spread.json" "$O/exl3-resume.log"
grep -q '"status": "ok"' "$P/exl3/exl3_logits.json" || die "no usable reference"
