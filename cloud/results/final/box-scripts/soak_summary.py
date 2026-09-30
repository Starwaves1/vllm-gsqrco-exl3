"""Summarise a bench/soak.sh run dir into soak-summary.json (stdout too):
hours, requests by status, faults, restarts, GPU MiB and host RSS min/max (after the first hour
too), and per-hour throughput (completion tokens / 3600 s, requests). bad_output records that
ended on length with tokens are counted as reasoning_only: soak_load before cf8fbde looked only
at message.content, and vLLM returns content None when max_tokens ends inside the thinking.
  python3 soak_summary.py RUN_DIR"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
rows = [r.split(",") for r in (d / "monitor.csv").read_text().splitlines()[1:] if r]
recs = [json.loads(x) for x in (d / "load.jsonl").read_text().splitlines() if x.strip()]
faults = (d / "faults.log").read_text().splitlines() if (d / "faults.log").exists() else []
t0 = float(rows[0][0])


def mm(xs):
    return [min(xs), max(xs)] if xs else None


status = {}
for r in recs:
    s = r["status"]
    if s == "bad_output" and r.get("finish") == "length" and r.get("completion_tokens", 0) > 0:
        s = "reasoning_only"
    status[s] = status.get(s, 0) + 1
hours = {}
for r in recs:
    h = int((r["t"] - t0) // 3600)
    e = hours.setdefault(h, {"requests": 0, "completion_tokens": 0, "prompt_tokens": 0})
    e["requests"] += 1
    e["completion_tokens"] += r.get("completion_tokens", 0)
    e["prompt_tokens"] += r.get("prompt_tokens", 0)
warm = [r for r in rows if float(r[0]) - t0 >= 3600]
out = {
    "hours_monitored": round((float(rows[-1][0]) - t0) / 3600, 2),
    "monitor_rows": len(rows),
    "server_alive_all": all(r[1] == "1" for r in rows),
    "health_non_200_rows": sum(r[2] != "200" for r in rows),
    "restarts": int(rows[-1][9]),
    "fault_lines": faults,
    "requests": len(recs),
    "status": status,
    "by_kind": {k: sum(r["kind"] == k for r in recs) for k in sorted({r["kind"] for r in recs})},
    "tool_calls_parsed": sum(r.get("tool_calls", 0) > 0 for r in recs),
    "gpu_mib_min_max": mm([int(r[3]) for r in rows]),
    "gpu_mib_min_max_after_1h": mm([int(r[3]) for r in warm]),
    "rss_gib_min_max": [round(x / 2**20, 2) for x in mm([int(r[4]) for r in rows])],
    "rss_gib_min_max_after_1h": [round(x / 2**20, 2) for x in mm([int(r[4]) for r in warm])] if warm else None,
    "preemptions_total": float(rows[-1][8]),
    "per_hour": [{"hour": h, **v, "completion_tok_per_s": round(v["completion_tokens"] / 3600, 1)}
                 for h, v in sorted(hours.items())],
}
if (d / "report.json").exists():
    out["soak_sh_report"] = json.loads((d / "report.json").read_text())
json.dump(out, open(d / "soak-summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
