"""Report and verdict of a torture run: report.json and REPORT.md in the run dir (`torture report DIR`).

A run dir holds plan.json, load.jsonl, phases.jsonl, monitor.csv and faults.log; a switch run holds
switch.jsonl plus one such dir per leg. The verdict passes only if every criterion holds:
  faults    faults.log is empty (device faults, EngineDeadError, Traceback, OOM, server exited,
            restarted, unhealthy or never came up)
  alive     every monitor row: /health 200 and (server pid given) alive; 0 restarts; monitored
            >= 98% of the plan (less 2 min, the monitor's 60 s granularity)
  errors    every response is in OK_CLASSES (client cancels and planned client timeouts are fine;
            reasoning-only, EOS-first and short whitespace outputs are counted, not failed)
  memory    after warm-up (first hour, or first quarter of shorter runs): server GPU MiB grows
            <= 256 and host anon RSS <= 1 GiB (shmem is the CPU KV tier, bounded by its size);
            only with --server-pid
  drift     greedy probe TPOT median, last hour vs first hour, <= +15% (runs >= 3 h)
  idle      probe TPOT right after idle gaps <= +15% over the other probes (>= 3 such probes)
  coverage  every phase type ran (runs >= 6 h)
  switch    per leg (switch mode): healthy, stopped without SIGKILL, no process left, port free,
            GPU back to idle (+256 MiB), no new /dev/shm files; the two models' KV-tier dirs disjoint
"""

import json
import statistics
from pathlib import Path

from load import OK_CLASSES, TYPES

LIMITS = {"gpu_mib_growth": 256, "anon_rss_kib_growth": 1 << 20, "probe_tpot_drift": 0.15, "idle_tpot_ratio": 0.15,
          "monitored_fraction": 0.98, "drift_min_hours": 3, "coverage_min_hours": 6, "gpu_idle_slack_mib": 256}


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def pct(xs: list[float], q: float):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def growth(ts: list[float], ys: list[float]):
    """Least-squares slope times the span: growth that is a trend, not a spike."""
    if len(ts) < 3 or ts[-1] == ts[0]:
        return None
    mt, my = statistics.fmean(ts), statistics.fmean(ys)
    sxx = sum((t - mt) ** 2 for t in ts)
    return sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / sxx * (ts[-1] - ts[0])


def ms(x):
    return None if x is None else round(x * 1000, 1)


def phase_rows(recs: list[dict], phases: list[dict]) -> list[dict]:
    rows = []
    for ph in phases:
        rs = [r for r in recs if r.get("phase") == ph["i"] and r["kind"] not in ("tokenize", "metrics", "probe")]
        ttft = [r["ttft"] for r in rs if r.get("ttft") is not None and r["class"] in OK_CLASSES]
        tpot = [r["tpot"] for r in rs if r.get("tpot") is not None and r["class"] in OK_CLASSES]
        m0, m1 = ph.get("metrics0", {}), ph.get("metrics1", {})
        d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        hit = lambda p: round(d[p + "hits"] / d[p + "queries"], 3) if d.get(p + "queries") else None  # noqa: E731
        longs = [r for r in rs if r["kind"] == "long"]
        fails: dict[str, int] = {}
        for r in recs:
            if r.get("phase") == ph["i"] and r["class"] not in OK_CLASSES:
                fails[r["class"]] = fails.get(r["class"], 0) + 1
        rows.append({
            "i": ph["i"], "type": ph["type"], "minutes": round(ph["dur"] / 60, 1),
            "drain_s": round(ph["t_drained"] - ph["t_end"], 1) if "t_drained" in ph else None,
            "requests": len(rs), "fails": fails,
            "classes": {c: sum(r["class"] == c for r in rs) for c in sorted({r["class"] for r in rs})},
            "cancelled": sum(r["class"] == "cancelled" for r in rs), "timeouts": sum(r["class"] == "client_timeout" for r in rs),
            "ttft_s": [pct(ttft, q) for q in (0.5, 0.95, 0.99)], "tpot_ms": [ms(pct(tpot, q)) for q in (0.5, 0.95, 0.99)],
            "prefix_hit": hit("vllm:prefix_cache_"), "external_hit": hit("vllm:external_prefix_cache_"),
            "planned_hit": ph.get("planned_hit"), "long_sent_as_hit": round(sum(bool(r.get("hit")) for r in longs) / len(longs), 3) if longs else None,
            "preemptions": d.get("vllm:num_preemptions"), "corrupted": d.get("vllm:corrupted_requests"),
            "tier_failures": {k: v for k, v in d.items() if v and ("fail" in k or "lost" in k or "dropped" in k or "breaker" in k)},
            "mtp_accept_len": round(1 + d["vllm:spec_decode_num_accepted_tokens"] / d["vllm:spec_decode_num_drafts"], 3)
            if d.get("vllm:spec_decode_num_drafts") else None,
        })
    return rows


def monitor_stats(mon: list[dict], warm_from: float) -> dict:
    out = {}
    for col in (c for c in (mon[0] if mon else {}) if c != "t"):
        try:
            ys = [float(r[col]) for r in mon]
        except ValueError:
            continue
        w = [(float(r["t"]) / 3600, float(r[col])) for r in mon if float(r["t"]) >= warm_from]
        g = growth([t for t, _ in w], [y for _, y in w])
        out[col] = {"min": min(ys), "max": max(ys), "growth_after_warmup": None if g is None else round(g, 3)}
    return out


def crit(ok: bool, value, limit=None, applies=True) -> dict:
    return {"pass": bool(ok) or not applies, "applies": applies, "value": value, "limit": limit}


def run_report(d: Path) -> dict:
    plan = json.loads((d / "plan.json").read_text()) if (d / "plan.json").exists() else {"total_s": 0, "phases": []}
    recs, phases = jsonl(d / "load.jsonl"), jsonl(d / "phases.jsonl")
    lines = (d / "monitor.csv").read_text().splitlines() if (d / "monitor.csv").exists() else []
    mon = [dict(zip(lines[0].split(","), x.split(","))) for x in lines[1:] if x] if lines else []
    faults = [x for x in (d / "faults.log").read_text().splitlines() if x.strip()] if (d / "faults.log").exists() else []
    t0 = float(mon[0]["t"]) if mon else 0.0
    span = float(mon[-1]["t"]) - t0 if mon else 0.0
    planned_h = plan["total_s"] / 3600
    warm_from = t0 + min(3600, span / 4)
    stats = monitor_stats(mon, warm_from)

    classes: dict[str, int] = {}
    for r in recs:
        classes[r["class"]] = classes.get(r["class"], 0) + 1
    bad = [r for r in recs if r["class"] not in OK_CLASSES]

    probes = [r for r in recs if r["kind"] == "probe" and r["class"] == "ok" and r.get("tpot")]
    r0 = min((r["t"] for r in recs), default=0.0)
    r1 = max((r["t"] for r in recs), default=0.0)
    first = [r["tpot"] for r in probes if r["t"] - r0 < 3600]
    last = [r["tpot"] for r in probes if r1 - r["t"] < 3600]
    drift = statistics.median(last) / statistics.median(first) - 1 if first and last else None
    after = [r["tpot"] for r in probes if r.get("after_idle")]
    other = [r["tpot"] for r in probes if not r.get("after_idle")]
    idle_ratio = statistics.median(after) / statistics.median(other) - 1 if after and other else None
    shas = [r.get("text_sha") for r in recs if r["kind"] == "probe" and r["class"] == "ok"]
    mode = max(set(shas), key=shas.count) if shas else None

    g_gpu = (stats.get("gpu_mib") or {}).get("growth_after_warmup")
    g_anon = (stats.get("rss_anon_kib") or {}).get("growth_after_warmup")
    ran = {p["type"] for p in phases}
    criteria = {
        "faults": crit(not faults, len(faults), 0),
        "alive": crit(bool(mon) and all(r["server_alive"] in ("1", "") and r["health"] == "200" for r in mon)
                      and int(float(mon[-1]["restarts"])) == 0 and span >= LIMITS["monitored_fraction"] * plan["total_s"] - 120,
                      {"rows": len(mon), "rows_not_alive_or_healthy": sum(r["server_alive"] not in ("1", "") or r["health"] != "200" for r in mon),
                       "restarts": int(float(mon[-1]["restarts"])) if mon else None, "hours": round(span / 3600, 2), "planned_hours": round(planned_h, 2)}),
        "errors": crit(not bad, {c: n for c, n in classes.items() if c not in OK_CLASSES}, 0),
        "memory": crit((g_gpu or 0) <= LIMITS["gpu_mib_growth"] and (g_anon or 0) <= LIMITS["anon_rss_kib_growth"],
                       {"gpu_mib_growth": g_gpu, "anon_rss_kib_growth": g_anon}, [LIMITS["gpu_mib_growth"], LIMITS["anon_rss_kib_growth"]],
                       applies="rss_anon_kib" in stats),
        "drift": crit(drift is not None and drift <= LIMITS["probe_tpot_drift"], None if drift is None else round(drift, 4),
                      LIMITS["probe_tpot_drift"], applies=planned_h >= LIMITS["drift_min_hours"]),
        "idle": crit(idle_ratio is not None and idle_ratio <= LIMITS["idle_tpot_ratio"], None if idle_ratio is None else round(idle_ratio, 4),
                     LIMITS["idle_tpot_ratio"], applies=len(after) >= 3),
        "coverage": crit(set(TYPES) | {"idle"} <= ran, sorted((set(TYPES) | {"idle"}) - ran), [],
                         applies=planned_h >= LIMITS["coverage_min_hours"]),
    }
    return {
        "mode": "run", "dir": str(d), "seed": plan.get("seed"), "planned_hours": round(planned_h, 2), "hours_monitored": round(span / 3600, 2),
        "requests": len(recs), "classes": classes, "phases_run": len(phases), "phase_types_run": sorted(ran),
        "probe": {"n": len(probes), "tpot_ms_first_hour": ms(statistics.median(first)) if first else None,
                  "tpot_ms_last_hour": ms(statistics.median(last)) if last else None, "after_idle_n": len(after),
                  "greedy_distinct_outputs": len(set(shas)), "greedy_mode_fraction": round(shas.count(mode) / len(shas), 3) if shas else None},
        "monitor": stats, "faults": faults[:50], "fail_samples": bad[:20],
        "phases": phase_rows(recs, phases), "criteria": criteria, "limits": LIMITS,
        "pass": all(c["pass"] for c in criteria.values()),
    }


def switch_report(d: Path) -> dict:
    legs = jsonl(d / "switch.jsonl")
    runs = {}
    for leg in legs:
        p = d / leg["dir"]
        runs[leg["dir"]] = run_report(p) if (p / "monitor.csv").exists() else None
    touched = {}
    for leg in legs:
        touched.setdefault(leg["side"], set()).update(leg.get("tier_dirs_touched", []))
    overlap = sorted(touched.get("a", set()) & touched.get("b", set()))
    rows = []
    for leg in legs:
        r = runs.get(leg["dir"])
        checks = {"healthy": leg.get("healthy_s") is not None, "clean_stop": leg.get("killed") == 0,
                  "no_leftover_procs": leg.get("leftover_procs") == 0, "port_free": leg.get("port_free") == 1,
                  "gpu_idle": None in (leg.get("gpu_mib_after"), leg.get("gpu_idle_mib"))  # no nvidia-smi: not measured
                  or leg["gpu_mib_after"] <= leg["gpu_idle_mib"] + LIMITS["gpu_idle_slack_mib"],
                  "no_shm_left": not leg.get("shm_left"),
                  "run": bool(r) and all(r["criteria"][k]["pass"] for k in ("faults", "alive", "errors", "memory"))}
        rows.append({**leg, "checks": checks, "pass": all(checks.values()),
                     "requests": r["requests"] if r else 0, "fail_classes": r["criteria"]["errors"]["value"] if r else None})
    criteria = {"legs": crit(bool(rows) and all(x["pass"] for x in rows), sum(not x["pass"] for x in rows), 0),
                "namespaces_disjoint": crit(not overlap, overlap, [])}
    return {"mode": "switch", "dir": str(d), "legs": rows, "tier_dirs": {k: sorted(v) for k, v in touched.items()},
            "runs": runs, "criteria": criteria, "limits": LIMITS, "pass": all(c["pass"] for c in criteria.values())}


def fmt(x):
    return "-" if x is None else (f"{x:.3g}" if isinstance(x, float) else str(x))


def markdown(rep: dict) -> str:
    out = [f"# Torture {rep['mode']}: {'PASS' if rep['pass'] else 'FAIL'}", "", f"`{rep['dir']}`", "", "| criterion | pass | value | limit |",
           "|---|---|---|---|"]
    for k, c in rep["criteria"].items():
        out.append(f"| {k} | {'yes' if c['pass'] else '**NO**'}{'' if c['applies'] else ' (n/a)'} | {json.dumps(c['value'])} | {json.dumps(c['limit'])} |")
    if rep["mode"] == "switch":
        out += ["", "| leg | serve | start-to-healthy s | requests | failed checks | GPU MiB after / idle | shm left | tier dirs |", "|---|---|---|---|---|---|---|---|"]
        for x in rep["legs"]:
            failed = [k for k, v in x["checks"].items() if not v]
            out.append(f"| {x['dir']} | {Path(x['serve']).name} | {fmt(x.get('healthy_s'))} | {x['requests']} | {', '.join(failed) or '-'} "
                       f"| {fmt(x.get('gpu_mib_after'))} / {fmt(x.get('gpu_idle_mib'))} | {len(x.get('shm_left') or [])} | {', '.join(x.get('tier_dirs_touched') or []) or '-'} |")
        return "\n".join(out) + "\n"
    p = rep["probe"]
    out += ["", f"{rep['requests']} responses over {rep['hours_monitored']} h ({rep['phases_run']} phases). Classes: "
            + ", ".join(f"{k} {v}" for k, v in sorted(rep["classes"].items())),
            f"Greedy probe: TPOT {fmt(p['tpot_ms_first_hour'])} ms first hour, {fmt(p['tpot_ms_last_hour'])} ms last hour; "
            f"{p['greedy_distinct_outputs']} distinct outputs, mode {fmt(p['greedy_mode_fraction'])} (T=0 is not batch-invariant under MTP: recorded only).",
            "", "| # | type | min | drain s | req | fails | cancel | timeout | TTFT p50/95/99 s | TPOT p50/95/99 ms | prefix hit (ext) | planned hit | preempt | MTP len |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["phases"]:
        out.append(f"| {r['i']} | {r['type']} | {r['minutes']} | {fmt(r['drain_s'])} | {r['requests']} | {json.dumps(r['fails']) if r['fails'] else '-'} "
                   f"| {r['cancelled']} | {r['timeouts']} | {'/'.join(fmt(x) for x in r['ttft_s'])} | {'/'.join(fmt(x) for x in r['tpot_ms'])} "
                   f"| {fmt(r['prefix_hit'])} ({fmt(r['external_hit'])}) | {fmt(r['planned_hit'])} | {fmt(r['preemptions'])} | {fmt(r['mtp_accept_len'])} |")
    out += ["", "| monitor column | min | max | growth after warm-up |", "|---|---|---|---|"]
    out += [f"| {k} | {fmt(v['min'])} | {fmt(v['max'])} | {fmt(v['growth_after_warmup'])} |" for k, v in rep["monitor"].items()]
    if rep["faults"]:
        out += ["", "Faults:", "```", *rep["faults"], "```"]
    return "\n".join(out) + "\n"


def write(d: Path) -> dict:
    rep = switch_report(d) if (d / "switch.jsonl").exists() else run_report(d)
    (d / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    md = markdown(rep)
    (d / "REPORT.md").write_text(md)
    print("\n\n".join(md.split("\n\n")[:3]))
    return rep
