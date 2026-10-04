"""Validation kit bookkeeping: environment record, power/clock gate, results.json and summary.md.

  python kit/report.py env       --out F.json [--gpu I] [--no-gpu]   card, driver, CUDA, software, commits
  python kit/report.py gpucheck  --out F.json [--gpu I] [--allow-capped] [--idle-mib N] [--wait S]
        exit 0: go; 3: power-capped below the default limit (refused without --allow-capped);
        4: the card is busy (memory or utilization above the idle bar for S seconds)
  python kit/report.py collect   DIR      every artefact in DIR -> DIR/results.json + DIR/summary.md
  python kit/report.py selftest  DIR      fake artefacts in DIR, then collect (CPU dry run)

Standard library only (env imports torch/vllm when present, for their versions).
"""

import argparse
import collections
import csv
import glob
import json
import os
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHORT = {"lcpp_mmvq": "mmvq", "lcpp_mmq": "mmq", "own_vec": "own", "iq3_vec": "iq3-dp4a",
         "iq3_vec_mma": "iq3-mma", "iq3_vec_mma_packed": "iq3-mma-p", "iq3_tiled_packed": "iq3-tiled-p",
         "mma_k": "mma_k", "mma_k_chunked64": "mma_k/64", "lcpp_mmvq_chunked8": "mmvq/8", "stock_mmvq": "s-mmvq", "stock_mmq": "s-mmq",
         "stock_dq": "s-dq+cublas"}
BASELINE = ("lcpp_mmvq", "lcpp_mmq", "lcpp_mmvq_chunked8")  # vendored llama.cpp b11211 MMVQ / MMQ


def sh(*cmd, timeout=60):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def smi_xml(gpu: int):
    out = sh("nvidia-smi", "-q", "-x", "-i", str(gpu))
    return ET.fromstring(out).find("gpu") if out else None


def _num(s):
    m = re.match(r"\s*([-\d.]+)", s or "")
    return float(m.group(1)) if m else None


def gpu_state(gpu: int) -> dict:
    g = smi_xml(gpu)
    if g is None:
        return {}
    t = lambda p: (g.findtext(p) or "").strip()  # noqa: E731
    pw = "power_readings" if g.find("power_readings") is not None else "gpu_power_readings"
    reasons = g.find("clocks_event_reasons") or g.find("clocks_throttle_reasons")
    active = [e.tag for e in (reasons if reasons is not None else []) if (e.text or "").strip() == "Active"]
    return {
        "name": t("product_name"), "uuid": t("uuid"), "pci_bus_id": t("pci/pci_bus_id"), "vbios": t("vbios_version"),
        "power_limit_w": _num(t(f"{pw}/current_power_limit") or t(f"{pw}/power_limit")),
        "default_power_limit_w": _num(t(f"{pw}/default_power_limit")),
        "max_power_limit_w": _num(t(f"{pw}/max_power_limit")),
        "power_draw_w": _num(t(f"{pw}/average_power_draw") or t(f"{pw}/power_draw") or t(f"{pw}/instant_power_draw")),
        "sm_clock_mhz": _num(t("clocks/sm_clock")), "mem_clock_mhz": _num(t("clocks/mem_clock")),
        "max_sm_clock_mhz": _num(t("max_clocks/sm_clock")), "max_mem_clock_mhz": _num(t("max_clocks/mem_clock")),
        "temperature_c": _num(t("temperature/gpu_temp")), "utilization_pct": _num(t("utilization/gpu_util")),
        "memory_used_mib": _num(t("fb_memory_usage/used")), "memory_total_mib": _num(t("fb_memory_usage/total")),
        "persistence_mode": t("persistence_mode"), "clock_event_reasons_active": active,
        "processes": [{"pid": p.findtext("pid"), "name": p.findtext("process_name"), "used": p.findtext("used_memory")}
                      for p in g.findall("processes/process_info")],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def cmd_env(a):
    e = {"host": socket.gethostname(), "date": time.strftime("%Y-%m-%d"), "platform": platform.platform(),
         "python": platform.python_version(), "cpu": "", "gpu_index": a.gpu}
    try:
        e["cpu"] = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    if not a.no_gpu:
        top = sh("nvidia-smi")
        m = re.search(r"Driver Version:\s*([\d.]+).*?CUDA Version:\s*([\d.]+)", top)
        e["driver"] = {"version": m.group(1), "cuda": m.group(2)} if m else {}
        e["gpu"] = gpu_state(a.gpu)
        e["gpu"]["compute_capability"] = sh("nvidia-smi", "-i", str(a.gpu), "--query-gpu=compute_cap",
                                            "--format=csv,noheader")
    sw = {}
    from importlib import metadata

    for pkg in ("torch", "vllm", "triton", "flashinfer-python", "gguf", "safetensors", "numpy",
                "vllm-gguf-plugin", "vllm-exl3-plugin", "nvidia-cuda-runtime", "nvidia-cuda-runtime-cu12"):
        try:
            sw[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        sw["torch_cuda"] = torch.version.cuda
        sw["torch_arch_list"] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    except Exception as ex:  # noqa: BLE001
        sw["torch_error"] = repr(ex)[:200]
    e["software"] = sw
    e["repo"] = {"commit": sh("git", "-C", str(ROOT), "rev-parse", "HEAD"),
                 "branch": sh("git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"),
                 "dirty": bool(sh("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no")),
                 "remote": sh("git", "-C", str(ROOT), "remote", "get-url", "origin")}
    Path(a.out).write_text(json.dumps(e, indent=1))
    print(json.dumps({k: e.get(k) for k in ("host", "driver")}), e.get("gpu", {}).get("name", "no gpu"))


def cmd_gpucheck(a):
    deadline = time.time() + a.wait
    while True:
        samples = []
        for _ in range(3):
            samples.append(gpu_state(a.gpu))
            time.sleep(1)
        s = samples[-1]
        if not s:
            sys.exit("nvidia-smi gave no data")
        busy_mem = max(x["memory_used_mib"] or 0 for x in samples)
        busy_util = max(x["utilization_pct"] or 0 for x in samples)
        idle = busy_mem <= a.idle_mib and busy_util <= a.idle_util
        if idle or time.time() >= deadline:
            break
        print(f"waiting: GPU {a.gpu} busy ({busy_mem:.0f} MiB used, {busy_util:.0f}% util)", flush=True)
        time.sleep(min(30, max(0, deadline - time.time())))
    lim, dflt = s["power_limit_w"], s["default_power_limit_w"]
    capped = lim is not None and dflt is not None and lim < dflt - 1
    s.update(capped=capped, allow_capped=a.allow_capped, idle=idle, idle_bar={"mib": a.idle_mib, "util": a.idle_util},
             max_memory_used_mib=busy_mem, max_utilization_pct=busy_util)
    Path(a.out).write_text(json.dumps(s, indent=1))
    print(f"GPU {a.gpu} {s['name']}: power limit {lim} W (default {dflt} W){' CAPPED' if capped else ''}, "
          f"SM {s['sm_clock_mhz']}/{s['max_sm_clock_mhz']} MHz, mem {s['mem_clock_mhz']}/{s['max_mem_clock_mhz']} MHz, "
          f"{s['temperature_c']} C, {busy_mem:.0f} MiB used, {busy_util:.0f}% util")
    if capped and not a.allow_capped:
        print("refusing: the card is power-capped below its default limit; rerun with --allow-capped to record "
              "numbers anyway (they will be labelled capped)")
        sys.exit(3)
    if not idle:
        print(f"refusing: GPU {a.gpu} is not idle (bar: <= {a.idle_mib} MiB used, <= {a.idle_util}% util)")
        sys.exit(4)


# ----------------------------------------------------------------------------------- collect


def clocks_summary(path: Path) -> dict:
    """1 Hz samples: timestamp, clocks.sm, clocks.mem, power.draw, utilization.gpu, temperature.gpu,
    clocks_event_reasons.active; busy = util >= 50 %."""
    rows = []
    for line in path.read_text().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 6 and _num(p[1]) is not None:
            rows.append(p)
    busy = [r for r in rows if (_num(r[4]) or 0) >= 50]
    if not busy:
        return {"samples": len(rows), "busy_s": 0}
    med = lambda i: statistics.median(_num(r[i]) for r in busy)  # noqa: E731
    reasons = collections.Counter(r[6] for r in busy if len(r) > 6 and r[6] not in ("0x0000000000000000", "[N/A]"))
    return {"samples": len(rows), "busy_s": len(busy), "median_sm_mhz": med(1), "median_mem_mhz": med(2),
            "median_power_w": med(3), "max_power_w": max(_num(r[3]) for r in busy),
            "max_temp_c": max(_num(r[5]) or 0 for r in busy), "event_reasons_busy": dict(reasons)}


def junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    out = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "duration_s": 0.0, "by_test": {}, "failures": []}
    for s in suites:
        out["duration_s"] += float(s.get("time", 0))
        for c in s.findall("testcase"):
            fn = c.get("name", "").split("[")[0]
            st = ("failed" if c.find("failure") is not None else "errors" if c.find("error") is not None
                  else "skipped" if c.find("skipped") is not None else "passed")
            out[st] += 1
            out["by_test"].setdefault(fn, collections.Counter())[st] += 1
            if st in ("failed", "errors") and len(out["failures"]) < 60:
                el = c.find("failure") if st == "failed" else c.find("error")
                out["failures"].append({"test": c.get("name"), "message": (el.get("message") or "")[:300]})
    out["by_test"] = {k: dict(v) for k, v in out["by_test"].items()}
    out["duration_s"] = round(out["duration_s"], 1)
    return out


def gguf_micro(d: Path) -> dict:
    meta = json.loads((d / "gguf_micro.json").read_text()) if (d / "gguf_micro.json").exists() else {}
    cells = list(csv.DictReader((d / "gguf_micro.tsv").open(), delimiter="\t"))
    bw = meta.get("copy_GBps")
    by = collections.defaultdict(dict)  # (type, rows, K, n) -> variant -> us
    shape_cnt, route_of, bytes_of = {}, {}, {}
    for c in cells:
        key = (c["type"], int(c["rows"]), int(c["K"]), int(c["n"]))
        shape_cnt[key[:3]] = int(c["main"]) + int(c["mtp"])
        if c["is_route"] == "1":
            route_of[key] = c["variant"]
        if c["us"]:
            by[key][c["variant"]] = float(c["us"])
            if c["GBps"]:
                bytes_of[key[:3]] = float(c["GBps"]) * 1e9 * float(c["us"]) * 1e-6
    out_cells, dispatch, model = [], [], []
    tn = collections.defaultdict(list)
    for key, v in sorted(by.items()):
        win = min(v, key=v.get)
        r = route_of.get(key)
        base = min((v[b] for b in BASELINE if b in v), default=v.get(r))  # no vendored op: the route (IQ1_M dequant)
        out_cells.append({"type": key[0], "rows": key[1], "K": key[2], "n": key[3], "winner": win, "winner_us": v[win],
                          "route": r, "route_us": v.get(r), "baseline_us": base})
        tn[(key[0], key[3])].append((key, v))
    for (typ, n), lst in sorted(tn.items()):
        common = set.intersection(*(set(v) for _, v in lst))
        tot = {var: sum(shape_cnt[k[:3]] * v[var] for k, v in lst) for var in common}
        rt = [shape_cnt[k[:3]] * v[route_of[k]] for k, v in lst if route_of.get(k) in v]
        base = [shape_cnt[k[:3]] * min((v[b] for b in BASELINE if b in v), default=v.get(route_of.get(k), 0)) for k, v in lst]
        win = min(tot, key=tot.get) if tot else None
        dispatch.append({"type": typ, "n": n, "winner": win, "winner_us": round(tot[win], 2) if win else None,
                         "route_us": round(sum(rt), 2) if len(rt) == len(lst) else None,
                         "baseline_us": round(sum(base), 2) if len(base) == len(lst) else None,
                         "route_mix": sorted({route_of.get(k) for k, _ in lst} - {None})})
    for n in sorted({k[3] for k in by}):
        ks = [k for k in by if k[3] == n]
        cnt = lambda k: shape_cnt[k[:3]]  # noqa: E731
        route = sum(cnt(k) * by[k][route_of[k]] for k in ks if route_of.get(k) in by[k])
        best = sum(cnt(k) * min(by[k].values()) for k in ks)
        base = sum(cnt(k) * min((by[k][b] for b in BASELINE if b in by[k]), default=by[k].get(route_of.get(k), 0)) for k in ks)
        floor = sum(cnt(k) * bytes_of.get(k[:3], 0) for k in ks) / (bw * 1e9) * 1e3 if bw else None
        model.append({"n": n, "route_ms": round(route / 1e3, 3), "best_ms": round(best / 1e3, 3),
                      "baseline_ms": round(base / 1e3, 3), "floor_ms": round(floor, 3) if floor else None})
    exact = [c for c in cells if c.get("graph_exact") == "0"]
    na = [c for c in cells if not c["us"]]
    return {**meta, "cells": len(cells), "per_cell": out_cells, "dispatch": dispatch, "model": model,
            "graph_mismatches": [{k: c[k] for k in ("type", "rows", "K", "n", "variant")} for c in exact],
            "not_applicable": [{k: c[k] for k in ("type", "rows", "K", "n", "variant", "note")} for c in na][:80]}


def exl3_counts():
    """(k, n, K) -> [target products, draft products] from the real checkpoint's headers."""
    heads = json.loads((ROOT / "hf-config/Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6/safetensors_headers.json").read_text())
    out = collections.defaultdict(lambda: [0, 0])
    for h in heads.values():
        for key, v in h["tensors"].items():
            if key.endswith(".trellis") and "visual" not in key:
                kt, nt, w = v["shape"]
                out[(kt * 16, nt * 16, w // 16)][1 if key.startswith("mtp.") else 0] += 1
    return out


def exl3_micro(d: Path, bw) -> dict:
    rows = list(csv.DictReader((d / "shapes.tsv").open(), delimiter="\t"))
    cnt = exl3_counts()
    by = collections.defaultdict(dict)
    for r in rows:
        by[(int(r["k"]), int(r["n"]), int(r["K"]), int(r["rows"]))][r["route"]] = float(r["us"])
    per, model = [], collections.defaultdict(lambda: {"best_us": 0.0, "gemm_us": 0.0, "floor_us": 0.0, "mix": collections.Counter()})
    for (k, n, K, m), v in sorted(by.items()):
        win = min(v, key=v.get)
        nbytes = k * n * K / 8
        per.append({"k": k, "n": n, "K": K, "rows": m, "winner": win, "us": v,
                    "eff": round(nbytes / (v[win] * 1e-6) / (bw * 1e9), 3) if bw else None})
        c = sum(cnt.get((k, n, K), [1, 0]))
        model[m]["best_us"] += c * v[win]
        model[m]["gemm_us"] += c * v.get("gemm", v.get("dequant", 0))
        model[m]["floor_us"] += c * nbytes / (bw * 1e3) if bw else 0
        model[m]["mix"][win] += c
    win_k = collections.defaultdict(collections.Counter)
    for p in per:
        win_k[(p["K"], p["rows"])][p["winner"]] += 1
    return {"per_shape": per,
            "winner_by_K_rows": [{"K": K, "rows": m, "winners": dict(c)} for (K, m), c in sorted(win_k.items())],
            "model": [{"rows": m, "best_ms": round(v["best_us"] / 1e3, 3), "gemm_or_dequant_ms": round(v["gemm_us"] / 1e3, 3),
                       "floor_ms": round(v["floor_us"] / 1e3, 3), "mix": dict(v["mix"])} for m, v in sorted(model.items())]}


def load_json(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def cmd_collect(a):
    d = Path(a.dir)
    res = {"kit_format": 1, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    res.update(load_json(d / "env.json") or {})
    res["bootstrap"] = load_json(d / "bootstrap.json")
    res["steps"] = load_json(d / "steps.json") or {}
    res["power_checks"] = {Path(p).stem.removeprefix("gpucheck-"): load_json(Path(p)) for p in sorted(glob.glob(f"{d}/gpucheck-*.json"))}
    res["capped"] = any((c or {}).get("capped") for c in res["power_checks"].values())
    res["clocks"] = {Path(p).stem.removeprefix("clocks-"): clocks_summary(Path(p)) for p in sorted(glob.glob(f"{d}/clocks-*.csv"))}
    t1 = {}
    for name in ("gguf", "exl3"):
        p = d / "tier1" / f"parity-{name}.xml"
        if p.exists():
            t1.setdefault("parity", {})[name] = junit(p)
    bw = None
    if (d / "tier1/gguf_micro.tsv").exists():
        t1["gguf_micro"] = gguf_micro(d / "tier1")
        bw = t1["gguf_micro"].get("copy_GBps")
    if (d / "tier1/exl3_micro/shapes.tsv").exists():
        t1["exl3_micro"] = exl3_micro(d / "tier1/exl3_micro", bw)
    if t1:
        res["tier1"] = t1
    for tier in ("tier2", "tier3"):
        runs = {}
        for p in sorted(glob.glob(f"{d}/{tier}/*/ladder.json")):
            run = Path(p).parent
            runs[run.name] = {"ladder": load_json(Path(p)), "serve": load_json(run / "serve.json"),
                              "corruption": [json.loads(line) for line in (run / "corruption.jsonl").read_text().splitlines()
                                             if line.startswith("{")] if (run / "corruption.jsonl").exists() else None}
        if runs:
            res[tier] = runs
    res["cc"] = res.get("gpu", {}).get("compute_capability", "")
    res["tiers"] = tier_status(res)
    (d / "results.json").write_text(json.dumps(res, indent=1))
    (d / "summary.md").write_text(render(res))
    print(f"wrote {d / 'results.json'} and {d / 'summary.md'}")


def tier_status(res: dict) -> dict:
    """{"1": "pass" | "partial" | "fail"}: pass = every step of the tier ok (parity with 0 failures),
    fail = none ok, partial otherwise (unsupported / failed / skipped steps are listed in steps)."""
    out = {}
    for t in ("1", "2", "3"):
        st = [v.get("status") for k, v in res.get("steps", {}).items() if k.startswith(f"tier{t}.")]
        if st:
            out[t] = "pass" if all(s == "ok" for s in st) else "fail" if not any(s == "ok" for s in st) else "partial"
    return out


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def render(r: dict) -> str:
    g, drv, sw, repo = r.get("gpu", {}), r.get("driver", {}), r.get("software", {}), r.get("repo", {})
    pc = next(iter(r.get("power_checks", {}).values()), None) or g
    L = [f"# Validation kit: {g.get('name', '?')} (sm{g.get('compute_capability', '?').replace('.', '')}), {r.get('host', '?')}, {r.get('date', '?')}", ""]
    L += ["| | |", "|---|---|",
          f"| card | {g.get('name', '?')}, {fmt(g.get('memory_total_mib'), 0)} MiB, compute capability {g.get('compute_capability', '?')}, PCI {g.get('pci_bus_id', '?')} |",
          f"| driver / CUDA | {drv.get('version', '?')} / {drv.get('cuda', '?')} |",
          f"| power limit | {fmt(pc.get('power_limit_w'), 0)} W (default {fmt(pc.get('default_power_limit_w'), 0)} W){' **CAPPED, numbers labelled capped**' if r.get('capped') else ''} |",
          f"| max clocks | SM {fmt(pc.get('max_sm_clock_mhz'), 0)} MHz, mem {fmt(pc.get('max_mem_clock_mhz'), 0)} MHz |",
          f"| software | torch {sw.get('torch', '?')} (CUDA {sw.get('torch_cuda', '?')}), vLLM {sw.get('vllm', '?')}, triton {sw.get('triton', '?')} |",
          f"| repo | {repo.get('remote', '?')} @ {repo.get('commit', '?')[:12]}{' (dirty)' if repo.get('dirty') else ''} |"]
    clk = {k: v for k, v in r.get("clocks", {}).items() if v.get("busy_s")}
    if clk:
        L += ["", "Clocks while busy (util >= 50 %):", "", "| step | busy s | median SM MHz | median mem MHz | median / max W | max C | clock-event reasons |", "|---|---|---|---|---|---|---|"]
        for k, v in clk.items():
            L.append(f"| {k} | {v['busy_s']} | {fmt(v['median_sm_mhz'], 0)} | {fmt(v['median_mem_mhz'], 0)} | "
                     f"{fmt(v['median_power_w'])} / {fmt(v['max_power_w'])} | {fmt(v['max_temp_c'], 0)} | "
                     f"{', '.join(f'{a} x{b}' for a, b in v['event_reasons_busy'].items()) or 'none'} |")
    if r.get("steps"):
        L += ["", "Tiers: " + ", ".join(f"tier {k} **{v}**" for k, v in r.get("tiers", {}).items()), "",
              "| step | status | note |", "|---|---|---|"]
        L += [f"| {k} | {v.get('status')} | {v.get('note', '')} |" for k, v in r["steps"].items()]
    b = r.get("bootstrap") or {}
    if b.get("builds"):
        L += ["", "| build | status | note |", "|---|---|---|"]
        L += [f"| {k} | {v.get('status')} | {v.get('note', '')} |" for k, v in b["builds"].items()]
    t1 = r.get("tier1", {})
    for name, p in t1.get("parity", {}).items():
        L += ["", f"## Tier 1: {name.upper()} kernel parity (synthetic weights at the 27B shapes)", "",
              f"{p['passed']} passed, {p['failed']} failed, {p['errors']} errors, {p['skipped']} skipped, {p['duration_s']} s", "",
              "| test | passed | failed | errors | skipped |", "|---|---|---|---|---|"]
        L += [f"| {k} | {v.get('passed', 0)} | {v.get('failed', 0)} | {v.get('errors', 0)} | {v.get('skipped', 0)} |"
              for k, v in sorted(p["by_test"].items())]
        if p["failures"]:
            L += ["", "First failures:", ""] + [f"- `{f['test']}`: {f['message'][:160]}" for f in p["failures"][:12]]
    m = t1.get("gguf_micro")
    if m:
        ns = sorted({x["n"] for x in m["dispatch"]})
        L += ["", "## Tier 1: GGUF routing winner per (type, rows) on this card", "",
              f"Copy bandwidth (floor) {fmt(m.get('copy_GBps'), 0)} GB/s; X {m.get('x_dtype')}; Route L "
              f"{'built' if m.get('route_l') else 'NOT built (stock kernels only)'}. Cell: the fastest kernel summed over the "
              "type's 27B shapes (weighted by tensor count); `(+x%)` = how much slower the current routing is there.", "",
              "| type | " + " | ".join(f"n={n}" for n in ns) + " |", "|---|" + "---|" * len(ns)]
        for typ in sorted({x["type"] for x in m["dispatch"]}):
            row = []
            for n in ns:
                x = next((y for y in m["dispatch"] if y["type"] == typ and y["n"] == n), None)
                if not x or not x["winner"]:
                    row.append("-")
                    continue
                loss = (x["route_us"] / x["winner_us"] - 1) * 100 if x["route_us"] and x["winner_us"] else None
                row.append(SHORT.get(x["winner"], x["winner"]) + (f" (+{loss:.0f}%)" if loss and loss >= 3 else ""))
            L.append(f"| {typ} | " + " | ".join(row) + " |")
        L += ["", "Whole-model GEMM time per forward (all 27B linears incl. lm_head and the MTP layer), ms:", "",
              "| rows | current routing | best per cell | vendored llama.cpp | copy-bandwidth floor | routing / floor |", "|---|---|---|---|---|---|"]
        L += [f"| {x['n']} | {x['route_ms']} | {x['best_ms']} | {x['baseline_ms']} | {fmt(x['floor_ms'], 3)} | "
              f"{fmt(x['route_ms'] / x['floor_ms'], 2) if x['floor_ms'] else '-'} |" for x in m["model"]]
        L += ["", f"CUDA-graph replay bit-identical to eager: {m['cells'] - len(m['graph_mismatches']) - len(m['not_applicable'])} "
              f"of {m['cells'] - len(m['not_applicable'])} timed cells" + (f"; mismatches: {m['graph_mismatches'][:6]}" if m["graph_mismatches"] else "")]
        if m.get("fatal"):
            L.append(f"\n**Device error stopped the micro-benchmark:** {m['fatal']}")
    e = t1.get("exl3_micro")
    if e:
        rows = sorted({x["rows"] for x in e["winner_by_K_rows"]})
        L += ["", "## Tier 1: EXL3 route winner per (bits K, rows)", "", "| K | " + " | ".join(f"{m_}" for m_ in rows) + " |",
              "|---|" + "---|" * len(rows)]
        for K in sorted({x["K"] for x in e["winner_by_K_rows"]}):
            cells = []
            for m_ in rows:
                x = next((y for y in e["winner_by_K_rows"] if y["K"] == K and y["rows"] == m_), None)
                cells.append("/".join(f"{k}{'' if len(x['winners']) == 1 else f' {v}'}" for k, v in x["winners"].items()) if x else "-")
            L.append(f"| {K} | " + " | ".join(cells) + " |")
        L += ["", "| rows | best route ms per forward | exl3_gemm (dequant above 144) ms | floor ms |", "|---|---|---|---|"]
        L += [f"| {x['rows']} | {x['best_ms']} | {x['gemm_or_dequant_ms']} | {x['floor_ms']} |" for x in e["model"]]
    for tier in ("tier2", "tier3"):
        for name, run in (r.get(tier) or {}).items():
            lad = run.get("ladder") or {}
            L += ["", f"## {tier.title()}: {name}", "", f"{lad.get('model', '?')}; {lad.get('note', '')}", "",
                  "| c | tok/s | ms/step | tok/step | mean TPOT ms |", "|---|---|---|---|---|"]
            for c, v in sorted((lad.get("pass2") or {}).items(), key=lambda kv: int(kv[0])):
                L.append(f"| {c} | {fmt(v.get('tok_s'))} | {fmt(v.get('ms_step'), 2)} | {fmt(v.get('tok_step'), 2)} | {fmt(v.get('mean_tpot_ms'), 2)} |")
            for cr in run.get("corruption") or []:
                L.append(f"\nCorruption check ({cr.get('config')}): {cr.get('corrupt')} of {cr.get('requests')} flagged, "
                         f"by kind {cr.get('by_kind')}, T=0 token mismatches vs eager reference {cr.get('t0_token_mismatch')}")
    return "\n".join(L) + "\n"


def cmd_selftest(a):
    """Fake artefacts in the shape the real steps write, then collect: exercises the writer and renderer."""
    d = Path(a.dir)
    (d / "tier1/exl3_micro").mkdir(parents=True, exist_ok=True)
    (d / "tier2/gguf-4b-q4km-k3").mkdir(parents=True, exist_ok=True)
    (d / "env.json").write_text(json.dumps({"host": "fakehost", "date": "2026-10-04", "driver": {"version": "595.84", "cuda": "13.4"},
                                            "gpu": {"name": "NVIDIA GeForce RTX 3070", "compute_capability": "8.6", "memory_total_mib": 8192},
                                            "software": {"torch": "2.13.0", "torch_cuda": "13.0", "vllm": "0.30.1rc1"},
                                            "repo": {"commit": "0" * 40, "remote": "https://github.com/Starwaves1/vllm-gsqrco-exl3"}}))
    (d / "gpucheck-tier1.json").write_text(json.dumps({"power_limit_w": 220.0, "default_power_limit_w": 220.0, "capped": False,
                                                       "max_sm_clock_mhz": 2100, "max_mem_clock_mhz": 7001}))
    (d / "clocks-tier1.csv").write_text("".join(f"2026/10/04 12:00:{i:02d}.000, 1905 MHz, 6801 MHz, 180.5 W, 97 %, 66, 0x0000000000000000\n" for i in range(20)))
    (d / "steps.json").write_text(json.dumps({"tier1.parity-gguf": {"status": "ok"}, "tier1.parity-exl3": {"status": "unsupported", "note": "EXL3 build failed on sm75"}}))
    (d / "tier1/parity-gguf.xml").write_text(
        '<testsuites><testsuite name="pytest" tests="3" time="12.5"><testcase classname="t" name="test_lcpp_mmvq[IQ3_S-1-bfloat16]"/>'
        '<testcase classname="t" name="test_lcpp_mmq[Q4_K-8-float16]"><failure message="AssertionError: too far"/></testcase>'
        '<testcase classname="t" name="test_lcpp_mma_k[IQ2_S-9]"><skipped message="x"/></testcase></testsuite></testsuites>')
    lines = ["type\trows\tK\tmain\tmtp\tn\tvariant\tis_route\tus\tGBps\teff\tgraph_exact\tnote"]
    for typ, rows, k, cnt in (("IQ3_S", 17408, 5120, 40), ("IQ3_S", 5120, 17408, 22), ("Q4_K", 17408, 5120, 5)):
        for n in (1, 8, 32):
            vs = {"lcpp_mmvq": 60.0 + n, "lcpp_mmq": 80.0 + n / 2, ("iq3_vec_mma_packed" if typ == "IQ3_S" else "own_vec"): 50.0 + n}
            if n > 8:
                vs.pop("lcpp_mmvq")
            route = min(vs, key=vs.get) if n < 32 else "lcpp_mmq"
            for v, us in vs.items():
                gb = rows * k * 0.43 / (us * 1e-6) / 1e9
                lines.append(f"{typ}\t{rows}\t{k}\t{cnt}\t0\t{n}\t{v}\t{int(v == route)}\t{us}\t{gb:.0f}\t{gb / 400:.3f}\t1\t")
    lines.append("Q4_K\t17408\t5120\t5\t0\t32\tmma_k\t0\t\t\t\t\tRuntimeError: guard")
    (d / "tier1/gguf_micro.tsv").write_text("\n".join(lines) + "\n")
    (d / "tier1/gguf_micro.json").write_text(json.dumps({"copy_GBps": 400.0, "x_dtype": "bfloat16", "route_l": True, "fatal": None}))
    ex = ["k\tn\tK\ttarget\tdraft\trows\troute\tus\tGBps\tfloor_pct"]
    for m_ in (1, 17, 145):
        ex.append(f"5120\t17408\t3\t1\t0\t{m_}\tgemm\t{40 + m_}\t300\t50")
        ex.append(f"5120\t17408\t3\t1\t0\t{m_}\tmr\t{45 + m_ / 2}\t300\t50")
    (d / "tier1/exl3_micro/shapes.tsv").write_text("\n".join(ex) + "\n")
    t2 = d / "tier2/gguf-4b-q4km-k3"
    (t2 / "ladder.json").write_text(json.dumps({"model": "unsloth/Qwen3.5-4B-MTP-GGUF Qwen3.5-4B-Q4_K_M.gguf", "note": "k=3",
                                                "pass2": {str(c): {"tok_s": 100.0 * c ** 0.8, "tok_step": 2.6, "ms_step": 9.5 + c,
                                                                   "mean_tpot_ms": 4.0} for c in (1, 2, 4, 8)}}))
    (t2 / "corruption.jsonl").write_text(json.dumps({"config": "k3", "requests": 200, "corrupt": 0, "by_kind": {}, "t0_token_mismatch": 3}) + "\n")
    cmd_collect(argparse.Namespace(dir=str(d)))
    r = json.loads((d / "results.json").read_text())
    disp = {(x["type"], x["n"]): x for x in r["tier1"]["gguf_micro"]["dispatch"]}
    assert disp[("IQ3_S", 1)]["winner"] == "iq3_vec_mma_packed", disp[("IQ3_S", 1)]
    assert disp[("IQ3_S", 32)]["route_us"] > disp[("IQ3_S", 32)]["winner_us"]
    assert r["tier1"]["parity"]["gguf"]["failed"] == 1 and r["clocks"]["tier1"]["busy_s"] == 20
    assert "| IQ3_S |" in (d / "summary.md").read_text()
    assert r["cc"] == "8.6" and r["tiers"] == {"1": "partial"}, (r["cc"], r["tiers"])
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("env")
    e.add_argument("--out", required=True)
    e.add_argument("--gpu", type=int, default=0)
    e.add_argument("--no-gpu", action="store_true")
    g = sub.add_parser("gpucheck")
    g.add_argument("--out", required=True)
    g.add_argument("--gpu", type=int, default=0)
    g.add_argument("--allow-capped", action="store_true")
    g.add_argument("--idle-mib", type=float, default=1024)
    g.add_argument("--idle-util", type=float, default=10)
    g.add_argument("--wait", type=float, default=0)
    for name in ("collect", "selftest"):
        sub.add_parser(name).add_argument("dir")
    a = ap.parse_args()
    {"env": cmd_env, "gpucheck": cmd_gpucheck, "collect": cmd_collect, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
