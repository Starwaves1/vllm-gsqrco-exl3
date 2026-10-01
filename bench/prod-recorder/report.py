#!/usr/bin/env python3
"""Corruption report over recorded production traffic. Stdlib only.

    python3 report.py [--data data] [--since 1h] [--until ...] [--reclassify]
    python3 report.py --compare WINDOW_A WINDOW_B

A window is comma-separated terms, all of which must hold:
    FROM..TO      times: 2026-10-01T11:00, 11:00 (today), -2h, now, epoch
    key=value     record tag: async, spec, k_cfg, label, pid, start_ts
e.g.  --compare spec=mtp spec=off      --compare label=asis label=k3
      --compare 09:00..10:30 11:00..12:30
Writes REPORT.md + report.json (or COMPARE.md + compare.json) to --out.
"""
import argparse
import glob
import gzip
import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classify  # noqa: E402

GEN_PATHS = ("/v1/chat/completions", "/v1/completions")


# ------------------------------------------------------------ statistics

def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def newcombe(k1, n1, k2, n2):
    """Difference p2 - p1 with Newcombe's hybrid score 95% interval."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p2 - p1
    return d, d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2), d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)


def pct(x):
    return f"{100 * x:.2f}%"


def rate_str(k, n):
    p, lo, hi = wilson(k, n)
    return f"{pct(p)} [{pct(lo)}, {pct(hi)}]" if n else "n/a"


# --------------------------------------------------------------- loading

def parse_time(s, now=None):
    now = now or time.time()
    s = s.strip()
    if s in ("", "now"):
        return now
    m = re.fullmatch(r"-?(\d+(?:\.\d+)?)([smhd])", s)
    if m:
        return now - float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    if re.fullmatch(r"\d{9,}(\.\d+)?", s):
        return float(s)
    if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", s):
        s = time.strftime("%Y-%m-%dT") + s
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(s, fmt))
        except ValueError:
            pass
    raise SystemExit(f"cannot parse time {s!r}")


def load(data, since=None, until=None):
    recs = []
    files = glob.glob(os.path.join(data, "requests-*.jsonl")) + \
        glob.glob(os.path.join(data, "requests-*.jsonl.gz"))
    for f in sorted(files):
        m = re.search(r"requests-(\d{8}-\d{2})\.jsonl", f)
        if m and since:
            hour_end = time.mktime(time.strptime(m.group(1), "%Y%m%d-%H")) + 3600
            if hour_end < since:
                continue
        opener = gzip.open if f.endswith(".gz") else open
        with opener(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                t = r.get("t0") or 0
                if (since and t < since) or (until and t > until):
                    continue
                recs.append(r)
    return recs


def is_generation(r):
    return (r.get("path") in GEN_PATHS and r.get("status") == 200 and r.get("req_info")
            and not r.get("client_aborted") and not r.get("error")
            and not r.get("resp_truncated") and r.get("flags") is not None)


def response_choices(r):
    raw = r.get("resp_text")
    if raw is None:
        return None
    return classify.parse_response(raw.encode("utf-8"), r.get("ctype") or "")


def reclassify(r, allow):
    if r.get("resp_stored_truncated"):
        return r["flags"]
    p = response_choices(r)
    if p is None:
        return r["flags"]
    lp = (r.get("parse") or {}).get("lp")
    p["lp"] = lp
    return classify.classify(p, r["req_info"], r.get("prompt_info") or {}, allow)


# ------------------------------------------------------------------ bins

def b_temp(r):
    t = r["req_info"].get("temperature")
    return "T=0" if t is not None and float(t) == 0 else "T>0"


def b_k(r):
    c = r.get("cond") or {}
    if c.get("k_mixed"):
        return "k mixed"
    k = c.get("k_mode")
    return f"k={k}" if k is not None else "k=?"


def b_running(r):
    m = (r.get("cond") or {}).get("running_max")
    if m is None:
        return "?"
    m = int(round(m))
    return "<=1" if m <= 1 else "2-4" if m <= 4 else "5-8" if m <= 8 else "9-16" if m <= 16 else "17+"


def b_prompt(r):
    n = r.get("prompt_tokens")
    if n is None:
        return "?"
    for hi, name in ((1024, "<1k"), (4096, "1k-4k"), (16384, "4k-16k"), (65536, "16k-64k")):
        if n < hi:
            return name
    return "64k+"


def b_plp(r):
    if r["req_info"].get("prompt_logprobs"):
        return "self"
    return "yes" if (r.get("cond") or {}).get("other_prompt_logprobs_inflight") else "no"


def b_echo(r):
    return "yes" if (r.get("cond") or {}).get("other_echo_inflight") else "no"


def b_stream(r):
    return "stream" if r["req_info"].get("stream") else "non-stream"


def b_endpoint(r):
    return r["req_info"].get("type", "?")


def b_preempt(r):
    d = (r.get("cond") or {}).get("preempt_delta")
    return "?" if d is None else ("yes" if d > 0 else "no")


def b_hour(r):
    return time.strftime("%H", time.localtime(r["t0"]))


def b_upstream(r):
    u = r.get("up") or {}
    return f"async={u.get('async')} spec={u.get('spec')} k={u.get('k_cfg')}"


def b_label(r):
    return str((r.get("up") or {}).get("label"))


BINS = [  # (key, title, fn, split by temperature)
    ("temperature", "Temperature (T=0 vs T>0)", b_temp, False),
    ("k", "k in use (from running count via the MTP schedule) x temperature", b_k, True),
    ("running", "Running requests (max during generation) x temperature", b_running, True),
    ("prompt", "Prompt length (tokens)", b_prompt, True),
    ("prompt_logprobs", "prompt_logprobs request in flight", b_plp, True),
    ("echo", "echo request in flight", b_echo, False),
    ("stream", "Streaming", b_stream, True),
    ("endpoint", "Endpoint", b_endpoint, False),
    ("preempt", "Preemption during generation", b_preempt, False),
    ("upstream", "Server config (from the vLLM argv)", b_upstream, False),
    ("label", "Recorder label", b_label, False),
    ("hour", "Hour of day", b_hour, False),
]


def _sortkey(label):
    m = re.search(r"\d+", label)
    return (0, int(m.group()), label) if m else (1, 0, label)


def binned(gens, fn, split_temp):
    out = {}
    for r in gens:
        b = fn(r)
        e = out.setdefault(b, {"n": 0, "k": 0, "T=0": [0, 0], "T>0": [0, 0]})
        e["n"] += 1
        e["k"] += bool(r["flags"])
        if split_temp:
            t = e[b_temp(r)]
            t[0] += 1
            t[1] += bool(r["flags"])
    return {b: out[b] for b in sorted(out, key=_sortkey)}


# ---------------------------------------------------------------- report

def excerpt(r, flag, width=160):
    f = r["flags"][flag]
    p = response_choices(r)
    ch = p["choices"][f.get("choice", 0)] if p and p["choices"] else None
    field = f.get("field")
    if ch is None:
        return "(response not stored)"
    if field == "all" or field not in ("content", "reasoning", "text"):
        text = ch["content"] or ch["reasoning"] or ch["text"]
        return repr(text[:width])
    text = ch[field]
    span = f.get("span") or [len(text), len(text)]
    a, b = span
    lo, hi = max(0, a - width), min(len(text), b + width // 2)
    s = ("..." if lo else "") + text[lo:a] + "⟦" + text[a:b] + "⟧" + text[b:hi] \
        + ("..." if hi < len(text) else "")
    return f"[{field}] " + s.replace("```", "'''")


def pick_examples(gens, n=20):
    flagged = sorted((r for r in gens if r["flags"]), key=lambda r: -r["t0"])
    by_flag = {}
    for r in flagged:
        by_flag.setdefault(next(iter(r["flags"])), []).append(r)
    out, seen = [], set()
    while len(out) < n and any(by_flag.values()):
        for f in list(by_flag):
            if by_flag[f] and len(out) < n:
                r = by_flag[f].pop(0)
                if r["rid"] not in seen:
                    seen.add(r["rid"])
                    out.append(r)
    return out


def split_stats(gens):
    st = [r for r in gens if r.get("utf8_split") is not None]
    with_split = [r for r in st if r["utf8_split"]]
    fffd = [r for r in gens if "replacement_char" in r["flags"]]
    return {
        "chunked_responses": len(st),
        "with_utf8_split": len(with_split),
        "utf8_split_boundaries": sum(r["utf8_split"] for r in st),
        "sse_events_split_across_chunks": sum(r.get("sse_split") or 0 for r in st),
        "fffd_flagged": len(fffd),
        "fffd_flagged_with_split": sum(1 for r in fffd if r.get("utf8_split")),
        "fffd_flagged_without_split": sum(1 for r in fffd if not r.get("utf8_split")),
    }


def summarize(gens):
    flags = {f: sum(1 for r in gens if f in r["flags"]) for f in classify.FLAG_NAMES}
    by_temp = {}
    for r in gens:
        e = by_temp.setdefault(b_temp(r), {f: 0 for f in classify.FLAG_NAMES})
        e["_n"] = e.get("_n", 0) + 1
        for f in r["flags"]:
            e[f] += 1
    return {"n": len(gens), "flagged": sum(1 for r in gens if r["flags"]),
            "flags": flags, "flags_by_temp": by_temp}


def table_rows(bins, split):
    rows = []
    for b, e in bins.items():
        row = f"| {b} | {e['n']} | {e['k']} | {rate_str(e['k'], e['n'])} |"
        if split:
            for t in ("T=0", "T>0"):
                n, k = e[t]
                row += f" {k}/{n} = {pct(k / n) if n else 'n/a'} |"
        rows.append(row)
    return rows


def write_report(recs, gens, out, args, events, status):
    s = summarize(gens)
    lines = ["# Production corruption report", ""]
    t0 = min((r["t0"] for r in recs), default=None)
    t1 = max((r["t0"] for r in recs), default=None)
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) if t else "-"  # noqa: E731
    lines += [f"Window: {fmt(t0)} .. {fmt(t1)}; {len(recs)} requests recorded, "
              f"{s['n']} completed generations classified.", ""]
    lines += [f"**Corrupted (any flag): {s['flagged']}/{s['n']} = {rate_str(s['flagged'], s['n'])}** "
              "(95% Wilson interval)", ""]
    lines += ["## Flags", "", "| flag | count | rate [95% CI] | T=0 | T>0 |", "|---|---|---|---|---|"]
    bt = s["flags_by_temp"]
    for f, k in s["flags"].items():
        cells = []
        for t in ("T=0", "T>0"):
            e = bt.get(t)
            cells.append(f"{e[f]}/{e['_n']}" if e else "0/0")
        lines.append(f"| {f} | {k} | {rate_str(k, s['n'])} | {cells[0]} | {cells[1]} |")
    lines.append("")
    report = {"summary": s, "bins": {}, "split": split_stats(gens), "examples": []}
    for key, title, fn, split in BINS:
        bins = binned(gens, fn, split)
        report["bins"][key] = bins
        lines += [f"## {title}", ""]
        hdr = "| bin | n | flagged | rate [95% CI] |" + (" T=0 | T>0 |" if split else "")
        lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        lines += table_rows(bins, split)
        lines.append("")
    sp = report["split"]
    lines += ["## UTF-8 chunk splits (transport vs server)", "",
              f"- Chunked responses: {sp['chunked_responses']}; with >= 1 HTTP chunk boundary inside a "
              f"multi-byte UTF-8 character: {sp['with_utf8_split']} ({sp['utf8_split_boundaries']} boundaries). "
              f"SSE events spanning two chunks: {sp['sse_events_split_across_chunks']}.",
              f"- U+FFFD-flagged responses: {sp['fffd_flagged']}; of those with a split boundary: "
              f"{sp['fffd_flagged_with_split']}, without: {sp['fffd_flagged_without_split']}.",
              "- The recorder reassembles bytes before decoding, so a U+FFFD in the recorded text came from "
              "the server. A split boundary only produces U+FFFD in clients that decode chunk by chunk "
              "without an incremental decoder.", ""]
    ex = pick_examples(gens)
    lines += [f"## Flagged examples ({len(ex)} of {s['flagged']}, newest first per flag)", ""]
    for r in ex:
        c = r.get("cond") or {}
        ri = r["req_info"]
        lines.append(f"### {fmt(r['t0'])} rid={r['rid']} flags={','.join(r['flags'])}")
        lines.append(f"T={ri.get('temperature')} top_p={ri.get('top_p')} top_k={ri.get('top_k')} "
                     f"k={c.get('k_mode')} k_set={c.get('k_set')} running={c.get('running_min')}-"
                     f"{c.get('running_max')} prompt_tokens={r.get('prompt_tokens')} "
                     f"completion_tokens={r.get('completion_tokens')} max_tokens={ri.get('max_tokens')} "
                     f"finish={r.get('finish_reasons')} stream={ri.get('stream')} "
                     f"plp_inflight={c.get('other_prompt_logprobs_inflight')} up={b_upstream(r)}")
        for f in r["flags"]:
            lines += ["```", f"{f}: {excerpt(r, f)}", "```"]
        report["examples"].append({"rid": r["rid"], "t0": r["t0"], "flags": r["flags"],
                                   "excerpts": {f: excerpt(r, f) for f in r["flags"]}})
        lines.append("")
    other = [r for r in recs if not is_generation(r)]
    lines += ["## Recorder health", "",
              f"- Not classified: {len(other)} (other paths {sum(r.get('path') not in GEN_PATHS for r in other)}, "
              f"client aborts {sum(bool(r.get('client_aborted')) for r in other)}, "
              f"errors/non-200 {sum(bool(r.get('error')) or r.get('status') != 200 for r in other)}).",
              f"- status.json: {json.dumps({k: status.get(k) for k in ('recorded', 'record_errors', 'proxy_errors', 'upstream_errors', 'sampler_errors')}) if status else 'n/a'}",
              "- Upstream configs seen:"]
    for e in events:
        if e.get("kind") == "upstream":
            lines.append(f"  - {fmt(e['ts'])} ({e.get('why')}): pid={e.get('pid')} async={e.get('async')} "
                         f"spec={json.dumps(e.get('spec'))} label={e.get('label')}")
    lines.append("")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (Path(out) / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


# --------------------------------------------------------------- compare

def window_filter(spec):
    terms = [t for t in spec.split(",") if t]
    tests = []
    for t in terms:
        if ".." in t:
            a, b = t.split("..", 1)
            lo, hi = parse_time(a) if a else 0, parse_time(b) if b else float("inf")
            tests.append(lambda r, lo=lo, hi=hi: lo <= r["t0"] <= hi)
        elif "=" in t:
            k, v = t.split("=", 1)
            tests.append(lambda r, k=k, v=v: str((r.get("up") or {}).get(k, r.get(k))) == v)
        else:
            raise SystemExit(f"bad window term {t!r} (want FROM..TO or key=value)")
    return lambda r: all(f(r) for f in tests)


def write_compare(gens, a, b, out):
    A = [r for r in gens if window_filter(a)(r)]
    B = [r for r in gens if window_filter(b)(r)]
    lines = ["# Corruption: window comparison", "", f"- A = `{a}`: {len(A)} generations",
             f"- B = `{b}`: {len(B)} generations", "",
             "Difference is B - A with a 95% Newcombe interval; * = interval excludes 0.", ""]
    res = {"A": a, "B": b, "rows": []}

    def section(title, rows):
        lines.extend([f"## {title}", "", "| | A | B | B - A [95% CI] |", "|---|---|---|---|"])
        for name, sa, sb in rows:
            ka, na = sum(1 for r in sa if r["flags"]), len(sa)
            kb, nb = sum(1 for r in sb if r["flags"]), len(sb)
            if name.startswith("flag:"):
                f = name[5:]
                ka, kb = sum(1 for r in sa if f in r["flags"]), sum(1 for r in sb if f in r["flags"])
            if na and nb:
                d, lo, hi = newcombe(ka, na, kb, nb)
                dd = f"{pct(d)} [{pct(lo)}, {pct(hi)}]{' *' if lo > 0 or hi < 0 else ''}"
            else:
                d = lo = hi = None
                dd = "n/a"
            lines.append(f"| {name} | {ka}/{na} = {rate_str(ka, na)} | {kb}/{nb} = {rate_str(kb, nb)} | {dd} |")
            res["rows"].append({"section": title, "row": name, "A": [ka, na], "B": [kb, nb],
                                "diff": d, "ci": [lo, hi]})
        lines.append("")

    section("Overall", [("any flag", A, B)] + [(f"flag:{f}", A, B) for f in classify.FLAG_NAMES])
    for key, title, fn, _ in BINS[:4]:
        labels = sorted({fn(r) for r in A + B}, key=_sortkey)
        section(title.split(" x ")[0], [(lab, [r for r in A if fn(r) == lab], [r for r in B if fn(r) == lab])
                                        for lab in labels])
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "COMPARE.md").write_text("\n".join(lines), encoding="utf-8")
    (Path(out) / "compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--out", default=None, help="output dir (default: --data)")
    ap.add_argument("--since", default=None, help="e.g. 1h, 30m, 2026-10-01T09:00")
    ap.add_argument("--until", default=None)
    ap.add_argument("--reclassify", action="store_true", help="recompute flags from the stored responses")
    ap.add_argument("--allow-scripts", default="greek")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    out = a.out or a.data
    recs = load(a.data, parse_time(a.since) if a.since else None, parse_time(a.until) if a.until else None)
    gens = [r for r in recs if is_generation(r)]
    if a.reclassify:
        allow = frozenset(x for x in a.allow_scripts.split(",") if x)
        for r in gens:
            r["flags"] = reclassify(r, allow)
    if a.compare:
        res = write_compare(gens, *a.compare, out)
        for row in res["rows"][:1 + len(classify.FLAG_NAMES)]:
            print(f"{row['row']:32s} A {row['A'][0]}/{row['A'][1]}  B {row['B'][0]}/{row['B'][1]}")
        print(f"wrote {Path(out) / 'COMPARE.md'}")
        return
    events = []
    ev = Path(a.data) / "events.jsonl"
    if ev.exists():
        events = [json.loads(x) for x in ev.read_text().splitlines() if x.strip()]
    st = {}
    sp = Path(a.data) / "status.json"
    if sp.exists():
        try:
            st = json.loads(sp.read_text())
        except ValueError:
            pass
    rep = write_report(recs, gens, out, a, events, st)
    s = rep["summary"]
    print(f"{s['flagged']}/{s['n']} generations flagged = {rate_str(s['flagged'], s['n'])}")
    for f, k in s["flags"].items():
        if k:
            print(f"  {f}: {k}")
    print(f"wrote {Path(out) / 'REPORT.md'}")


if __name__ == "__main__":
    main()
