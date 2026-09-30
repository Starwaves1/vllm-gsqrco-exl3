#!/usr/bin/env python3
"""Model-matrix table from runs/<cfg>/{job.log,server.log,speed/summary.txt} (+ the cited rows).

Decode = pass 2 of production's run_benchmarks.sh single, T=0 cohorts, decode(C/meanTPOT) tok/s.
ms/step = C * 1000 / tok/s * tok/step (tok/step 1 without MTP: no drafts, one token per sequence
per engine step). GB = weight bytes the target model streams per step (layers + output head;
embedding gather, vision tower and MTP block excluded; inventory.txt). eff GB/s = GB / ms/step;
ms/GB = ms/step / GB. With MTP the step also runs k=3 draft passes (MTP block + draft lm_head),
so eff GB/s understates those rows' weight streaming rate.
usage: table.py RUNS_DIR
"""
import re, sys
from pathlib import Path

GB = {"swift": 11.353, "base": 11.353, "w4a16": 13.252, "official": 15.139}   # read/step, target
MTPGB = {"swift": 0.348, "base": 0.348, "w4a16": 0.327, "official": 0.849}
NAME = {"swift": "Swift GGUF", "base": "Base GGUF", "w4a16": "prod W4A16", "official": "Official INT4 stock"}
CITED = {  # measured earlier, same box / power limit
    "swift-mtp": dict(src="Integration 2 (4cbd091)", tps=[110.3, 192.7, 348.1, 541.3], ts=[3.08, 3.04, 3.11, 2.99],
                      pf={8192: 1248, 180000: 644}, kv=253906, vram="22551 MiB", acc="0.650", mml=200000),
    "w4a16-mtp": dict(src="phase 1b", tps=[94.1, 194.4, 345.1, 505.4], ts=[2.60, 2.65, 2.59, 2.61],
                      pf={8192: 1108, 180000: 603}, kv=207812, vram="n/r", acc="0.522 (T=default run)", mml=200000),
}
CS = [1, 2, 4, 8]


def parse(d: Path):
    s = (d / "speed/summary.txt").read_text()
    tps, ts, cur = {}, {}, None
    for line in s.splitlines():
        m = re.match(r"# production run_benchmarks.sh single, pass (\d)", line)
        if m:
            cur = int(m.group(1))
        m = re.match(r"ROW cohort C(\d) real prompts T=0 .*decode\(C/meanTPOT\)=([\d.]+) \| tok/step=([\d.]+|-)", line)
        if m and cur == 2:
            tps[int(m.group(1))] = float(m.group(2))
            ts[int(m.group(1))] = 1.0 if m.group(3) == "-" else float(m.group(3))
    pf = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"ROW prefill len=(\d+) conc=1 .*?\| (\d+) tok/s", s)}
    acc = re.search(r"acceptance rate ([\d.]+)", s)
    job = (d / "job.log").read_text() if (d / "job.log").exists() else ""
    srv = (d / "server.log").read_text(errors="replace")
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", srv)
    vram = re.search(r"VRAM after load: (\d+) MiB", job)
    mml = re.search(r"max_model_len=(\d+)", job)
    return dict(src="this run", tps=[tps.get(c) for c in CS], ts=[ts.get(c) for c in CS], pf=pf,
                kv=int(kv[-1].replace(",", "")) if kv else None, vram=f"{vram.group(1)} MiB" if vram else "?",
                acc=acc.group(1) if acc else "-", mml=int(mml.group(1)) if mml else None)


runs = Path(sys.argv[1])
rows = []
for model in ["swift", "base", "w4a16", "official"]:
    for spec in ["mtp", "nomtp"]:
        cfg = f"{model}-{spec}"
        if (runs / cfg / "speed/summary.txt").exists():
            r = parse(runs / cfg)
        elif cfg in CITED:
            r = CITED[cfg]
        else:
            continue
        rows.append((model, spec, r))

f = lambda x, p=1: "-" if x is None else f"{x:.{p}f}"
print(f"{'row':28s} {'decode tok/s c=1/2/4/8':30s} {'tok/step':22s} {'ms/step':26s} {'eff GB/s':24s} {'ms/GB':22s}")
for model, spec, r in rows:
    ms = [None if t is None else c * 1000 / t * s for c, t, s in zip(CS, r["tps"], r["ts"])]
    g = GB[model]
    print(f"{NAME[model] + ' ' + spec:28s} "
          f"{' / '.join(f(x) for x in r['tps']):30s} {' / '.join(f(x, 2) for x in r['ts']):22s} "
          f"{' / '.join(f(x) for x in ms):26s} {' / '.join(f(None if x is None else g / x * 1000, 0) for x in ms):24s} "
          f"{' / '.join(f(None if x is None else x / g, 2) for x in ms):22s}")
print()
print(f"{'row':28s} {'prefill 8k':>10s} {'180k':>6s} {'KV tokens':>10s} {'max len':>8s} {'VRAM':>10s} {'acceptance':>12s} {'GB/step (+MTP)':>16s}  source")
for model, spec, r in rows:
    print(f"{NAME[model] + ' ' + spec:28s} {r['pf'].get(8192, '-'):>10} {r['pf'].get(180000, '-'):>6} {r['kv'] or '-':>10} "
          f"{r['mml'] or '-':>8} {r['vram']:>10} {r['acc']:>12} {f'{GB[model]:.2f} (+{MTPGB[model]:.2f})':>16s}  {r['src']}")
print()
ref = {(m, s): r for m, s, r in rows}
def per_step(key):
    m, s = key
    r = ref.get(key)
    return None if r is None else [None if t is None else c * 1000 / t * x for c, t, x in zip(CS, r["tps"], r["ts"])]
for a, b in [(("swift", "nomtp"), ("w4a16", "nomtp")), (("base", "nomtp"), ("w4a16", "nomtp")),
             (("base", "nomtp"), ("swift", "nomtp")), (("base", "mtp"), ("swift", "mtp")),
             (("official", "nomtp"), ("w4a16", "nomtp")), (("swift", "nomtp"), ("official", "nomtp"))]:
    x, y = per_step(a), per_step(b)
    if x and y:
        print(f"ms/step {NAME[a[0]]} {a[1]} / {NAME[b[0]]} {b[1]}: " + " / ".join(
            "-" if p is None or q is None else f"{p / q:.3f}" for p, q in zip(x, y)) + "  (c=1/2/4/8; >1 = slower)")
