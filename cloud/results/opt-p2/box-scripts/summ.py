"""Decode table from bench/speed/run.sh summaries: pass 2 T=0 decode(C/meanTPOT) tok/s per
cohort, [pass 1], tok/step, ms/step = C * 1000 / tok/s * tok/step, MTP acceptance.
usage: summ.py RUN_DIR... (each with summary.txt)"""
import re, sys
PROD = {1: 27.6, 2: 27.3, 4: 30.0, 8: 41.3}  # prod W4A16 ms/step
for d in sys.argv[1:]:
    rows, cur = {}, None
    for line in open(f"{d}/summary.txt"):
        m = re.match(r"# production run_benchmarks.sh single, pass (\d)", line)
        if m:
            cur = int(m.group(1))
        m = re.match(r"ROW cohort C(\d) real prompts T=0 .*decode\(C/meanTPOT\)=([\d.]+) \| tok/step=([\d.]+)", line)
        if m and cur:
            rows[(cur, int(m.group(1)))] = (float(m.group(2)), float(m.group(3)))
    acc = re.search(r"acceptance rate ([\d.]+)", open(f"{d}/summary.txt").read())
    out = []
    for c in (1, 2, 4, 8):
        t2, s2 = rows.get((2, c), (float("nan"),) * 2)
        t1 = rows.get((1, c), (float("nan"),))[0]
        out.append(f"c={c} {t2:6.1f} ({t1:6.1f}) {s2:.2f} tok/step {c * 1000 / t2 * s2:5.1f} ms")
    print(f"{d.rstrip('/').split('/')[-1]:22} " + " | ".join(out) + f" | acc {acc.group(1) if acc else '-'}")
