"""ms/step per cohort from bench/speed/run.sh summaries, one run dir per EXL3_MR mode: pass 2,
T=0 cohorts, decode(C/meanTPOT) tok/s and tok/step; ms/step = C * 1000 / (tok/s) * tok/step
(GPU time of one target+draft step, the GSQ metric); pass-1 tok/s in brackets; delta vs the
first dir (EXL3_MR=0). Also MTP acceptance (it should not move: the drafts are the same model).
usage: ladder_summ.py RUN_DIR... (each with summary.txt)"""
import re
import sys

rows = {}
for d in sys.argv[1:]:
    name = d.rstrip("/").split("/")[-1].replace("06-ladder", "mr0(06)")
    try:
        text = open(f"{d}/summary.txt").read()
    except OSError:
        continue
    cur, r = None, {}
    for line in text.splitlines():
        m = re.match(r"# production run_benchmarks.sh single, pass (\d)", line)
        if m:
            cur = int(m.group(1))
        m = re.match(r"ROW cohort C(\d) real prompts T=0 .*decode\(C/meanTPOT\)=([\d.]+) \| tok/step=([\d.]+)", line)
        if m and cur:
            r[(cur, int(m.group(1)))] = (float(m.group(2)), float(m.group(3)))
    acc = re.search(r"acceptance rate ([\d.]+)", text)
    rows[name] = (r, acc.group(1) if acc else "-")
base = None
for name, (r, acc) in rows.items():
    cells, ms = [], {}
    for c in (1, 2, 4, 8):
        t2, s2 = r.get((2, c), (float("nan"), float("nan")))
        t1 = r.get((1, c), (float("nan"),))[0]
        ms[c] = c * 1000 / t2 * s2 if t2 == t2 and t2 else float("nan")
        delta = f" {ms[c] - base[c]:+5.2f}" if base else ""
        cells.append(f"c={c} {t2:6.1f} tok/s ({t1:6.1f}) {s2:.2f} tok/step {ms[c]:5.2f} ms{delta}")
    base = base or ms
    print(f"{name:8} " + " | ".join(cells) + f" | acc {acc}")
