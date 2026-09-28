"""Clock/power medians inside each pass-2 T=default decode cohort window (log mtime minus benchmark
duration). Run from cloud/results/phase1b/runs after a pull that preserved mtimes (rsync -a)."""
import datetime as dt, os, re, statistics
for k in ("gsq", "baseline"):
    rows = []
    for l in open(f"p1b-speed-{k}/clocks-decode-pass2.csv"):
        p = l.strip().split(", ")
        if len(p) == 5:
            t = dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            rows.append((t, float(p[1].split()[0]), float(p[3].split()[0]), float(p[4].split()[0])))
    for c in ("c1", "c2"):
        f = f"p1b-speed-{k}/prod-pass2/cohort_Tdefault_{c}.log"
        end = os.path.getmtime(f); dur = float(re.search(r"Benchmark duration \(s\):\s+([\d.]+)", open(f).read()).group(1))
        sel = [r for r in rows if end - dur <= r[0] <= end]
        med = lambda i: statistics.median(r[i] for r in sel)
        print(f"{k} T=default {c}: {len(sel)} samples, SM {med(1):.0f} MHz, {med(2):.1f} W, util {med(3):.0f}%, "
              f"{sum(r[2] >= 345 for r in sel)} samples >= 345 W")
