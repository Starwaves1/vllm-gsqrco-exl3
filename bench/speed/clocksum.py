"""Median SM/mem clock, power, util over busy samples (util >= 50%) of each clocks-*.csv."""
import glob, statistics, sys
for f in sorted(sum((glob.glob(a) for a in sys.argv[1:]), [])):
    rows = [l.strip().split(", ") for l in open(f) if l.count(",") == 4]
    busy = [r for r in rows if float(r[4].split()[0]) >= 50]
    if not busy:
        continue
    med = lambda i: statistics.median(float(r[i].split()[0]) for r in busy)
    print(f"{f.split('/')[-2]}/{f.split('/')[-1]}: {len(busy)} s busy, median SM {med(1):.0f} MHz, mem {med(2):.0f} MHz, {med(3):.1f} W, util {med(4):.0f}%")
