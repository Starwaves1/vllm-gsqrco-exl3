import re, sys, collections
best = collections.defaultdict(float); full = collections.defaultdict(float); bestname = {}
route = collections.defaultdict(float)
for line in open(sys.argv[1]):
    m = re.match(r"(\w+) n=(\d+) torch\.(\w+) mmq=(\w+): (.*)", line.strip())
    if m:
        name, n, dt, mmq, rest = m.groups()
        errs = dict((k, float(v)) for k, v in re.findall(r"(\w+)=([\d.e+-]+)", rest))
        key = (name, "mmq" if mmq == "True" else "mmvq", dt)
        b = min(v for k, v in errs.items() if k != "full"); bn = min((v, k) for k, v in errs.items() if k != "full")[1]
        if b >= best[key]: best[key] = b; bestname[key] = bn
        full[key] = max(full[key], errs["full"])
        continue
    m = re.match(r"(\w+) rows=(\d+) n=(\d+): rel err vs full ([\d.e+-]+)", line.strip())
    if m:
        route[m.group(1)] = max(route[m.group(1)], float(m.group(4)))
print(f"{'type':8} {'op':5} {'dtype':9} {'max best-model err':>19} {'(model)':8} {'max err vs full':>16}")
for k in sorted(best):
    print(f"{k[0]:8} {k[1]:5} {k[2]:9} {best[k]:19.2e} {bestname[k]:8} {full[k]:16.2e}")
print("\nrouting (_fused_mul_mat_gguf, whole tensor, bf16) max rel err vs full:")
for k in sorted(route): print(f"  {k:8} {route[k]:.2e}")
