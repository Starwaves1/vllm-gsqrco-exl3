"""Per-kernel difference per decode step between two torch-profiler traces (5 complete steps
each): launches and ms, A -> B, sorted by |delta ms|. usage: kdiff.py A.json B.json [n]"""
import collections, json, sys


def per_step(path):
    ev = json.load(open(path))["traceEvents"]
    rt = {e["args"]["correlation"]: e["ts"] for e in ev
          if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    st = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
    t0, t1, n = st[0]["ts"], st[-1]["ts"], len(st) - 1
    agg = collections.defaultdict(lambda: [0.0, 0.0])
    for e in ev:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X" and t0 <= rt.get(e["args"].get("correlation"), e["ts"]) < t1:
            a = agg[e["name"][:110]]; a[0] += 1 / n; a[1] += e["dur"] / n / 1e3
    return agg


A, B = per_step(sys.argv[1]), per_step(sys.argv[2])
rows = [(k, A.get(k, [0, 0]), B.get(k, [0, 0])) for k in set(A) | set(B)]
rows.sort(key=lambda r: -abs(r[2][1] - r[1][1]))
print(f"total ms/step {sum(v[1] for v in A.values()):.3f} -> {sum(v[1] for v in B.values()):.3f}; launches {sum(v[0] for v in A.values()):.0f} -> {sum(v[0] for v in B.values()):.0f}")
for k, a, b in rows[:int(sys.argv[3]) if len(sys.argv) > 3 else 20]:
    print(f"{a[0]:6.0f} -> {b[0]:6.0f}  {a[1]:7.3f} -> {b[1]:7.3f} ({b[1] - a[1]:+.3f})  {k}")
