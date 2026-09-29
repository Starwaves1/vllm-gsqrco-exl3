"""GPU idle per decode step attributed to the host op running at the time: for each gap between
GPU activities (kernels, memcpy, memset) inside the 5 complete steps, the innermost-first list
of cpu_op / user_annotation events covering the gap's midpoint; gap time is summed per the
outermost vllm/_C_gguf/aten op name (and per top-level annotation).
usage: gaps.py TRACE.json [min_gap_us]"""
import bisect, collections, json, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
mn = float(sys.argv[2]) if len(sys.argv) > 2 else 5
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
t0, t1, n = steps[0]["ts"], steps[-1]["ts"], len(steps) - 1
gpu = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and t0 <= e["ts"] < t1)
cpu = sorted(((e["ts"], e["ts"] + e.get("dur", 0), e["name"]) for e in ev if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function") and t0 - 1e5 <= e["ts"] < t1), key=lambda c: c[0])
starts = [c[0] for c in cpu]
by_op, by_top, total = collections.Counter(), collections.Counter(), 0.0
end = gpu[0][1]
for s, e in gpu[1:]:
    if s - end >= mn:
        mid = (s + end) / 2
        i = bisect.bisect_right(starts, mid)
        cover = [c for c in cpu[max(0, i - 4000):i] if c[1] >= mid]  # events containing mid, outermost first
        names = [c[2] for c in cover if not c[2].startswith("execute_context")]
        pick = next((x for x in names if x.startswith(("vllm::", "_C_gguf::", "_C::", "_C_cache_ops::"))), None) or (names[0] if names else "(none)")
        by_op[pick[:90]] += s - end
        by_top[(names[0] if names else "(none)")[:90]] += s - end
        total += s - end
    end = max(end, e)
print(f"{n} steps; GPU idle in gaps >= {mn} us: {total / n / 1e3:.2f} ms/step")
print("by first vllm/_C op covering the gap (ms/step):")
for k, v in by_op.most_common(25):
    print(f"  {v / n / 1e3:7.3f}  {k}")
print("by outermost host event (ms/step):")
for k, v in by_top.most_common(15):
    print(f"  {v / n / 1e3:7.3f}  {k}")
