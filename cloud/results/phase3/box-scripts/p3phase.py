"""Split each c=1 MTP decode step of a vLLM torch-profiler trace into phases and report GPU busy
vs idle per phase (ms per step, averaged over the complete steps):
  target: from the first to the last GPU activity launched inside execute_context (the target
          forward, piecewise CUDA graphs + eager attention/GDN between pieces);
  rest:   from there to the first GPU activity of the next step (target lm_head/sampling,
          rejection, the 3 MTP draft passes, scheduling, input prep).
usage: p3phase.py TRACE.json"""
import json, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
launch = {e["args"]["correlation"]: e["ts"] for e in ev
          if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"), key=lambda e: e["ts"])
def busy(a, b):
    tot, cur = 0.0, a
    for e in gpu:
        s, f = max(e["ts"], cur), min(e["ts"] + e["dur"], b)
        if f > s: tot += f - s; cur = f
    return tot
rows = []
for s, nxt in zip(steps, steps[1:]):
    t = [e for e in gpu if s["ts"] <= launch.get(e.get("args", {}).get("correlation"), -1) < s["ts"] + s["dur"]]
    t0, t1 = t[0]["ts"], max(e["ts"] + e["dur"] for e in t)
    n0 = min(e["ts"] for e in gpu if launch.get(e.get("args", {}).get("correlation"), -1) >= nxt["ts"])
    rows.append((t1 - t0, busy(t0, t1), n0 - t1, busy(t1, n0)))
n = len(rows)
avg = [sum(r[i] for r in rows) / n / 1e3 for i in range(4)]
print(f"{n} complete steps, ms per step:")
print(f"  target: wall {avg[0]:6.2f}  busy {avg[1]:6.2f}  idle {avg[0]-avg[1]:5.2f}")
print(f"  rest:   wall {avg[2]:6.2f}  busy {avg[3]:6.2f}  idle {avg[2]-avg[3]:5.2f}")
print(f"  step:   wall {avg[0]+avg[2]:6.2f}  busy {avg[1]+avg[3]:6.2f}  idle {avg[0]+avg[2]-avg[1]-avg[3]:5.2f}")
# idle inside CUDA graph replays: gaps between consecutive kernels of the same cudaGraphLaunch
# (kernels of one graph launch share its correlation id); the GPU's per-node cost, not the host's
w0, w1 = steps[0]["ts"], steps[-1]["ts"]
g = [e for e in gpu if w0 <= launch.get(e.get("args", {}).get("correlation"), -1) < w1]
graph = {e["args"]["correlation"] for e in ev if e.get("cat") == "cuda_runtime" and e["name"] == "cudaGraphLaunch"}
ig = [(b["ts"] - a["ts"] - a["dur"]) for a, b in zip(g, g[1:])
      if a["args"].get("correlation") == b["args"].get("correlation") in graph]
print(f"  in-graph gaps: {sum(max(0, x) for x in ig) / n / 1e3:.2f} ms over {len(ig) / n:.0f} kernel boundaries per step")
