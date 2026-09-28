"""Which kernels run inside CUDA graphs vs eager launches, split into the target forward
(kernels whose host launch falls inside the execute_context annotation) and the rest of the step (draft, sampling, bookkeeping).
usage: p3launch.py TRACE.json"""
import collections, json, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
rt = {e["args"]["correlation"]: (e["name"], e["ts"]) for e in ev
      if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"), key=lambda e: e["ts"])
t0 = steps[0]["ts"]; t1 = steps[-1]["ts"] + steps[-1]["dur"]
inside = lambda t: any(s["ts"] <= t < s["ts"] + s["dur"] for s in steps)
def cls(n):
    if "mul_mat_vec_q" in n or "mul_mat_q<" in n or "stream_k_fixup" in n: return "gemm_lcpp"
    if "quantize" in n and "q8_1" in n: return "q8_1 quantize"
    if "Marlin" in n or "marlin" in n: return "gemm_marlin"
    if "gemm" in n.lower() or "cutlass" in n or "xmma" in n or "ampere" in n: return "gemm_cublas"
    return "other"
agg = collections.defaultdict(lambda: [0, 0.0])
first_last = collections.defaultdict(list)
for e in gpu:
    if not (t0 <= e["ts"] < t1): continue
    api, launched = rt.get(e.get("args", {}).get("correlation"), ("?", e["ts"]))
    region = "target" if inside(launched) else "outside"  # by host launch time; the GPU lags
    k = (region, api, cls(e["name"]))
    agg[k][0] += 1; agg[k][1] += e["dur"]
n = len(steps)
print(f"{n} steps; per step: region | launched by | class | count | GPU ms")
for k in sorted(agg, key=lambda k: -agg[k][1]):
    print(f"  {k[0]:8} {k[1]:22} {k[2]:14} {agg[k][0]/n:7.1f} {agg[k][1]/n/1e3:8.3f}")
for r in ("target", "outside"):
    print(f"{r}: GPU busy per step {sum(v[1] for k, v in agg.items() if k[0] == r) / n / 1e3:.2f} ms")
