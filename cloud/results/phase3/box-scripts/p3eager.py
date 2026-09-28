"""Eagerly launched (cudaLaunchKernel, i.e. outside CUDA graphs) GEMM kernels in a vLLM
torch-profiler trace: count, mean us, name, grid. These are the lm_head calls (target and draft).
usage: p3eager.py TRACE.json"""
import collections, json, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
rt = {e["args"]["correlation"]: e["name"] for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
c = collections.defaultdict(list)
for e in ev:
    if e.get("cat") == "kernel" and rt.get(e["args"].get("correlation")) == "cudaLaunchKernel" and ("mul_mat" in e["name"] or "arlin" in e["name"]):
        c[(e["name"].split("(")[0][-70:], str(e["args"].get("grid")))].append(e["dur"])
for k, v in sorted(c.items(), key=lambda kv: -sum(kv[1])):
    print(len(v), round(sum(v) / len(v)), k)
