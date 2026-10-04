"""Per-step GPU summary of a vLLM torch-profiler trace of MTP decode steps on the EXL3 plugin
(integration-2's pstep.py windowing: complete steps only, from the first execute_context to the
last one's start; GPU activities attributed by their host launch time).
Per step: launches and ms per class, the top non-GEMM kernels, busy/span/idle.
usage: exl3_pstep.py TRACE.json"""
import collections
import json
import sys

ev = json.load(open(sys.argv[1]))["traceEvents"]
rt = {e["args"]["correlation"]: e["ts"] for e in ev
      if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")),
               key=lambda e: e["ts"])
t0, t1, n = steps[0]["ts"], steps[-1]["ts"], len(steps) - 1
gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"
              and t0 <= rt.get(e.get("args", {}).get("correlation"), e["ts"]) < t1), key=lambda e: e["ts"])


def cls(e):
    m = e["name"]
    low = m.lower()
    if e["cat"] != "kernel":
        return e["cat"]
    if "exl3_gemm_kernel" in m or "exl3_gemv" in m or "exl3_mgemm" in m:
        return "gemm exl3"
    if "trellis_exl3_marlin" in m or "Marlin<" in m:
        return "gemm mr"
    if "trellis_had_r_128" in m:
        return "had mr-in"
    if "had_hf_r_128" in m or "had_ff_r_128" in m:
        return "had dequant-route"
    if "reconstruct" in m:
        return "dequant"
    if "hgemm" in low or "cutlass" in m or "xmma" in m or "ampere_" in m or "sm80_" in m or "cublas" in low:
        return "gemm dense"
    if "CatArrayBatchedCopy" in m:
        return "cat parts"
    if "direct_copy" in m or ("copy" in low and "elementwise" in low):
        return "cast/copy"
    if "flash" in low or "fmha" in low or "attention" in low or "paged" in low:
        return "attention"
    if "gated_delta" in low or "recurrent" in low or "conv1d" in low or "chunk_" in low:
        return "gdn"
    if "sampl" in low or "topk" in low or "argmax" in low or "softmax" in low:
        return "sampling"
    return "other"


tot, cnt = collections.Counter(), collections.Counter()
names = collections.defaultdict(lambda: [0, 0.0])
for e in gpu:
    c = cls(e)
    tot[c] += e["dur"]
    cnt[c] += 1
    if not c.startswith("gemm"):
        names[e["name"][:110]][0] += 1
        names[e["name"][:110]][1] += e["dur"]
busy = sum(e["dur"] for e in gpu)
span = gpu[-1]["ts"] + gpu[-1]["dur"] - gpu[0]["ts"]
print(f"{sys.argv[1]}\n{n} complete steps; per step: {len(gpu) / n:.0f} GPU activities, busy {busy / n / 1e3:.2f} ms, "
      f"span {span / n / 1e3:.2f} ms (idle {(span - busy) / n / 1e3:.2f})")
for c in sorted(tot, key=lambda c: -tot[c]):
    print(f"  {c:18} {cnt[c] / n:7.1f} launches {tot[c] / n / 1e3:7.3f} ms")
print("top non-GEMM kernels per step (launches, ms):")
for m, (k, d) in sorted(names.items(), key=lambda kv: -kv[1][1])[:15]:
    print(f"  {k / n:6.1f} {d / n / 1e3:7.3f}  {m}")
