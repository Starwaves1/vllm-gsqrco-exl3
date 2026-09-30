"""(integration-2: + mma_k and the packed IQ3 kernels; integration-1: + iq3_mma and own_mul_mat_vec) Per-step GPU summary of a vLLM torch-profiler trace of c=1 MTP decode steps (phase 3's
p3trace.py + p3launch.py windowing): complete steps only (first execute_context start to the
last one's start), GPU activities attributed by their host launch time.
Prints per step: launches and ms per class, the top non-GEMM kernels, idle.
usage: pstep.py TRACE.json"""
import collections, json, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
rt = {e["args"]["correlation"]: e["ts"] for e in ev
      if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
t0, t1, n = steps[0]["ts"], steps[-1]["ts"], len(steps) - 1
gpu = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"
              and t0 <= rt.get(e.get("args", {}).get("correlation"), e["ts"]) < t1), key=lambda e: e["ts"])
def cls(e):
    m = e["name"]
    if e["cat"] != "kernel": return e["cat"]
    if "iq3_mul_mat_vec" in m: return "gemm iq3"
    if "iq3_packed" in m: return "gemm iq3_tiled"
    if "iq3_mma_packed" in m: return "gemm iq3_packed"
    if "iq3_mma" in m: return "gemm iq3_mma"
    if "mma_k" in m: return "gemm mma_k"
    if "own_mul_mat_vec" in m: return "gemm own"
    if "mul_mat_vec_q" in m: return "gemm mmvq"
    if "mul_mat_q<" in m or "stream_k_fixup" in m: return "gemm mmq"
    if "Marlin" in m or "marlin" in m or "gemm" in m.lower() or "cutlass" in m or "xmma" in m or "ampere" in m: return "gemm other"
    if "quantize" in m and "q8_1" in m: return "q8_1 quantize"
    if "CatArrayBatchedCopy" in m: return "cat"
    if "direct_copy" in m or ("copy" in m.lower() and "elementwise" in m): return "copy/cast"
    return "other"
tot, cnt, names = collections.Counter(), collections.Counter(), collections.defaultdict(lambda: [0, 0.0])
for e in gpu:
    c = cls(e); tot[c] += e["dur"]; cnt[c] += 1
    if not c.startswith("gemm"):
        names[e["name"][:110]][0] += 1; names[e["name"][:110]][1] += e["dur"]
busy = sum(e["dur"] for e in gpu)
span = gpu[-1]["ts"] + gpu[-1]["dur"] - gpu[0]["ts"]
print(f"{sys.argv[1]}\n{n} complete steps; per step: {len(gpu)/n:.0f} GPU activities, busy {busy/n/1e3:.2f} ms, "
      f"span {span/n/1e3:.2f} ms (idle {(span-busy)/n/1e3:.2f})")
for c in sorted(tot, key=lambda c: -tot[c]):
    print(f"  {c:14} {cnt[c]/n:7.1f} launches {tot[c]/n/1e3:7.3f} ms")
print("top non-GEMM kernels per step (launches, ms):")
for m, (k, d) in sorted(names.items(), key=lambda kv: -kv[1][1])[:15]:
    print(f"  {k/n:6.1f} {d/n/1e3:7.3f}  {m}")
