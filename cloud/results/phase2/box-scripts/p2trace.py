"""Summarise a vLLM torch-profiler trace of c=1 MTP decode steps: kernels per step, GEMM vs
shim overhead (casts, q8_1 quantize, memsets) vs everything else, and the lm_head reads."""
import collections, glob, json, sys
path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("/workspace/runs/p2-profile/trace/**/*.json", recursive=True))[-1]
ev = json.load(open(path))["traceEvents"]
k = sorted((e for e in ev if e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy") and e.get("ph") == "X"), key=lambda e: e["ts"])
def cls(e):
    n = e["name"]
    if e["cat"] == "gpu_memset": return "memset"
    if e["cat"] == "gpu_memcpy": return "memcpy"
    if "mul_mat_vec_q" in n or "mul_mat_q<" in n or "mul_mat_q_stream_k_fixup" in n: return "gemm_lcpp"
    if "quantize_q8_1" in n or "quantize_mmq_q8_1" in n: return "shim_quantize"
    if "direct_copy" in n or ("copy" in n.lower() and "elementwise" in n): return "copy/cast"
    if "gemm" in n.lower() or "cutlass" in n or "sm80_xmma" in n or "ampere" in n: return "gemm_cublas"
    if "dequantize" in n: return "dequant"
    return "other"
tot = collections.Counter(); cnt = collections.Counter(); names = collections.Counter()
for e in k:
    c = cls(e); tot[c] += e["dur"]; cnt[c] += 1
    if c == "other": names[e["name"][:90]] += e["dur"]
# shim casts: a copy right before a q8_1 quantize (X -> fp32) or right after an lcpp GEMM (Y -> 16-bit)
shim_cast = 0.0; nshim = 0
for i, e in enumerate(k):
    if cls(e) != "copy/cast": continue
    nxt = cls(k[i + 1]) if i + 1 < len(k) else ""
    prv = cls(k[i - 1]) if i else ""
    if nxt == "shim_quantize" or prv == "gemm_lcpp":
        shim_cast += e["dur"]; nshim += 1
# lm_head: lcpp MMVQ launches with a grid.x beyond any 17408-row layer
lm = [e for e in k if "mul_mat_vec_q" in e["name"] and (e.get("args", {}).get("grid") or [0])[0] > 40000]
span = (k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]) if k else 0
busy = sum(e["dur"] for e in k)
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 6  # --profiler-config max_iterations
print(f"trace {path}\n{len(k)} GPU activities over {span/1e3:.1f} ms wall, {busy/1e3:.1f} ms busy; per step (/{nsteps}):")
for c in sorted(tot, key=lambda c: -tot[c]):
    print(f"  {c:14} {cnt[c]/nsteps:7.0f} launches  {tot[c]/nsteps/1e3:7.3f} ms")
print(f"  shim casts (copy next to quantize/lcpp gemm): {nshim/nsteps:.0f} launches {shim_cast/nsteps/1e3:.3f} ms")
print(f"  lm_head MMVQ launches: {len(lm)/nsteps:.1f} per step, {sum(e['dur'] for e in lm)/nsteps/1e3:.3f} ms; "
      f"durations us {sorted(round(e['dur']) for e in lm)[:12]}")
print(f"  idle (wall - busy): {(span-busy)/nsteps/1e3:.3f} ms")
print("top 'other' kernels (ms per step):")
for n, d in names.most_common(15): print(f"  {d/nsteps/1e3:7.3f}  {n}")
