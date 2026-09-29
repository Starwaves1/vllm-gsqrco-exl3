"""pstep.py plus a per-kernel-family, per-ggml-type GEMM breakdown (for the c=4/c=8 MMQ path).
usage: pstep2.py TRACE.json"""
import collections, json, re, runpy, sys
runpy.run_path(sys.argv[0].replace("pstep2.py", "pstep.py"), run_name="__main__")
TYPES = {10: "Q2_K", 12: "Q4_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 21: "IQ3_S",
         22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M"}
ev = json.load(open(sys.argv[1]))["traceEvents"]
rt = {e["args"]["correlation"]: e["ts"] for e in ev
      if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
t0, t1, n = steps[0]["ts"], steps[-1]["ts"], len(steps) - 1
agg = collections.defaultdict(lambda: [0, 0.0])
for e in ev:
    if e.get("cat") != "kernel" or e.get("ph") != "X" or not (t0 <= rt.get(e["args"].get("correlation"), e["ts"]) < t1):
        continue
    m = e["name"]
    fam = ("iq3" if "iq3_mul_mat_vec" in m else "iq3_mma" if "iq3_mma" in m else "own" if "own_mul_mat_vec" in m
           else "mmvq" if "mul_mat_vec_q" in m else "mmq fixup" if "stream_k_fixup" in m
           else "mmq" if "mul_mat_q<" in m else "quantize mmq" if "quantize_mmq" in m else None)
    if fam is None:
        continue
    t = re.search(r"\(ggml_type\)(\d+)", m)
    agg[(fam, TYPES.get(int(t.group(1)), t.group(1)) if t else "-")][0] += 1
    agg[(fam, TYPES.get(int(t.group(1)), t.group(1)) if t else "-")][1] += e["dur"]
print("GEMM family / type per step (launches, ms):")
for (fam, typ), (c, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"  {fam:12} {typ:8} {c/n:6.1f} {d/n/1e3:7.3f}")
