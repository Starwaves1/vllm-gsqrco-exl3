"""Per-step GPU time by kernel family and quant type from a vLLM torch-profiler trace (K3, c=8 steps).
usage: k3step.py TRACE.json"""
import collections, json, re, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
rt = {e["args"]["correlation"]: e["ts"] for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
t0, t1, n = steps[0]["ts"], steps[-1]["ts"], len(steps) - 1
k = sorted((e for e in ev if e.get("cat") == "kernel" and t0 <= rt.get(e.get("args", {}).get("correlation"), e["ts"]) < t1), key=lambda e: e["ts"])
PATS = ((r"mma_k_fixup", "mma_k fixup"), (r"mma_k<\(ggml_type\)(\d+), (\d+)", "mma_k"), (r"mul_mat_q<\(ggml_type\)(\d+), (\d+)", "mmq"),
        (r"mul_mat_vec_q<\(ggml_type\)(\d+), (\d+)", "mmvq"), (r"iq3_mma<\(ggml_type\)(\d+)", "iq3 mma"),
        (r"iq3_mul_mat_vec<\(ggml_type\)(\d+), (\d+)", "iq3"), (r"own_mul_mat_vec<\(ggml_type\)(\d+), (\d+)", "own"),
        (r"stream_k_fixup", "mmq fixup"), (r"quantize", "quantize"))
def fam(m):
    for pat, name in PATS:
        mm = re.search(pat, m)
        if mm:
            return name + "".join(f" {g}" for g in mm.groups())
    return "other"
t, c = collections.Counter(), collections.Counter()
for e in k:
    f = fam(e["name"]); t[f] += e["dur"]; c[f] += 1
busy = sum(e["dur"] for e in k)
span = k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]
print(f"{sys.argv[1]}\n{n} steps; per step busy {busy/n/1e3:.2f} ms, span {span/n/1e3:.2f} ms")
for f in sorted(t, key=lambda f: -t[f]):
    print(f"  {f:22} {c[f]/n:7.1f} launches {t[f]/n/1e3:7.3f} ms")
