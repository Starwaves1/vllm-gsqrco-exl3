"""Target-pass GEMM kernel time per quant type (ggml_type id) and MMVQ/MMQ, ms per complete
step, from a vLLM torch-profiler trace (item 5: plus the shim's iq3_mul_mat_vec). usage: p3pertype.py TRACE.json"""
import json, re, collections, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
steps = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
launch = {e["args"]["correlation"]: e["ts"] for e in ev if e.get("cat") in ("cuda_runtime","cuda_driver") and "correlation" in e.get("args", {})}
inside = lambda t: any(s["ts"] <= t < s["ts"] + s["dur"] for s in steps[:-1])
c = collections.Counter(); n = collections.Counter(); names = {}
for e in ev:
    if e.get("cat") != "kernel": continue
    nm = e["name"]
    if not ("mul_mat_vec_q<" in nm or "mul_mat_q<" in nm or "iq3_mul_mat_vec<" in nm): continue
    if not inside(launch.get(e["args"].get("correlation"), -1)): continue
    m = re.search(r"<\(ggml_type\)(\d+)", nm)
    k = (m.group(1) if m else nm[:40], "iq3" if nm.startswith("void iq3_") else "mmvq" if "vec" in nm else "mmq")
    c[k] += e["dur"]; n[k] += 1; names[k] = nm[:160]
S = len(steps) - 1
for k, v in c.most_common(): print(k, f"{v/S/1e3:.3f} ms/step", n[k]//S, "calls", names[k])
