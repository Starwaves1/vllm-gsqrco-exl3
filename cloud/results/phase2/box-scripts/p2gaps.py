"""Inter-kernel gaps, lm_head launches and MMVQ/MMQ instance counts in the phase-2 decode-step trace."""
import json, glob, collections
p = sorted(glob.glob("/workspace/runs/p2-profile/trace/*.pt.trace.json"))[-1]
ev = json.load(open(p))["traceEvents"]
k = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
big = [e for e in k if "mul_mat" in e["name"] and (e.get("args", {}).get("grid") or [0])[0] > 40000]
for e in big[:8]:
    print(round(e["dur"]), e["args"].get("grid"), e["args"].get("block"), e["name"][:150])
# per-step: gaps > 1 ms between kernels
gaps = [(k[i+1]["ts"] - (k[i]["ts"] + k[i]["dur"]), k[i]["name"][:60], k[i+1]["name"][:60]) for i in range(len(k)-1)]
gaps.sort(reverse=True)
print("largest gaps (us):")
for g in gaps[:12]: print(round(g[0]), "|", g[1], "->", g[2])
c = collections.Counter()
for e in k:
    if "mul_mat_vec_q<" in e["name"] or "mul_mat_q<" in e["name"]:
        c[e["name"].split("(")[0][:110]] += 1
for n, v in c.most_common(12): print(v, n)
