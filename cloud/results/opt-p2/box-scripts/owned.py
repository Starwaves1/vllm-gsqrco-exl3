"""Per decode step (5 complete steps of a torch-profiler trace): GPU time by owner.
plugin GEMM (Route L kernels), plugin plumbing (quantize, output casts, the cat of a fused
layer's runs, stream-k fixups, in_proj_ba BF16 product, IQ1_M, embedding), and everything
else (vLLM: GDN core, attention, norms, activation, sampling, copies), plus busy/idle.
usage: owned.py TRACE.json"""
import collections, json, re, sys
ev = json.load(open(sys.argv[1]))["traceEvents"]
rt = {e["args"]["correlation"]: e["ts"] for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
st = sorted((e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("execute_context")), key=lambda e: e["ts"])
t0, t1, n = st[0]["ts"], st[-1]["ts"], len(st) - 1
RULES = [  # (owner, label, regex on kernel name); first match wins
    ("plugin GEMM", "stream-k fixup (vendored MMQ)", r"stream_k_fixup"),
    ("plugin GEMM", "IQ1_M MMVQ", r"mul_mat_vec_q<\(ggml_type\)29"),
    ("plugin GEMM", "MMQ", r"mul_mat_q<"),
    ("plugin GEMM", "MMVQ", r"mul_mat_vec_q<"),
    ("plugin GEMM", "owned IQ3 dp4a", r"iq3_mul_mat_vec<"),
    ("plugin GEMM", "owned IQ3 mma", r"iq3_mma"),
    ("plugin GEMM", "owned Q4_K/IQ2_S", r"own_mul_mat_vec"),
    ("plugin plumbing", "q8_1 quantize (MMVQ/owned)", r"quantize_q8_1_x"),
    ("plugin plumbing", "q8_1 quantize (MMQ)", r"quantize_mmq_q8_1_x"),
    ("plugin plumbing", "fp32->16-bit output cast", r"bfloat16_copy_kernel|Half_copy"),
    ("plugin plumbing", "cat of a fused layer's runs", r"triton_poi_fused_cat"),
    ("plugin plumbing", "in_proj_ba BF16 product", r"gemvx|wmma_tensorop_bf16|splitKreduce"),
    ("plugin plumbing", "IQ1_M dequant + cuBLAS", r"dequantize_block_iq1_m|ampere_bf16_s16816gemm"),
    ("plugin plumbing", "embedding (gather + dequant)", r"indexSelectSmallIndex|dequantize_block_iq2_s"),
    ("plugin plumbing", "MMQ tail memset", r"memset32"),
]
agg = collections.defaultdict(lambda: [0.0, 0.0]); busy = []
for e in ev:
    if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X" and t0 <= rt.get(e["args"].get("correlation"), e["ts"]) < t1:
        owner, label = next(((o, l) for o, l, r in RULES if re.search(r, e["name"])), ("vLLM / other", "-"))
        a = agg[(owner, label)]; a[0] += 1 / n; a[1] += e["dur"] / n / 1e3
        busy.append((e["ts"], e["ts"] + e["dur"]))
busy.sort(); tot, end = 0.0, 0.0
for s, f in busy:
    if f > end: tot += f - max(s, end); end = f
span = (t1 - t0) / n / 1e3
print(f"per step: span {span:.2f} ms, GPU busy {tot / n / 1e3:.2f} ms, idle {span - tot / n / 1e3:.2f} ms")
for owner in ("plugin GEMM", "plugin plumbing", "vLLM / other"):
    rows = {l: v for (o, l), v in agg.items() if o == owner}
    print(f"{owner:16} {sum(v[1] for v in rows.values()):7.3f} ms  {sum(v[0] for v in rows.values()):6.0f} launches")
    if owner != "vLLM / other":
        for l, (c, d) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
            print(f"   {d:7.3f} ms {c:6.0f}  {l}")
