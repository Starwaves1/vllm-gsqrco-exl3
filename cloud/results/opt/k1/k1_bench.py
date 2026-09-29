"""K1 A/B microbench: interleaved rounds of CUDA-graph timings per variant, median us.
python k1_bench.py OUT TYPES TOKENS CFGS [lm]   (K1_SHAPES=ROWSxK,... overrides the shapes)"""
import os, sys, statistics, time
sys.path.insert(0, os.environ["K1_ROOT"] + "/bench/micro"); sys.path.insert(0, os.environ["K1_ROOT"] + "/tests/gpu")
import torch, gguf, gemm
from gsq_gpu import GGUF
out, types, tokens, cfgs = sys.argv[1], sys.argv[2].split(","), [int(t) for t in sys.argv[3].split(",")], sys.argv[4].split(",")
lm = len(sys.argv) > 5 and sys.argv[5] == "lm"
C = torch.ops._C_gguf
reader = gguf.GGUFReader(str(GGUF))
shapes = [tuple(map(int, x.split("x"))) for x in os.environ["K1_SHAPES"].split(",")] if os.environ.get("K1_SHAPES") else gemm.SHAPES
cases = [(t, s) for t in types for s in shapes] + ([("Q4_K", gemm.LM_HEAD)] if lm else [])
lines = ["type\trows\tK\tn\tvariant\tmedian_us\tmin_us\tGBps"]
PER = 10
for name, (rows, k) in cases:
    w, qt = gemm.weight(reader, name, rows, k)
    for n in tokens:
        x = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        graphs = {}
        vs = [("mmvq", None), ("mmq", None)] + [(f"own{c}", c) for c in cfgs]
        for vname, c in vs:
            if c is None:
                fn = (lambda w, x: C.lcpp_mul_mat_vec_q(w, x, qt, rows)) if vname == "mmvq" else (lambda w, x: C.lcpp_mul_mat_q(w, x, qt, rows))
            else:
                os.environ["K1_CFG"] = c
                fn = lambda w, x: C.lcpp_mul_mat_vec_own(w, x, qt, rows)
            fn(w, x); torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(PER):
                    fn(w, x)
            g.replay(); torch.cuda.synchronize()
            graphs[vname] = g
        os.environ.pop("K1_CFG", None)
        res = {v: [] for v in graphs}
        reps = 3 if rows > 100000 else 20
        for _ in range(7):
            for v, g in graphs.items():
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(reps):
                    g.replay()
                e.record(); torch.cuda.synchronize()
                res[v].append(s.elapsed_time(e) * 1e3 / (reps * PER))
        row = []
        for v, ts in res.items():
            m = statistics.median(ts)
            lines.append(f"{name}\t{rows}\t{k}\t{n}\t{v}\t{m:.1f}\t{min(ts):.1f}\t{w.numel()/(m*1e-6)/1e9:.0f}")
            row.append(f"{v} {m:.1f}")
        print(f"{name} {rows}x{k} n={n}: " + " | ".join(row), flush=True)
        del graphs
    del w; torch.cuda.empty_cache()
open(out, "w").write("\n".join(lines) + "\n")
