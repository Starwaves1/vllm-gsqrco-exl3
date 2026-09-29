"""Per-layer GEMM microbenchmark: stock plugin kernels vs Route L (lcpp) on real GGUF blocks.

  GSQ_ALLOW_GPU=1 .venv/bin/python bench/micro/gemm.py [--out results.tsv]

For each type and weight shape (rows x K), the weight is filled cyclically with the type's
real blocks from the GGUF (so decode paths see real data), X is bf16 (production's dtype).
Variants (only where production could route them):
  stock_mmvq     ops.ggml_mul_mat_vec_a8 (e2b8ad5 MMVQ, one grid slice per token), n <= 16
  stock_mmq      ops.ggml_mul_mat_a8 (e2b8ad5 MMQ), K-quants only
  stock_dq       ops.ggml_dequantize to bf16 + x @ W.T (cuBLAS), what IQ types get above 8-16 rows
  lcpp_mmvq      torch.ops._C_gguf.lcpp_mul_mat_vec_q, n <= 8
  lcpp_mmq       torch.ops._C_gguf.lcpp_mul_mat_q
  lcpp_iq3       torch.ops._C_gguf.lcpp_mul_mat_vec_iq3 (owned IQ3_S/IQ3_XXS kernel), n <= 8
  lcpp_iq3_mma   torch.ops._C_gguf.lcpp_mul_mat_vec_iq3_mma (the same on int8 tensor cores), n <= 8
  lcpp_iq3_mma_packed  the same on W packed by quantization/iq3_pack.py, n <= 32
Times: "graph" = GPU time per call, 10 calls captured in one CUDA graph and replayed (no CPU
launch cost; decode runs under CUDA graphs up to 32 tokens); "eager" = wall per call of plain
back-to-back calls (prefill chunks above 32 tokens run eager). GB/s = weight bytes / graph time.
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "gpu"))
from gsq_gpu import GGUF  # noqa: E402  (imports no_gpu unless GSQ_ALLOW_GPU=1)

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm_gguf_plugin import ops  # noqa: E402
from vllm_gguf_plugin.quantization import iq3_pack  # noqa: E402

TYPES = ["IQ3_S", "IQ3_XXS", "IQ4_XS", "Q4_K"]
SHAPES = [(17408, 5120), (5120, 17408)]
LM_HEAD = (248320, 5120)             # output.weight (Q4_K in this GGUF)
TOKENS = [1, 2, 4, 8, 16, 32, 128, 2048]
PER_GRAPH = 10


def weight(reader, name, rows, k):
    qt = gguf.GGMLQuantizationType[name]
    bsz = gguf.GGML_QUANT_SIZES[qt][1]
    blocks = np.concatenate([np.asarray(t.data).reshape(-1, bsz) for t in reader.tensors
                             if t.tensor_type.name == name and len(t.shape) == 2][:4])
    need = rows * (k // 256)
    idx = np.arange(need) % len(blocks)
    return torch.from_numpy(np.ascontiguousarray(blocks[idx]).reshape(rows, -1)).cuda(), int(qt)


def variants(name, qt, rows, k, n, packed=None):
    C = torch.ops._C_gguf
    v = {}
    if n <= 16:
        v["stock_mmvq"] = lambda w, x: ops.ggml_mul_mat_vec_a8(w, x, qt, rows)
    if name.startswith("Q"):
        v["stock_mmq"] = lambda w, x: ops.ggml_mul_mat_a8(w, x, qt, rows)
    v["stock_dq"] = lambda w, x: x @ ops.ggml_dequantize(w, qt, rows, k, x.dtype).T
    if n <= 8:
        v["lcpp_mmvq"] = lambda w, x: C.lcpp_mul_mat_vec_q(w, x, qt, rows)
    v["lcpp_mmq"] = lambda w, x: C.lcpp_mul_mat_q(w, x, qt, rows)
    if n <= 8 and name.startswith("IQ3"):
        v["lcpp_iq3"] = lambda w, x: C.lcpp_mul_mat_vec_iq3(w, x, qt, rows)
        if hasattr(C, "lcpp_mul_mat_vec_iq3_mma"):
            v["lcpp_iq3_mma"] = lambda w, x: C.lcpp_mul_mat_vec_iq3_mma(w, x, qt, rows)
    if n <= 32 and name.startswith("IQ3") and hasattr(C, "lcpp_mul_mat_vec_iq3_mma_packed"):
        v["lcpp_iq3_mma_packed"] = lambda w, x: C.lcpp_mul_mat_vec_iq3_mma_packed(packed, x, qt, rows)
    return v


def time_graph(fn, w, x):
    fn(w, x)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(PER_GRAPH):
            fn(w, x)
    g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    reps = max(3, min(200, int(0.3 / max(time.perf_counter() - t0, 1e-6))))
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    del g
    return s.elapsed_time(e) * 1e3 / (reps * PER_GRAPH)


def time_eager(fn, w, x, calls):
    for _ in range(3):
        fn(w, x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(calls):
        fn(w, x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--types", default=",".join(TYPES), help="comma-separated subset of TYPES")
    ap.add_argument("--tokens", default=",".join(map(str, TOKENS)))
    ap.add_argument("--variants", help="comma-separated subset of variant names")
    args = ap.parse_args()
    types = args.types.split(",")
    tokens = [int(t) for t in args.tokens.split(",")]
    if not hasattr(torch.ops._C_gguf, "lcpp_mul_mat_q"):
        sys.exit("_C_gguf built without VLLM_GGUF_BUILD_LCPP=1")
    reader = gguf.GGUFReader(str(GGUF))
    cases = [(t, s) for t in types for s in SHAPES] + ([("Q4_K", LM_HEAD)] if "Q4_K" in types else [])
    lines = ["type\trows\tK\tn\tvariant\tgraph_us\teager_us\tGBps"]
    print(f"{torch.cuda.get_device_name()}  X bf16, {PER_GRAPH} calls per graph", flush=True)
    for name, (rows, k) in cases:
        w, qt = weight(reader, name, rows, k)
        packed = iq3_pack.pack(w, qt) if name.startswith("IQ3") else None
        mb = w.numel() / 1e6
        print(f"\n{name} {rows}x{k} ({mb:.1f} MB)", flush=True)
        for n in tokens:
            x = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            row = []
            for vname, fn in variants(name, qt, rows, k, n, packed).items():
                if args.variants and vname not in args.variants.split(","):
                    continue
                g = time_graph(fn, w, x)
                e = time_eager(fn, w, x, calls=max(5, min(100, int(2e5 / max(g, 1)))))
                gbps = w.numel() / (g * 1e-6) / 1e9
                lines.append(f"{name}\t{rows}\t{k}\t{n}\t{vname}\t{g:.1f}\t{e:.1f}\t{gbps:.0f}")
                row.append(f"{vname} {g:8.1f}us {gbps:4.0f}GB/s (eager {e:.0f})")
            print(f"  n={n:5d}  " + " | ".join(row), flush=True)
        del w, packed
        torch.cuda.empty_cache()
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
