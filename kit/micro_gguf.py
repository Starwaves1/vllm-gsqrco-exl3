"""Tier 1 GGUF micro-benchmark: every candidate kernel per (type, weight shape, rows), on synthetic
blocks at the Swift-1.5-Qwen3.8-27B shapes (kit/data/swift-27b-gguf-tensors.tsv, incl. the
248320-row lm_head), against the vendored llama.cpp b11211 MMVQ/MMQ.

  GSQ_ALLOW_GPU=1 python kit/micro_gguf.py --out DIR [--rows 1,2,...] [--types IQ3_S,...]

Per cell: GPU time per call (PER_GRAPH calls captured in one CUDA graph and replayed, cycling
through enough copies of the weight to defeat L2), GB/s of weight bytes, efficiency = that over
the card's measured device-to-device copy bandwidth (the floor; measured first), and whether
the graph replay's output is bit-identical to an eager call. "route" marks the op production's
routing (linear._lcpp_op, or the stock branch of _fused_mul_mat_gguf without Route L) picks.
X is bf16 on sm80+, fp16 below (the dtype vLLM serves with there). Route L variants need
VLLM_GGUF_LCPP=1 and the VLLM_GGUF_BUILD_LCPP=1 build; without them only the stock kernels run.

Writes DIR/gguf_micro.tsv (every cell) and DIR/gguf_micro.json (copy bandwidth, per-cell winner).
A CUDA device error stops the run (the context is dead) and is recorded; a kernel's own input
guard rejecting a cell is recorded as n/a.
"""

import argparse
import collections
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kit"))
sys.path.insert(0, str(ROOT / "tests" / "gpu"))
from gsq_gpu import GPU  # noqa: E402,F401  (imports no_gpu unless GSQ_ALLOW_GPU=1)

import torch  # noqa: E402

import synth  # noqa: E402

ROWS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128]
PER_GRAPH = 10
FATAL = ("illegal memory access", "misaligned address", "unspecified launch failure", "illegal instruction",
         "device-side assert", "CUDA error: an illegal", "uncorrectable ECC")
OPS = {"lcpp_mmvq": "lcpp_mul_mat_vec_q", "lcpp_mmq": "lcpp_mul_mat_q", "own_vec": "lcpp_mul_mat_vec_own",
       "iq3_vec": "lcpp_mul_mat_vec_iq3", "iq3_vec_mma": "lcpp_mul_mat_vec_iq3_mma",
       "iq3_vec_mma_packed": "lcpp_mul_mat_vec_iq3_mma_packed", "iq3_tiled_packed": "lcpp_mul_mat_iq3_packed",
       "mma_k": "lcpp_mul_mat_mma_k"}


def shapes():
    """(type, rows, K) -> [main products, MTP products], from the real GGUF's tensor list."""
    out = collections.defaultdict(lambda: [0, 0])
    for name, typ, rows, k in synth.gguf_tensors():
        if name != "token_embd.weight":
            out[(typ, rows, k)][1 if name.startswith("blk.64.") else 0] += 1
    return dict(sorted(out.items()))


def variants(typ, qt, rows, k, n, lcpp):
    """name -> fn(w, packed, x); only where production could route the op (gemm.py, r3gemm.py)."""
    from vllm_gguf_plugin import ops

    C = torch.ops._C_gguf if lcpp else None
    v = {}
    if n <= 16:
        v["stock_mmvq"] = lambda w, p, x: ops.ggml_mul_mat_vec_a8(w, x, qt, rows)
    if typ in ("Q2_K", "Q4_K", "Q6_K"):
        v["stock_mmq"] = lambda w, p, x: ops.ggml_mul_mat_a8(w, x, qt, rows)
    if n >= 16 and rows <= 20000:  # dequantize + cuBLAS; above 20000 rows the bf16 copy alone is 2.5 GB
        v["stock_dq"] = lambda w, p, x: x @ ops.ggml_dequantize(w, qt, rows, k, x.dtype).T
    if not lcpp:
        return v
    if n <= 8:
        v["lcpp_mmvq"] = lambda w, p, x: C.lcpp_mul_mat_vec_q(w, x, qt, rows)
    elif typ == "IQ1_M" and n <= 32:  # production's IQ1_M route: MMVQ in 8-row calls
        v["lcpp_mmvq_chunked8"] = lambda w, p, x: torch.cat(
            [C.lcpp_mul_mat_vec_q(w, x[i:i + 8], qt, rows) for i in range(0, x.shape[0], 8)])
    if typ != "IQ1_M":  # llama.cpp has no IQ1_M MMQ
        v["lcpp_mmq"] = lambda w, p, x: C.lcpp_mul_mat_q(w, x, qt, rows)
    if typ in ("IQ3_S", "IQ3_XXS"):
        if n <= 8:
            v["iq3_vec"] = lambda w, p, x: C.lcpp_mul_mat_vec_iq3(w, x, qt, rows)
            v["iq3_vec_mma"] = lambda w, p, x: C.lcpp_mul_mat_vec_iq3_mma(w, x, qt, rows)
        if rows % 16 == 0:
            if n <= 32:
                v["iq3_vec_mma_packed"] = lambda w, p, x: C.lcpp_mul_mat_vec_iq3_mma_packed(p, x, qt, rows)
            v["iq3_tiled_packed"] = lambda w, p, x: C.lcpp_mul_mat_iq3_packed(p, x, qt, rows)
    if typ in ("Q4_K", "IQ2_S") and n <= 8:
        v["own_vec"] = lambda w, p, x: C.lcpp_mul_mat_vec_own(w, x, qt, rows)
    if typ in ("Q4_K", "IQ4_XS", "IQ2_S"):
        if n <= 64:
            v["mma_k"] = lambda w, p, x: C.lcpp_mul_mat_mma_k(w, x, qt, rows)
        else:  # the kernel takes <= 64 rows: 64-row chunks, W read once per chunk
            v["mma_k_chunked64"] = lambda w, p, x: torch.cat(
                [C.lcpp_mul_mat_mma_k(w, x[i:i + 64], qt, rows) for i in range(0, x.shape[0], 64)])
    return v


def route(typ, qt, rows, k, n, lcpp):
    """The variant production's _fused_mul_mat_gguf runs for this cell."""
    from vllm_gguf_plugin.quantization import linear, utils

    if lcpp:
        op = linear._lcpp_op(n, qt, rows, k, typ in ("IQ3_S", "IQ3_XXS") and rows % 16 == 0)
        if op == "lcpp_mul_mat_vec_q" and n > 8:
            return "lcpp_mmvq_chunked8"
        if op is not None:
            return {v: k_ for k_, v in OPS.items()}[op]
    safe = (8 if rows > 5120 else 16) if qt in utils.IMATRIX_QUANT_TYPES else (2 if rows > 5120 else 6)
    if n <= safe and qt in utils.MMVQ_QUANT_TYPES:
        return "stock_mmvq"
    return "stock_mmq" if qt in utils.MMQ_QUANT_TYPES else "stock_dq"


def copy_bandwidth():
    """Device-to-device copy GB/s (read + write bytes), best of 10, on 2 x 256 MiB."""
    n = 256 * 2**20
    a, b = torch.empty(n, dtype=torch.uint8, device="cuda"), torch.empty(n, dtype=torch.uint8, device="cuda")
    a.fill_(1)
    b.copy_(a)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(10):
        s.record()
        for _ in range(5):
            b.copy_(a)
        e.record()
        e.synchronize()
        best = min(best, s.elapsed_time(e) / 5)
    del a, b
    torch.cuda.empty_cache()
    return 2 * n / (best * 1e-3) / 1e9


def measure(fn, ws, ps, x):
    """(us per call in a replayed graph, graph output bit-identical to an eager call)."""
    ref = fn(ws[0], ps[0], x).clone()
    for w, p in zip(ws, ps):
        fn(w, p, x)
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn(ws[0], ps[0], x)
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [fn(ws[i % len(ws)], ps[i % len(ps)], x) for i in range(PER_GRAPH)]
    g.replay()
    torch.cuda.synchronize()
    exact = bool(torch.equal(outs[0], ref))
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(5):
        a.record()
        for _ in range(4):
            g.replay()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / (4 * PER_GRAPH))
    del g, outs
    return best, exact


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--types", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("KIT_MEM_FRACTION", "0.8")))
    from vllm_gguf_plugin import ops

    lcpp = bool(ops.LCPP_ENABLED)
    rows_list = [int(r) for r in a.rows.split(",")]
    types = set(a.types.split(",")) if a.types else None
    cc = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if cc >= (8, 0) else torch.float16
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    bw = copy_bandwidth()
    print(f"{torch.cuda.get_device_name()} sm{cc[0]}{cc[1]}: copy {bw:.0f} GB/s, L2 {l2 / 2**20:.1f} MiB, "
          f"X {str(dtype)[6:]}, Route L {'on' if lcpp else 'NOT BUILT (stock kernels only)'}", flush=True)
    cols = ["type", "rows", "K", "main", "mtp", "n", "variant", "is_route", "us", "GBps", "eff", "graph_exact", "note"]
    tsv = a.out / "gguf_micro.tsv"
    lines = ["\t".join(cols)]
    fatal = None
    rng = __import__("numpy").random.default_rng(synth.SEED)
    for (typ, rows, k), (n_main, n_mtp) in shapes().items():
        if types and typ not in types:
            continue
        if fatal:
            break
        blocks, qt = synth.random_blocks(rng, typ, rows, k)
        w0 = torch.from_numpy(blocks).cuda()
        nbytes = w0.numel()
        ws = [w0] + [w0.clone() for _ in range(max(1, -(-4 * l2 // nbytes)) - 1)]
        ps = [None] * len(ws)
        if lcpp and typ in ("IQ3_S", "IQ3_XXS") and rows % 16 == 0:
            from vllm_gguf_plugin.quantization import iq3_pack

            ps = [iq3_pack.pack(w, qt) for w in ws]
        print(f"\n{typ} {rows}x{k} ({nbytes / 2**20:.1f} MiB, {n_main} main + {n_mtp} mtp, x{len(ws)} copies)", flush=True)
        for n in rows_list:
            x = torch.randn(n, k, device="cuda").to(dtype)
            r = route(typ, qt, rows, k, n, lcpp)
            cells = []
            for vname, fn in variants(typ, int(qt), rows, k, n, lcpp).items():
                try:
                    us, exact = measure(fn, ws, ps, x)
                    gbps = nbytes / (us * 1e-6) / 1e9
                    lines.append("\t".join(map(str, [typ, rows, k, n_main, n_mtp, n, vname, int(vname == r),
                                                     f"{us:.2f}", f"{gbps:.0f}", f"{gbps / bw:.3f}", int(exact), ""])))
                    cells.append(f"{'*' if vname == r else ''}{vname} {us:.1f}us {gbps / bw:.2f}{'' if exact else ' GRAPH!=EAGER'}")
                except Exception as e:  # noqa: BLE001
                    msg = " ".join(str(e).split())[:120]
                    lines.append("\t".join(map(str, [typ, rows, k, n_main, n_mtp, n, vname, int(vname == r),
                                                     "", "", "", "", msg])))
                    cells.append(f"{vname} n/a")
                    if any(f in str(e) for f in FATAL):
                        fatal = f"{typ} {rows}x{k} n={n} {vname}: {msg}"
                        break
                    torch.cuda.synchronize()
            print(f"  n={n:3d} route={r}: " + " | ".join(cells), flush=True)
            if fatal:
                break
        del ws, ps, w0
        torch.cuda.empty_cache()
        tsv.write_text("\n".join(lines) + "\n")
    tsv.write_text("\n".join(lines) + "\n")
    (a.out / "gguf_micro.json").write_text(json.dumps(
        {"copy_GBps": round(bw, 1), "l2_bytes": l2, "x_dtype": str(dtype)[6:], "route_l": lcpp,
         "rows": rows_list, "fatal": fatal, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    if fatal:
        sys.exit(f"device error, run stopped: {fatal}")


if __name__ == "__main__":
    main()
