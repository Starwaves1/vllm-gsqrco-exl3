"""One out-of-bounds / layout case against the plugin's CUDA ops, in its own process
(CUDA errors are sticky: an illegal memory access poisons the context for the rest of the
process). Prints one JSON line: status = ok (ran, result correct) | rejected (clean Python
exception before any bad access) | mismatch (ran, silently wrong).

  python tests/gpu/_guard_case.py CASE TYPE OP     OP = mmvq | mmq | lcpp_mmvq | lcpp_mmq | lcpp_iq3 | lcpp_iq3_mma | lcpp_own
                                                   | lcpp_mma_k | lcpp_iq3_mma_packed | lcpp_iq3_packed
                                                   (the packed ops: W packed first)

Cases (the kernels use data_ptr() only; no contiguity, stride or alignment checks,
gguf_kernel.cu:98,118-285 per STATUS):
  x_noncontig      X is a transposed view (column-major)
  x_misaligned     X starts 1 element into its storage (2-byte offset)
  w_narrow_view    W is weight[:, :bytes] of a wider buffer (row stride > row bytes), as
                   GGUFLinearMethod.apply() would pass without its .contiguous()
  w_misaligned     W starts 1 byte into its storage
  row_too_big      row argument > W rows (reads past W)
  k_mismatch       X has fewer columns than W's rows hold (reads past X)
  x_rowstride      X is x[:, :k] of a wider buffer (row stride > k, unit inner stride)
  graph_replay     capture the op in a CUDA graph, replay with new X contents, compare
  x_q8_short       pre-quantized X (x_q8, the 1..8-row lcpp ops only) one byte short
  x_q8_misaligned  x_q8 starts 1 byte into its storage
  x_q8_dtype, x_q8_2d, x_q8_strided, x_q8_cpu   x_q8 as int8, 2-D, stride 2, on the CPU
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import _refs  # noqa: E402
from vllm_gguf_plugin import ops  # noqa: E402

ROWS, N = 256, 4


def main() -> None:
    case, name, op = sys.argv[1:4]
    path = os.environ["GSQ_GGUF"]
    r = gguf.GGUFReader(path)
    t = next(t for t in r.tensors if t.tensor_type.name == name and len(t.shape) == 2)
    raw = np.ascontiguousarray(t.data[:ROWS])
    qt = int(gguf.GGMLQuantizationType[name])
    k = int(t.shape[0])
    fn = {"mmvq": ops.ggml_mul_mat_vec_a8, "mmq": ops.ggml_mul_mat_a8,
          "lcpp_mmvq": getattr(torch.ops._C_gguf, "lcpp_mul_mat_vec_q", None),
          "lcpp_mmq": getattr(torch.ops._C_gguf, "lcpp_mul_mat_q", None),
          "lcpp_iq3": getattr(torch.ops._C_gguf, "lcpp_mul_mat_vec_iq3", None),
          "lcpp_iq3_mma": getattr(torch.ops._C_gguf, "lcpp_mul_mat_vec_iq3_mma", None),
          "lcpp_iq3_mma_packed": getattr(torch.ops._C_gguf, "lcpp_mul_mat_vec_iq3_mma_packed", None),
          "lcpp_iq3_packed": getattr(torch.ops._C_gguf, "lcpp_mul_mat_iq3_packed", None),
          "lcpp_own": getattr(torch.ops._C_gguf, "lcpp_mul_mat_vec_own", None),
          "lcpp_mma_k": getattr(torch.ops._C_gguf, "lcpp_mul_mat_mma_k", None)}[op]
    if fn is None:
        raise SystemExit(f"{op}: _C_gguf built without VLLM_GGUF_BUILD_LCPP=1")
    # lcpp MMQ at 5 rows: below upstream's J_max tail, where only the shim's zeroed tail
    # keeps the tile reads defined
    n = {"mmvq": N, "lcpp_mmvq": N, "lcpp_iq3": N, "lcpp_iq3_mma": N, "lcpp_own": N, "mmq": 64,
         "lcpp_mmq": 5, "lcpp_mma_k": 16, "lcpp_iq3_mma_packed": 16, "lcpp_iq3_packed": 129}[op]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, k, generator=g).to(torch.bfloat16)
    w = torch.from_numpy(raw).cuda()
    if op in ("lcpp_iq3_mma_packed", "lcpp_iq3_packed"):
        from vllm_gguf_plugin.quantization import iq3_pack

        w = iq3_pack.pack(w, qt)
    xc = x.cuda()
    row = w.shape[0]
    ref_raw, ref_x = raw, x
    extra = ()

    if case == "x_noncontig":
        xc = x.t().contiguous().cuda().t()
    elif case == "x_misaligned":
        buf = torch.empty(n * k + 1, dtype=torch.bfloat16, device="cuda")
        xc = buf[1:].view(n, k)
        xc.copy_(x.cuda())
    elif case == "w_narrow_view":
        wide = torch.zeros(w.shape[0], w.shape[1] + 256, dtype=torch.uint8, device="cuda")
        wide[:, : w.shape[1]] = w
        w = wide[:, : w.shape[1]]
    elif case == "w_misaligned":
        buf = torch.empty(w.numel() + 1, dtype=torch.uint8, device="cuda")
        w2 = buf[1:].view(w.shape)
        w2.copy_(w)
        w = w2
    elif case == "row_too_big":
        row = w.shape[0] + 64
    elif case == "k_mismatch":
        xc = xc[:, : k // 2].contiguous()
    elif case == "x_rowstride":
        wide = torch.zeros(n, k + 512, dtype=torch.bfloat16, device="cuda")
        wide[:, :k] = xc
        xc = wide[:, :k]
    elif case.startswith("x_q8_"):
        q8 = torch.ops._C_gguf.lcpp_quantize_q8_1(xc, qt, False, False)
        q8 = {"x_q8_short": lambda: q8[:-1].clone(),  # its own, short allocation
              "x_q8_misaligned": lambda: torch.cat([q8.new_zeros(1), q8])[1:],
              "x_q8_dtype": lambda: q8.view(torch.int8),
              "x_q8_2d": lambda: q8.view(1, -1),
              "x_q8_strided": lambda: torch.stack([q8, q8], 1).view(-1)[::2],
              "x_q8_cpu": lambda: q8.cpu()}[case]()
        extra = (q8,)
    elif case not in ("graph_replay", "graph_first"):
        raise SystemExit(f"unknown case {case}")

    try:
        if case in ("graph_replay", "graph_first"):
            static_x = xc.clone()
            if case == "graph_replay":
                fn(w, static_x, qt, row)  # warm-up outside capture
            torch.cuda.synchronize()  # graph_first: the op's first call in this process is captured
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_y = fn(w, static_x, qt, row)
            ref_x = torch.randn(n, k, generator=g).to(torch.bfloat16)
            static_x.copy_(ref_x.cuda())
            graph.replay()
            y = static_y
        else:
            y = fn(w, xc, qt, row, *extra)
        torch.cuda.synchronize()
    except (RuntimeError, ValueError, TypeError) as e:
        if "CUDA" in str(e) or "illegal" in str(e).lower():
            raise  # a device fault is not a clean rejection
        print(json.dumps({"case": case, "status": "rejected", "error": str(e)[:300]}))
        return
    if case in ("row_too_big", "k_mismatch"):
        # ran without a device fault; there is no correct answer to compare with
        print(json.dumps({"case": case, "status": "mismatch", "note": "accepted invalid shapes silently"}))
        return
    # MMQ's q8_1 layout
    mmq = op.endswith(("mmq", "mma_k")) or op == "lcpp_iq3_packed"
    refs = _refs.refs(ref_raw, name, ref_x, mmq=mmq, lcpp=op.startswith("lcpp"))
    err = min(_refs.rel_err(y, v) for kk, v in refs.items() if kk != "full")
    print(json.dumps({"case": case, "status": "ok" if err < 5e-3 else "mismatch", "rel_err": err}))


if __name__ == "__main__":
    main()
