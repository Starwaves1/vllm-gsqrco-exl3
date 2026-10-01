# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One bad-input case against a _C_gguf op, in its own process (a device fault
poisons the CUDA context for the rest of the process). Prints one JSON line with
status ok (ran, result equal to the op on clean inputs), rejected (a clean
exception before any launch) or mismatch (ran, silently wrong).

    python tests/kernel_guard_case.py CASE TYPE OP

OP is the name of a _C_gguf op taking (W, X, type, row). Cases:
  x_noncontig     X is a transposed view (column-major)
  x_misaligned    X starts one element into its storage
  x_rowstride     X is x[:, :k] of a wider buffer (row stride > k)
  w_narrow_view   W is weight[:, :bytes] of a wider buffer (row stride > bytes)
  w_misaligned    W starts one byte into its storage
  row_too_big     the row argument exceeds W's rows (reads past W)
  k_mismatch      X has fewer columns than W's rows hold (reads past X)
  graph_replay    capture in a CUDA graph, replay with new X, compare to eager
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tests.utils import get_gguf_sample_tensors  # noqa: E402
from vllm_gguf_plugin import ops  # noqa: E402, F401  (loads _C_gguf)

ROWS = 256
N = {"ggml_mul_mat_a8": 64}  # activation rows per op; 4 otherwise


def main() -> None:
    case, name, op = sys.argv[1:4]
    qt = gguf.GGMLQuantizationType[name]
    t = get_gguf_sample_tensors(1024, qt)[0]
    raw = np.ascontiguousarray(t.data[:ROWS])
    k = raw.shape[1] // gguf.GGML_QUANT_SIZES[qt][1] * gguf.GGML_QUANT_SIZES[qt][0]
    fn = getattr(torch.ops._C_gguf, op)
    n = N.get(op, 4)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, k, generator=g).to(torch.bfloat16).cuda()
    w = torch.from_numpy(raw).cuda()
    row = w.shape[0]
    clean = fn(w, x, int(qt), row)  # before any bad input can fault
    torch.cuda.synchronize()
    xc, wc = x, w

    if case == "x_noncontig":
        xc = x.t().contiguous().t()
    elif case == "x_misaligned":
        xc = torch.empty(n * k + 1, dtype=x.dtype, device="cuda")[1:].view(n, k)
        xc.copy_(x)
    elif case == "x_rowstride":
        xc = torch.zeros(n, k + 512, dtype=x.dtype, device="cuda")[:, :k]
        xc.copy_(x)
    elif case == "w_narrow_view":
        wc = torch.zeros(row, w.shape[1] + 256, dtype=w.dtype, device="cuda")
        wc = wc[:, : w.shape[1]]
        wc.copy_(w)
    elif case == "w_misaligned":
        wc = torch.empty(w.numel() + 1, dtype=w.dtype, device="cuda")[1:]
        wc = wc.view(w.shape)
        wc.copy_(w)
    elif case == "row_too_big":
        row += 64
    elif case == "k_mismatch":
        xc = x[:, : k // 2].contiguous()
    elif case != "graph_replay":
        raise SystemExit(f"unknown case {case}")

    try:
        if case == "graph_replay":
            static_x = x.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_y = fn(w, static_x, int(qt), row)
            x2 = torch.randn(n, k, generator=g).to(torch.bfloat16).cuda()
            static_x.copy_(x2)
            graph.replay()
            y, clean = static_y, fn(w, x2, int(qt), row)
        else:
            y = fn(wc, xc, int(qt), row)
        torch.cuda.synchronize()
    except (RuntimeError, ValueError, TypeError) as e:
        if "CUDA error" in str(e) or "illegal" in str(e).lower():
            raise  # a device fault is not a clean rejection
        print(json.dumps({"case": case, "status": "rejected", "error": str(e)[:300]}))
        return
    ok = case not in ("row_too_big", "k_mismatch") and torch.equal(y, clean)
    print(json.dumps({"case": case, "status": "ok" if ok else "mismatch"}))


if __name__ == "__main__":
    main()
