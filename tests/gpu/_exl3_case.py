"""One EXL3 shim case in a fresh process (empty warmup registry; a sticky CUDA error cannot
reach the rest of the suite). Prints one JSON line: status = ok | rejected | mismatch,
the error text, and healthy_after (a small eager exl3_gemm still works afterwards).

  python tests/gpu/_exl3_case.py CASE TID      (TID from tests/gpu/exl3_cases.py)

Cases:
  unwarmed_capture        exl3_gemm inside a CUDA graph capture with no exl3_warmup: rejected
  hgemm_unwarmed_capture  exl3_hgemm inside a capture with no warmup: rejected
  first_call_in_capture   exl3_warmup, then the first exl3_gemm call of the process happens
                          inside a capture (1, 4, 16 and 48 rows): ok if replay == eager
  x_misaligned, x_noncontig, x_bf16, k_mismatch, suh_wrong_size, trellis_misaligned,
  dequant_unaligned, dequant_out_of_range   bad CUDA inputs: rejected before any launch
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import torch  # noqa: E402

import exl3_cases as C  # noqa: E402
from vllm_exl3_plugin import _C_exl3  # noqa: E402,F401

OPS = torch.ops._C_exl3


def offset(t, elems=1):
    """Same values, data pointer `elems` elements past a 16-byte boundary."""
    buf = torch.empty(t.numel() + elems, dtype=t.dtype, device=t.device)
    v = buf[elems:].view(t.shape)
    v.copy_(t)
    return v


def capture(fn):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def main():
    case, tid = sys.argv[1:3]
    w = C.load(torch, tid)
    tr, suh, svh, mcg, mul1 = w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"]
    k, n = tr.shape[0] * 16, tr.shape[1] * 16
    x = C.make_x(torch, tid, 4)
    res = {"case": case, "tid": tid}
    try:
        if case == "unwarmed_capture":
            capture(lambda: OPS.exl3_gemm(x, tr, suh, svh, mcg, mul1, True))
            res["status"] = "mismatch"  # captured without warmup: the guard did not fire
        elif case == "hgemm_unwarmed_capture":
            a, b = torch.zeros(160, k, dtype=torch.half, device="cuda"), torch.zeros(k, 256, dtype=torch.half, device="cuda")
            capture(lambda: OPS.exl3_hgemm(a, b))
            res["status"] = "mismatch"
        elif case == "first_call_in_capture":
            OPS.exl3_warmup(tr, suh, svh, mcg, mul1, [1, 2, 4, 8, 16], True)
            detail = {}
            for m in (1, 4, 16, 48):
                xs = C.make_x(torch, tid, m)
                g, y = capture(lambda: OPS.exl3_gemm(xs, tr, suh, svh, mcg, mul1, True))
                xs.copy_(torch.randn(m, k, generator=torch.Generator().manual_seed(m)).half())
                g.replay()
                torch.cuda.synchronize()
                eager = OPS.exl3_gemm(xs, tr, suh, svh, mcg, mul1, True)
                d = C.err_stats(torch, y, eager.double())
                detail[m] = {"bit_identical": bool(torch.equal(y, eager)), **d}
                if not (d["finite"] and d["rel_rms"] <= 1e-3):
                    res["status"] = "mismatch"
            res.setdefault("status", "ok")
            res["detail"] = detail
        else:
            if case == "x_misaligned":
                call = lambda: OPS.exl3_gemm(offset(x), tr, suh, svh, mcg, mul1, True)  # noqa: E731
            elif case == "x_noncontig":
                xt = torch.empty(k, 4, dtype=torch.half, device="cuda").t()
                call = lambda: OPS.exl3_gemm(xt, tr, suh, svh, mcg, mul1, True)  # noqa: E731
            elif case == "x_bf16":
                call = lambda: OPS.exl3_gemm(x.bfloat16(), tr, suh, svh, mcg, mul1, True)  # noqa: E731
            elif case == "k_mismatch":
                call = lambda: OPS.exl3_gemm(x[:, : k - 128].contiguous(), tr, suh, svh, mcg, mul1, True)  # noqa: E731
            elif case == "suh_wrong_size":
                call = lambda: OPS.exl3_gemm(x, tr, suh[:-128].contiguous(), svh, mcg, mul1, True)  # noqa: E731
            elif case == "trellis_misaligned":
                call = lambda: OPS.exl3_gemm(x, offset(tr), suh, svh, mcg, mul1, True)  # noqa: E731
            elif case == "dequant_unaligned":
                call = lambda: OPS.exl3_dequant(tr, suh, svh, mcg, mul1, 64, 128, True)  # noqa: E731
            elif case == "dequant_out_of_range":
                call = lambda: OPS.exl3_dequant(tr, suh, svh, mcg, mul1, n - 128, 256, True)  # noqa: E731
            else:
                raise SystemExit(f"unknown case {case}")
            call()
            torch.cuda.synchronize()
            res["status"] = "mismatch"  # ran on bad input without complaint
    except RuntimeError as e:
        res.setdefault("status", "rejected")
        res["error"] = str(e).splitlines()[0][:300]
    try:  # the context must still work
        torch.cuda.synchronize()
        y = OPS.exl3_gemm(C.make_x(torch, tid, 1), tr, suh, svh, mcg, mul1, True)
        torch.cuda.synchronize()
        res["healthy_after"] = bool(torch.isfinite(y).all().item())
    except RuntimeError as e:
        res["healthy_after"] = False
        res["after_error"] = str(e).splitlines()[0][:300]
    print(json.dumps(res))


if __name__ == "__main__":
    main()
