# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input checks of the lcpp ops (csrc/lcpp_shim.cu), on the CPU.

As for the dense ops (test_kernel_guards_cpu.py): the ops are registered for
CPU tensors, run every check and then reject the call with "must be CUDA
tensors", so a call ending in that message passed all checks. Skipped unless
the extension was built with VLLM_GGUF_BUILD_LCPP=1.
"""

import pytest
import torch

from vllm_gguf_plugin import ops

IQ3_S, IQ3_XXS, Q4_K, IQ1_M, IQ4_XS, IQ2_S, IQ1_S = 21, 18, 12, 29, 23, 22, 19
TS = {IQ3_S: 110, IQ3_XXS: 98, Q4_K: 144, IQ4_XS: 136, IQ2_S: 82, IQ1_M: 56}
K = 5120
CUDA_ONLY = "must be CUDA tensors"
OPS = [
    "lcpp_mul_mat_vec_q",
    "lcpp_mul_mat_q",
    "lcpp_mul_mat_vec_iq3",
    "lcpp_mul_mat_vec_iq3_mma",
    "lcpp_mul_mat_vec_iq3_mma_packed",
    "lcpp_mul_mat_iq3_packed",
    "lcpp_mul_mat_vec_own",
    "lcpp_mul_mat_mma_k",
]


def _lcpp():
    if not (ops._CUDA_AVAILABLE and hasattr(torch.ops._C_gguf, "lcpp_mul_mat_q")):
        pytest.skip("_C_gguf built without VLLM_GGUF_BUILD_LCPP=1")
    return torch.ops._C_gguf


def _case(op, type=IQ3_S, rows=256, row_bytes=None, n=4, k=K, row=None, **kw):
    rb = row_bytes if row_bytes is not None else K // 256 * TS.get(type, 110)
    return dict(
        op=op,
        type=type,
        rows=rows,
        row_bytes=rb,
        n=n,
        k=k,
        row=rows if row is None else row,
        **kw,
    )


CASES = {}  # name -> (case, expected message fragment)
for op in ("lcpp_mul_mat_vec_q", "lcpp_mul_mat_q"):
    n = 4 if op == "lcpp_mul_mat_vec_q" else 64
    for t in (IQ3_S, Q4_K):
        CASES[f"{op}-valid-{t}"] = (_case(op, t, n=n), CUDA_ONLY)
        CASES[f"{op}-row_strided-{t}"] = (
            _case(op, t, n=n, stride=2 * K // 256 * TS[t]),
            "rows must be contiguous",
        )
        CASES[f"{op}-w_narrow_view-{t}"] = (
            _case(op, t, n=n, stride=K // 256 * TS[t] + 256),
            "row stride",
        )
    CASES[f"{op}-fp32_x"] = (_case(op, n=n, x_dtype="float32"), CUDA_ONLY)
    CASES[f"{op}-iq1_s"] = (_case(op, IQ1_S, n=n), "unsupported ggml type")
    CASES[f"{op}-w_float"] = (_case(op, n=n, w_dtype="float32"), "W must be uint8")
    CASES[f"{op}-w_1d"] = (_case(op, n=n, w_1d=True), "must be 2-D")
    CASES[f"{op}-x_int"] = (_case(op, n=n, x_dtype="int32"), "X must be fp32")
    CASES[f"{op}-row_bytes_not_blocks"] = (
        _case(op, n=n, row_bytes=2201),
        "not a multiple of the block size",
    )
    CASES[f"{op}-k_not_512"] = (
        _case(op, n=n, row_bytes=110, k=256),
        "must be a multiple of 512",
    )
    CASES[f"{op}-row_too_big"] = (_case(op, n=n, row=256 + 64), "out of range")
    CASES[f"{op}-row_zero"] = (_case(op, n=n, row=0), "out of range")
    CASES[f"{op}-w_transposed"] = (
        _case(op, n=n, rows=2200, row_bytes=256, w_t=True, row=256),
        "W inner stride",
    )
    CASES[f"{op}-w_misaligned"] = (_case(op, n=n, w_offset=1), "16-byte aligned")
    CASES[f"{op}-k_mismatch"] = (_case(op, n=n, k=K // 2), "columns, W rows hold")
    CASES[f"{op}-x_noncontig"] = (_case(op, n=n, x_t=True), "X inner stride")
CASES["lcpp_mul_mat_vec_q-9_rows"] = (
    _case("lcpp_mul_mat_vec_q", n=9),
    "at most 8 rows",
)
CASES["lcpp_mul_mat_q-9_rows"] = (_case("lcpp_mul_mat_q", n=9), CUDA_ONLY)
CASES["lcpp_mul_mat_q-1_row"] = (_case("lcpp_mul_mat_q", n=1), CUDA_ONLY)
# IQ1_M: MMVQ only (llama.cpp has no IQ1_M MMQ)
CASES["lcpp_mul_mat_vec_q-iq1_m"] = (_case("lcpp_mul_mat_vec_q", IQ1_M), CUDA_ONLY)
CASES["lcpp_mul_mat_q-iq1_m"] = (
    _case("lcpp_mul_mat_q", IQ1_M, n=64),
    "no MMQ for IQ1_M",
)
# the owned IQ3 kernels share check_inputs and add a type check
for op in (
    "lcpp_mul_mat_vec_iq3",
    "lcpp_mul_mat_vec_iq3_mma",
    "lcpp_mul_mat_vec_iq3_mma_packed",
    "lcpp_mul_mat_iq3_packed",
):
    for t in (IQ3_S, IQ3_XXS):
        CASES[f"{op}-valid-{t}"] = (_case(op, t), CUDA_ONLY)
        CASES[f"{op}-row_strided-{t}"] = (
            _case(op, t, stride=2 * K // 256 * TS[t]),
            "rows must be contiguous",
        )
    CASES[f"{op}-q4_k"] = (_case(op, Q4_K), "IQ3_S or IQ3_XXS only")
    CASES[f"{op}-1_row"] = (_case(op, n=1), CUDA_ONLY)
    CASES[f"{op}-k_not_512"] = (
        _case(op, row_bytes=110, k=256),
        "must be a multiple of 512",
    )
    CASES[f"{op}-w_misaligned"] = (_case(op, w_offset=1), "16-byte aligned")
    CASES[f"{op}-k_mismatch"] = (_case(op, k=K // 2), "columns, W rows hold")
    if op.endswith("_packed"):  # 16-row tiles; the vec kernel to 32 rows
        CASES[f"{op}-32_rows"] = (_case(op, n=32), CUDA_ONLY)
        CASES[f"{op}-row_not_16"] = (_case(op, row=200), "must be a multiple of 16")
        if op == "lcpp_mul_mat_iq3_packed":
            CASES[f"{op}-2048_rows"] = (_case(op, n=2048), CUDA_ONLY)
        else:
            CASES[f"{op}-33_rows"] = (_case(op, n=33), "at most 32 rows")
    else:
        CASES[f"{op}-9_rows"] = (_case(op, n=9), "at most 8 rows")
# the owned Q4_K / IQ2_S kernel: the same, with its own type check
op = "lcpp_mul_mat_vec_own"
for t in (Q4_K, IQ2_S):
    CASES[f"{op}-valid-{t}"] = (_case(op, t), CUDA_ONLY)
    CASES[f"{op}-row_strided-{t}"] = (
        _case(op, t, stride=2 * K // 256 * TS[t]),
        "rows must be contiguous",
    )
CASES[f"{op}-iq3_s"] = (_case(op, IQ3_S), "Q4_K or IQ2_S only")
CASES[f"{op}-iq4_xs"] = (_case(op, IQ4_XS), "Q4_K or IQ2_S only")
CASES[f"{op}-1_row"] = (_case(op, Q4_K, n=1), CUDA_ONLY)
CASES[f"{op}-9_rows"] = (_case(op, Q4_K, n=9), "at most 8 rows")
CASES[f"{op}-k_not_512"] = (
    _case(op, Q4_K, row_bytes=144, k=256),
    "must be a multiple of 512",
)
CASES[f"{op}-w_misaligned"] = (_case(op, Q4_K, w_offset=1), "16-byte aligned")
CASES[f"{op}-k_mismatch"] = (_case(op, Q4_K, k=K // 2), "columns, W rows hold")
# the owned int8 tensor-core kernel: MMQ's checks, its own type check, at
# most 64 rows
op = "lcpp_mul_mat_mma_k"
for t in (Q4_K, IQ4_XS, IQ2_S):
    CASES[f"{op}-valid-{t}"] = (_case(op, t, n=16), CUDA_ONLY)
    CASES[f"{op}-row_strided-{t}"] = (
        _case(op, t, n=16, stride=2 * K // 256 * TS[t]),
        "rows must be contiguous",
    )
CASES[f"{op}-iq3_s"] = (_case(op, IQ3_S, n=16), "Q4_K, IQ4_XS or IQ2_S only")
CASES[f"{op}-1_row"] = (_case(op, Q4_K, n=1), CUDA_ONLY)
CASES[f"{op}-64_rows"] = (_case(op, Q4_K, n=64), CUDA_ONLY)
CASES[f"{op}-65_rows"] = (_case(op, Q4_K, n=65), "at most 64 rows")
CASES[f"{op}-k_not_512"] = (
    _case(op, Q4_K, n=16, row_bytes=144, k=256),
    "must be a multiple of 512",
)
CASES[f"{op}-w_misaligned"] = (_case(op, Q4_K, n=16, w_offset=1), "16-byte aligned")
CASES[f"{op}-k_mismatch"] = (_case(op, Q4_K, n=16, k=K // 2), "columns, W rows hold")


def _w(rows, row_bytes, stride=None, offset=0, dtype=torch.uint8):
    stride = stride or row_bytes
    buf = torch.zeros(rows * stride + offset + 64, dtype=dtype)
    return buf[offset : offset + rows * stride].view(rows, stride)[:, :row_bytes]


def test_ops_registered_without_cuda_init():
    C = _lcpp()
    assert all(hasattr(C, op) for op in OPS)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("name", list(CASES))
def test_guard(name):
    C = _lcpp()
    c, want = CASES[name]
    w = _w(
        c["rows"],
        c["row_bytes"],
        c.get("stride"),
        c.get("w_offset", 0),
        getattr(torch, c.get("w_dtype", "uint8")),
    )
    if c.get("w_t"):
        w = w.t()
    if c.get("w_1d"):
        w = w.reshape(-1)
    x = torch.zeros(c["n"], c["k"], dtype=getattr(torch, c.get("x_dtype", "bfloat16")))
    if c.get("x_t"):
        x = torch.zeros(c["k"], c["n"], dtype=x.dtype).t()
    with pytest.raises(RuntimeError) as e:
        getattr(C, c["op"])(w, x, c["type"], c["row"])
    assert want in str(e.value), str(e.value)[:300]
    assert not torch.cuda.is_initialized()
