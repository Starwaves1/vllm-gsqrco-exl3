# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Input checks of the _C_gguf ops, on the CPU.

The ops are registered for CPU tensors too, where they run every shape, dtype,
stride and alignment check and then reject the call with "must be CUDA
tensors" (or "must be a CUDA tensor"). So a CPU call that ends in that message
passed all checks, and any other message names the check that fired. No kernel
runs and CUDA is never initialised.
"""

import pytest
import torch

from vllm_gguf_plugin import ops

IQ3_S, Q4_K, Q6_K, IQ4_XS, F16, IQ1_M = 21, 12, 14, 23, 1, 29
TS = {IQ3_S: 110, Q4_K: 144, Q6_K: 210, IQ4_XS: 136, IQ1_M: 56}  # bytes / 256 values
K = 1024
CUDA_ONLY = ("must be CUDA tensors", "must be a CUDA tensor")


def _registered(name: str) -> bool:
    return ops._CUDA_AVAILABLE and hasattr(torch.ops._C_gguf, name)


def _w(rows, row_bytes, stride=None, offset=0, dtype=torch.uint8):
    """[rows, row_bytes] view of a zeroed buffer: row stride `stride`, starting
    `offset` elements into the buffer (the buffer itself is 64-byte aligned)."""
    stride = stride or row_bytes
    buf = torch.zeros(rows * stride + offset + 64, dtype=dtype)
    return buf[offset : offset + rows * stride].view(rows, stride)[:, :row_bytes]


def _mm_case(t=Q4_K, rows=256, row_bytes=None, n=4, k=K, row=None, **kw):
    return dict(
        t=t,
        rows=rows,
        row_bytes=row_bytes or k // 256 * TS[t],
        n=n,
        k=k,
        row=rows if row is None else row,
        **kw,
    )


MM_CASES = {  # name -> (case, expected message fragment)
    "valid": (_mm_case(), CUDA_ONLY),
    "valid_iq3_s": (_mm_case(IQ3_S), CUDA_ONLY),
    "fp32_x": (_mm_case(x_dtype=torch.float32), CUDA_ONLY),
    "unsupported_type": (_mm_case(t=F16, row_bytes=2 * K), ("unsupported ggml type",)),
    "w_float": (_mm_case(w_dtype=torch.float32), ("W must be uint8",)),
    "w_1d": (_mm_case(w_1d=True), ("must be 2-D",)),
    "x_int": (_mm_case(x_dtype=torch.int32), ("X must be fp32, fp16 or bf16",)),
    "row_bytes_not_blocks": (_mm_case(row_bytes=577), ("not a multiple of the block",)),
    "k_mismatch": (
        _mm_case(k=K // 2, row_bytes=K // 256 * 144),
        ("columns, W rows hold",),
    ),
    "row_too_big": (_mm_case(row=256 + 64), ("out of range",)),
    "row_zero": (_mm_case(row=0), ("out of range",)),
    "w_narrow_view": (
        _mm_case(stride=K // 256 * 144 + 256),
        ("W rows must be contiguous",),
    ),
    "w_transposed": (_mm_case(w_t=True), ("W rows must be contiguous",)),
    "w_misaligned": (_mm_case(w_offset=1), ("16-byte aligned",)),
    "x_noncontig": (_mm_case(x_t=True), ("X rows must be contiguous",)),
    "x_rowstride": (_mm_case(x_stride=K + 64), ("X rows must be contiguous",)),
    "one_row_x": (_mm_case(n=1), CUDA_ONLY),
}


def _mm_inputs(c):
    w = _w(
        c["rows"],
        c["row_bytes"],
        c.get("stride"),
        c.get("w_offset", 0),
        c.get("w_dtype", torch.uint8),
    )
    if c.get("w_t"):
        w = torch.zeros(c["row_bytes"], c["rows"], dtype=torch.uint8).t()
    if c.get("w_1d"):
        w = w.reshape(-1)
    dtype = c.get("x_dtype", torch.bfloat16)
    x = torch.zeros(c["n"], c.get("x_stride", c["k"]), dtype=dtype)[:, : c["k"]]
    if c.get("x_t"):
        x = torch.zeros(c["k"], c["n"], dtype=dtype).t()
    return w, x


@pytest.mark.parametrize("op", ["ggml_mul_mat_vec_a8", "ggml_mul_mat_a8"])
@pytest.mark.parametrize("name", list(MM_CASES))
def test_mul_mat_input_checks(op, name):
    if not _registered(op):
        pytest.skip("_C_gguf not built")
    case, want = MM_CASES[name]
    if op == "ggml_mul_mat_a8" and case["t"] == IQ3_S:
        want = ("unsupported ggml type",)  # no IQ MMQ kernel
    w, x = _mm_inputs(case)
    with pytest.raises(RuntimeError) as e:
        getattr(torch.ops._C_gguf, op)(w, x, case["t"], case["row"])
    assert any(m in str(e.value) for m in want), str(e.value)[:300]
    assert not torch.cuda.is_initialized()


def test_mul_mat_vec_row_limit():
    if not _registered("ggml_mul_mat_vec_a8"):
        pytest.skip("_C_gguf not built")
    w, x = _mm_inputs(_mm_case(n=65536, k=256))
    with pytest.raises(RuntimeError, match="at most 65535 rows"):
        torch.ops._C_gguf.ggml_mul_mat_vec_a8(w, x, Q4_K, w.shape[0])


DQ_CASES = {  # name -> (W, type, m, n, dtype, expected message fragment)
    "valid": (lambda: _w(4, 4 * 144), Q4_K, 4, K, None, CUDA_ONLY),
    # the embedding path: m = hidden size, n = tokens (any count)
    "valid_embedding": (lambda: _w(3, 4 * 144), Q4_K, K, 3, None, CUDA_ONLY),
    "valid_3d": (lambda: _w(4, 4 * 144).view(2, 2, -1), Q4_K, 4, K, None, CUDA_ONLY),
    "unsupported_type": (
        lambda: _w(4, 2 * K),
        F16,
        4,
        K,
        None,
        ("unsupported ggml type",),
    ),
    "n_not_blocks": (
        lambda: _w(4, 4 * 144),
        Q4_K,
        4,
        K - 32,
        None,
        ("must be a multiple",),
    ),
    "w_too_small": (lambda: _w(3, 4 * 144), Q4_K, 4, K, None, ("bytes",)),
    "w_strided": (
        lambda: _w(4, 4 * 144, stride=4 * 144 + 16),
        Q4_K,
        4,
        K,
        None,
        ("must be contiguous",),
    ),
    "w_misaligned": (
        lambda: _w(4, 4 * 144, offset=1),
        Q4_K,
        4,
        K,
        None,
        ("16-byte aligned",),
    ),
    "w_float": (
        lambda: _w(4, 4 * 144, dtype=torch.float32),
        Q4_K,
        4,
        K,
        None,
        ("W must be uint8",),
    ),
    "dtype_int": (lambda: _w(4, 4 * 144), Q4_K, 4, K, torch.int32, ("dtype must be",)),
}


@pytest.mark.parametrize("name", list(DQ_CASES))
def test_dequantize_input_checks(name):
    if not _registered("ggml_dequantize"):
        pytest.skip("_C_gguf not built")
    w, t, m, n, dtype, want = DQ_CASES[name]
    with pytest.raises(RuntimeError) as e:
        torch.ops._C_gguf.ggml_dequantize(w(), t, m, n, dtype)
    assert any(s in str(e.value) for s in want), str(e.value)[:300]
    assert not torch.cuda.is_initialized()
