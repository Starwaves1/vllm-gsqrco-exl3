# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CUDA kernels against CPU reference models (tests/kernel_refs.py), per
quant type, with tolerances calibrated on an RTX 3090.

ggml_dequantize against gguf-py, MMVQ (ggml_mul_mat_vec_a8) at 1..16 rows,
MMQ (ggml_mul_mat_a8) at 16..512 rows, and the routing function linear.py
calls (_fused_mul_mat_gguf: MMVQ, MMQ or dequantize + x @ W.T by type and
row count) on whole tensors. The types are the ten the tolerances were
calibrated on; weights are blocks of the sample GGUFs (tests/utils.py).
"""

import gguf
import pytest
import torch

from .kernel_refs import (
    LOOSE,
    LOOSE_XSUM,
    check,
    dequant,
    make_x,
    poison_cuda_allocator,
    rel_err,
    sample_weight,
)

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

QUANT_TYPES = [
    "IQ1_M",
    "IQ2_S",
    "IQ2_XS",
    "IQ2_XXS",
    "IQ3_S",
    "IQ3_XXS",
    "IQ4_XS",
    "Q2_K",
    "Q4_K",
    "Q6_K",
]
ROWS, BLOCKS = 512, 20  # weight rows and K / 256 per kernel test
MMVQ_TOKENS = [1, 2, 3, 4, 8, 16]
MMQ_TOKENS = [16, 64, 512]
ROUTE_TOKENS = [1, 4, 8, 9, 16, 32, 128, 512]
# the dequantize kernels compute legacy and K-quants in fp16, ggml in fp32
FP16_DEQUANT = ("Q2_K", "Q4_K", "Q6_K")  # of QUANT_TYPES


def _qt(name: str) -> int:
    return int(gguf.GGMLQuantizationType[name])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str
)
@pytest.mark.parametrize("name", QUANT_TYPES)
def test_dequantize(name, dtype):
    from vllm_gguf_plugin import ops

    raw = sample_weight(name, ROWS, BLOCKS)
    ref = dequant(raw, name)
    w = torch.from_numpy(raw).cuda()
    poison_cuda_allocator()
    out = ops.ggml_dequantize(w, _qt(name), *ref.shape, dtype).cpu()
    if name in FP16_DEQUANT:
        # not bit-exact: bounded by 2**-8 of the row's absmax, plus half an
        # output ulp for the final rounding (tests/test_dequant_host.py)
        tol = 2.0**-8 * ref.abs().amax(dim=1, keepdim=True)
        tol = tol + torch.finfo(out.dtype).eps / 2 * ref.abs()
        assert ((out.double() - ref).abs() <= tol).all()
    elif dtype == torch.float32:
        torch.testing.assert_close(out, ref.float(), rtol=2e-7, atol=0)
    else:
        torch.testing.assert_close(out, ref.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("n", MMVQ_TOKENS)
@pytest.mark.parametrize("name", QUANT_TYPES)
def test_mmvq(name, n, dtype):
    from vllm_gguf_plugin import ops

    raw = sample_weight(name, ROWS, BLOCKS)
    x = make_x(n, BLOCKS * 256, dtype, seed=n)
    w = torch.from_numpy(raw).cuda()
    y = ops.ggml_mul_mat_vec_a8(w, x.cuda(), _qt(name), w.shape[0])
    assert y.shape == (n, ROWS) and y.dtype == dtype
    check(y, raw, name, x, mmq=False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("n", MMQ_TOKENS)
@pytest.mark.parametrize("name", [q for q in QUANT_TYPES if not q.startswith("IQ")])
def test_mmq(name, n, dtype):
    from vllm_gguf_plugin import ops

    raw = sample_weight(name, ROWS, BLOCKS)
    x = make_x(n, BLOCKS * 256, dtype, seed=100 + n)
    w = torch.from_numpy(raw).cuda()
    y = ops.ggml_mul_mat_a8(w, x.cuda(), _qt(name), w.shape[0])
    check(y, raw, name, x, mmq=True)


@pytest.mark.parametrize("rows", [2048, 18944])  # MMVQ limits differ above 5120
@pytest.mark.parametrize("n", ROUTE_TOKENS)
@pytest.mark.parametrize("name", QUANT_TYPES)
def test_routing_whole_tensor(name, n, rows):
    """_fused_mul_mat_gguf on a whole weight, against x @ W.T in float64 with
    W from the kernel's own dequantize (checked above): a CPU dequant per case
    would dominate the run time."""
    from vllm_gguf_plugin import ops
    from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf

    raw = sample_weight(name, rows, BLOCKS)
    w = torch.from_numpy(raw).cuda()
    x = make_x(n, BLOCKS * 256, torch.bfloat16, seed=200 + n).cuda()
    y = _fused_mul_mat_gguf(x, w, _qt(name))
    W = ops.ggml_dequantize(w, _qt(name), rows, BLOCKS * 256, torch.float32)
    ref = (x.double() @ W.double().T).cpu()
    assert rel_err(y, ref) <= (LOOSE_XSUM if name in ("Q4_K", "Q5_K") else LOOSE)
