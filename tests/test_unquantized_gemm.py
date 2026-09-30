# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF F32/F16/BF16 linears: GGUFUnquantizedLinearMethod and _unquantized_gemm,
which runs a weight of at most 128 rows at up to 8 activation rows as a batched
gemv and everything else as F.linear."""

import pytest
import torch
import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod

from vllm_gguf_plugin.quantization.config import GGUFConfig
from vllm_gguf_plugin.quantization.linear import (
    GGUFUnquantizedLinearMethod,
    _unquantized_gemm,
)


def test_skipped_layers_get_the_gguf_unquantized_method(monkeypatch):
    for module in (linear_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    layer = ReplicatedLinear(64, 96, bias=False)
    method = GGUFConfig(["model.layers.0.in_proj_ba"]).get_quant_method(
        layer, "model.layers.0.in_proj_ba"
    )
    assert isinstance(method, GGUFUnquantizedLinearMethod)
    assert isinstance(method, UnquantizedLinearMethod)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 9])
def test_unquantized_gemm_cpu(n):
    """Same product as F.linear; F.linear itself above 8 rows, above 128
    weight rows and with a bias."""
    g = torch.Generator().manual_seed(n)
    w = torch.randn(96, 512, generator=g)
    x = torch.randn(n, 512, generator=g)
    y = _unquantized_gemm(x, w)
    assert y.shape == (n, 96) and y.is_contiguous()
    torch.testing.assert_close(
        y, torch.nn.functional.linear(x, w), rtol=1e-5, atol=1e-4
    )
    big = torch.cat([w, w])  # 192 rows
    assert torch.equal(_unquantized_gemm(x, big), torch.nn.functional.linear(x, big))
    bias = torch.randn(96, generator=g)
    assert torch.equal(
        _unquantized_gemm(x, w, bias), torch.nn.functional.linear(x, w, bias)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("n", [1, 2, 4, 8, 9])
def test_unquantized_gemm_cuda(n):
    """A 96 x 5120 bf16 weight (the shape of Qwen3.5's GDN in_proj_ba): the gemv
    (<= 8 rows) and F.linear (9) both accumulate in fp32, so both are within
    bf16 output rounding of the fp64 product; the result is a contiguous
    [n, 96] bf16 tensor."""
    g = torch.Generator().manual_seed(n)
    w = (torch.randn(96, 5120, generator=g) * 0.02).bfloat16().cuda()
    x = torch.randn(n, 5120, generator=g).bfloat16().cuda()
    y = _unquantized_gemm(x, w)
    ref = x.double() @ w.double().T
    assert y.shape == (n, 96) and y.dtype == torch.bfloat16 and y.is_contiguous()
    err = ((y.double() - ref).norm(dim=-1) / ref.norm(dim=-1)).max().item()
    assert err <= 4e-3
    big = torch.cat([w, w])
    assert torch.equal(_unquantized_gemm(x, big), torch.nn.functional.linear(x, big))
