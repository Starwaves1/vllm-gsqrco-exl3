# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CUDA dequantize kernels, compiled for the host (tests/dequant_host.py),
against gguf-py on the sample GGUFs, on the CPU.

The IQ kernels compute in fp32 as ggml does and are bit-exact. The legacy
(Q4_0 ... Q8_0) and K-quant kernels compute in fp16 (__hmul / __hsub on half,
__int2half_rn(sc * q)) where ggml uses fp32: they are not bit-exact (strict
xfail), and test_fp16_dequant_error_bound bounds the difference.
"""

import shutil

import gguf
import numpy as np
import pytest
import torch

from .dequant_host import build, dequantize

T = gguf.GGMLQuantizationType
FP16_TYPES = ["Q4_0", "Q5_0", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]
EXACT_TYPES = [
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ4_NL",
    "IQ4_XS",
]
XFAIL = pytest.mark.xfail(strict=True, reason="dequantize.cuh computes in fp16")
TYPES = EXACT_TYPES + [pytest.param(t, marks=XFAIL) for t in FP16_TYPES]


@pytest.fixture(scope="module")
def lib():
    if shutil.which("g++") is None:
        pytest.skip("needs g++")
    return build()


def _samples(name):
    from .utils import get_gguf_sample_tensors

    qt = T[name]
    for t in get_gguf_sample_tensors(256, qt)[:2]:  # 768 and 1024 rows of K 256
        raw = np.ascontiguousarray(t.data)
        yield raw, gguf.quants.dequantize(raw, qt).astype(np.float32)


@pytest.mark.parametrize("name", TYPES)
def test_fp32_bit_exact(lib, name):
    for raw, ref in _samples(name):
        y = dequantize(lib, raw, int(T[name]), ref.size, "float32")
        np.testing.assert_array_equal(
            y.view(np.uint32), ref.reshape(-1).view(np.uint32)
        )


@pytest.mark.parametrize("name", TYPES)
def test_bf16_is_rounded_reference(lib, name):
    for raw, ref in _samples(name):
        y = dequantize(lib, raw, int(T[name]), ref.size, "bfloat16")
        want = torch.from_numpy(ref.reshape(-1)).bfloat16().view(torch.int16).numpy()
        np.testing.assert_array_equal(y, want.view(np.uint16))


@pytest.mark.parametrize("name", FP16_TYPES)
def test_fp16_dequant_error_bound(lib, name):
    """At most 2**-8 of the row's largest magnitude. To first order the fp16
    roundings (d * sc, dmin * m, the product, the difference) add at most
    3 * 2**-11 * (|y| + |dmin * m|), and the min offset dmin * m can reach
    about the row's largest magnitude. Measured: up to 1.04 * 2**-9 (Q5_K)."""
    for raw, ref in _samples(name):
        y = dequantize(lib, raw, int(T[name]), ref.size, "float32").reshape(ref.shape)
        err = np.abs(y.astype(np.float64) - ref).max(axis=1)
        bound = 2.0**-8 * np.abs(ref).max(axis=1)
        assert (err <= bound).all(), (err / bound).max()
