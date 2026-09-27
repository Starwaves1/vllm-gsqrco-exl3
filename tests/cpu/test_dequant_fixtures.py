"""Dequant of the model's 10 quantized ggml types, bit-exact against gguf-py.

Fixtures: tests/fixtures/dequant/*.npz (tools/make_dequant_fixtures.py), real
rows from the Swift GSQ-RCO IQ3_S-mtp GGUF. References checked here:
  * gguf-py 0.19.0 (== llama.cpp b11211 gguf-py): the fixture's `ref`;
  * libggml-base.so from b11211 (C `to_float`);
Plugin code under test (both run on the CPU):
  * the Triton dequant kernels (triton/dequantize), in TRITON_INTERPRET mode;
  * the CUDA dequant kernels (csrc/gguf/dequantize.cuh) compiled for the host
    by tools/cuda_dequant_host.py. This is the path linear.py uses for IQ
    weights above the MMVQ batch limit, and for embeddings.
"""

import glob
import json
import os

import numpy as np
import pytest
import torch
from conftest import FIXTURES
from gguf import GGMLQuantizationType as T
from gguf.quants import dequantize as gguf_dequantize

import cuda_dequant_host
import ggml_ref

FILES = sorted(glob.glob(os.path.join(FIXTURES, "*.npz")))
IDS = [os.path.basename(f).removesuffix(".npz") for f in FILES]
K_QUANTS = {int(T.Q2_K), int(T.Q3_K), int(T.Q4_K), int(T.Q5_K), int(T.Q6_K)}
MODEL_TYPES = {"IQ1_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S", "IQ3_XXS", "IQ4_XS", "Q2_K", "Q4_K", "Q6_K"}


def _load(path):
    z = np.load(path)
    return z["raw"], z["ref"], int(z["ggml_type"]), str(z["tensor"])


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint32 if a.dtype == np.float32 else np.uint16)


def _bf16_bits(x: np.ndarray) -> np.ndarray:
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def test_manifest_covers_all_model_types():
    m = json.load(open(os.path.join(FIXTURES, "manifest.json")))
    assert set(m["types"]) == MODEL_TYPES
    assert {f["type"] for f in m["fixtures"]} == MODEL_TYPES
    assert len(FILES) == len(m["fixtures"])


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_fixture_ref_is_gguf_py(path):
    raw, ref, qt, _ = _load(path)
    np.testing.assert_array_equal(_bits(gguf_dequantize(raw, T(qt)).reshape(-1)), _bits(ref))


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_libggml_matches_gguf_py(path):
    raw, ref, qt, _ = _load(path)
    np.testing.assert_array_equal(_bits(ggml_ref.dequantize(raw, qt, ref.size)), _bits(ref))


# ---------------------------------------------------------------- Triton path
@pytest.fixture(scope="module")
def triton_dequant():
    from vllm_gguf_plugin.triton.dequantize import interface, utils

    orig = utils.validate_dequant_args

    def cpu_validate(W, quant_type, m, n, dtype):
        # The real check also rejects non-CUDA tensors; interpreter mode runs on CPU.
        meta = torch.empty(W.shape, dtype=W.dtype, device="meta")
        try:
            orig(meta, quant_type, m, n, dtype)
        except ValueError as e:
            if "CUDA" not in str(e):
                raise
        return W.contiguous(), int(m) * int(n), dtype or torch.float16

    utils.validate_dequant_args = cpu_validate
    yield interface.ggml_dequantize_triton
    utils.validate_dequant_args = orig


@pytest.mark.parametrize("path", FILES, ids=IDS)
def test_triton_dequant_fp32_bit_exact(path, triton_dequant):
    raw, ref, qt, _ = _load(path)
    y = triton_dequant(torch.from_numpy(raw.copy()), qt, 1, ref.size, torch.float32)
    np.testing.assert_array_equal(_bits(y.numpy().reshape(-1)), _bits(ref))


# No bf16 Triton test: the interpreter's implicit fp32->bf16 store truncates
# (triton/runtime/interpreter.py _convert_float with rounding_mode None), while
# compiled Triton rounds to nearest even. bf16 output would test the interpreter.


# ------------------------------------------------------------------ CUDA path
@pytest.fixture(scope="module")
def cuda_host():
    return cuda_dequant_host.build()


def _kquant_xfail(path):
    qt = int(np.load(path)["ggml_type"])
    if qt in K_QUANTS:
        return pytest.param(path, marks=pytest.mark.xfail(
            strict=True,
            reason="dequantize.cuh does K-quant math in fp16 (__hmul/__hsub on half), "
                   "gguf-py/ggml in fp32",
        ))
    return path


@pytest.mark.parametrize("path", [_kquant_xfail(f) for f in FILES], ids=IDS)
def test_cuda_dequant_fp32_bit_exact(path, cuda_host):
    raw, ref, qt, _ = _load(path)
    y = cuda_dequant_host.dequantize(cuda_host, raw, qt, ref.size, "float32")
    np.testing.assert_array_equal(_bits(y), _bits(ref))


@pytest.mark.parametrize("path", [_kquant_xfail(f) for f in FILES], ids=IDS)
def test_cuda_dequant_bf16_is_rounded_ref(path, cuda_host):
    raw, ref, qt, _ = _load(path)
    y = cuda_dequant_host.dequantize(cuda_host, raw, qt, ref.size, "bfloat16")
    np.testing.assert_array_equal(y, _bf16_bits(ref))


@pytest.mark.parametrize("path", [f for f in FILES if int(np.load(f)["ggml_type"]) in K_QUANTS],
                         ids=[i for f, i in zip(FILES, IDS) if int(np.load(f)["ggml_type"]) in K_QUANTS])
def test_cuda_kquant_dequant_error_is_fp16_sized(path, cuda_host):
    """The fp16 K-quant path is not bit-exact. Bound it at 2**-9 of the row's
    largest magnitude (about half a bf16 ulp at the top of the row): the fp16
    roundings of d*sc, dmin*m, the product and the difference each add up to
    2**-11 of intermediates that can exceed the output (the min offset)."""
    raw, ref, qt, _ = _load(path)
    y = cuda_dequant_host.dequantize(cuda_host, raw, qt, ref.size, "float32")
    err = np.abs(y.astype(np.float64) - ref)
    assert err.max() <= 2.0**-9 * np.abs(ref).max(), err.max() / np.abs(ref).max()
