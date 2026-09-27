"""The plugin's CUDA lookup tables (csrc/gguf/ggml-common.h, from llama.cpp
b2899 plus e2b8ad5's IQ3_S fix) against llama.cpp b11211's ggml-common.h and
gguf-py 0.19.0, which also feeds the plugin's Triton tables (iq_tables.py).
"""

import os
import re

import numpy as np
import pytest
from conftest import LLAMA_CPP, ROOT
from gguf.quants import IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL

PLUGIN_H = os.path.join(ROOT, "plugin/vllm_gguf_plugin/csrc/gguf/ggml-common.h")
B11211_H = os.path.join(LLAMA_CPP, "ggml/src/ggml-common.h")


def _parse(path, pattern):
    src = open(path).read()
    src = re.sub(r"//[^\n]*", "", src)
    out = {}
    for ctype, name, size, body in re.findall(pattern, src, re.S):
        vals = [int(v, 0) for v in re.findall(r"-?(?:0x[0-9a-fA-F]+|\d+)", body)]
        out[name] = (ctype, size, vals)
    return out


@pytest.fixture(scope="module")
def plugin():
    return _parse(PLUGIN_H, r"static const __device__ (\w+) (\w+)\[(\w+)\] = \{(.*?)\};")


@pytest.fixture(scope="module")
def b11211():
    return _parse(B11211_H, r"GGML_TABLE_BEGIN\((\w+), (\w+), (\w+)\)(.*?)GGML_TABLE_END\(\)")


def _bytes(vals, dtype):
    return np.array(vals, dtype=dtype).view(np.uint8).reshape(len(vals), -1)


def _grid(cls):
    cls.init_grid()
    return cls.grid[0, 0].astype(np.int64).reshape(cls.grid_shape)


@pytest.mark.parametrize("name", ["iq2xxs_grid", "iq2xs_grid", "iq2s_grid", "iq3xxs_grid",
                                  "ksigns_iq2xs", "ksigns64", "kmask_iq2xs", "kvalues_iq4nl"])
def test_tables_equal_b11211(plugin, b11211, name):
    assert len(plugin[name][2]) == int(plugin[name][1]) > 0
    assert plugin[name][2] == b11211[name][2]


def test_iq3xs_grid_is_4x_b11211_iq3s_grid(plugin, b11211):
    """The plugin keeps the old 4x-scaled IQ3_S grid and compensates with
    (0.5 + s) * 0.5 instead of b11211's (1 + 2 s); 4*(0.5+s)*0.5 == 1 + 2 s."""
    cuda = _bytes(plugin["iq3xs_grid"][2], np.uint32).astype(np.int64)
    ref = _bytes(b11211["iq3s_grid"][2], np.uint32).astype(np.int64)
    assert cuda.shape == ref.shape == (512, 4)
    bad = np.argwhere(cuda != 4 * ref)
    assert bad.size == 0, f"{len(bad)} entries differ, first {bad[:4].tolist()}"
    assert int(cuda.max()) <= 127  # still a valid int8 for dp4a


def test_iq1s_grid_gpu_truncation(plugin, b11211):
    """iq1s_grid_gpu is declared uint64 in the plugin but read into uint32
    (dequantize.cuh/vecdotq.cuh); every entry must fit in 32 bits and equal
    b11211's uint32 table."""
    vals = plugin["iq1s_grid_gpu"][2]
    assert plugin["iq1s_grid_gpu"][0] == "uint64_t" and len(vals) == 2048
    assert max(vals) < 2**32
    assert [v & 0xFFFFFFFF for v in vals] == b11211["iq1s_grid_gpu"][2]


def test_iq1s_grid_gpu_decodes_to_gguf_py_grid(plugin):
    """grid32 & 0x0f0f0f0f gives elements 0..3, (grid32 >> 4) & 0x0f0f0f0f gives 4..7;
    each is the IQ1 grid value + 1 (0/1/2 for -1/0/+1)."""
    g = np.array(plugin["iq1s_grid_gpu"][2], dtype=np.uint64).astype(np.uint32)
    lo = (g & 0x0F0F0F0F).view(np.uint8).reshape(-1, 4)
    hi = ((g >> 4) & 0x0F0F0F0F).view(np.uint8).reshape(-1, 4)
    dec = np.concatenate([lo, hi], axis=1).astype(np.int64) - 1
    np.testing.assert_array_equal(dec, _grid(IQ1_S))


@pytest.mark.parametrize("name,cls,dtype,scale", [
    ("iq2xxs_grid", IQ2_XXS, np.uint64, 1),
    ("iq2xs_grid", IQ2_XS, np.uint64, 1),
    ("iq2s_grid", IQ2_S, np.uint64, 1),
    ("iq3xxs_grid", IQ3_XXS, np.uint32, 1),
    ("iq3xs_grid", IQ3_S, np.uint32, 4),
])
def test_grids_equal_gguf_py(plugin, name, cls, dtype, scale):
    np.testing.assert_array_equal(_bytes(plugin[name][2], dtype).astype(np.int64), scale * _grid(cls))


def test_ksigns_and_kvalues_equal_gguf_py(plugin):
    assert plugin["ksigns_iq2xs"][2] == list(IQ2_XXS.ksigns)
    assert plugin["kvalues_iq4nl"][2] == list(IQ4_NL.kvalues)
    # ksigns64 is ksigns_iq2xs expanded to one 0x00/0xff byte per sign bit.
    k64 = _bytes(plugin["ksigns64"][2], np.uint64)
    bits = (np.array(plugin["ksigns_iq2xs"][2])[:, None] >> np.arange(8)) & 1
    np.testing.assert_array_equal(k64, bits * 0xFF)


def test_triton_tables_are_gguf_py():
    from vllm_gguf_plugin.triton.gemm.iq_quant.iq_tables import _cpu_iq_tables

    t = _cpu_iq_tables()
    for key, cls in [("iq2xxs_grid", IQ2_XXS), ("iq2xs_grid", IQ2_XS), ("iq2s_grid", IQ2_S),
                     ("iq3xxs_grid", IQ3_XXS), ("iq3s_grid", IQ3_S)]:
        np.testing.assert_array_equal(t[key].reshape(-1).astype(np.int64), _grid(cls).reshape(-1))
    np.testing.assert_array_equal(t["iq1s_grid"].reshape(-1).astype(np.int64), _grid(IQ1_S).reshape(-1) + 1)
