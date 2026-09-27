"""Bit-exact dequant reference from llama.cpp b11211's own C code (libggml-base.so).

Calls `ggml_get_type_traits(type)->to_float` through ctypes. libggml-base is
CPU-only (no CUDA, no FMA-specific paths), so this is the C reference that
gguf-py's numpy dequant must match. No GPU involvement.
"""

import ctypes
import os

import numpy as np

LIB = os.environ.get(
    "GGML_BASE_LIB", os.path.expanduser("~/llama.cpp-b11211/build/bin/libggml-base.so")
)

_to_float_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)


class _Traits(ctypes.Structure):  # ggml.h: struct ggml_type_traits
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


_lib = None


def _load():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(LIB)
        _lib.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
        _lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    return _lib


def traits(ggml_type: int) -> _Traits:
    return _load().ggml_get_type_traits(int(ggml_type)).contents


def dequantize(raw: np.ndarray, ggml_type: int, n_elements: int) -> np.ndarray:
    """raw: contiguous uint8 bytes holding n_elements of ggml_type. Returns float32."""
    t = traits(ggml_type)
    assert n_elements % t.blck_size == 0
    assert raw.nbytes == n_elements // t.blck_size * t.type_size, (raw.nbytes, t.type_size)
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    out = np.empty(n_elements, dtype=np.float32)
    _to_float_t(t.to_float)(raw.ctypes.data, out.ctypes.data, n_elements)
    return out
