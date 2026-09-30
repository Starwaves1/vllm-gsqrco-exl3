# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the CUDA dequantize kernels (csrc/gguf/dequantize.cuh) on the CPU.

Compiles ggml-common.h + dequantize.cuh unmodified with g++, except that each
`kernel<<<grid, block, 0, stream>>>(args);` launch becomes a serial loop over
blockIdx.x / threadIdx.x. A small shim supplies the CUDA types the kernels use:
`half` = _Float16 with one correctly rounded operation per __hmul / __hsub /
... (as the CUDA intrinsics, which are never contracted into an FMA),
c10::BFloat16 with round-to-nearest-even, and uint3 blockIdx / threadIdx.
Built with -ffp-contract=off.

The real build uses --use_fast_math (FTZ, fmad). The dequantize kernels have
no float multiply-add expressions and real weights produce no denormals there,
so the host result is expected to equal the GPU's; that is inferred from the
code, not measured.
"""

import ctypes
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

CSRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vllm_gguf_plugin/csrc/gguf",
)

SHIM = r"""
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#define __device__
#define __global__
#define __forceinline__ inline
typedef void * cudaStream_t;
struct uint3_ { unsigned int x = 0, y = 0, z = 0; };
static uint3_ threadIdx, blockIdx, blockDim;

typedef _Float16 half;
struct half2 { half x, y; };
// one round-to-nearest-even rounding each, as the CUDA intrinsics
static inline half __double2half_rn(double v) { return (half)v; }
static inline float __half2float(half h) { return (float)h; }
static inline half __int2half_rn(int i) { return (half)(double)i; }
static inline half __hmul(half a, half b) { return (half)((double)a * b); }
static inline half __hadd(half a, half b) { return (half)((double)a + b); }
static inline half __hsub(half a, half b) { return (half)((double)a - b); }
static inline half __low2half(half2 h) { return h.x; }
static inline half __high2half(half2 h) { return h.y; }
static inline float __low2float(half2 h) { return (float)h.x; }
static inline half2 __floats2half2_rn(float a, float b) {
  return {(half)a, (half)b};
}
static inline half2 __hmul2(half2 a, half2 b) {
  return {__hmul(a.x, b.x), __hmul(a.y, b.y)};
}
static inline half2 __hadd2(half2 a, half2 b) {
  return {__hadd(a.x, b.x), __hadd(a.y, b.y)};
}
static inline half2 __hsub2(half2 a, half2 b) {
  return {__hsub(a.x, b.x), __hsub(a.y, b.y)};
}

namespace c10 {
struct BFloat16 {
  uint16_t x;
  BFloat16() = default;
  BFloat16(float f) {  // c10::detail::round_to_nearest_even
    uint32_t u; std::memcpy(&u, &f, 4);
    if (std::isnan(f)) { x = 0x7fc0; return; }
    u += 0x7fffu + ((u >> 16) & 1u);
    x = (uint16_t)(u >> 16);
  }
};
}  // namespace c10
"""

WRAPPER = r"""
template <typename T>
static int run(int type, const void* x, void* y, int64_t k) {
  auto f = ggml_get_to_cuda<T>(type);
  if (!f) return -1;
  f(x, (T*)y, k, nullptr);
  return 0;
}
extern "C" int host_dequant(int type, const void* x, void* y, int64_t k,
                            int dst) {
  if (dst == 0) return run<float>(type, x, y, k);
  if (dst == 1) return run<c10::BFloat16>(type, x, y, k);
  if (dst == 2) return run<half>(type, x, y, k);
  return -2;
}
"""

_LAUNCH = re.compile(
    r"^(\s*)(.+?)<<<\s*([^,]+?)\s*,\s*([^,]+?)\s*,[^>]*>>>\s*\((.*)\);\s*$", re.M
)


def _rewrite_launches(src: str) -> str:
    def sub(m):
        ind, kern, grid, block, args = m.groups()
        return (
            f"{ind}{{ const int64_t _g = ({grid}); const unsigned _b = ({block});"
            f" blockDim.x = _b; for (int64_t _i = 0; _i < _g; ++_i) {{"
            f" blockIdx.x = (unsigned)_i; for (unsigned _t = 0; _t < _b; ++_t) {{"
            f" threadIdx.x = _t; {kern}({args}); }} }} }}"
        )

    out, n = _LAUNCH.subn(sub, src)
    if "<<<" in out or n == 0:
        raise RuntimeError("unrewritten CUDA launch left in dequantize.cuh")
    return out


def build(cxx: str = "g++") -> ctypes.CDLL:
    with open(os.path.join(CSRC, "ggml-common.h")) as f:
        common = f.read()
    with open(os.path.join(CSRC, "dequantize.cuh")) as f:
        deq = _rewrite_launches(f.read())
    d = tempfile.mkdtemp(prefix="gguf-dequant-host-")
    try:
        cpp, so = os.path.join(d, "deq.cpp"), os.path.join(d, "deq.so")
        with open(cpp, "w") as f:
            f.write(SHIM + "\n" + common + "\n" + deq + "\n" + WRAPPER)
        subprocess.run(
            [
                cxx,
                "-std=c++17",
                "-O1",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-w",
                "-shared",
                "-fPIC",
                "-o",
                so,
                cpp,
            ],
            check=True,
        )
        lib = ctypes.CDLL(so)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    lib.host_dequant.restype = ctypes.c_int
    lib.host_dequant.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    return lib


def dequantize(lib, raw: np.ndarray, ggml_type: int, n: int, dst: str) -> np.ndarray:
    """dst: float32 | bfloat16 (returned as uint16 bit patterns) | float16."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    code = {"float32": 0, "bfloat16": 1, "float16": 2}[dst]
    out = np.zeros(n, dtype={0: np.float32, 1: np.uint16, 2: np.float16}[code])
    rc = lib.host_dequant(int(ggml_type), raw.ctypes.data, out.ctypes.data, n, code)
    if rc != 0:
        raise ValueError(f"host_dequant rc={rc} for type {ggml_type}")
    return out
