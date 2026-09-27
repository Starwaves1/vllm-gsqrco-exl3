"""Run the plugin's CUDA dequant kernels (csrc/gguf/dequantize.cuh) on the CPU.

Compiles ggml-common.h + dequantize.cuh unmodified, except that each
`kernel<<<grid, block, 0, stream>>>(args);` launch is rewritten into a serial
loop over blockIdx.x/threadIdx.x. A small shim supplies the CUDA types the
kernels use: `half` = _Float16 with one correctly rounded operation per
__hmul/__hsub/... (same as the CUDA intrinsics, which are documented as never
contracted into FMA), `c10::BFloat16` with round-to-nearest-even, and
uint3 blockIdx/threadIdx. Built with -ffp-contract=off.

Deviation from the real build: nvcc uses --use_fast_math (FTZ, fmad). The
dequant kernels have no mul+add float expressions on these paths and produce no
denormals for real weights, so the host result is expected to equal the GPU's;
that equality is INFERRED, not measured (no GPU use in Phase A).

No GPU, no CUDA headers or libraries: plain g++.
"""

import ctypes
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSRC = os.path.join(ROOT, "plugin/vllm_gguf_plugin/csrc/gguf")

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
static inline half  __double2half_rn(double v) { return (half)v; }  // one RNE rounding
static inline float __half2float(half h) { return (float)h; }
static inline half  __int2half_rn(int i) { return (half)(double)i; }
static inline half  __hmul(half a, half b) { return __double2half_rn((double)a * (double)b); }
static inline half  __hadd(half a, half b) { return __double2half_rn((double)a + (double)b); }
static inline half  __hsub(half a, half b) { return __double2half_rn((double)a - (double)b); }
static inline half  __low2half(half2 h) { return h.x; }
static inline half  __high2half(half2 h) { return h.y; }
static inline float __low2float(half2 h) { return (float)h.x; }
static inline half2 __floats2half2_rn(float a, float b) { return {(half)a, (half)b}; }
static inline half2 __hmul2(half2 a, half2 b) { return {__hmul(a.x, b.x), __hmul(a.y, b.y)}; }
static inline half2 __hadd2(half2 a, half2 b) { return {__hadd(a.x, b.x), __hadd(a.y, b.y)}; }
static inline half2 __hsub2(half2 a, half2 b) { return {__hsub(a.x, b.x), __hsub(a.y, b.y)}; }

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
extern "C" int gsq_dequant(int type, const void * x, void * y, int64_t k, int dst) {
  cudaStream_t s = nullptr;
  if (dst == 0) { auto f = ggml_get_to_cuda<float>(type); if (!f) return -1; f(x, (float *)y, k, s); }
  else if (dst == 1) { auto f = ggml_get_to_cuda<c10::BFloat16>(type); if (!f) return -1; f(x, (c10::BFloat16 *)y, k, s); }
  else if (dst == 2) { auto f = ggml_get_to_cuda<half>(type); if (!f) return -1; f(x, (half *)y, k, s); }
  else return -2;
  return 0;
}
"""

_LAUNCH = re.compile(r"^(\s*)(.+?)<<<\s*([^,]+?)\s*,\s*([^,]+?)\s*,[^>]*>>>\s*\((.*)\);\s*$", re.M)


def _rewrite_launches(src: str) -> str:
    def sub(m):
        ind, kern, grid, block, args = m.groups()
        return (
            f"{ind}{{ const int64_t _g = ({grid}); const unsigned _b = ({block}); blockDim.x = _b;"
            f" for (int64_t _i = 0; _i < _g; ++_i) {{ blockIdx.x = (unsigned)_i;"
            f" for (unsigned _t = 0; _t < _b; ++_t) {{ threadIdx.x = _t; {kern}({args}); }} }} }}"
        )

    out, n = _LAUNCH.subn(sub, src)
    if "<<<" in out or n == 0:
        raise RuntimeError("unrewritten CUDA launch left in dequantize.cuh")
    return out


def build(cxx: str = "g++-13") -> ctypes.CDLL:
    common = open(os.path.join(CSRC, "ggml-common.h")).read()
    deq = _rewrite_launches(open(os.path.join(CSRC, "dequantize.cuh")).read())
    d = tempfile.mkdtemp(prefix="gsq-tests-cudahost-")
    try:
        cpp, so = os.path.join(d, "deq.cpp"), os.path.join(d, "deq.so")
        with open(cpp, "w") as f:
            f.write(SHIM + "\n" + common + "\n" + deq + "\n" + WRAPPER)
        subprocess.run(
            [cxx, "-std=c++17", "-O1", "-ffp-contract=off", "-fno-fast-math", "-w",
             "-shared", "-fPIC", "-o", so, cpp],
            check=True,
        )
        lib = ctypes.CDLL(so)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    lib.gsq_dequant.restype = ctypes.c_int
    lib.gsq_dequant.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]
    return lib


_DST = {np.dtype(np.float32): 0, "bfloat16": 1, np.dtype(np.float16): 2}


def dequantize(lib, raw: np.ndarray, ggml_type: int, n: int, dst: str = "float32") -> np.ndarray:
    """dst: float32 | bfloat16 (returned as uint16 bit patterns) | float16."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    code = {"float32": 0, "bfloat16": 1, "float16": 2}[dst]
    out = np.zeros(n, dtype={0: np.float32, 1: np.uint16, 2: np.float16}[code])
    rc = lib.gsq_dequant(int(ggml_type), raw.ctypes.data, out.ctypes.data, n, code)
    if rc != 0:
        raise ValueError(f"gsq_dequant rc={rc} for type {ggml_type}")
    return out
