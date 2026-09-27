"""Import this FIRST (before torch or vllm) in Phase A (CPU-only) scripts.

It guarantees the process never talks to the GPU while a live benchmark runs:
  * CUDA_VISIBLE_DEVICES="" (torch sees no devices);
  * ctypes refuses to load NVML (libnvidia-ml, what nvidia-smi and vLLM's
    platform detection use) and the driver (libcuda). pynvml then raises
    NVMLError_LibraryNotFound, which vLLM treats as "no CUDA platform".
`assert_no_gpu_libs()` checks that NVML was never mapped and that torch never
initialized CUDA. (`import torch` itself dlopens libcuda lazily; with no visible
device the driver cannot open a context, so that mapping is allowed.)
Scripts that need a vLLM platform object must set one explicitly.
"""

import ctypes
import os
import sys

if "vllm" in sys.modules or "torch" in sys.modules:
    raise RuntimeError("import no_gpu before torch/vllm")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

_BLOCKED = ("libnvidia-ml", "nvml.dll", "libcuda.so")
_RealCDLL = ctypes.CDLL


class _GuardedCDLL(_RealCDLL):
    def __init__(self, name, *args, **kwargs):
        if name and any(b in os.path.basename(str(name)) for b in _BLOCKED):
            raise OSError(f"{name}: blocked by tools/no_gpu.py (Phase A: no GPU use)")
        super().__init__(name, *args, **kwargs)


ctypes.CDLL = _GuardedCDLL
ctypes.cdll = ctypes.LibraryLoader(_GuardedCDLL)


def assert_no_gpu_libs() -> None:
    with open("/proc/self/maps") as f:
        if "libnvidia-ml" in f.read():
            raise RuntimeError("NVML (libnvidia-ml) was loaded")
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError("torch initialized CUDA")
