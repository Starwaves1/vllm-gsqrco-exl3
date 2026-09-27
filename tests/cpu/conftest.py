"""CPU-only tests. Run: GSQ_LIGHT=1 tools/capped tools/pytest tests/cpu

no_gpu is imported before anything can pull in torch or vLLM. Triton runs in
interpreter mode (numpy on the CPU), which must be set before triton imports.
At session end we assert NVML was never mapped, torch never initialized CUDA,
and no /dev/nvidia* file was ever open at the end.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["TRITON_INTERPRET"] = "1"

import no_gpu  # noqa: E402

FIXTURES = os.path.join(ROOT, "tests/fixtures/dequant")
LLAMA_CPP = os.path.expanduser("~/llama.cpp-b11211")


def _nvidia_fds():
    out = []
    for fd in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            out.append(target)
    return out


def pytest_sessionfinish(session, exitstatus):
    no_gpu.assert_no_gpu_libs()
    fds = _nvidia_fds()
    if fds:
        raise RuntimeError(f"GPU device files open: {fds}")
