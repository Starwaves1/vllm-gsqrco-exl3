"""Out-of-bounds, contiguity and alignment behaviour of the plugin's CUDA ops, each case
in a subprocess (tests/gpu/_guard_case.py) so a sticky CUDA error cannot take the rest of
the suite with it.

Pass = the op either rejects the input with a clean Python exception, or runs and returns
the right answer. Fail = a device fault (illegal memory access, launch failure, crash) or
a silently wrong result. With the plugin as of e2b8ad5 several cases are EXPECTED to fail:
the kernels do no contiguity/stride/alignment/shape checks (STATUS "Not started"). These
tests are the acceptance test for the guards (localweights-style), not a baseline.

GSQ_COMPUTE_SANITIZER=/path/to/compute-sanitizer additionally runs each case under
memcheck, which also catches out-of-bounds reads that happen not to fault. Memcheck only sees
cudaMalloc boundaries, so the case then runs with PYTORCH_NO_CUDA_MEMORY_CACHING=1 (one
allocation per tensor; graph_replay excepted, as capture cannot cudaMalloc); under torch's
caching allocator a read past a tensor stays inside its segment and goes unreported.
Cases failing at e2b8ad5 are recorded in STATUS.md (Phase 1). The Route L ops
(lcpp_mul_mat_vec_q / lcpp_mul_mat_q / lcpp_mul_mat_vec_iq3[_mma[_packed]] / lcpp_mul_mat_iq3_packed /
lcpp_mul_mat_vec_own / lcpp_mul_mat_mma_k, csrc/lcpp_shim.cu) check every one of these before
launching, so their rows must all pass; they skip without the VLLM_GGUF_BUILD_LCPP=1 build.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gsq_gpu import GGUF, ROOT

CASES = ["x_noncontig", "x_misaligned", "x_rowstride", "w_narrow_view", "w_misaligned", "row_too_big",
         "k_mismatch", "graph_replay", "x_q8_short", "x_q8_misaligned", "x_q8_dtype", "x_q8_2d",
         "x_q8_strided", "x_q8_cpu"]
TYPES_OPS = [("IQ3_S", "mmvq"), ("IQ4_XS", "mmvq"), ("Q4_K", "mmvq"), ("Q4_K", "mmq"), ("Q6_K", "mmq"),
             ("IQ3_S", "lcpp_mmvq"), ("IQ4_XS", "lcpp_mmvq"), ("Q4_K", "lcpp_mmvq"),
             ("IQ3_S", "lcpp_mmq"), ("IQ3_XXS", "lcpp_mmq"), ("Q2_K", "lcpp_mmq"), ("Q4_K", "lcpp_mmq"),
             ("Q6_K", "lcpp_mmq"), ("IQ3_S", "lcpp_iq3"), ("IQ3_XXS", "lcpp_iq3"),
             ("IQ3_S", "lcpp_iq3_mma"), ("IQ3_XXS", "lcpp_iq3_mma"), ("Q4_K", "lcpp_own"), ("IQ2_S", "lcpp_own"),
             ("Q4_K", "lcpp_mma_k"), ("IQ4_XS", "lcpp_mma_k"), ("IQ2_S", "lcpp_mma_k"),
             ("IQ3_S", "lcpp_iq3_mma_packed"), ("IQ3_XXS", "lcpp_iq3_mma_packed"),
             ("IQ3_S", "lcpp_iq3_packed"), ("IQ3_XXS", "lcpp_iq3_packed")]
FAULT = ("illegal memory access", "misaligned address", "unspecified launch failure", "CUDA error", "an illegal instruction")


@pytest.mark.parametrize("type_op", TYPES_OPS, ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("case", CASES)
def test_guard_case(case, type_op):
    res = _run_case(case, type_op)
    if case.startswith("x_q8"):  # every bad x_q8 must be refused before a launch
        assert res["status"] == "rejected", res
    assert res["status"] in ("ok", "rejected"), f"silently wrong: {res}"


def _run_case(case, type_op):
    if not GGUF.exists():
        pytest.skip(f"GGUF not found: {GGUF}")
    name, op = type_op
    if case.startswith("x_q8") and op not in ("lcpp_mmvq", "lcpp_iq3", "lcpp_iq3_mma", "lcpp_own"):
        pytest.skip("x_q8 is an argument of the 1..8-row Route L ops only")
    cmd = [sys.executable, str(Path(__file__).with_name("_guard_case.py")), case, name, op]
    san = os.environ.get("GSQ_COMPUTE_SANITIZER")
    if san:
        cmd = [san, "--tool", "memcheck", "--error-exitcode", "99"] + cmd
    env = dict(os.environ, GSQ_GGUF=str(GGUF), CUDA_LAUNCH_BLOCKING="1",
               PYTHONPATH=os.pathsep.join([str(ROOT / "tools"), os.environ.get("PYTHONPATH", "")]))
    if san and not case.startswith("graph"):  # a graph cannot capture cudaMalloc
        env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    tail = (p.stdout + p.stderr)[-2000:]
    if "built without VLLM_GGUF_BUILD_LCPP" in p.stderr:
        pytest.skip("_C_gguf built without VLLM_GGUF_BUILD_LCPP=1")
    assert not any(f in p.stderr for f in FAULT), f"device fault:\n{tail}"
    assert p.returncode == 0, f"exit {p.returncode}:\n{tail}"
    return json.loads(next(ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")))


@pytest.mark.parametrize("name", ["IQ3_S", "IQ3_XXS"])
def test_iq3_mma_first_call_in_capture(name):
    """The mma op sets its launch attributes (dynamic shared memory, resident CTAs) on its first
    call in a process; that call may happen inside CUDA-graph capture."""
    res = _run_case("graph_first", (name, "lcpp_iq3_mma"))
    assert res["status"] == "ok", res


@pytest.mark.parametrize("name", ["IQ3_S", "IQ3_XXS"])
def test_iq3_mma_packed_first_call_in_capture(name):
    """The same for the packed op (its launch attributes are set per column-group variant)."""
    res = _run_case("graph_first", (name, "lcpp_iq3_mma_packed"))
    assert res["status"] == "ok", res


@pytest.mark.parametrize("name", ["IQ3_S", "IQ3_XXS"])
def test_iq3_packed_first_call_in_capture(name):
    """The same for the tiled packed op (launch attributes per tile shape, 129 rows: the split
    tiles' fixup kernel is captured too)."""
    res = _run_case("graph_first", (name, "lcpp_iq3_packed"))
    assert res["status"] == "ok", res
