# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bad inputs to the CUDA ops, one subprocess per case (tests/kernel_guard_case.py),
so that a sticky CUDA error cannot take the rest of the suite with it.

Pass: the op rejects the input with a clean exception, or runs and returns what
it returns on clean inputs. Fail: a device fault, a crash, or a silently wrong
result. The lcpp ops (built with VLLM_GGUF_BUILD_LCPP=1, skipped
otherwise) check all of these before launching, so every case must pass.

GGUF_COMPUTE_SANITIZER=/path/to/compute-sanitizer also runs each case under
memcheck, which catches out-of-bounds reads that happen not to fault. Memcheck
only sees cudaMalloc boundaries, so the case then runs with
PYTORCH_NO_CUDA_MEMORY_CACHING=1 (except the graph cases: capture cannot
cudaMalloc).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

CASES = [
    "x_noncontig",
    "x_misaligned",
    "x_rowstride",
    "w_narrow_view",
    "w_misaligned",
    "row_too_big",
    "k_mismatch",
    "graph_replay",
]
TYPES_OPS = [
    ("IQ3_S", "ggml_mul_mat_vec_a8"),
    ("IQ4_XS", "ggml_mul_mat_vec_a8"),
    ("Q4_K", "ggml_mul_mat_vec_a8"),
    ("Q4_K", "ggml_mul_mat_a8"),
    ("Q6_K", "ggml_mul_mat_a8"),
    ("IQ3_S", "lcpp_mul_mat_vec_q"),
    ("IQ4_XS", "lcpp_mul_mat_vec_q"),
    ("Q4_K", "lcpp_mul_mat_vec_q"),
    ("IQ3_S", "lcpp_mul_mat_q"),
    ("IQ3_XXS", "lcpp_mul_mat_q"),
    ("Q2_K", "lcpp_mul_mat_q"),
    ("Q4_K", "lcpp_mul_mat_q"),
    ("Q6_K", "lcpp_mul_mat_q"),
    ("IQ3_S", "lcpp_mul_mat_vec_iq3"),
    ("IQ3_XXS", "lcpp_mul_mat_vec_iq3"),
    ("IQ3_S", "lcpp_mul_mat_vec_iq3_mma"),
    ("IQ3_XXS", "lcpp_mul_mat_vec_iq3_mma"),
    ("Q4_K", "lcpp_mul_mat_vec_own"),
    ("IQ2_S", "lcpp_mul_mat_vec_own"),
    ("Q4_K", "lcpp_mul_mat_mma_k"),
    ("IQ4_XS", "lcpp_mul_mat_mma_k"),
    ("IQ2_S", "lcpp_mul_mat_mma_k"),
    ("IQ3_S", "lcpp_mul_mat_vec_iq3_mma_packed"),
    ("IQ3_XXS", "lcpp_mul_mat_vec_iq3_mma_packed"),
    ("IQ3_S", "lcpp_mul_mat_iq3_packed"),
    ("IQ3_XXS", "lcpp_mul_mat_iq3_packed"),
]
FAULT = (
    "illegal memory access",
    "misaligned address",
    "unspecified launch failure",
    "CUDA error",
    "illegal instruction",
)


def run_case(case: str, name: str, op: str) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("kernel_guard_case.py")),
        case,
        name,
        op,
    ]
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1")
    sanitizer = os.environ.get("GGUF_COMPUTE_SANITIZER")
    if sanitizer:
        cmd = [sanitizer, "--tool", "memcheck", "--error-exitcode", "99"] + cmd
        if not case.startswith("graph"):
            env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    tail = (p.stdout + p.stderr)[-2000:]
    if "not built" in p.stderr:
        pytest.skip(f"{op} not built (VLLM_GGUF_BUILD_LCPP=1)")
    assert not any(f in p.stderr for f in FAULT), f"device fault:\n{tail}"
    assert p.returncode == 0, f"exit {p.returncode}:\n{tail}"
    line = next(ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{"))
    return json.loads(line)


@pytest.mark.parametrize("type_op", TYPES_OPS, ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("case", CASES)
def test_bad_input(case, type_op):
    res = run_case(case, *type_op)
    assert res["status"] in ("ok", "rejected"), f"silently wrong: {res}"


@pytest.mark.parametrize(
    "op",
    [
        "lcpp_mul_mat_vec_iq3_mma",
        "lcpp_mul_mat_vec_iq3_mma_packed",
        "lcpp_mul_mat_iq3_packed",
    ],
)
@pytest.mark.parametrize("name", ["IQ3_S", "IQ3_XXS"])
def test_first_call_in_capture(name, op):
    """The mma ops set their launch attributes (dynamic shared memory, resident
    CTAs) on their first call in a process, which may happen inside a
    CUDA-graph capture."""
    assert run_case("graph_first", name, op)["status"] == "ok"
