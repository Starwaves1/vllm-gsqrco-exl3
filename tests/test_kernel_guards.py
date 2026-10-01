# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bad inputs to the CUDA ops, one subprocess per case (tests/kernel_guard_case.py),
so that a sticky CUDA error cannot take the rest of the suite with it.

Pass: the op rejects the input with a clean exception, or runs and returns what
it returns on clean inputs. Fail: a device fault, a crash, or a silently wrong
result.

GGUF_COMPUTE_SANITIZER=/path/to/compute-sanitizer also runs each case under
memcheck, which catches out-of-bounds reads that happen not to fault. Memcheck
only sees cudaMalloc boundaries, so the case then runs with
PYTORCH_NO_CUDA_MEMORY_CACHING=1 (except graph_replay: capture cannot cudaMalloc).
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
        if case != "graph_replay":
            env["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    tail = (p.stdout + p.stderr)[-2000:]
    assert not any(f in p.stderr for f in FAULT), f"device fault:\n{tail}"
    assert p.returncode == 0, f"exit {p.returncode}:\n{tail}"
    line = next(ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{"))
    return json.loads(line)


@pytest.mark.parametrize("type_op", TYPES_OPS, ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("case", CASES)
def test_bad_input(case, type_op):
    res = run_case(case, *type_op)
    assert res["status"] in ("ok", "rejected"), f"silently wrong: {res}"
