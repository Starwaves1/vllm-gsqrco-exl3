"""Out-of-bounds, contiguity and alignment behaviour of the plugin's CUDA ops, each case
in a subprocess (tests/gpu/_guard_case.py) so a sticky CUDA error cannot take the rest of
the suite with it.

Pass = the op either rejects the input with a clean Python exception, or runs and returns
the right answer. Fail = a device fault (illegal memory access, launch failure, crash) or
a silently wrong result. With the plugin as of e2b8ad5 several cases are EXPECTED to fail:
the kernels do no contiguity/stride/alignment/shape checks (STATUS "Not started"). These
tests are the acceptance test for the guards (localweights-style), not a baseline.

GSQ_COMPUTE_SANITIZER=/path/to/compute-sanitizer additionally runs each case under
memcheck, which also catches out-of-bounds reads that happen not to fault.
Cases failing at e2b8ad5 are recorded in STATUS.md (Phase 1). TODO: add the guards in
plugin/ and turn this suite green.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gsq_gpu import GGUF, ROOT

CASES = ["x_noncontig", "x_misaligned", "w_narrow_view", "w_misaligned", "row_too_big", "k_mismatch", "graph_replay"]
TYPES_OPS = [("IQ3_S", "mmvq"), ("IQ4_XS", "mmvq"), ("Q4_K", "mmvq"), ("Q4_K", "mmq"), ("Q6_K", "mmq")]
FAULT = ("illegal memory access", "misaligned address", "unspecified launch failure", "CUDA error", "an illegal instruction")


@pytest.mark.parametrize("type_op", TYPES_OPS, ids=lambda p: f"{p[0]}-{p[1]}")
@pytest.mark.parametrize("case", CASES)
def test_guard_case(case, type_op):
    if not GGUF.exists():
        pytest.skip(f"GGUF not found: {GGUF}")
    name, op = type_op
    cmd = [sys.executable, str(Path(__file__).with_name("_guard_case.py")), case, name, op]
    san = os.environ.get("GSQ_COMPUTE_SANITIZER")
    if san:
        cmd = [san, "--tool", "memcheck", "--error-exitcode", "99"] + cmd
    env = dict(os.environ, GSQ_GGUF=str(GGUF), CUDA_LAUNCH_BLOCKING="1",
               PYTHONPATH=os.pathsep.join([str(ROOT / "tools"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    tail = (p.stdout + p.stderr)[-2000:]
    assert not any(f in p.stderr for f in FAULT), f"device fault:\n{tail}"
    assert p.returncode == 0, f"exit {p.returncode}:\n{tail}"
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["status"] in ("ok", "rejected"), f"silently wrong: {res}"
