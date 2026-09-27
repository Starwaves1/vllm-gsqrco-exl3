"""Route L shim guards (plugin/vllm_gguf_plugin/csrc/lcpp_shim.cu), CPU only.

The shim registers its ops for CPU too: every shape/stride/alignment guard runs,
then the call is rejected with "must be CUDA tensors". So a CPU call that ends in
that message passed all guards; any other message names the guard that fired.
No kernel runs. Everything happens in one subprocess that imports no_gpu first
and asserts torch never initialized CUDA.

Needs the extension built with VLLM_GGUF_BUILD_LCPP=1 (skips otherwise).
"""

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SO = os.path.join(ROOT, "plugin/vllm_gguf_plugin/_C_gguf.abi3.so")

IQ3_S, Q4_K, IQ1_M = 21, 12, 29
TS = {IQ3_S: 110, Q4_K: 144}  # bytes per 256-value block
K = 5120

_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[2])
import no_gpu
import torch
torch.ops.load_library(sys.argv[1])
ops = torch.ops._C_gguf
cases = json.loads(sys.argv[3])
out = {"registered": [n for n in ("lcpp_mul_mat_vec_q", "lcpp_mul_mat_q") if hasattr(ops, n)]}

def w(rows, row_bytes, stride=None, offset=0, dtype=torch.uint8):
    stride = stride or row_bytes
    buf = torch.zeros(rows * stride + offset + 64, dtype=dtype)
    return buf[offset:offset + rows * stride].view(rows, stride)[:, :row_bytes]

res = {}
for name, c in cases.items():
    W = w(c["rows"], c["row_bytes"], c.get("stride"), c.get("w_offset", 0),
          getattr(torch, c.get("w_dtype", "uint8")))
    if c.get("w_t"):
        W = W.t()
    if c.get("w_1d"):
        W = W.reshape(-1)
    X = torch.zeros(c["n"], c["k"], dtype=getattr(torch, c.get("x_dtype", "bfloat16")))
    if c.get("x_t"):
        X = torch.zeros(c["k"], c["n"], dtype=X.dtype).t()
    try:
        getattr(ops, c["op"])(W, X, c["type"], c["row"])
        res[name] = "no error"
    except RuntimeError as e:
        res[name] = str(e).splitlines()[0][:300]
out["results"] = res
no_gpu.assert_no_gpu_libs()
out["cuda_initialized"] = torch.cuda.is_initialized()
print(json.dumps(out))
"""


def _case(op, type=IQ3_S, rows=256, row_bytes=None, n=4, k=K, row=None, **kw):
    rb = row_bytes if row_bytes is not None else K // 256 * TS.get(type, 110)
    return dict(op=op, type=type, rows=rows, row_bytes=rb, n=n, k=k,
                row=rows if row is None else row, **kw)


# name -> (case, expected message fragment)
CASES = {}
for op in ("lcpp_mul_mat_vec_q", "lcpp_mul_mat_q"):
    n = 4 if op == "lcpp_mul_mat_vec_q" else 64
    for t in (IQ3_S, Q4_K):
        CASES[f"{op}-valid-{t}"] = (_case(op, t, n=n), "must be CUDA tensors")
        CASES[f"{op}-row_strided-{t}"] = (
            _case(op, t, n=n, stride=2 * K // 256 * TS[t]), "must be CUDA tensors")
        CASES[f"{op}-w_narrow_view-{t}"] = (
            _case(op, t, n=n, stride=K // 256 * TS[t] + 256), "row stride")
    CASES[f"{op}-fp32_x"] = (_case(op, n=n, x_dtype="float32"), "must be CUDA tensors")
    CASES[f"{op}-iq1_m"] = (_case(op, IQ1_M, n=n), "unsupported ggml type")
    CASES[f"{op}-w_float"] = (_case(op, n=n, w_dtype="float32"), "W must be uint8")
    CASES[f"{op}-w_1d"] = (_case(op, n=n, w_1d=True), "must be 2-D")
    CASES[f"{op}-x_int"] = (_case(op, n=n, x_dtype="int32"), "X must be fp32, fp16 or bf16")
    CASES[f"{op}-row_bytes_not_blocks"] = (_case(op, n=n, row_bytes=2201), "not a multiple of the block size")
    CASES[f"{op}-k_not_512"] = (_case(op, n=n, row_bytes=110, k=256), "must be a multiple of 512")
    CASES[f"{op}-row_too_big"] = (_case(op, n=n, row=256 + 64), "out of range")
    CASES[f"{op}-row_zero"] = (_case(op, n=n, row=0), "out of range")
    CASES[f"{op}-w_transposed"] = (_case(op, n=n, rows=2200, row_bytes=256, w_t=True, row=256), "W inner stride")
    CASES[f"{op}-w_misaligned"] = (_case(op, n=n, w_offset=1), "16-byte aligned")
    CASES[f"{op}-k_mismatch"] = (_case(op, n=n, k=K // 2), "columns, W rows hold")
    CASES[f"{op}-x_noncontig"] = (_case(op, n=n, x_t=True), "X inner stride")
CASES["lcpp_mul_mat_vec_q-9_rows"] = (_case("lcpp_mul_mat_vec_q", n=9), "at most 8 rows")
CASES["lcpp_mul_mat_q-9_rows"] = (_case("lcpp_mul_mat_q", n=9), "must be CUDA tensors")
CASES["lcpp_mul_mat_q-1_row"] = (_case("lcpp_mul_mat_q", n=1), "must be CUDA tensors")


@pytest.fixture(scope="module")
def child():
    if not os.path.exists(SO):
        pytest.skip(f"not built: {SO}")
    p = subprocess.run(
        [sys.executable, "-c", _CHILD, SO, os.path.join(ROOT, "tools"),
         json.dumps({k: v[0] for k, v in CASES.items()})],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 0, p.stderr[-3000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    if not out["registered"]:
        pytest.skip("_C_gguf built without VLLM_GGUF_BUILD_LCPP=1")
    return out


def test_ops_registered_without_cuda_init(child):
    assert child["registered"] == ["lcpp_mul_mat_vec_q", "lcpp_mul_mat_q"]
    assert child["cuda_initialized"] is False


@pytest.mark.parametrize("name", list(CASES))
def test_guard(child, name):
    got = child["results"][name]
    assert CASES[name][1] in got, got
