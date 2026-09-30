"""EXL3 shim guards (plugin-exl3/vllm_exl3_plugin/csrc/exl3_shim.cu), CPU only.

The shim registers its ops for CPU too: every shape/dtype/stride/alignment guard runs, then
the call is rejected with "must be CUDA tensors". So a CPU call that ends in that message
passed all guards; any other message names the guard that fired. No kernel runs. Everything
happens in one subprocess that imports no_gpu first and asserts torch never initialized CUDA.

Needs _C_exl3 built (VLLM_EXL3_BUILD=1, see EXL3.md); skips otherwise.
"""

import glob
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, "plugin-exl3/vllm_exl3_plugin")
OPS = ("exl3_gemm", "exl3_dequant", "exl3_had_r_128", "exl3_hgemm", "exl3_warmup")
K_IN, N_OUT = 5120, 1024  # k, n of a valid weight (k_proj-like), K=4 unless a case says otherwise

_CHILD = r"""
import ctypes, json, sys
sys.path.insert(0, sys.argv[2])
import no_gpu
import torch
torch.ops.load_library(sys.argv[1])
ops = torch.ops._C_exl3
cases = json.loads(sys.argv[3])
out = {"registered": [n for n in json.loads(sys.argv[4]) if hasattr(ops, n)]}
libc = ctypes.CDLL(None)
libc.getenv.restype = ctypes.c_char_p
out["int8_gemv_env"] = (libc.getenv(b"EXL3_INT8_GEMV") or b"").decode()

def t(spec):
    if spec is None:
        return None
    shape, dtype = spec["shape"], getattr(torch, spec.get("dtype", "float16"))
    off = spec.get("offset", 0)
    n = 1
    for s in shape:
        n *= s
    buf = torch.zeros(n + off + 64, dtype=dtype)
    x = buf[off:off + n].view(shape)
    if spec.get("t"):  # same shape, not contiguous
        x = torch.zeros(list(reversed(shape[:2])) + list(shape[2:]), dtype=dtype).transpose(0, 1)
    return x

res = {}
for name, c in cases.items():
    a = c["args"]
    try:
        op = c["op"]
        if op == "exl3_gemm":
            ops.exl3_gemm(t(a["x"]), t(a["trellis"]), t(a["suh"]), t(a["svh"]), a["mcg"], a["mul1"],
                          a["out_fp32"])
        elif op == "exl3_dequant":
            ops.exl3_dequant(t(a["trellis"]), t(a["suh"]), t(a["svh"]), a["mcg"], a["mul1"],
                             a["n_start"], a["n_count"], a["had"])
        elif op == "exl3_had_r_128":
            ops.exl3_had_r_128(t(a["x"]), t(a["pre"]), t(a["post"]), 1.0)
        elif op == "exl3_hgemm":
            ops.exl3_hgemm(t(a["a"]), t(a["b"]))
        elif op == "exl3_warmup":
            ops.exl3_warmup(t(a["trellis"]), t(a["suh"]), t(a["svh"]), a["mcg"], a["mul1"], a["rows"],
                            a["out_fp32"])
        res[name] = "no error"
    except RuntimeError as e:
        res[name] = str(e).splitlines()[0][:300]
out["results"] = res
no_gpu.assert_no_gpu_libs()
out["cuda_initialized"] = torch.cuda.is_initialized()
print(json.dumps(out))
"""


def T(*shape, dtype="float16", **kw):
    return dict(shape=list(shape), dtype=dtype, **kw)


def trellis(k=K_IN, n=N_OUT, width=64, **kw):
    return T(k // 16, n // 16, width, dtype=kw.pop("dtype", "int16"), **kw)


def gemm(x=None, tr=None, suh=None, svh=None, mcg=False, mul1=True, out_fp32=True):
    return dict(op="exl3_gemm", args=dict(
        x=x or T(4, K_IN), trellis=tr or trellis(),
        suh=suh or T(K_IN), svh=svh or T(N_OUT), mcg=mcg, mul1=mul1, out_fp32=out_fp32))


def dequant(n_start=0, n_count=N_OUT, had=True, tr=None, mcg=False, mul1=True):
    return dict(op="exl3_dequant", args=dict(trellis=tr or trellis(), suh=T(K_IN), svh=T(N_OUT),
                                             mcg=mcg, mul1=mul1, n_start=n_start, n_count=n_count, had=had))


def warmup(rows=(1, 2, 4, 8, 16), out_fp32=True, tr=None):
    return dict(op="exl3_warmup", args=dict(trellis=tr or trellis(), suh=T(K_IN), svh=T(N_OUT),
                                            mcg=False, mul1=True, rows=list(rows), out_fp32=out_fp32))


CUDA = "must be CUDA tensors"
# name -> (case, expected message fragment)
CASES = {
    # exl3_gemm: valid calls reach the device check
    "gemm-valid-K4-out-fp32": (gemm(), CUDA),
    "gemm-valid-out-fp16": (gemm(out_fp32=False), CUDA),
    "gemm-valid-1-row": (gemm(x=T(1, K_IN)), CUDA),
    "gemm-valid-0-rows": (gemm(x=T(0, K_IN)), CUDA),
    "gemm-valid-200-rows": (gemm(x=T(200, K_IN)), CUDA),
    "gemm-valid-K3": (gemm(tr=trellis(width=48)), CUDA),
    "gemm-valid-K6": (gemm(tr=trellis(width=96)), CUDA),
    "gemm-valid-K3.5-mul1": (gemm(tr=trellis(width=56)), CUDA),
    "gemm-valid-3inst": (gemm(mul1=False), CUDA),
    "gemm-valid-mcg": (gemm(mcg=True, mul1=False), CUDA),
    "gemm-K3.5-mcg": (gemm(tr=trellis(width=56), mcg=True, mul1=False), "half-integer bit widths need the mul1"),
    "gemm-mcg-and-mul1": (gemm(mcg=True, mul1=True), "mcg and mul1 are exclusive"),
    "gemm-trellis-2d": (gemm(tr=T(320, 4096, dtype="int16")), "trellis must be 3-D"),
    "gemm-trellis-int32": (gemm(tr=trellis(dtype="int32")), "trellis must be int16"),
    "gemm-trellis-noncontig": (gemm(tr=trellis(t=True)), "trellis must be contiguous"),
    "gemm-trellis-width-50": (gemm(tr=trellis(width=50)), "is not 16*K or 16*K+8"),
    "gemm-trellis-K9": (gemm(tr=trellis(width=144)), "unsupported bit width"),
    "gemm-trellis-K4.5": (gemm(tr=trellis(width=72)), "unsupported bit width"),
    "gemm-n-not-128": (gemm(tr=trellis(n=64), svh=T(64)), "multiples of 128"),
    "gemm-k-not-128": (gemm(x=T(4, 5104), tr=trellis(k=5104), suh=T(5104)), "multiples of 128"),
    "gemm-trellis-misaligned": (gemm(tr=trellis(offset=1)), "trellis must be 16-byte aligned"),
    "gemm-x-k-mismatch": (gemm(x=T(4, 4096)), "columns, the weight has k=5120"),
    "gemm-x-fp32": (gemm(x=T(4, K_IN, dtype="float32")), "x must be fp16"),
    "gemm-x-bf16": (gemm(x=T(4, K_IN, dtype="bfloat16")), "x must be fp16"),
    "gemm-x-1d": (gemm(x=T(K_IN)), "x must be 2-D"),
    "gemm-x-noncontig": (gemm(x=T(4, K_IN, t=True)), "x must be contiguous"),
    "gemm-x-misaligned": (gemm(x=T(4, K_IN, offset=1)), "x must be 16-byte aligned"),
    "gemm-suh-size": (gemm(suh=T(4096)), "suh must be 1-D of size 5120"),
    "gemm-suh-bf16": (gemm(suh=T(K_IN, dtype="bfloat16")), "suh must be fp16"),
    "gemm-suh-misaligned": (gemm(suh=T(K_IN, offset=1)), "suh must be 16-byte aligned"),
    "gemm-svh-size": (gemm(svh=T(K_IN)), "svh must be 1-D of size 1024"),
    "gemm-svh-2d": (gemm(svh=T(1, N_OUT)), "svh must be 1-D"),
    # exl3_dequant
    "dequant-valid-had": (dequant(), CUDA),
    "dequant-valid-rotated": (dequant(had=False), CUDA),
    "dequant-valid-slice": (dequant(n_start=512, n_count=512), CUDA),
    "dequant-start-not-128": (dequant(n_start=64, n_count=128), "must be a 128-aligned range"),
    "dequant-count-not-128": (dequant(n_count=100), "must be a 128-aligned range"),
    "dequant-past-end": (dequant(n_start=512, n_count=1024), "must be a 128-aligned range"),
    "dequant-count-0": (dequant(n_count=0), "must be a 128-aligned range"),
    "dequant-trellis-int32": (dequant(tr=trellis(dtype="int32")), "trellis must be int16"),
    "dequant-K3.5-3inst": (dequant(tr=trellis(width=56), mul1=False), "half-integer bit widths need the mul1"),
    # exl3_had_r_128
    "had-valid-pre": (dict(op="exl3_had_r_128", args=dict(x=T(4, K_IN), pre=T(K_IN), post=None)), CUDA),
    "had-valid-post": (dict(op="exl3_had_r_128", args=dict(x=T(4, K_IN), pre=None, post=T(K_IN))), CUDA),
    "had-valid-none": (dict(op="exl3_had_r_128", args=dict(x=T(4, K_IN), pre=None, post=None)), CUDA),
    "had-cols-not-128": (dict(op="exl3_had_r_128", args=dict(x=T(4, 100), pre=None, post=None)), "multiple of 128"),
    "had-bf16": (dict(op="exl3_had_r_128", args=dict(x=T(4, K_IN, dtype="bfloat16"), pre=None, post=None)),
                 "contiguous 2-D fp16"),
    "had-both-scales": (dict(op="exl3_had_r_128", args=dict(x=T(4, K_IN), pre=T(K_IN), post=T(K_IN))),
                        "at most one of pre_scale and post_scale"),
    "had-pre-size": (dict(op="exl3_had_r_128", args=dict(x=T(4, K_IN), pre=T(128), post=None)),
                     "pre_scale must be 1-D of size 5120"),
    # exl3_hgemm
    "hgemm-valid": (dict(op="exl3_hgemm", args=dict(a=T(256, K_IN), b=T(K_IN, 1024))), CUDA),
    "hgemm-mismatch": (dict(op="exl3_hgemm", args=dict(a=T(256, K_IN), b=T(4096, 1024))), "a is"),
    "hgemm-bf16": (dict(op="exl3_hgemm", args=dict(a=T(256, K_IN, dtype="bfloat16"), b=T(K_IN, 1024))),
                   "must be fp16"),
    "hgemm-noncontig": (dict(op="exl3_hgemm", args=dict(a=T(256, K_IN), b=T(K_IN, 1024, t=True))),
                        "must be contiguous"),
    # exl3_warmup
    "warmup-valid": (warmup(), CUDA),
    "warmup-valid-out-fp16": (warmup(out_fp32=False), CUDA),
    "warmup-rows-0": (warmup(rows=(0, 1)), "row counts must be >= 1"),
    "warmup-bad-trellis": (warmup(tr=trellis(width=50)), "is not 16*K or 16*K+8"),
}


def _so():
    found = glob.glob(os.path.join(PKG, "_C_exl3*.so"))
    return found[0] if found else None


@pytest.fixture(scope="module")
def child():
    so = _so()
    if so is None:
        pytest.skip(f"not built: {PKG}/_C_exl3*.so (VLLM_EXL3_BUILD=1)")
    p = subprocess.run(
        [sys.executable, "-c", _CHILD, so, os.path.join(ROOT, "tools"),
         json.dumps({k: v[0] for k, v in CASES.items()}), json.dumps(OPS)],
        capture_output=True, text=True, timeout=300,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert p.returncode == 0, p.stderr[-3000:]
    return json.loads(p.stdout.strip().splitlines()[-1])


def test_ops_registered_without_cuda_init(child):
    assert child["registered"] == list(OPS)
    assert child["cuda_initialized"] is False


def test_int8_gemv_switched_off(child):
    """exllamav3 d3739fd turns its int8-activation GEMV on by default (mode 2); the shim
    sets EXL3_INT8_GEMV=0 when it loads (not graph-capturable, first-call cudaMalloc)."""
    assert child["int8_gemv_env"] == "0"


@pytest.mark.parametrize("name", list(CASES))
def test_guard(child, name):
    got = child["results"][name]
    assert CASES[name][1] in got, got
