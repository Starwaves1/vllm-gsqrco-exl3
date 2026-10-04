"""EXL3 kernel parity and plumbing on the GPU (EXL3.md, GPU phase 1): the plugin's
torch.ops._C_exl3 (plugin-exl3/vllm_exl3_plugin/csrc/exl3_shim.cu) on real checkpoint tensors,
against exllamav3 d3739fd itself (tests/gpu/exl3_ref_dump.py, run first in the reference venv;
its JSON in EXL3_REF_DIR) and against fp64.

  dequant     exl3_dequant(had=True / False) bit-exact with exllamav3's reconstruct_had_slice /
              reconstruct_slice, every 32768-column slice (sha256 over the fp16 bytes)
  gemm        the routed op (ops.exl3_linear: exl3_gemm to 144 rows, reconstruct + hgemm above,
              fused from 1024) at fp16 and fp32 output, vs fp64: error inside exllamav3's own
              error at the same rows (x1.5, floor 0.2 % of RMS); bit-identity with exllamav3's
              output is reported (non-strict xfail: it needs the same autotuned kernel, i.e. a
              shared EXLLAMAV3_TUNE_CACHE)
  routes      at 145 and 1024 rows the reconstruct routes agree with exl3_gemm forced at the
              same rows; fp32 and fp16 outputs agree
  graphs      after exl3_warmup: capture exl3_gemm at 1..48 rows, replay with new inputs,
              compare with eager; in fresh processes (tests/gpu/_exl3_case.py): capture before
              warmup is refused cleanly, the first call inside a capture after warmup works,
              and the shim's guards reject bad CUDA inputs without a device fault

Environment: EXL3_MODEL (checkpoint dir), EXL3_REF_DIR, EXLLAMAV3_TUNE_CACHE (share with the
reference run). Skips without the checkpoint or the built _C_exl3; reference-dependent checks
skip without the reference JSON.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gsq_gpu import ROOT
import exl3_cases as C

sys.path.insert(0, str(ROOT / "plugin-exl3"))

SLACK, FLOOR_RMS, FLOOR_MAX = 1.5, 2e-3, 2e-2
NOREF_RMS, NOREF_MAX = 5e-3, 5e-2  # no reference JSON: exllamav3's documented ~1 % of RMS (H_ACC)
GRAPH_ROWS = [1, 2, 4, 8, 16, 17, 48]
GRAPH_TIDS = [*C.PER_K, C.HEAD]
WARM_ROWS = [1, 2, 4, 8, 16]
FAULT = ("illegal memory access", "misaligned address", "unspecified launch failure", "CUDA error",
         "an illegal instruction", "operation not permitted when stream is capturing")
SUB_CASES = ["unwarmed_capture", "hgemm_unwarmed_capture", "first_call_in_capture", "x_misaligned",
             "x_noncontig", "x_bf16", "k_mismatch", "suh_wrong_size", "trellis_misaligned",
             "dequant_unaligned", "dequant_out_of_range"]


@pytest.fixture(scope="session")
def ops():
    import torch

    if not C.MODEL.exists():
        pytest.skip(f"EXL3 checkpoint not found: {C.MODEL} (EXL3_MODEL)")
    from vllm_exl3_plugin import ops as o

    if not o.OPS_AVAILABLE:
        pytest.skip("_C_exl3 not built (VLLM_EXL3_BUILD=1)")
    # phase 1's routes (EXL3_MR=0); the multi-row kernel is test_exl3_mr.py's
    o.MR_MODE, o.MULTI_ROW_OP, o.MULTI_ROW_MIN, o.MR_GLUE = 0, None, 17, False
    torch.manual_seed(0)
    return o


_W = {}


def weights(tid):
    import torch

    if tid not in _W:
        _W.clear()  # one tensor on the GPU at a time (the lm_head trellis is 0.9 GiB)
        torch.cuda.empty_cache()
        _W[tid] = C.load(torch, tid)
    return _W[tid]


def ref(tid):
    p = C.REF_DIR / f"{tid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def dequant_fn(w, had):
    import torch

    return lambda s, c: torch.ops._C_exl3.exl3_dequant(w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"],
                                                       s, c, had)


@pytest.mark.parametrize("had", [True, False], ids=["had", "rot"])
@pytest.mark.parametrize("tid", list(C.TENSORS))
def test_dequant_bitexact(ops, tid, had):
    r = ref(tid)
    if r is None:
        pytest.skip(f"no exllamav3 reference in {C.REF_DIR} (run tests/gpu/exl3_ref_dump.py)")
    w = weights(tid)
    got = C.hash_slices(dequant_fn(w, had), C.TENSORS[tid][3])
    assert got == r["dq_had_sha256" if had else "dq_rot_sha256"], f"{tid} dequant ({'had' if had else 'rot'}) differs from exllamav3"


@pytest.mark.parametrize("out_fp32", [True, False], ids=["fp32", "fp16"])
@pytest.mark.parametrize("m", C.ROWS)
@pytest.mark.parametrize("tid", list(C.TENSORS))
def test_gemm_vs_fp64(ops, tid, m, out_fp32):
    import torch

    w = weights(tid)
    n = C.TENSORS[tid][3]
    x = C.make_x(torch, tid, m)
    y = ops.exl3_linear(x, w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"], out_fp32)
    assert y.shape == (m, n) and y.dtype == (torch.float if out_fp32 else torch.half)
    s = C.err_stats(torch, y, C.fp64_ref(torch, x, dequant_fn(w, True), n))
    r = ref(tid)
    rs = r["gemm"][str(m)]["fp32" if out_fp32 else "fp16"] if r else None
    print(f"\n{tid} m={m} route={ops._exl3_op(m)} plugin {s} exllamav3 {rs}")
    assert s["finite"], "non-finite output"
    if rs:
        assert s["rel_rms"] <= max(SLACK * rs["rel_rms"], FLOOR_RMS), (s, rs)
        assert s["max_rel"] <= max(SLACK * rs["max_rel"], FLOOR_MAX), (s, rs)
    else:
        assert s["rel_rms"] <= NOREF_RMS and s["max_rel"] <= NOREF_MAX, s


@pytest.mark.xfail(strict=False, reason="informational: bit identity needs the same autotuned kernel "
                                       "(shared EXLLAMAV3_TUNE_CACHE) and the same output dtype path")
@pytest.mark.parametrize("out_fp32", [True, False], ids=["fp32", "fp16"])
@pytest.mark.parametrize("m", C.ROWS)
@pytest.mark.parametrize("tid", list(C.TENSORS))
def test_gemm_bits_match_exllamav3(ops, tid, m, out_fp32):
    import torch

    r = ref(tid)
    if r is None:
        pytest.skip("no exllamav3 reference")
    w = weights(tid)
    y = ops.exl3_linear(C.make_x(torch, tid, m), w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"], out_fp32)
    assert C.sha_tensor(y) == r["gemm"][str(m)]["fp32" if out_fp32 else "fp16"]["sha256"]


@pytest.mark.parametrize("m", [145, 1024])
@pytest.mark.parametrize("tid", C.PER_K)
def test_routes_agree(ops, tid, m):
    import torch

    w = weights(tid)
    x = C.make_x(torch, tid, m)
    args = (w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"])
    routed = ops.exl3_linear(x, *args, True)
    direct = torch.ops._C_exl3.exl3_gemm(x, *args, True)  # the gemm kernel at >144 rows (16-row passes)
    assert ops._exl3_op(m) != ops.EXL3_GEMM
    d = C.err_stats(torch, routed, direct.double())
    # two routes, each ~2.5e-3 of RMS from fp64 (exllamav3's own error: fp16-accumulate MMA on
    # sm86); their difference is bounded by the sum (measured 2.3-2.5e-3, box run 1)
    assert d["rel_rms"] <= 5e-3 and d["max_rel"] <= 3e-2, d


@pytest.mark.parametrize("m", [1, 8, 16, 48])
@pytest.mark.parametrize("tid", ["K3-down", "K4-kproj", C.HEAD])
def test_fp32_fp16_outputs_agree(ops, tid, m):
    import torch

    w = weights(tid)
    x = C.make_x(torch, tid, m)
    args = (w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"])
    y32, y16 = torch.ops._C_exl3.exl3_gemm(x, *args, True), torch.ops._C_exl3.exl3_gemm(x, *args, False)
    d = C.err_stats(torch, y16, y32.double())
    # fp16 and fp32 outputs are autotuned separately (the tune key holds the output dtype), so
    # from 16 rows they can run different tile shapes: measured 1.2e-3 at 16/48 rows (box run 1)
    assert d["rel_rms"] <= 3e-3, d


@pytest.mark.parametrize("m", GRAPH_ROWS)
@pytest.mark.parametrize("tid", GRAPH_TIDS)
def test_graph_replay(ops, tid, m):
    """exl3_gemm captured after exl3_warmup, replayed with new inputs == eager on those inputs."""
    import torch

    w = weights(tid)
    args = (w["trellis"], w["suh"], w["svh"], w["mcg"], w["mul1"])
    torch.ops._C_exl3.exl3_warmup(*args, WARM_ROWS, True)
    x_static = C.make_x(torch, tid, m)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            ops.exl3_linear(x_static, *args, True)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y_static = ops.exl3_linear(x_static, *args, True)
    for i in range(3):
        x_new = torch.randn(m, x_static.shape[1], generator=torch.Generator().manual_seed(i + 7)).half().cuda()
        x_static.copy_(x_new)
        g.replay()
        torch.cuda.synchronize()
        eager = ops.exl3_linear(x_new, *args, True)
        d = C.err_stats(torch, y_static, eager.double())
        print(f"\n{tid} m={m} replay {i}: bit-identical={torch.equal(y_static, eager)} {d}")
        assert d["rel_rms"] <= 1e-3 and d["finite"], d
    del g


@pytest.mark.parametrize("case", SUB_CASES)
def test_subprocess_case(ops, case):
    """One case per fresh process (tests/gpu/_exl3_case.py): an empty warmup registry, and a
    device fault cannot poison the rest of the session."""
    cmd = [sys.executable, str(Path(__file__).with_name("_exl3_case.py")), case, "K4-kproj"]
    env = dict(os.environ, CUDA_LAUNCH_BLOCKING="1",
               PYTHONPATH=os.pathsep.join([str(ROOT / "tools"), str(ROOT / "plugin-exl3"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    tail = (p.stdout + p.stderr)[-3000:]
    assert not any(f in p.stderr for f in FAULT[:5]), f"device fault:\n{tail}"
    assert p.returncode == 0, f"exit {p.returncode}:\n{tail}"
    res = json.loads(next(ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("{")))
    want = "ok" if case == "first_call_in_capture" else "rejected"
    assert res["status"] == want, res
    assert res.get("healthy_after", True), f"CUDA context unusable after the case: {res}"
