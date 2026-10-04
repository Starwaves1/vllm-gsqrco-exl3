"""tests/gpu/exl3_cases.py (the EXL3 GPU kernel cases) against each checkpoint's metadata, CPU
only: every case tensor exists with the pinned K, k, n and the mul1 codebook, every text-model
bit width has a case, and the row list covers the routing boundaries."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests/gpu"))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))
import pytest  # noqa: E402

import exl3_cases as C  # noqa: E402


@pytest.fixture(autouse=True)
def _phase1_routing(monkeypatch):
    """Phase 1's table (EXL3_MR=0): the GPU case table is phase 1's; EXL3_MR is test_exl3_mr.py's."""
    from vllm_exl3_plugin import ops

    monkeypatch.setattr(ops, "MR_MODE", 0)
    monkeypatch.setattr(ops, "MULTI_ROW_OP", None)
    monkeypatch.setattr(ops, "MULTI_ROW_MIN", 17)
    monkeypatch.setattr(ops, "MR_GLUE", False)


@pytest.mark.parametrize("ckpt", list(C.CHECKPOINTS))
def test_case_tensors_match_checkpoint(ckpt):
    hf = os.path.join(ROOT, "hf-config", ckpt)
    heads = json.load(open(os.path.join(hf, "safetensors_headers.json")))
    wm = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
    shapes = {}
    for info in heads.values():
        for name, t in (info.get("tensors") or info).items():
            if isinstance(t, dict) and "shape" in t:
                shapes[name] = t["shape"]
    ks = set()
    for tid, (prefix, K, k, n) in C.CHECKPOINTS[ckpt].items():
        assert shapes[f"{prefix}.trellis"] == [k // 16, n // 16, 16 * K], tid
        assert shapes[f"{prefix}.suh"] == [k] and shapes[f"{prefix}.svh"] == [n], tid
        assert f"{prefix}.mul1" in wm and f"{prefix}.mcg" not in wm, tid
        ks.add(K)
    text = {s[2] // 16 for n, s in shapes.items() if n.endswith(".trellis") and "visual" not in n}
    assert ks == text  # every bit width the text model uses (swift 2..5, turboderp 3..6)


def test_rows_cover_routing_boundaries():
    from vllm_exl3_plugin import ops

    assert set(range(1, 18)) <= set(C.ROWS)
    assert {ops.GEMM_MAX_ROWS + 1, ops.FUSED_RECON_MIN_ROWS} <= set(C.ROWS)
    assert {ops._exl3_op(m) for m in C.ROWS} == {ops.EXL3_GEMM, ops.RECON_HGEMM, ops.RECON_HAD_HGEMM}
    assert [s for s, _ in C.slices(248320)] == [i * 32768 for i in range(8)]
