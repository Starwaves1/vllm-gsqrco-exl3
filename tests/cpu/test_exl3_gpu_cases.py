"""tests/gpu/exl3_cases.py (the EXL3 GPU kernel cases) against the checkpoint's metadata, CPU
only: every case tensor exists with the pinned K, k, n and the mul1 codebook, and the row list
covers the routing boundaries."""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests/gpu"))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))
HF = os.path.join(ROOT, "hf-config/Qwen3.8-27B-exl3-3.50bpw")


def test_case_tensors_match_checkpoint():
    import exl3_cases as C

    heads = json.load(open(os.path.join(HF, "safetensors_headers.json")))
    wm = json.load(open(os.path.join(HF, "model.safetensors.index.json")))["weight_map"]
    shapes = {}
    for info in heads.values():
        for name, t in (info.get("tensors") or info).items():
            if isinstance(t, dict) and "shape" in t:
                shapes[name] = t["shape"]
    ks = set()
    for tid, (prefix, K, k, n) in C.TENSORS.items():
        assert shapes[f"{prefix}.trellis"] == [k // 16, n // 16, 16 * K], tid
        assert shapes[f"{prefix}.suh"] == [k] and shapes[f"{prefix}.svh"] == [n], tid
        assert f"{prefix}.mul1" in wm and f"{prefix}.mcg" not in wm, tid
        ks.add(K)
    assert ks == {3, 4, 5, 6}  # every bit width in the 3.50bpw checkpoint


def test_rows_cover_routing_boundaries():
    import exl3_cases as C
    from vllm_exl3_plugin import ops

    assert set(range(1, 18)) <= set(C.ROWS)
    assert {ops.GEMM_MAX_ROWS + 1, ops.FUSED_RECON_MIN_ROWS} <= set(C.ROWS)
    assert {ops._exl3_op(m) for m in C.ROWS} == {ops.EXL3_GEMM, ops.RECON_HGEMM, ops.RECON_HAD_HGEMM}
    assert [s for s, _ in C.slices(248320)] == [i * 32768 for i in range(8)]
