"""tools/exl3_draft_head.py on a synthetic EXL3 checkpoint, CPU only.

The dequant kernel is replaced by a stand-in that returns slices of a known dense W[in, out];
the tool must write exactly W.T[ids] as bf16 under mtp.draft_lm_head.weight, register the
file in the index (replacing a symlinked index rather than writing through it), and save the
sorted ids. Checked: across dequant slices, with ids in any order, on a rerun, and the
rejects (duplicate / out-of-range ids).
"""

import json
import os
import sys

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))

K, N = 128, 1024


@pytest.fixture
def ckpt(tmp_path):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(K, N, generator=g).half()  # original-basis W[in, out]
    blob = tmp_path / "blob"
    blob.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    shard = {"lm_head.trellis": torch.zeros(K // 16, N // 16, 96, dtype=torch.int16),
             "lm_head.suh": torch.ones(K, dtype=torch.half), "lm_head.svh": torch.ones(N, dtype=torch.half),
             "lm_head.mul1": torch.tensor(-2082680531, dtype=torch.int32),
             "model.norm.weight": torch.ones(K, dtype=torch.bfloat16)}
    save_file(shard, str(model / "model-00001-of-00001.safetensors"))
    index = {"metadata": {}, "weight_map": {n: "model-00001-of-00001.safetensors" for n in shard}}
    # as in an HF cache snapshot: the index is a symlink into a blob store
    (blob / "index").write_text(json.dumps(index))
    os.symlink(blob / "index", model / "model.safetensors.index.json")
    calls = []

    def dequant(trellis, suh, svh, mcg, mul1, n_start, n_count):
        calls.append((n_start, n_count, mcg, mul1))
        assert trellis.shape == (K // 16, N // 16, 96)
        return w[:, n_start:n_start + n_count].clone()

    return model, blob, w, dequant, calls


def test_writes_rows(ckpt):
    from exl3_draft_head import build_draft_head

    model, blob, w, dequant, calls = ckpt
    ids = torch.tensor([1000, 3, 511, 512, 0, 1023, 700])
    info = build_draft_head(str(model), ids, dequant=dequant, device="cpu", slice_n=512)
    assert info == {"rows": 7, "hidden": K, "vocab": N, "bytes": 7 * K * 2}
    assert calls == [(0, 512, False, True), (512, 512, False, True)]
    want_ids = torch.sort(ids).values
    with safe_open(str(model / "mtp_draft_head.safetensors"), framework="pt") as f:
        rows = f.get_tensor("mtp.draft_lm_head.weight")
    assert rows.dtype == torch.bfloat16 and rows.shape == (7, K)
    assert torch.equal(rows, w.t()[want_ids].to(torch.bfloat16))
    assert torch.equal(torch.load(model / "mtp_draft_vocab_ids.pt"), want_ids)
    index = json.loads((model / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["mtp.draft_lm_head.weight"] == "mtp_draft_head.safetensors"
    assert not (model / "model.safetensors.index.json").is_symlink()
    assert "mtp.draft_lm_head.weight" not in json.loads((blob / "index").read_text())["weight_map"]
    assert "mtp.draft_lm_head.weight" not in json.loads(
        (model / "model.safetensors.index.json.orig").read_text())["weight_map"]


def test_skips_slices_without_ids_and_reruns(ckpt):
    from exl3_draft_head import build_draft_head

    model, _, w, dequant, calls = ckpt
    build_draft_head(str(model), torch.tensor([600, 601]), dequant=dequant, device="cpu", slice_n=512)
    assert calls == [(512, 512, False, True)]
    build_draft_head(str(model), torch.tensor([5]), dequant=dequant, device="cpu", slice_n=512)
    with safe_open(str(model / "mtp_draft_head.safetensors"), framework="pt") as f:
        assert torch.equal(f.get_tensor("mtp.draft_lm_head.weight"), w.t()[[5]].to(torch.bfloat16))
    index = json.loads((model / "model.safetensors.index.json").read_text())
    assert list(index["weight_map"]).count("mtp.draft_lm_head.weight") == 1


@pytest.mark.parametrize("ids,match", [([1, 1], "unique"), ([N], r"\[0, 1024\)"), ([-1], r"\[0, 1024\)"),
                                       ([], "no draft ids")])
def test_rejects(ckpt, ids, match):
    from exl3_draft_head import build_draft_head

    model, _, _, dequant, _ = ckpt
    with pytest.raises(ValueError, match=match):
        build_draft_head(str(model), torch.tensor(ids, dtype=torch.int64), dequant=dequant, device="cpu")


def test_load_ids(tmp_path):
    from exl3_draft_head import load_ids

    (tmp_path / "ids.json").write_text(json.dumps([3, 1, 2]))
    torch.save(torch.tensor([4, 5], dtype=torch.int32), tmp_path / "ids.pt")
    assert load_ids(str(tmp_path / "ids.json")).tolist() == [3, 1, 2]
    assert load_ids(str(tmp_path / "ids.pt")).dtype == torch.int64
