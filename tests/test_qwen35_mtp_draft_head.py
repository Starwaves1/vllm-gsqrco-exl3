# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm_gguf_plugin.weights_adapter.qwen3_5 as qwen35
from vllm_gguf_plugin.gguf_files import GGUFModelFiles

FILES = GGUFModelFiles(backbone=("/models/qwen3.5-mtp.gguf",))
TENSORS = ["blk.1.nextn.eh_proj.weight", "output.weight", "token_embd.weight"]


def _config(model_dir):
    hf_config = SimpleNamespace(model_type="qwen3_5_mtp")
    return SimpleNamespace(model=str(model_dir), hf_config=hf_config)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(qwen35, "get_gguf_tensor_names", lambda files: TENSORS)
    monkeypatch.setattr(qwen35, "_find_nextn_block_index", lambda files: 1)
    return qwen35.Qwen35MtpGGUFAdapter()


def test_draft_lm_head_stays_a_gguf_placeholder():
    assert qwen35.Qwen35MtpGGUFAdapter.extra_unquantized_modules == ("embed_tokens",)


def test_pruned_draft_head_rows(adapter, tmp_path):
    """With mtp_draft_vocab_ids.pt in the model dir, output.weight maps to
    mtp.draft_lm_head.weight and keeps exactly those rows (raw GGUF blocks, in
    id order)."""
    ids = torch.tensor([0, 3, 7, 8])
    torch.save(ids, tmp_path / "mtp_draft_vocab_ids.pt")
    assert adapter.build_name_map(FILES, _config(tmp_path))["output.weight"] == (
        "mtp.draft_lm_head.weight"
    )
    w = torch.arange(10 * 144, dtype=torch.int32).to(torch.uint8).view(10, 144)
    weights = [
        ("mtp.draft_lm_head.weight_type", torch.tensor(12)),
        ("mtp.draft_lm_head.weight", w),
    ]
    out = dict(adapter.transform_weights(iter(weights), _config(tmp_path)))
    assert torch.equal(out["mtp.draft_lm_head.weight"], w[ids])
    assert int(out["mtp.draft_lm_head.weight_type"]) == 12


@pytest.mark.parametrize("off", ["no_file", "env_off"])
def test_full_draft_head_without_ids(adapter, tmp_path, monkeypatch, off):
    if off == "env_off":
        torch.save(torch.tensor([0, 1]), tmp_path / "mtp_draft_vocab_ids.pt")
        monkeypatch.setenv("MTP_DRAFT_VOCAB", "0")
    assert "output.weight" not in adapter.build_name_map(FILES, _config(tmp_path))
