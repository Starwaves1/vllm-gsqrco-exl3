# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

import vllm_gguf_plugin.weights_adapter.qwen3_5 as qwen35
from vllm_gguf_plugin.gguf_files import GGUFModelFiles

TENSORS = ["token_embd.weight", "blk.0.attn_norm.weight", "output.weight"]
BACKBONE = GGUFModelFiles(backbone=("/models/qwen3.5-text.gguf",))


@pytest.fixture
def adapter(monkeypatch):
    """The adapter on a backbone GGUF with no mm_proj beside it (no file I/O)."""
    monkeypatch.setattr(qwen35, "get_gguf_tensor_names", lambda files: TENSORS)
    monkeypatch.setattr(
        qwen35,
        "maybe_patch_hf_config_from_gguf",
        lambda path, config, mmproj_path=None: config,
    )
    return qwen35.Qwen35GGUFAdapter()


def _model_config(hf_config, limits):
    """limits: modality -> --limit-mm-per-prompt, or None for no multimodal config."""
    mm_config = (
        None
        if limits is None
        else SimpleNamespace(get_limit_per_prompt=lambda modality: limits[modality])
    )
    return SimpleNamespace(hf_config=hf_config, multimodal_config=mm_config)


def test_multimodal_config_without_mm_proj_keeps_vision_config(adapter):
    patched = adapter.patch_hf_config(BACKBONE, Qwen3_5Config())
    assert patched.vision_config is not None
    assert patched.architectures == ["Qwen3_5ForConditionalGeneration"]


def test_multimodal_config_without_mm_proj_serves_text_only(adapter):
    """With image and video limits of 0 (or --language-model-only) the text
    weights map under the multimodal model's language_model prefix."""
    config = _model_config(Qwen3_5Config(), {"image": 0, "video": 0})
    name_map = adapter.build_name_map(BACKBONE, config)
    assert name_map["token_embd.weight"] == ("model.language_model.embed_tokens.weight")
    assert name_map["blk.0.attn_norm.weight"] == (
        "model.language_model.layers.0.input_layernorm.weight"
    )
    assert name_map["output.weight"] == "lm_head.weight"


@pytest.mark.parametrize(
    "limits",
    [{"image": 1, "video": 0}, {"image": 0, "video": 1}, None],
    ids=["image", "video", "no_multimodal_config"],
)
def test_multimodal_config_without_mm_proj_needs_vision_disabled(adapter, limits):
    config = _model_config(Qwen3_5Config(), limits)
    with pytest.raises(RuntimeError, match="mm_proj"):
        adapter.build_name_map(BACKBONE, config)


def test_text_config_without_mm_proj_is_unchanged(adapter):
    config = _model_config(Qwen3_5TextConfig(), None)
    name_map = adapter.build_name_map(BACKBONE, config)
    assert name_map["token_embd.weight"] == "model.embed_tokens.weight"
