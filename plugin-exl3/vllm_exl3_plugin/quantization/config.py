# SPDX-License-Identifier: Apache-2.0
"""QuantizationConfig for exllamav3 EXL3 checkpoints (quant_method "exl3")."""

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

from .. import ops
from ..format import QUANT_METHOD, EXL3QuantConfig
from ..weights_adapter.qwen3_5 import DRAFT_HEAD, is_unquantized_module


class EXL3Config(QuantizationConfig):
    """config.json's `quantization_config` block; the per-tensor bit widths come from
    each tensor's trellis shape at load time.

    Methods by layer: linears and the lm_head get EXL3LinearMethod, except the modules the
    checkpoint stores unquantized (weights_adapter.qwen3_5.is_unquantized_module: GDN
    in_proj_ba, the vision tower, the pruned MTP draft head), which get vLLM's unquantized
    methods; the input embedding stays bf16, in page-locked host memory (EXL3_EMBED_HOST, default
    on; the MTP draft's own copy stays plain: vLLM replaces it with the target's)."""

    def __init__(self, quant: EXL3QuantConfig) -> None:
        super().__init__()
        self.quant = quant

    def __repr__(self) -> str:
        return f"EXL3Config({self.quant})"

    def get_name(self) -> QuantizationMethods:
        return QUANT_METHOD

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # the kernels take fp16 activations; bf16 is cast at the op boundary
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # config.json's quantization_config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EXL3Config":
        return cls(EXL3QuantConfig.from_dict(config))

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        from .linear import EXL3LinearMethod

        if isinstance(layer, LinearBase):
            if "visual" in prefix.split(".") and self.quant.vision_bits is not None:
                _refuse_quantized_vision_tower(prefix, self.quant.vision_bits)
            if is_unquantized_module(prefix):
                return UnquantizedLinearMethod()
            return EXL3LinearMethod(self)
        if isinstance(layer, ParallelLMHead):
            if is_unquantized_module(prefix):
                if ops.DRAFT_FP8 and DRAFT_HEAD in prefix.split("."):
                    from .draft_head import EXL3DraftHeadFp8Method

                    return EXL3DraftHeadFp8Method()  # the MTP draft head in fp8 (EXL3_DRAFT_FP8)
                return None  # vLLM's UnquantizedEmbeddingMethod
            return EXL3LinearMethod(self)
        if ops.EMBED_HOST and isinstance(layer, VocabParallelEmbedding) and "mtp" not in prefix.split("."):
            from .embedding import EXL3HostEmbeddingMethod

            return EXL3HostEmbeddingMethod()  # bf16, in pinned host memory
        return None  # VocabParallelEmbedding: bf16 in the checkpoint, on the GPU


def _refuse_quantized_vision_tower(prefix: str, bits: float) -> None:
    """An EXL3 vision tower (erlidev's "V6" quants) is not loadable here: vLLM's tower wants a
    fused bf16 qkv and 4304 MLP rows, the checkpoint has EXL3 q/k/v and 4352 padded rows. Text-only
    serving (image and video limits 0, as production's argv) never builds the tower, and vLLM's
    loader skips its tensors, so only refuse when the tower is being built."""
    from vllm.config import get_current_vllm_config_or_none

    vc = get_current_vllm_config_or_none()
    mm = vc.model_config.multimodal_config if vc is not None and vc.model_config else None
    if mm is not None and all(mm.get_limit_per_prompt(m) == 0 for m in ("image", "video")):
        return
    raise NotImplementedError(
        f"{prefix}: this checkpoint's vision tower is EXL3-quantized (vision_bits {bits:g}), which "
        "the EXL3 plugin cannot load; serve text-only with --limit-mm-per-prompt "
        "'{\"image\": 0, \"video\": 0}' (--enable-mm-embeds still takes precomputed embeddings)")
