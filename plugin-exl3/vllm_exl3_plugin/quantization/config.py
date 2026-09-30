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
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from ..format import QUANT_METHOD, EXL3QuantConfig
from ..weights_adapter.qwen3_5 import is_unquantized_module


class EXL3Config(QuantizationConfig):
    """config.json's `quantization_config` block; the per-tensor bit widths come from
    each tensor's trellis shape at load time.

    Methods by layer: linears and the lm_head get EXL3LinearMethod, except the modules the
    checkpoint stores unquantized (weights_adapter.qwen3_5.is_unquantized_module: GDN
    in_proj_ba, the vision tower, the pruned MTP draft head), which get vLLM's unquantized
    methods; the input embedding stays bf16 (unquantized, on the GPU)."""

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
            if is_unquantized_module(prefix):
                return UnquantizedLinearMethod()
            return EXL3LinearMethod(self)
        if isinstance(layer, ParallelLMHead):
            if is_unquantized_module(prefix):
                return None  # vLLM's UnquantizedEmbeddingMethod
            return EXL3LinearMethod(self)
        return None  # VocabParallelEmbedding: bf16 in the checkpoint
