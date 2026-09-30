# SPDX-License-Identifier: Apache-2.0
"""vLLM general plugin entry point (pyproject: vllm.general_plugins exl3).

Registers quant method "exl3" and nothing else: an EXL3 checkpoint is a normal HF directory
whose config.json carries quantization_config.quant_method "exl3", so vLLM's own config
detection and default safetensors loader serve it. No monkeypatches."""

from vllm.model_executor.layers.quantization import register_quantization_config

from .format import QUANT_METHOD


def register() -> None:
    """Register the EXL3 quantization config with vLLM (re-registering overwrites)."""
    from .quantization import EXL3Config

    register_quantization_config(QUANT_METHOD)(EXL3Config)
