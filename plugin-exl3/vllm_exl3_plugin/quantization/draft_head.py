# SPDX-License-Identifier: Apache-2.0
"""MTP draft head in fp8 (EXL3_DRAFT_FP8): weight-only e4m3 with one scale per vocab row, run by
vLLM's fp8 Marlin kernel (bf16 activations, Ampere-capable), so each draft step reads 0.21 GB
instead of the bf16 head's 0.42 GB. Only the drafter's proposals change (acceptance is measured);
the target's logits are untouched. The head stays on the GPU."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

FP8_MAX = 448.0  # e4m3


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """w [n, k] -> (e4m3 [n, k], fp32 scale [n]) with w ~ q * scale, scale = max|row| / 448."""
    scale = (w.float().abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)
    q = (w.float() / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


class EXL3DraftHeadFp8Method(UnquantizedEmbeddingMethod):
    marlin: torch.nn.Module | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        if w.device.type != "cuda":
            return  # CPU (tests): stays bf16
        from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_layer_for_marlin

        # Quantized in row chunks staged on the host, then the bf16 head is freed before the fp8 copy and its
        # Marlin repack are allocated, so they reuse its block: under vLLM's cumem allocator (production's argv)
        # freed weight memory stays in its pool until the load ends, and the whole-head fp32 temporaries (0.8 GiB
        # each) OOMed there. Same GPU arithmetic per row as one call.
        (n, k), dtype, dev = w.shape, w.dtype, w.device
        chunks = [[t.cpu() for t in quantize_rows(w[i:i + 1024])] for i in range(0, n, 1024)]
        q, scale = (torch.cat(ts) for ts in zip(*chunks))
        w.untyped_storage().resize_(0)
        layer.weight.data = torch.empty((0, k), dtype=dtype, device=dev)
        m = torch.nn.Module()  # what prepare_fp8_layer_for_marlin reads
        m.output_size_per_partition, m.input_size_per_partition, m.orig_dtype = n, k, dtype
        m.weight = torch.nn.Parameter(q.to(dev), requires_grad=False)
        m.weight_scale = torch.nn.Parameter(scale.to(dev), requires_grad=False)
        prepare_fp8_layer_for_marlin(m, size_k_first=False)
        m.workspace = marlin_make_workspace_new(dev)
        self.marlin = m

    def apply(self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.marlin is None:
            return super().apply(layer, x, bias)
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import apply_fp8_marlin_linear

        m = self.marlin
        return apply_fp8_marlin_linear(x, m.weight, m.weight_scale, m.workspace, m.output_size_per_partition,
                                       m.input_size_per_partition, bias)
