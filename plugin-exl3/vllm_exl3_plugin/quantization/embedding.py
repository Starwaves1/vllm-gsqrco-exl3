# SPDX-License-Identifier: Apache-2.0
"""Token embedding in pinned host memory (EXL3_EMBED_HOST=1, ops.EMBED_HOST).

The checkpoint's bf16 embedding (248320 x 5120, 2.37 GiB) loads as vLLM's unquantized one, then
process_weights_after_loading copies it into pinned host memory and frees the GPU copy; each
forward gathers its rows to the GPU with torch.ops._C_exl3.exl3_embed_host (a kernel reading the
pinned table over PCIe, graph-capturable), about 10 KB per token. Same values. The table's
address goes to the op as an int, so no CPU tensor enters the compiled graph; the pinned tensor
is kept alive here. The MTP draft shares the target's embedding module (vLLM), the draft head
stays on the GPU (it is a ParallelLMHead, not this method).
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod


class EXL3HostEmbeddingMethod(UnquantizedEmbeddingMethod):
    host: torch.Tensor | None = None
    table: tuple[int, int, int] | None = None  # (address, rows, cols): ints, constants to dynamo

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        if w.device.type != "cuda" or w.dtype != torch.bfloat16:
            return  # CPU (tests) or an unexpected dtype: stays a plain embedding
        self.host = torch.empty(w.shape, dtype=w.dtype, pin_memory=True)
        self.host.copy_(w)
        self.table = (self.host.data_ptr(), w.shape[0], w.shape[1])
        w.untyped_storage().resize_(0)  # frees the 2.37 GiB even if something still aliases it
        layer.weight.data = torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if self.table is None:
            return super().embedding(layer, input_)
        return torch.ops.vllm._exl3_embed_host(input_, *self.table)
