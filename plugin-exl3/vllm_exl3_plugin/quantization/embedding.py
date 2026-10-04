# SPDX-License-Identifier: Apache-2.0
"""Token embedding in page-locked host memory (EXL3_EMBED_HOST, default on; ops.EMBED_HOST).

The checkpoint's bf16 embedding (248320 x 5120, 2.37 GiB) loads as vLLM's unquantized one, then
process_weights_after_loading copies it to the CPU, page-locks that copy in place
(exl3_embed_host_register: exactly its size) and frees the GPU copy; each forward gathers its rows
to the GPU with torch.ops._C_exl3.exl3_embed_host (a kernel reading the table over PCIe,
graph-capturable), about 10 KB per token. Same values. The op takes the table's registration id,
not its address: vLLM's AOT compile cache reloads graphs without guards, and the id is the same
in every process. Only the target model's embedding uses it: the MTP draft shares the target's
module (vLLM) and its own copy is dropped at load; the draft head stays on the GPU (a
ParallelLMHead).
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod


class EXL3HostEmbeddingMethod(UnquantizedEmbeddingMethod):
    table: tuple[int, int] | None = None  # (registration id, cols): ints, constants to dynamo

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight.data
        if w.device.type != "cuda" or w.dtype != torch.bfloat16:
            return  # CPU (tests) or an unexpected dtype: stays a plain embedding
        self.table = (torch.ops._C_exl3.exl3_embed_host_register(w.cpu()), w.shape[1])
        w.untyped_storage().resize_(0)  # frees the 2.37 GiB even if something still aliases it
        layer.weight.data = torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device)

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if self.table is None:
            return super().embedding(layer, input_)
        return torch.ops.vllm._exl3_embed_host(input_, *self.table)
