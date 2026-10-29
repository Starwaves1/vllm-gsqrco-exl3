# SPDX-License-Identifier: Apache-2.0
"""Attention backend for vLLM's ``int4_per_token_head`` KV cache.

Everything but decode attention is inherited from the Triton backend: cache
layout, scale views, the store kernel, the metadata builder and CUDA graph
support. Batches whose queries are at most ``MAX_QUERY_LEN`` tokens long
(decode and MTP verify) run ``csrc/int4kv/int4kv_attention.cu``; longer ones
(prefill) run the inherited Triton int4 kernel.

Select with ``--attention-backend CUSTOM --kv-cache-dtype int4_per_token_head``.
"""

from typing import ClassVar

import torch
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs

from . import _C_int4kv  # noqa: F401  registers torch.ops._C_int4kv

# 24 query rows per KV head (GQA 6 x MTP k=3 verify) is the kernel's limit.
MAX_QUERY_ROWS = 24


class Int4KVAttentionBackend(TritonAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["int4_per_token_head"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["Int4KVAttentionImpl"]:
        return Int4KVAttentionImpl

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 256

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major >= 8


class Int4KVAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_query_len = MAX_QUERY_ROWS // (self.num_heads // self.num_kv_heads)
        self._cuda_ok = (
            self.sliding_window in (None, (-1, -1))
            and not self.logits_soft_cap
            and self.alibi_slopes is None
            and self.sinks is None
            and self.chunk_lookback == -1
        )
        self._rht_signs: torch.Tensor | None = None

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if (
            attn_metadata is None
            or not self._cuda_ok
            or attn_metadata.max_query_len > self._max_query_len
            or not attn_metadata.causal
            or attn_metadata.mm_prefix_range_tensor is not None
            or attn_metadata.rswa_prefix_lens is not None
            or output_scale is not None
            or output_block_scale is not None
        ):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale, output_block_scale,
            )
        if self._rht_signs is None:
            self._rht_signs = _get_rht_signs(self.head_size, 0, query.device)
        n = attn_metadata.num_actual_tokens
        torch.ops._C_int4kv.int4kv_attention(
            output[:n].view(n, self.num_heads, self.head_size),
            query[:n],
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            self._rht_signs,
            self.scale,
            attn_metadata.max_query_len,
        )
        return output
