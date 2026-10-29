# SPDX-License-Identifier: Apache-2.0
"""Attention backend for vLLM's ``int4_per_token_head`` KV cache.

Cache layout, scale views, the store kernel and CUDA graph support are
inherited from the Triton backend; attention never runs the Triton kernel:

- First prefill chunks (no cached context) attend over their own unquantized
  K/V with FlashAttention, like vLLM's TurboQuant backend.
- Everything else (decode, MTP verify, chunks that extend a cached prefix)
  runs ``csrc/int4kv/int4kv_attention.cu`` over the int4 cache. Its CTAs take
  at most ``MAX_QUERY_ROWS`` query rows per KV head, so the metadata builder
  cuts longer queries into causal pieces of ``MAX_QUERY_ROWS // group`` tokens.

The runner orders a batch as decodes, extends, then first chunks, so the two
paths cover two contiguous token ranges.

Select with ``--attention-backend CUSTOM --kv-cache-dtype int4_per_token_head``;
with MTP, also put ``"attention_backend": "CUSTOM"`` in ``--speculative-config``
or the drafter's layer picks its own backend.
"""

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np
import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.int4_per_token_head import _get_rht_signs

from . import _C_int4kv  # noqa: F401  registers torch.ops._C_int4kv

# Query rows per KV head one CTA of the kernel holds (GQA 6 x MTP k=3 verify).
MAX_QUERY_ROWS = 24


def split_queries(query_start_loc: np.ndarray, max_len: int) -> tuple[np.ndarray, ...]:
    """Cut each request's query into consecutive pieces of <= ``max_len`` tokens.

    Returns, per piece: its request, its first token, and how many of the
    request's query tokens follow it (so its causal end is seq_len - tail).
    """
    q_lens = np.diff(query_start_loc)
    pieces = -(-q_lens // max_len)
    req = np.repeat(np.arange(len(q_lens)), pieces)
    rank = np.arange(len(req)) - np.repeat(np.cumsum(pieces) - pieces, pieces)
    start = query_start_loc[req] + max_len * rank
    tail = query_start_loc[req + 1] - np.minimum(start + max_len, query_start_loc[req + 1])
    return req, start, tail


@dataclass
class Int4KVMetadata(TritonAttentionMetadata):
    # Kernel part: tokens [0, kernel_tokens), queries of <= kernel_max_query_len.
    kernel_tokens: int = 0
    kernel_max_query_len: int = 0
    kernel_query_start_loc: torch.Tensor | None = None
    kernel_seq_lens: torch.Tensor | None = None
    kernel_block_table: torch.Tensor | None = None
    # First chunks: tokens [kernel_tokens, num_actual_tokens), FlashAttention.
    fresh_query_start_loc: torch.Tensor | None = None
    fresh_max_query_len: int = 0


class Int4KVMetadataBuilder(TritonAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        heads = vllm_config.model_config.get_num_attention_heads(vllm_config.parallel_config)
        group = heads // kv_cache_spec.num_kv_heads
        self.max_query_len = MAX_QUERY_ROWS // group
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
        if self.reorder_batch_threshold > self.max_query_len:
            raise ValueError(
                f"CUSTOM int4 attention: {group} query heads per KV head x "
                f"{self.reorder_batch_threshold} tokens per decode step is more than "
                f"the kernel's {MAX_QUERY_ROWS} rows; lower num_speculative_tokens"
            )

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        meta = {f.name: getattr(base, f.name) for f in fields(base)}
        cam = common_attn_metadata
        if cam.max_query_len <= self.max_query_len:  # decode / verify: graph-safe tensors
            return Int4KVMetadata(
                **meta,
                kernel_tokens=cam.num_actual_tokens,
                kernel_max_query_len=cam.max_query_len,
                kernel_query_start_loc=base.query_start_loc,
                kernel_seq_lens=base.seq_lens,
                kernel_block_table=base.block_table,
            )

        qsl = cam.query_start_loc_cpu[: cam.num_reqs + 1].numpy()
        q_lens = np.diff(qsl)
        fresh = cam.seq_lens_cpu_upper_bound[: cam.num_reqs].numpy() == q_lens
        cached = np.flatnonzero(~fresh)
        n = int(cached[-1]) + 1 if cached.size else 0  # first chunks are last
        dev = base.seq_lens.device
        req, start, tail = split_queries(qsl[: n + 1], self.max_query_len)
        req_t = torch.from_numpy(req).to(dev)
        return Int4KVMetadata(
            **meta,
            kernel_tokens=int(qsl[n]),
            kernel_max_query_len=int(min(q_lens[:n].max(), self.max_query_len)) if n else 0,
            kernel_query_start_loc=torch.from_numpy(np.append(start, qsl[n]).astype(np.int32)).to(dev),
            kernel_seq_lens=base.seq_lens[req_t] - torch.from_numpy(tail.astype(np.int32)).to(dev),
            kernel_block_table=base.block_table[req_t],
            fresh_query_start_loc=torch.from_numpy((qsl[n:] - qsl[n]).astype(np.int32)).to(dev),
            fresh_max_query_len=int(q_lens[n:].max()) if n < cam.num_reqs else 0,
        )


class Int4KVAttentionBackend(TritonAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["int4_per_token_head"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["Int4KVAttentionImpl"]:
        return Int4KVAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Int4KVMetadataBuilder]:
        return Int4KVMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 256

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        return capability.major >= 8

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_rswa(cls) -> bool:
        return False


class Int4KVAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        unsupported = {
            "logits soft cap": bool(self.logits_soft_cap),
            "ALiBi": self.alibi_slopes is not None,
            "attention sinks": self.sinks is not None,
            "chunked local attention": self.chunk_lookback != -1,
        }
        if any(unsupported.values()):
            names = ", ".join(k for k, v in unsupported.items() if v)
            raise NotImplementedError(f"CUSTOM int4 attention does not support {names}")
        self._rht_signs: torch.Tensor | None = None

    def fused_output_quant_supported(self, quant_key) -> bool:
        return False

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata: Int4KVMetadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if attn_metadata is None:  # profiling run
            return output
        m = attn_metadata
        if self._rht_signs is None:
            self._rht_signs = _get_rht_signs(self.head_size, 0, query.device)
        out = output.view(-1, self.num_heads, self.head_size)
        n, end = m.kernel_tokens, m.num_actual_tokens
        if n:
            torch.ops._C_int4kv.int4kv_attention(
                out[:n], query[:n], kv_cache, m.kernel_block_table, m.kernel_seq_lens,
                m.kernel_query_start_loc, self._rht_signs, self.scale, m.kernel_max_query_len,
            )
        if end > n:
            flash_attn_varlen_func(
                q=query[n:end], k=key[n:end], v=value[n:end], out=out[n:end],
                cu_seqlens_q=m.fresh_query_start_loc, cu_seqlens_k=m.fresh_query_start_loc,
                max_seqlen_q=m.fresh_max_query_len, max_seqlen_k=m.fresh_max_query_len,
                softmax_scale=self.scale, causal=True, fa_version=2,
            )
        return output
