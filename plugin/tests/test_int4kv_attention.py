# SPDX-License-Identifier: Apache-2.0
"""CUDA int4_per_token_head decode attention vs vLLM's Triton int4 kernel.

Both read the same packed cache (written by vLLM's own store kernel); an exact
fp32 attention over the dequantized cache is the referee.
"""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm_gguf_plugin._C_int4kv")
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl  # noqa: E402
from vllm.v1.attention.ops.int4_per_token_head import (  # noqa: E402
    _get_rht_signs,
    reshape_and_cache_int4,
    unified_attention_int4,
)

import vllm_gguf_plugin.int4kv_attention  # noqa: E402,F401

HEAD, KV_HEADS, GROUP = 256, 4, 6
SLOT = 2 * (HEAD // 2 + 4)


def build_case(num_seqs, ctx, q_len, layout="NHD", block_size=1616, dtype=torch.bfloat16, seed=0):
    """Packed cache with shuffled blocks, seq lens in (7/8 ctx, ctx], and Q."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dev = torch.device("cuda")
    lens = [ctx - int(torch.randint(0, max(1, ctx // 8), (1,), generator=gen, device=dev)) for _ in range(num_seqs)]
    lens[0] = ctx
    blocks = [math.ceil(n / block_size) for n in lens]
    num_blocks = sum(blocks) + 3
    if layout == "HND":
        kv = torch.zeros(num_blocks, KV_HEADS, block_size, SLOT, dtype=torch.uint8, device=dev)
    else:
        kv = torch.zeros(num_blocks, block_size, KV_HEADS, SLOT, dtype=torch.uint8, device=dev).transpose(1, 2)
    perm = torch.randperm(num_blocks, generator=gen, device=dev).to(torch.int32)
    table = torch.zeros(num_seqs, max(blocks), dtype=torch.int32, device=dev)
    scales = SimpleNamespace(_k_scale_cache=None, _v_scale_cache=None)
    TritonAttentionImpl._ensure_scale_caches(scales, kv)
    key_cache, value_cache = kv.transpose(1, 2).split(SLOT // 2, dim=-1)
    # Per-head magnitudes and a few needle keys aligned with the queries so the
    # softmax is neither flat nor one-hot.
    head_scale = torch.tensor([0.5, 1.0, 2.0, 4.0], device=dev)[None, :, None]
    q = torch.randn(sum([q_len] * num_seqs), KV_HEADS * GROUP, HEAD, generator=gen, device=dev)
    q = (q * 2.0).to(dtype)
    start = 0
    for s, n in enumerate(lens):
        table[s, : blocks[s]] = perm[start : start + blocks[s]]
        start += blocks[s]
        pos = torch.arange(n, device=dev)
        slots = table[s, pos // block_size].long() * block_size + pos % block_size
        for c0 in range(0, n, 65536):
            c1 = min(n, c0 + 65536)
            k = torch.randn(c1 - c0, KV_HEADS, HEAD, generator=gen, device=dev) * head_scale
            v = torch.randn(c1 - c0, KV_HEADS, HEAD, generator=gen, device=dev) * head_scale
            needles = torch.randint(0, c1 - c0, (8,), generator=gen, device=dev)
            qs = q[s * q_len].float().view(KV_HEADS, GROUP, HEAD)[:, 0]
            k[needles] = qs * 0.5
            reshape_and_cache_int4(
                k.to(dtype), v.to(dtype), key_cache, value_cache, slots[c0:c1],
                k_scale_cache=scales._k_scale_cache, v_scale_cache=scales._v_scale_cache,
            )
    return SimpleNamespace(
        q=q, kv=kv, key_cache=key_cache, value_cache=value_cache,
        k_scale=scales._k_scale_cache, v_scale=scales._v_scale_cache, table=table,
        seq_lens=torch.tensor(lens, dtype=torch.int32, device=dev),
        qsl=torch.arange(0, (num_seqs + 1) * q_len, q_len, dtype=torch.int32, device=dev),
        q_len=q_len, max_len=max(lens), scale=HEAD**-0.5, block_size=block_size,
    )


def triton_segments(case, threshold=32, segments=16):
    """The split-KV buffers TritonAttentionMetadataBuilder allocates."""
    hq = case.q.shape[1]
    return dict(
        seq_threshold_3D=threshold,
        num_par_softmax_segments=segments,
        softmax_segm_output=torch.empty(threshold, hq, segments, HEAD, dtype=torch.float32, device="cuda"),
        softmax_segm_max=torch.empty(threshold, hq, segments, dtype=torch.float32, device="cuda"),
        softmax_segm_expsum=torch.empty(threshold, hq, segments, dtype=torch.float32, device="cuda"),
    )


def run_triton(case, out=None, segm=None):
    out = torch.empty_like(case.q) if out is None else out
    unified_attention_int4(
        case.q, case.key_cache, case.value_cache, out,
        cu_seqlens_q=case.qsl, max_seqlen_q=case.q_len, seqused_k=case.seq_lens,
        max_seqlen_k=case.max_len, softmax_scale=case.scale, window_size=(-1, -1),
        block_table=case.table, softcap=0.0, sinks=None, alibi_slopes=None,
        use_alibi_sqrt=False, qq_bias=None, output_scale=None, mm_prefix_range=None,
        k_scale_cache=case.k_scale, v_scale_cache=case.v_scale, **(segm or {}),
    )
    return out


def run_cuda(case, out=None):
    out = torch.empty_like(case.q) if out is None else out
    torch.ops._C_int4kv.int4kv_attention(
        out, case.q, case.kv, case.table, case.seq_lens, case.qsl,
        _get_rht_signs(HEAD, 0, case.q.device), case.scale, case.q_len,
    )
    return out


def hadamard(n=HEAD):
    h = torch.ones(1, 1, device="cuda")
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def unpack(rows):
    """[n, 132] packed half-slots -> fp32 [n, 256] in the rotated basis."""
    codes = torch.stack([rows[:, :128] & 0xF, rows[:, :128] >> 4], -1).reshape(-1, HEAD).float()
    bits = rows[:, 128:132].contiguous().view(torch.int32)
    scale = (bits & ~0xF).view(torch.float32)
    return scale * (codes - (bits & 0xF).float())


def reference(case):
    """Exact fp32 attention over the dequantized cache."""
    h, signs = hadamard(), _get_rht_signs(HEAD, 0, case.q.device)
    out = torch.empty(case.q.shape, dtype=torch.float32, device="cuda")
    for s, n in enumerate(case.seq_lens.tolist()):
        pos = torch.arange(n, device="cuda")
        blk = case.table[s, pos // case.block_size].long()
        slot = case.kv[blk, :, pos % case.block_size]  # [n, kv heads, 264]
        for kh in range(KV_HEADS):
            k = unpack(slot[:, kh, :132])
            v = unpack(slot[:, kh, 132:])
            q = case.q[s * case.q_len : (s + 1) * case.q_len, kh * GROUP : (kh + 1) * GROUP].float()
            qr = (q * signs) @ h  # [q_len, group, 256]
            logits = torch.einsum("lgd,nd->lgn", qr, k) * case.scale / HEAD
            causal = pos[None, :] > (n - case.q_len + torch.arange(case.q_len, device="cuda"))[:, None]
            logits.masked_fill_(causal[:, None, :], float("-inf"))
            o = torch.einsum("lgn,nd->lgd", logits.softmax(-1), v)
            out[s * case.q_len : (s + 1) * case.q_len, kh * GROUP : (kh + 1) * GROUP] = (o @ h) * signs / HEAD
    return out


def rel_err(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def check(case):
    ref = reference(case)
    cuda, triton = run_cuda(case), run_triton(case)
    e_cuda, e_triton = rel_err(cuda, ref), rel_err(triton, ref)
    # The Triton path rotates Q in bf16 and dots in TF32; the CUDA one rotates
    # in fp32, so it may only be closer to the exact result.
    assert e_cuda <= max(1.25 * e_triton, 4e-3), (e_cuda, e_triton)
    assert rel_err(cuda, triton) <= 2.0 * e_triton


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("num_seqs", [1, 2, 4, 8])
@pytest.mark.parametrize("ctx", [1024, 4096, 32768])
def test_matches_triton_and_reference(ctx, num_seqs, q_len, layout):
    check(build_case(num_seqs, ctx, q_len, layout))


@pytest.mark.parametrize("q_len", [1, 4])
@pytest.mark.parametrize("num_seqs", [1, 8])
@pytest.mark.parametrize("ctx", [131072, 262144])
def test_long_context(ctx, num_seqs, q_len):
    check(build_case(num_seqs, ctx, q_len))


def test_block_size_16_fp16_q_len_3():
    case = build_case(3, 5000, 3, block_size=16, dtype=torch.float16)
    assert rel_err(run_cuda(case), reference(case)) < 4e-3
