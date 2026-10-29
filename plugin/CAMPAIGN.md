# int4 KV attention campaign

Goal: make vLLM's `int4_per_token_head` KV cache fast on RTX 3090 (sm86) and RTX 4070 Ti
(sm89) with a from-scratch CUDA paged-attention kernel that reads exactly the cache
vLLM writes, living in this plugin so it can be upstreamed.

Model: Qwen3.8-27B hybrid (16 full-attention layers of 64, head_dim 256, 4 KV heads,
24 Q heads, GQA 6) with MTP k=3, so a verify step has 6 x 4 = 24 query rows per KV head.

## Rungs

| Rung | Target | Round 1 |
|---|---|---|
| R1-A correctness | matches the Triton int4 kernel on the same packed cache (q-len 1/4, batch 1-8, 1k-256k, NHD/HND); logprob parity and KLD vs bf16 KV equal to Triton int4 within noise | see below |
| R1-B kernel speed | >= 2x the Triton int4 kernel at 32k-128k on both cards | passed |
| R1-C server | ms/step at c = 1/2/4/8, ~1k/32k/128k, vs Triton int4 and fp8 FlashInfer, with the whole-step floor | see below |
| later | 80% of the attention floor; whole step within 1.3x of its floor at c=4 | |

## Design (round 1)

Code: `vllm_gguf_plugin/csrc/int4kv/int4kv_attention.cu` (kernel, combine, op),
`vllm_gguf_plugin/int4kv_attention.py` (backend), registered in `plugin.py` as
`AttentionBackendEnum.CUSTOM`. Select with `--attention-backend CUSTOM --kv-cache-dtype
int4_per_token_head`. Tests: `tests/test_int4kv_attention.py`; bench:
`tests/bench_int4kv_attention.py`.

Cache format (vLLM 0.30.1rc1 `v1/attention/ops/int4_per_token_head.py`): per token per KV
head a 264 B slot `[K 128 B | K scale | V 128 B | V scale]`, nibble codes u (byte j = dims
2j, 2j+1), x = s (u - zp), zp in the scale's low 4 mantissa bits, K and V stored after
the randomized Hadamard rotation H D (unnormalized).

Decisions and why:

- **Backend subclasses `TritonAttentionBackend`/`Impl`** and overrides only `forward`:
  layout, scale views, store kernel, metadata builder and CUDA graph support are
  inherited, so the cache is byte-for-byte vLLM's. Batches with <= 24 query rows per KV
  head (decode, MTP verify, the drafter's passes) go to the CUDA kernel; prefill and
  unsupported features (sliding window, softcap, sinks, alibi, mm prefix) fall back to
  the inherited Triton int4 kernel.
- **Transposed scores, S^T = K Q^T.** With K as the m16 operand, the query rows are the
  n8 side: 24 rows fill three n8 tiles exactly (an m16 formulation pads 24 to 32, and a
  q-len-1 batch's 6 rows to 16). P^T goes through shared memory and comes back with
  `ldmatrix.trans` as the B operand of O^T = V^T P^T.
- **Dequant in one function (`dequant8`).** `(w >> 4k) & 0x000F000F | 0x64006400` is
  fp16 1024 + u for two nibbles; a single `hsub2` with (1024 + zp) removes the bias and
  the zero point exactly, so there is no zp * sum(q) correction. For K both halves are
  the same token (zp splat); for V a `prmt` pairs the same byte of tokens 2t and 2t+1,
  so each half2 is one dim of two tokens and the zp pair is per token. The per-token K
  scale multiplies S after the MMA (fp32); the V scale is folded into P before it is
  rounded to fp16.
- **Q rotation in the prologue** (D signs, 256-point FWHT across a warp in fp32, x 1/16
  so it is orthonormal), stored in fp16 in a permuted column order that matches the K
  fragment, so the B fragment is a plain `ldmatrix`. The Triton path rotates Q in bf16;
  this one is closer to the exact result (tests below).
- **Split-KV + combine.** One CTA = (split, KV head, sequence), 8 warps, 128-token tiles,
  2-stage `cp.async` ring (8 B copies: slots are only 8 B aligned; each warp copies its
  own 16-token group, which never crosses a cache block). Splits are the fewest that
  fill the GPU to >= 95% in whole waves at one CTA per SM (SM count read at run time,
  nothing card-specific), computed from the batch size only, so a CUDA graph replays
  with any sequence lengths. The combine kernel merges splits (log2-domain LSE) and
  undoes the rotation (FWHT, D, / 256) per output row.
- Separate extension `_C_int4kv` (built for every non-ROCm build), so iterating on it
  does not rebuild the GGUF/llama.cpp kernels.

No external code was copied. llama.cpp's fattn and QServe's INT4KV decoder were not
needed for this round; the magic-number nibble conversion is the well-known trick from
FasterTransformer / Marlin (`interleaved_numeric_conversion.h`), written from its
description. Nothing found that drops in for this exact format (prior research: none in
FlashInfer, TRT-LLM, Strata).

## Round 1 results (2026-10-29)

### R1-A: kernel correctness

`tests/test_int4kv_attention.py`, 57 cases: q-len 1 and 4 (and 3 with fp16 and block
size 16), batch 1/2/4/8, contexts 1k/4k/32k/128k/256k, shuffled block tables, both NHD
and HND strides, seq lens varying within a batch. Cache written by vLLM's own
`reshape_and_cache_int4`. Referee: exact fp32 attention over the dequantized cache.
57/57 pass on the 3090 and on the 4070 Ti.

Max error relative to max |output| (smoke cases, 3090):

| case | CUDA vs exact | Triton int4 vs exact | CUDA vs Triton |
|---|---|---|---|
| 1 seq, 1k, q-len 4 | 2.6e-3 | 1.1e-2 | 9.6e-3 |
| 4 seqs, 4k, q-len 1, HND | 2.5e-3 | 1.3e-2 | 1.4e-2 |
| 8 seqs, 32k, q-len 4 | 3.1e-3 | 1.4e-2 | 1.4e-2 |

The CUDA error is the bf16 output rounding (2^-9); the Triton path's is ~4x larger
(bf16 Q rotation, TF32 dots).

### R1-B: kernel speed

`tests/bench_int4kv_attention.py`: one layer call, all 4 KV heads, c sequences of the
given length (rows: 6 x q-len per KV head), median of 15, CUDA events, L2 evicted with
clean lines before every call. Both columns include Q rotation, split-KV reduce and
output rotation. GB/s = K+V+scale bytes (264 B per token-head) / time. Floors: measured
streaming read, 891-892 GB/s (3090), 479 GB/s (4070 Ti, same Triton read kernel).

RTX 3090 (sm86, tensor-unlocked), ms per layer call:

| ctx | c | q-len | Triton int4 | CUDA | speedup | GB/s | of floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32k | 1 | 1 | 0.407 | 0.056 | 7.2x | 614 | 69% |
| 32k | 1 | 4 | 3.438 | 0.071 | 48.7x | 490 | 55% |
| 32k | 4 | 1 | 0.806 | 0.168 | 4.8x | 774 | 87% |
| 32k | 4 | 4 | 3.455 | 0.214 | 16.1x | 608 | 68% |
| 32k | 8 | 1 | 1.314 | 0.314 | 4.2x | 826 | 93% |
| 32k | 8 | 4 | 4.189 | 0.392 | 10.7x | 662 | 74% |
| 64k | 1 | 4 | 6.828 | 0.120 | 57.0x | 578 | 65% |
| 64k | 8 | 1 | 2.604 | 0.626 | 4.2x | 844 | 95% |
| 128k | 1 | 1 | 1.425 | 0.173 | 8.2x | 800 | 90% |
| 128k | 1 | 4 | 13.466 | 0.215 | 62.6x | 644 | 72% |
| 128k | 2 | 4 | 13.488 | 0.401 | 33.6x | 681 | 76% |
| 128k | 4 | 1 | 3.201 | 0.636 | 5.0x | 830 | 93% |
| 128k | 4 | 4 | 13.539 | 0.778 | 17.4x | 678 | 76% |
| 128k | 8 | 1 | 5.175 | 1.260 | 4.1x | 845 | 95% |
| 128k | 8 | 4 | 16.775 | 1.517 | 11.1x | 702 | 79% |

RTX 4070 Ti (sm89), ms per layer call:

| ctx | c | q-len | Triton int4 | CUDA | speedup | GB/s | of floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32k | 1 | 1 | 0.350 | 0.083 | 4.2x | 417 | 87% |
| 32k | 1 | 4 | 2.802 | 0.092 | 30.4x | 375 | 78% |
| 32k | 4 | 1 | 0.718 | 0.290 | 2.5x | 449 | 94% |
| 32k | 4 | 4 | 2.841 | 0.317 | 9.0x | 410 | 86% |
| 32k | 8 | 1 | 1.188 | 0.567 | 2.1x | 458 | 96% |
| 32k | 8 | 4 | 3.311 | 0.623 | 5.3x | 417 | 87% |
| 64k | 8 | 1 | 2.367 | 1.147 | 2.06x | 460 | 96% |
| 128k | 1 | 1 | 1.308 | 0.300 | 4.4x | 461 | 96% |
| 128k | 1 | 4 | 11.119 | 0.310 | 35.8x | 446 | 93% |
| 128k | 4 | 1 | 2.899 | 1.140 | 2.5x | 463 | 97% |
| 128k | 4 | 4 | 11.289 | 1.171 | 9.6x | 451 | 94% |
| 128k | 8 | 1 | 4.723 | 2.282 | 2.07x | 466 | 97% |
| 128k | 8 | 4 | 13.258 | 2.382 | 5.6x | 447 | 93% |

Full grids (8k-128k, c = 1/2/4/8): `~/3090-microbenchmark-repository/data/2026-10-29-int4kv-cuda-attention.md`.

Why the Triton baseline is so far behind at q-len 4: vLLM's unified attention refuses
the split-KV (3D) path whenever max_seqlen_q > 1, so every MTP verify step runs one
program per (16 query rows, KV head): 8 CTAs on the whole GPU at c=1. At q-len 1 it
does split (16 segments) and the gap is the plain 2-5x.

Against the old fp8 FlashInfer term (45-60% of floor, 18-19 ms/step of attention at a
200k context, `cloud/results/prod-profile-20261001.md` section 6-7): int4 moves 264 B per
token-head instead of 512, and this kernel runs at 72% of floor at q-len 4 on the 3090,
so 16 layers at 200k, c=1 extrapolate to ~5.4 ms/step (0.215 ms x 200/128 x 16).

What limits it now:

- q-len 1 (6 rows) is DRAM-bound: 90-97% of floor from 64k up on both cards; below
  that the fixed per-CTA cost (Q rotation, partial writes, combine) shows (8k c=1:
  36% on the 3090).
- q-len 4 (24 rows) on the 3090 is compute-bound. Per token-head it is 24.6 kFLOP of
  MMA for 264 B, i.e. 93 FLOP/B; at 892 GB/s that is 83 TFLOPS, 71% of the
  tensor-unlocked card's ~117 TFLOPS fp16/fp32-acc (it would be 117% of the stock 71).
  Per 128-token tile and SM: ~4000 clocks of MMA plus the dequant/softmax ALU work and
  three CTA barriers, against ~5600 clocks of DRAM time; the phases (QK on the tensor
  pipe, softmax, PV dequant + MMA) are serialized by the barriers with only 8 warps
  per SM, so the tile takes ~7000 clocks. The 4070 Ti has 2x more MMA per byte
  (half-rate fp32-acc ~80 TFLOPS vs 479 GB/s needs 45 TFLOPS) and reaches 93%.

### R1-C: server

(pending; see hand-off)

## What failed / gotchas

- `TORCH_CUDA_ARCH_LIST` build with the venv's CUDA 13.4 nvcc needs
  `-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` (the wheel's runtime headers are 13.0) and a
  `CUDA_HOME` with `lib64/libcudart.so` -> the wheel's `libcudart.so.13`; otherwise the
  extension silently links the system CUDA 12 runtime, which does not exist on MS5.
- L2 flush with `zero_()` before timing leaves 48 MB of dirty lines on the 4070 Ti whose
  write-back lands inside the timed call (c=1 32k read 299 GB/s instead of 416). The
  bench evicts with a read.

## Hand-off

(round 1 in progress)

## Round 0: root cause of the 2026-10-28 int4 PP collapse

**Not the int4 kernels.** It is the V1 model runner + GDN metadata under FULL cudagraphs with
MTP, and any KV dtype on TRITON_ATTN hits it. The new kernel inherits it if its backend keeps
FULL graphs with spec decode.

### What happened (data.db, 1 Hz)
- 19:55:39: stream B starts a turn with a 98,576-token prefix hit and prefills 8,084 tokens.
  A and C decode with normal acceptance until 19:56:13.
- 19:56:14: B's prefill ends. From 19:56:15 on, all three streams show accepted = 0 on every
  draft, at every position, until they were aborted at 20:56. A fresh 68-token request at
  20:55 drafted with acceptance about 2.5 while the three long streams stayed at 0. So one
  step permanently poisoned every request that was running, and new requests were unaffected.
- B's prompt was 98,576 + 8,084 = 106,660 tokens = 66 x 1616 + **4**. In align mode with
  other requests running, the last prefill chunk is the 4 tokens past the last block
  boundary. A and C verified 1 + 3 tokens each, so the step had 3 requests x 4 tokens.

### Mechanism
- `v1/worker/gpu_model_runner.py:3943` `_is_uniform_decode` decides from counts alone
  (max scheduled == 1 + k and num_tokens == (1 + k) * num_reqs). The 4-token prefill chunk
  passes, so the step replays a **FULL** decode graph. TRITON_ATTN keeps FULL_AND_PIECEWISE
  with spec decode. FlashInfer is downgraded to PIECEWISE ("not supported with spec-decode",
  journal 10-27), which is why the fp8 PP runs never hit this.
- `v1/attention/backends/gdn_attn.py:568` `_stage_spec_decode` stages the persistent FULL-graph
  buffers only when `num_prefills == 0`. Here num_prefills = 1, so the graph replays GDN
  spec-decode kernels on stale `spec_state_indices` / `num_accepted_tokens` / query starts
  left by an earlier step. That corrupts every running request's recurrent state, which
  shows up as NaN logits (`!!!!`) and exactly 0 acceptance.

### Reproduction (3090, base GSQ-RCO IQ3_S, int4 + TRITON_ATTN, MTP k=3, align prefix caching,
no async, 90-block pool; scripts /tmp/r0/{serve.sh,four.py,c3.py}, probe /tmp/r0/sc/sitecustomize.py)
Two long background completions decode while a probe prompt of exactly 6*1616 + R tokens prefills:

| R (last chunk) | FULL dispatches with a GDN prefill | probe | both background streams | acceptance after |
|---|---|---|---|---|
| 4 | 3 | `!!!!...` | turn to `!!!!` mid-answer | 0.00 |
| 5 | 0 | correct | coherent | 2.47 |

R=4 reproduced in 2 of 2 tries, R=5 in 0 of 2. A c=1 run with R=4 does not trigger, because
alone the final chunk is 1620 tokens.

### Ruled out
- Kernels: `/tmp/r0/kt.py` on MS5 (real store kernel, `_attn_packed`, 2D and 3D/`reduce_segments`,
  q-len 1 and 4, 430-block pool = prod's 695k tokens, random and high block ids, NHD and HND,
  bf16 and fp16, L from 1k to 260,600 including 65,535/65,536/~106k/~110k). Rel. error vs an
  fp32 reference on the same codes+scales is <= 0.50% everywhere, no NaN. RHT/hadacore is exact
  up to 1M rows (`/tmp/r0/rht.py`).
- `_ensure_scale_caches` `fill_(1.0)` captured into a graph: probed, the first call happens eagerly
  for all 17 impls (16 + MTP).
- CPU offload tier: raw page copies (`kv_offload/cpu/gpu_worker.py:833`), and ext_hits stayed 0
  (no loads ever happened). PP: not needed to reproduce. `spec-decode-int8-kv.patch`: gated on
  INT8_PER_TOKEN_HEAD, inert for int4.

### Fix
`~/qwen38-27b-rtx3090/patches/v1-uniform-decode-excludes-prefill.patch` (+9 lines in
`execute_model`): a step is uniform only if every request is a decode
(scheduled == 1 + its scheduled spec tokens), otherwise `force_uniform_decode=False`. Not applied
to the shared venv. Fallback without patching: `cudagraph_mode=PIECEWISE`.
Verified with the same logic injected at runtime (R0_FIX in the probe sitecustomize, same server):
R=4 twice and R=5 once all gave 0 FULL dispatches with a prefill, correct probes, coherent
background streams and acceptance 2.3-2.6 after the probe.

### Rules for the new kernel / CUSTOM backend
1. If the backend reports FULL cudagraph support with spec decode, run it only with the runner
   patch (or report UNIFORM_SINGLE_TOKEN_DECODE and get PIECEWISE). Otherwise R1-C server runs
   will reproduce this collapse, and it is not a kernel bug.
2. Do not read "uniform decode" as "every request is a spec decode": rows with q = 1 + k can be
   prefill tails. Attention handles them fine (the causal mask is the same). The kernel must also
   take any mix of q-lens <= 4 in one launch.
3. Keep one-time setup (scale views, `fill_`) out of graph capture. Today it runs eagerly; keep it
   that way.
4. Test gate for any server run: `python /tmp/r0/c3.py 4` must leave all three outputs coherent.
