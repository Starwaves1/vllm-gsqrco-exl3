# 07: Owned IQ3_S / IQ3_XXS kernels for 1..8 activation rows (dp4a and int8 mma)

Branch: `upstream/07-lcpp-owned-iq3` (3 commits, on 06). Depends on 05 (shim, routing) and 06 (the
quantizer both kernels read from).

**Title:** [Perf] lcpp: owned IQ3_S / IQ3_XXS decode kernels (dp4a and int8 tensor cores)

## Motivation

IQ3 weights dominate IQ3-based GGUFs and MTP decode runs 4 activation rows per sequence (8 at two
sequences). Vendored MMVQ is lookup-bound there (~380 GB/s at 4 rows on an RTX 3090, against ~625 for
IQ4_XS): each warp reuses an activation load over only 2 weight rows. (A SASS check showed nvcc already
shares the IQ3 decode across activation rows, so "decode once" alone is not the lever.)

## What changed

- `lcpp_mul_mat_vec_iq3` (dp4a, in the shim): CTAs of 4 warps x 4 weight rows; each lane decodes its
  32-value slice of its 4 rows once and dots it with every activation row, whose q8_1 chunks are staged
  in shared memory with the grid table; the sign step negates bytes directly (4 ops per word, from
  ninfer-all's `ggml_bridge_vec.cuh`, read, not copied). Writes X's dtype: no output cast.
- `lcpp_mul_mat_vec_iq3_mma` (`csrc/lcpp_owned_iq3_mma.cu`): int8 tensor cores (`mma.sync
  m16n8k32`), one mma per 16 weight rows x one q8_1 block x 8 columns, so its cost is flat in rows.
- Both compute each slice's scaled integer sum exactly as the vendored `vec_dot_iq3_*_q8_1`; only the
  fp32 order of a row's terms differs from MMVQ (~1e-7 relative).
- Routing: IQ3 at 1..5 rows to the dp4a kernel, 6..8 to the mma kernel; MMQ from 9 as before.
- Tests: against the reference models and vendored MMVQ at 1..8 rows on edge shapes (part-filled last
  CTA, short last chunk, K = 512, fewer / more tiles than CTAs, odd W size), 16-bit output equal to the
  fp32 output cast by torch, graph replay, first call inside a capture, input checks.

## How tested

- Build: clean. CPU tests: 295 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- Development branch (RTX 3090, real tensors): kernel parity 1152 passed / 16 skipped with the dp4a
  kernel, 1594 with the mma kernel added; compute-sanitizer memcheck + initcheck clean on 16 cases per
  op (1..8 rows, all tail shapes).

| (27B Qwen3.5-arch GGUF, MTP k=3, RTX 3090) | c=1 tok/s | c=2 tok/s | ms/step c=1 / c=2 |
| --- | --- | --- | --- |
| before (MMVQ / MMQ) | 83.5 | 146.7 | 36.6 / 41.6 |
| dp4a kernel at 1..8 rows | 90.9 | 154.2 | 33.1 / 37.9 |
| + mma kernel at 6..8 rows (same-session A/B) | 90.3 | 173.2 | 33.0 / 35.6 |

Op level at 4 rows: 541 / 520 GB/s for IQ3_S / IQ3_XXS (vendored MMVQ ~380).

## Risks

- Owned CUDA code to maintain (~250 lines dp4a, ~430 mma).
- The mma kernel needs sm_80+ (`mma.sync m16n8k32` s8; ptxas rejects it for sm_75). From this PR on, a
  `VLLM_GGUF_BUILD_LCPP=1` build needs compute capability 8.0+: setup.py refuses an explicit
  `TORCH_CUDA_ARCH_LIST` with an older numeric entry (an unset list is left to torch, which builds for
  the local GPU). The vendored-only build of 05 / 06 has no such limit. Pre-Ampere GPUs keep the
  default build. A multi-arch build including pre-Ampere archs would need arch guards in the owned
  kernels; not done.
- Routing thresholds (5 / 6 / 8 rows) were tuned on one RTX 3090.
