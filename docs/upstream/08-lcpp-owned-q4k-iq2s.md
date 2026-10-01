# 08: Owned Q4_K / IQ2_S kernel for 1..8 activation rows

Branch: `upstream/08-lcpp-owned-q4k-iq2s` (3 commits, on 07). Depends on 05 / 06; on 07 for the
shared CTA design and the owned-kernel test scaffolding.

**Title:** [Perf] lcpp: owned Q4_K / IQ2_S decode kernel

## Motivation

After 07, the largest decode GEMMs of the IQ3-based model are Q4_K and IQ2_S (FFN and attention
weights, and a 248320 x 5120 Q4_K lm_head) at 4..8 rows, on vendored MMVQ.

## What changed

- `csrc/lcpp_owned_k4.cu`: `lcpp_mul_mat_vec_own`, the IQ3 dp4a kernel's structure for Q4_K and
  IQ2_S (CTAs of 4 warps x 4 weight rows, each lane decoding its slice once for all activation rows,
  q8_1 chunks staged in shared memory, the integer sub-scale folded into the block scale). Each slice's
  integer sum is the vendored vec_dot's; the fp32 order and, for Q4_K, the rounding of the scale and
  min terms differ slightly.
- Routing (`_lcpp_op` now takes the weight's rows): Q4_K from 3 rows, IQ2_S from 1 row, up to 8, only
  on weights above 2048 rows (its 16-row CTAs underfill the GPU below). IQ4_XS gained nothing and is not
  taken.
- Tests: the owned-kernel tests of 07 extended to this op, 8192 rows in bf16 for its routed range;
  input checks.

## How tested

- Build: clean. CPU tests: 318 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- Development branch (RTX 3090): kernel parity 2036 passed / 119 skipped with 07 and this kernel (flag
  on), 1979 / 176 with the flag off; compute-sanitizer memcheck + initcheck clean, including this
  kernel at 8192 rows and inside `apply()`.

| per call at 4 rows, in situ (RTX 3090) | MMVQ | owned |
| --- | --- | --- |
| Q4_K 17408 x 5120 | 78.0 us | 65.2 us |
| IQ2_S 17408 x 5120 | 89.8 us | 56.9 us |
| Q4_K lm_head 248320 x 5120 | 1088 us | 836 us |

End to end (27B Qwen3.5-arch GGUF, MTP k=3): decode 88.1 -> 93.0 tok/s at c=1 (33.7 -> 31.8 ms per
step), 154.3 -> 158.9 at c=2 (38.5 -> 37.3 ms).

## Risks

- Routing floors (3 rows for Q4_K, 2048 weight rows) are tuned on one RTX 3090; at 3 rows Q4_K ties
  MMVQ below 12288 weight rows.
