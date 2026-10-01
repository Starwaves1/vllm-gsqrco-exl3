# 11a: Batched gemv for small unquantized GGUF linears at up to 8 rows

Branch: `upstream/11a-small-unquantized-gemv` (2 commits, on e2b8ad5). No dependency.

**Title:** [Perf] Run small F32/F16/BF16 GGUF linears at up to 8 rows as a batched gemv

## Motivation

GGUF F32 / F16 / BF16 tensors go to vLLM's `UnquantizedLinearMethod`, i.e. `F.linear`. For a small
weight at 2..4 activation rows cuBLAS picks a GEMM with a handful of one-warp CTAs. Qwen3.5's GDN
`in_proj_ba` (96 x 5120 bf16) at MTP decode's 4 rows: 48 such GEMMs per step at 36 us each, 1.73 ms of
a ~33 ms step.

| 96 x 5120 bf16, us per call (CUDA graph, RTX 3090) | n=1 | n=2 | n=3 | n=4 | n=5..8 |
| --- | --- | --- | --- | --- | --- |
| `x @ w.T` | 4.4 | 29.7 | 29.7 | 26.2 | 5.1-6 |
| `torch.bmm` over the rows (cuBLAS gemvx) | 4.4 | 4.9 | 5.4 | 5.5 | 5.7-6.9 |

(Padding X to 8 rows, a K-major weight or fp32 did not fix it across shapes.)

## What changed

- `quantization/linear.py`: `GGUFUnquantizedLinearMethod`, a subclass of vLLM's method that runs the
  product in a custom op (`_gguf_unquantized_gemm`), so the row-count choice is made per call and not
  fixed when the model is traced: `torch.bmm` for at most 8 rows times at most 128 weight rows without
  bias, `F.linear` otherwise; vLLM's own path for non-2-D input, non-CUDA platforms and
  `VLLM_BATCH_INVARIANT`.
- `quantization/config.py`: returned for skipped (unquantized) linear layers.
- `tests/test_unquantized_gemm.py`: CPU (method selection, equality with `F.linear`, the fallbacks)
  and CUDA (96 x 5120 bf16 at 1..9 rows within bf16 rounding of fp64).

## How tested

- `pytest tests --ignore=tests/test_kernels.py --ignore=tests/test_gguf_generation.py`: 110 passed, 11
  skipped on vLLM 0.27.1 and vLLM main.
- Development branch (RTX 3090): the CUDA test passed; profile of a c=1 MTP decode step: 48 gemvx
  launches 0.31 ms (was 48 GEMMs, 1.73 ms); decode 93.2 -> 96.4 tok/s at c=1 (32.6 -> 31.4 ms per
  step), c=2 unchanged (8 rows already took a fast GEMM).

## Risks

- This is a cuBLAS heuristic problem, not a GGUF one; vLLM's `UnquantizedLinearMethod` has the same
  cost for any small unquantized weight. It may belong in vLLM core instead; here it is limited to GGUF
  tensors.
- The gemv reads the weight once per activation row, which is why it is limited to 128 weight rows.
