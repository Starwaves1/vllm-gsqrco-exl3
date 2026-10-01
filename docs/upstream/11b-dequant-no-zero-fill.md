# 11b: Allocate ggml_dequantize's output uninitialised

Branch: `upstream/11b-dequant-no-zero-fill` (2 commits, on e2b8ad5). No dependency.

**Title:** [Perf] ggml_dequantize: no zero fill of the output

## Motivation

`ggml_dequantize` allocated its output with `new_zeros`, a memset of the whole m x n output that every
dequantize kernel then overwrites (each writes all m * n values: GGUF rows are whole blocks). In bf16 /
fp16 the output is 4-8x the quantized input, so the memset is a large share of the op's memory traffic
on the dequantize + GEMM path (IQ types above the MMVQ row limit, embeddings).

## What changed

- `csrc/gguf/gguf_kernel.cu`: `new_empty` instead of `new_zeros`.
- `tests/test_kernels.py`: `test_dequantize` fills the caching allocator's free blocks with 0xFF first,
  so an element a kernel leaves unwritten reads back as NaN and fails the comparison.

## How tested

- Build (sm_86): clean. CPU tests: 104 passed, 6 skipped on vLLM 0.27.1 and vLLM main
- Development branch (RTX 3090): the poisoned-allocator dequantize test passed for the ten types of the
  27B model in fp32 / fp16 / bf16. The upstream test in this PR covers all 17 types of the sample GGUFs
  and was not run on a GPU in this form. Not timed in isolation.

## Risks

- A dequantize kernel that skipped elements would now return garbage instead of zeros; the poisoned
  test is there to catch that for every type.
