# 06: Owned fp32 / fp16 / bf16 -> q8_1 quantizer for the lcpp ops

Branch: `upstream/06-lcpp-q8-quantizer` (2 commits, on 05). Depends on 05 (the shim).

**Title:** [Perf] lcpp: quantize 16-bit activations to q8_1 without a cast

## Motivation

The vendored quantizers take fp32 only, so every lcpp call on bf16 / fp16 activations first cast X
to fp32: one more kernel and a temporary per product. With MTP decode (4 activation rows per target
pass) the casts cost ~1.8 ms of a ~38 ms engine step.

## What changed

- `csrc/lcpp_shim.cu`: q8_1 quantizers for MMVQ's `block_q8_1` and MMQ's `block_q8_1_mmq` (all three
  ds layouts: D4, DS4, D2S6), templated on X's dtype. They copy the vendored `quantize_q8_1` /
  `quantize_mmq_q8_1` arithmetic and layout line for line, read X in its own dtype with scalar loads
  (no alignment or row-stride requirement beyond the element size), and use the vendored launch
  geometry. fp16 / bf16 to float is exact, so the bytes equal the vendored quantizers' on `X.float()`.
- A test-facing op `lcpp_quantize_q8_1(X, type, mmq, vendored)` returns either quantizer's bytes.
- `tests/test_lcpp_kernels.py`: the owned bytes equal the vendored bytes for every lcpp type, both
  layouts, fp32 / fp16 / bf16 and row-strided X, 1 / 4 / 9 rows, with all-zero blocks.

## How tested

- Build (`VLLM_GGUF_BUILD_LCPP=1`, sm_86): clean. CPU tests: 265 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- On the development branch (RTX 3090): the bytes were bit-identical to the vendored quantizer's in
  every case; kernel parity 776 -> 992 passed with the new tests; GPU input-check cases 64 of 64;
  compute-sanitizer memcheck and initcheck clean on 18 targeted cases.

| (27B Qwen3.5-arch GGUF, MTP k=3, RTX 3090) | before | after |
| --- | --- | --- |
| bf16 casts per decode step at c=1 | 1.8 ms | 0.7 ms (output casts only) |
| decode tok/s, c=1 / c=2 | 79.4 / 139.5 | 81.2 / 142.9 |
| ms per engine step, c=1 / c=2 | 37.8 / 43.2 | 36.9 / 42.1 |

## Risks

- A second copy of the quantizer to keep in sync with llama.cpp's on a tag bump; the equality test
  catches a divergence.
