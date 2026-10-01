# 04: Kernel parity tests against CPU reference models

Branch: `upstream/04-kernel-parity-harness` (2 commits, on `upstream/03-kernel-input-guards`; applies
cleanly on e2b8ad5 as well, checked with `git cherry-pick`). No functional dependency.

**Title:** [Test] Kernel parity tests against CPU reference models, and a CPU check of the dequantize kernels

## Motivation

`tests/test_kernels.py` compares the kernels with gguf-py at `rtol=4e-2`. That cannot tell a kernel bug
from the q8_1 rounding of the activations: the `iq3xs_grid` error fixed in #136 (3.3 % on some
elements) passed it. Modelling what the kernels compute allows tolerances two orders of magnitude
tighter.

## What changed

- `tests/kernel_refs.py`: CPU reference models in float64: `q81` (X rounded through q8_1 exactly as the
  kernels quantize it), `xsum` (the MMQ Q4_K / Q5_K min term from half(sum x), as ggml's MMQ does),
  `wround` (W rounded to the activation dtype, for dequantize + GEMM), and full precision. Tie-free
  activations (fast-math division may round a q8_1 tie either way), a weight builder from blocks of
  the sample GGUFs (any rows and K), and a helper that poisons the caching allocator's free blocks.
- `tests/test_kernel_parity.py` (CUDA): `ggml_dequantize` exact against gguf-py for IQ types and
  bounded for the fp16 ones; MMVQ at 1..16 rows and MMQ at 16..512 rows within 5e-3 (bf16) / 2.5e-3
  (fp16) of a reference model and 3e-2 of full precision (1.5e-1 for the MMQ min-term types); the
  routing function on whole weights of 2048 and 18944 rows. Ten types: those the tolerances were
  calibrated on.
- `tests/dequant_host.py` + `tests/test_dequant_host.py` (CPU): `dequantize.cuh` compiled with g++ for
  the host (each launch rewritten into a loop, `half` as `_Float16` with one rounding per intrinsic,
  `-ffp-contract=off`), compared with gguf-py on every sample GGUF.

## The fp16 dequantization, documented and bounded

The host check shows that the IQ kernels are bit-exact, but the legacy (Q4_0, Q5_0, Q8_0) and K-quant
(Q2_K ... Q6_K) kernels compute in fp16 (`__hmul`, `__hsub`, `__int2half_rn(sc * q)`) where ggml and
gguf-py use fp32: 22-99 % of values differ, by up to 1.02e-3 of the tensor's largest magnitude
(Q5_K). The test marks bit-exactness `xfail(strict=True)` for these eight types and bounds the error at
2^-8 of each row's largest magnitude (first-order error of the fp16 roundings; measured up to
1.04 x 2^-9, Q5_K). The fix (fp32 arithmetic with `__fmul_rn` / `__fsub_rn` to match ggml exactly) is
small per kernel but touches eight kernels' arithmetic, so it is left for a follow-up PR that this
harness can verify on the CPU; the strict xfail flags it when it lands.

## How tested

- `pytest tests/test_dequant_host.py`: 26 passed, 16 xfailed (CPU, needs g++ with `_Float16`).
- `pytest tests --ignore=tests/test_kernels.py --ignore=tests/test_gguf_generation.py`: 175 passed, 8 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
  (stacked on 03, so including its 45 CPU guard cases)
- The tolerances and the reference models come from the development suite, which ran the same tests on
  an RTX 3090 against the development model's tensors: 276 passed / 12 skipped at e2b8ad5 after
  calibration (worst reference-model error 2.5e-3 bf16 / 1.24e-3 fp16, worst error against full
  precision 1.5e-2, Q4_K MMQ 7e-2 from full but 2.5e-3 from the xsum model), and the stock-path tests
  later passed unchanged on every build (328 of them on the final one).
- In this form (sample-GGUF weights) the CUDA tests were collected, and their harness was run on the
  CPU with the CUDA ops replaced by the reference models; they have not run on a GPU in this form.

## Risks

- The sample GGUFs are random-normal weights, not a real model; a type whose error profile differs
  there could need a looser tolerance. Only the ten calibrated types are enabled.
- The host dequant test needs `g++` (skipped otherwise) and downloads the sample GGUFs on first use,
  like `tests/utils.py` already does.
