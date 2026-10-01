# 09: Owned 9..32-row int8 tensor-core kernel for Q4_K / IQ4_XS / IQ2_S

Branch: `upstream/09-lcpp-mma-k` (3 commits, on 08). Depends on 05 / 06 (shim, MMQ-layout quantizer);
on 08 only through routing and test scaffolding.

**Title:** [Perf] lcpp: owned int8 tensor-core kernel for Q4_K / IQ4_XS / IQ2_S at 9..32 rows

## Motivation

MTP verify passes at 3..8 sequences are 12..32 rows: small for MMQ's 128-column tiles, which spend
most of their time on tile overhead and the stream-k fixup.

## What changed

- `csrc/lcpp_owned_mma_k.cu`: `lcpp_mul_mat_mma_k`, up to 64 rows. MMQ's inputs (GGUF blocks, X in
  `block_q8_1_mmq`), each slice's term with the vendored MMQ vec_dot's expression, 16..64-column tiles,
  stream-K over persistent CTAs with a small fixup for tiles CTAs share (skipped when none is), output
  in X's dtype. Launch attributes are set once per device and instance, also when that first call is
  inside a CUDA-graph capture.
- Routing (`_mma_k_wins`; `_lcpp_op` now takes K): 9..32 rows on weights above 2048 rows; IQ4_XS at
  17..32 only from 12288 x 5120. From 33 rows it runs 64-column tiles and loses.
- Tests: fp32 X within 1e-5 of MMQ (2e-6 on whole 17408 x 5120 and 5120 x 17408 weights), 16-bit X
  against the reference models and within 1 ulp of MMQ, part-filled tiles, K 17408, layouts with and
  without shared tiles, graph replay, first calls inside a capture in a fresh process, input checks.

## How tested

- Build: clean. CPU tests: 338 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- Development branch (RTX 3090): kernel parity (lcpp and routing) 1516 passed / 61 skipped; fp32
  relative error to MMQ at most 5.4e-7 (129 cases); 16-bit output within 1 ulp of MMQ on all 222 cases;
  GPU input checks 120 passed; compute-sanitizer memcheck + initcheck clean on 36 cases.

| us per call, RTX 3090 (vendored MMQ) | 16 rows | 32 rows | 64 rows |
| --- | --- | --- | --- |
| Q4_K 17408 x 5120 | 80.7 (111.2) | 95.4 (129.5) | 169.6 (177.9) |
| Q4_K 5120 x 17408 | 84.4 (109.8) | 98.5 (129.2) | 169.8 (175.8) |
| IQ4_XS 17408 x 5120 | 83.4 (99.5) | 111.0 (126.0) | 165.3 (162.7) |
| Q4_K lm_head 248320 x 5120 | 975.9 (1364.0) | 1165.1 (1744.7) | 2209.2 (2345.6) |

End to end at c=4 (16 rows per target pass): these types' GEMM time 10.3 -> 9.0 ms per step;
ms/step 46.84 -> 45.44 (-3.0 %); c=8 -0.5 % (within noise).

## Risks

- sm_80+ only (see 07). The routing windows (9..32 rows, above 2048 weight rows, the IQ4_XS size cut)
  were measured on one RTX 3090.
