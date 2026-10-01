# 11e: IQ1_M on vendored MMVQ, 8 rows per call, up to 32 rows

Branch: `upstream/11e-lcpp-iq1m-mmvq` (3 commits, on 11d). Depends on 05 (MMVQ op, routing) and 11c
(it slices the shared `x_q8` per chunk).

**Title:** [Perf] lcpp: run IQ1_M on MMVQ up to 32 rows

## Motivation

llama.cpp has no IQ1_M MMQ, so IQ1_M stayed on the stock kernels: above the stock MMVQ limit the whole
weight is dequantized every forward. Inside CUDA-graph captures of 9..32 rows (MTP verify at 3..8
sequences) that also costs graph memory.

## What changed

- `csrc/lcpp_shim.cu`, `ops.py`: IQ1_M allowed in the lcpp ops for MMVQ; the MMQ op rejects it.
- `linear.py`: IQ1_M goes to MMVQ up to 32 rows, in 8-row calls above 8 (with the matching slices of a
  shared `x_q8`); above 32 rows the stock dequantize + GEMM, as before.
- Tests: MMVQ tests include IQ1_M; 9..32 rows in chunks equal the chunks' own products, with and
  without `x_q8`; routing and input checks.

## How tested

- Build: clean. CPU tests: 434 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- Development branch (RTX 3090): kernel parity 2007 passed / 146 skipped (flag on); compute-sanitizer
  clean.

| (27B model, MTP k=3, RTX 3090) | before | after |
| --- | --- | --- |
| IQ1_M time per step, c=1 / c=4 | | -0.12 / -0.27 ms |
| CUDA-graph memory | 0.35 GiB | 0.20 GiB |
| KV cache | | +4.7k tokens |

## Risks

- The 32-row limit was measured for one weight shape (the model has one IQ1_M tensor, 0.2 % of its
  bytes); above it the old path is kept.
