# 11c: One q8_1 quantization per fused-layer input

Branch: `upstream/11c-lcpp-shared-q8-input` (3 commits, on 10). Depends on 06 (the quantizer op) and
on 07-10, whose 1..32-row ops gain the `x_q8` argument.

**Title:** [Perf] lcpp: quantize a fused layer's input once for all its products

## Motivation

A fused layer with several shard runs (mixed types, e.g. GDN in_proj_qkvz: q/k/v in one type, z in
another) quantized its input once per run, although every q8_1-reading op makes the same bytes.

## What changed

- `csrc/lcpp_shim.cu`: the q8_1-reading ops (MMVQ, the dp4a and mma IQ3 kernels, the packed decode
  kernel, the Q4_K / IQ2_S kernel) take an optional `x_q8`, checked before any launch (uint8, 1-D,
  contiguous, large enough, 16-byte aligned, on X's device) and trusted to come from X.
- `linear.py`: `apply()` quantizes X once (the custom op `_quantize_x_q8_1`) when any run's op reads
  q8_1, and passes it to those runs; MMQ, mma_k and the tiled IQ3 kernel quantize for themselves in
  MMQ's layout. `_fused_mul_mat_gguf` takes `x_q8` after `packed`.
- Tests: the ops give identical results with and without `x_q8`; which type the shared quantize runs
  with, or that it stays unfilled when no run reads it (CPU, mocked op); a stock-path run beside an lcpp
  run; decode-path graph replay; every malformed `x_q8` rejected before a launch.

## How tested

- Build: clean. CPU tests: 420 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- Development branch (RTX 3090): bit-exact with per-run quantization; GPU input checks 168 passed / 30
  skipped (every bad `x_q8`, including on the CPU, rejected before a launch); compute-sanitizer clean on
  the `x_q8` and shared-quantize tests.

| per c=1 decode step (27B model, MTP k=3) | before | after |
| --- | --- | --- |
| q8_1 quantize launches | 375 | 276 |
| quantize time | 0.52 ms | 0.42 ms |

End to end the change is at the noise level (c=1 90.9 -> 92.3 tok/s, runs differ by ~1 %).

## Risks

- A signature change of the internal `_fused_mul_mat_gguf` / lcpp ops (optional trailing argument).
- The ops trust that `x_q8` is X's quantization; only `apply()` passes it.
