# 11d: Zero MMQ's read tail in the quantize kernel instead of a memset

Branch: `upstream/11d-lcpp-mmq-tail-fold` (1 commit, on 11c). Depends on 05 (the tail) and 06 (the
owned quantizer); stacked on 11c only for order.

**Title:** [Perf] lcpp: fold MMQ's read-tail zeroing into its quantize kernel

## Motivation

The shim zeroes MMQ's q8_1 read tail (see 05) with a `cudaMemsetAsync` on every MMQ call: one launch
per product, ~360 per engine step at 4 sequences.

## What changed

- `csrc/lcpp_shim.cu`: MMQ's quantize kernel zeroes the tail with its whole grid (a grid-stride loop
  over the tail's int4s), so the memset goes away. One commit: kernel change and its call site.

## How tested

- Build: clean. CPU tests: 419 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- The existing MMQ tests at 1..9 rows run with the allocator's free blocks poisoned, so an unzeroed tail
  shows up; on the development branch they pass, and compute-sanitizer memcheck + initcheck are clean
  on 53 cases including MMQ at 1..9 rows.

| per step at c=4 (27B model, MTP k=3, RTX 3090) | before | after |
| --- | --- | --- |
| launches | | -360 |
| memset time | 0.41 ms | 0 |
| quantize time | | +0.04 ms |
| total | | -0.37 ms |

## Risks

- None beyond the quantize kernel doing a little more work per call.
