# 10: Lossless load-time IQ3 repack and packed int8 tensor-core kernels (all row counts)

Branch: `upstream/10-lcpp-iq3-packed` (4 commits, on 09). Depends on 07 (the IQ3 mma kernel, which
the packed decode kernel extends and is tested against); on 08 / 09 only through routing and tests.

**Title:** [Perf] lcpp: repack IQ3 weights at load for int8 tensor-core kernels at every row count

## Motivation

IQ3 is most of the bytes of an IQ3-based GGUF. After 07, IQ3 decode at 1..8 rows is owned, but above
8 rows (MTP at 3+ sequences, prefill) IQ3 runs on vendored MMQ, whose tile loader re-derives the
signed-grid values from qs / qh / signs bytes for every tile. Reordering the bytes once, at load, into
the order an mma fragment consumes them removes that work from every forward.

## What changed

- `quantization/iq3_pack.py`: a bijective repack of each 16-row tile of an IQ3_S / IQ3_XXS weight (the
  same bytes, the same size, tile t where its rows were): per lane its grid-index bytes, the bits above
  them (IQ3_S) or re-coded sign bits (IQ3_XXS), sub-scales and scales, so a word's table index is one
  byte permute. `pack_` works in place a group of tiles at a time with scratch at most min(tensor,
  64 MiB) on the tensor's device.
- `lcpp_mul_mat_vec_iq3_mma_packed`: the 07 mma kernel reading packed W straight into registers, 1..32
  rows; bit-exact with the 07 kernel on GGUF bytes 8 rows at a time.
- `lcpp_mul_mat_iq3_packed`: a tiled kernel on packed W for any row count: 256-row x 16..64-column CTA
  tiles, activations staged by cp.async, stream-K with a split-sum kernel; the per-slice fp32 term
  equals vendored MMQ's, so results differ from MMQ only in summation order. Writes X's dtype.
- `linear.py`: with `VLLM_GGUF_LCPP=1`, `process_weights_after_loading` packs each IQ3 run of a layer in
  place, all or nothing per layer (rows % 16, 16-byte aligned), and sets `weight.iq3_packed`; routing
  sends packed IQ3 to the decode kernel up to 8 rows and the tiled one above. Methods that dequantize
  GGUF bytes (embedding, diffusion) opt out (`pack_iq3 = False`).
- Tests: `test_iq3_pack.py` (CPU: round trip on sample blocks and random bytes, tiles in place, `pack_`
  on a padded multi-shard view), the packed kernels against the 07 kernel and MMQ, whole packed weights
  through the routing, packed layers through `apply()`, `pack_`'s GPU scratch bound, first calls in a
  capture, input checks; `test_plugin.py` checks packing under the flag.

## How tested

- Build: clean. CPU tests: 411 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- Development branch (RTX 3090): parity `-k "pack or packed or iq3"` 2664 passed / 104 skipped; the
  pack round-trips on every IQ3 block of the 27B model (222 tensors); `pack_` scratch measured at most
  0.99x the tensor (was 9.2x before the bounded version), 0.59 s for all 5.45 GB of IQ3 at load, no host
  memory; compute-sanitizer memcheck + initcheck clean on the packed ops at 1 / 8 / 9 / 32 / 33 / 128
  rows.

| (27B Qwen3.5-arch GGUF, MTP k=3, RTX 3090) | c=1 | c=2 | c=4 | c=8 | prefill 8k |
| --- | --- | --- | --- | --- | --- |
| before (07-09 kernels, MMQ above 8 rows), tok/s | 90.6 | 172.7 | 259.9 | 439.6 | 1157 |
| this PR, tok/s | 99.1 | 188.5 | 319.5 | 533.7 | 1248 |

IQ3 GEMM time per step: 10.3 -> 7.8 ms at c=1, 19.8 -> 10.2 ms at c=4, ~23.3 -> 13.1 ms at c=8.

## Risks

- The in-memory weight is no longer GGUF layout for packed layers. Anything that reads those bytes
  (dequantize, a future method) must opt out or unpack (`iq3_pack.unpack`); two methods already do.
- sm_80+ only (see 07). The tiled kernel's tile-plan constants (unit cost per tile width, split cost)
  were calibrated on an RTX 3090; other GPUs get correct results with possibly suboptimal plans.
- Logits move slightly with the fp32 summation order: KLD against llama.cpp CUDA over 11 prompts went
  0.0237 -> 0.0249 between the builds without and with this PR and 09 together (deterministic run to
  run; the share of each was not separated).
