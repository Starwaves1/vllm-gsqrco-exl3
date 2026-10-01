# llama.cpp issue (ready to file): MMQ under-allocates the q8_1 read tail below 8 columns

Target: ggml-org/llama.cpp. Checked at tag b11211 (commit d7fb90e8e2494b2908934d956a3202fd60152ee0).
Not re-checked against master.

---

**Title:** CUDA MMQ: q8_1 buffer has no read tail when ne11 < 8, but the kernel reads a whole J-column tile

**Body:**

`ggml_cuda_mul_mat_q` sizes the quantized activation buffer as the quantized data plus a read tail
of `ggml_cuda_mmq_get_J_max(...)` blocks:

- `ggml/src/ggml-cuda/mmq.cu:188-190` (dense path):

  ```cpp
  const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
      ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
  ```

- `ggml/src/ggml-cuda/mmq.cu:257-259` (the `ids` path) sizes its tail the same way.

`ggml_cuda_mmq_get_J_max` (`ggml/src/ggml-cuda/mmq.cuh:372-381`) starts from `min(ne11, 512)`
rounded **down** to a multiple of 8:

```cpp
int ret = std::min(ne11, int64_t(512));
ret -= ret % 8;
for (;ret > 0; ret -= 8) { ... }
return ret;
```

so for `ne11 < 8` it returns 0: no tail at all. The kernel, however, runs whole J-column tiles:
`mul_mat_q_switch_J` (`mmq.cuh:1486-1512`) picks the J (a multiple of 8, from 8 to 128) that needs
the fewest tiles, which for `ne11 < 8` is at least 8, and the tile loaders read J columns of the
`block_q8_1_mmq` data, laid out column-fastest within each 128-value K block. For the last K block,
columns `ne11 .. J-1` lie past the end of the allocation. With `ne11 >= 8` the rounded-down J_max
covers the overshoot (a single tile overshoots by less than 8 columns, and J_max >= 8), so only
`ne11 < 8` is exposed.

**Who hits it.** llama.cpp's own dispatch sends `ne11 <= MMVQ_MAX_BATCH_SIZE` (8) to MMVQ
(`ggml-cuda.cu:1798-1799`), so the dense path is only reached with fewer than 8 columns by code that
calls MMQ directly. We hit it running the b11211 MMQ kernels from vLLM (vllm-gguf-plugin), which uses
MMQ from 8 rows and, in tests, at 1..7 rows. Maxwell-Lyu's vLLM bridge of the same kernels saw
illegal memory accesses and NaNs from the uninitialised pool bytes there (Maxwell-Lyu/vllm-gguf-plugin,
commit f1d38ffdd0). Whether the `ids` path can see `ne11 < 8` in practice was not checked.

**What we measured.** Our wrapper allocates 128 extra `block_q8_1_mmq` (18 KiB) after upstream's tail
and zeroes both tails. With that, MMQ at 1..7 columns is clean under compute-sanitizer memcheck (one
cudaMalloc per tensor) and initcheck, and parity tests pass at 1..9 columns with the caching
allocator's free blocks filled with 0xFF (RTX 3090, sm_86). We did not run upstream's sizing alone
under the sanitizer.

**Proposed fix.** Size the tail from the largest J the kernel can select, not from the rounded-down
`ne11`. The simplest rule is a constant: the J loop in `mul_mat_q_switch_J` stops at 128, so a tail of
128 blocks (18 KiB) always covers the overshoot:

```diff
-        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
-            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
+        // the kernel reads whole J-column tiles (J <= 128, mul_mat_q_switch_J)
+        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
+            128 * sizeof(block_q8_1_mmq);
```

(and the same at `mmq.cu:257`). Alternatively, keep `ggml_cuda_mmq_get_J_max` but round `ne11` up
to a multiple of 8, and return the smallest valid J at least that large when no smaller one is valid.
Whether the tail must also be zeroed depends on whether values read from it can reach written
outputs; we zero it to be safe and did not establish that it is needed.
