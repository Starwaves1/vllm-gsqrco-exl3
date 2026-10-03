# 11b: No zero fill in ggml_dequantize (FIXTURE)

This is a dashboard fixture, not a real report.

## Initial problem and concise proof

`ggml_dequantize` allocates its output with `torch::zeros` and then the kernel writes **every** element.
On an RTX 3090, a 4096 x 14336 Q4_K dequantize spends 31 us of 140 us in the fill:

| | before | after |
| --- | --- | --- |
| dequantize, 4096 x 14336 | 140 us | 109 us |
| output bytes compared | equal | equal |

## Problem mechanism (diagnosis, checked and verified)

1. The output is created zero-filled (`gguf_kernel.cu:812`).
2. The dequantize grid covers `m * n / QK` blocks and each thread writes its `QK` outputs, so no element
   keeps the fill value.
3. The fill is a separate `fill_` kernel launch on the same stream: pure overhead.

## Fix mechanism

Allocate with `torch::empty`. The grid-coverage argument in step 2 is the whole proof; the CPU test checks
that a NaN-prefilled output buffer has no NaN left after the call.

```cpp
- at::Tensor Y = torch::zeros({m, n}, options);
+ at::Tensor Y = torch::empty({m, n}, options);
```

## Final check agent's concise opinion

> Verified the grid covers every output for all supported types (including the tail block). No downside.

## Fable's comment

**Recommendation:** approve. **Confidence:** high.

Smallest possible diff, obvious rule, proof by construction plus a test. See [the branch](https://github.com/Starwaves1/vllm-gsqrco-exl3/tree/upstream/11b-dequant-no-zero-fill).
