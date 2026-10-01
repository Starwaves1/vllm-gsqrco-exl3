# 03: Check the inputs of the dense CUDA ops before launching

Branch: `upstream/03-kernel-input-guards` (3 commits, on e2b8ad5). No dependency.

**Title:** [BugFix] Check the inputs of ggml_mul_mat_vec_a8, ggml_mul_mat_a8 and ggml_dequantize

## Motivation

The dense ops only take `data_ptr()` of their tensors and trust the caller for everything else. Run
one bad input per subprocess on an RTX 3090 at e2b8ad5 (5 type/op combinations: IQ3_S, IQ4_XS, Q4_K
on MMVQ; Q4_K, Q6_K on MMQ):

| case | behaviour at e2b8ad5 |
| --- | --- |
| X a transposed (non-contiguous) view | silently wrong, relative error ~1.5 |
| W a narrow view of a wider buffer (row stride > row bytes) | NaN |
| W one byte off 16-byte alignment | device fault, "misaligned address" |
| row argument larger than W's rows | accepted, reads past W |
| X narrower than W's K | accepted, reads past X |
| unsupported type | all-zero result (`ggml_dequantize`: a call through a null function pointer) |
| X one element off alignment, CUDA-graph replay | fine |

`apply()` passes clean tensors today, so none of this shows in serving; it shows when a caller or a
refactor passes a view. The narrow-view NaN is exactly what `weight[start:end, :offset]` without
`.contiguous()` would produce.

## What changed

- `csrc/gguf/gguf_kernel.cu`: one check function for the two matmul ops (type supported by that op,
  2-D, W uint8, X fp32/fp16/bf16, K from W's row bytes equal to X's columns, `0 < row <= W rows`,
  W rows contiguous and 16-byte aligned, X rows contiguous, the MMVQ grid limit of 65535 rows, sizes
  within int, devices), and the matching checks in `ggml_dequantize` (type, m x n a whole number of
  blocks, W contiguous and large enough, alignment, dtype). All run before any launch; the device checks
  run last.
- `csrc/torch_bindings.cpp`: the three ops are also registered for CPU, where the checks run and a
  call that passes them ends in "must be CUDA tensors". The MoE ops are unchanged.
- `tests/test_kernel_guards_cpu.py` ((CPU, 45 cases)): which check fires for each bad input.
- `tests/kernel_guard_case.py` + `tests/test_kernel_guards.py` (CUDA): each bad input in its own
  process (a device fault is sticky); pass = clean rejection or the same result as on clean inputs.
  `GGUF_COMPUTE_SANITIZER=/path/to/compute-sanitizer` also runs each case under memcheck with the
  caching allocator off.

## How tested

- Build: `python setup.py build_ext --inplace` (CUDA 13.0, sm_86): clean, no new warnings.
- `pytest tests --ignore=tests/test_kernels.py --ignore=tests/test_gguf_generation.py`: 149 passed, 7
  skipped on vLLM 0.27.1 and vLLM main (the 45 CPU guard cases included; the CUDA module skips on a
  machine without a GPU).
- pre-commit hooks: clean.
- Not run on a GPU in this form: the table above comes from the same harness at e2b8ad5 with the
  development model's weights; the stock-op checks themselves are new here. The same kind of checks
  guard the lcpp ops of PR 05 and passed there (GPU guard cases 240 passed, compute-sanitizer memcheck
  and initcheck clean).

## Risks

- A caller that passed a row-strided or transposed X, or a W view with padding, used to get wrong
  numbers and now gets an exception. If maintainers prefer, `_fused_mul_mat_gguf` can make X contiguous
  before the call instead (a no-op when it already is).
- The 16-byte alignment check is stricter than some types strictly need; every caller in the plugin
  passes whole allocations or `.contiguous()` copies (256-byte aligned).
