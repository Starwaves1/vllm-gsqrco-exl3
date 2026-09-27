# Route L: llama.cpp b11211 MMVQ/MMQ behind a shim (compile-only)

Branch `route-l`. Built and checked on the CPU only. **No kernel has run.**
Every numeric property is untested.

## What is here

- `plugin/vllm_gguf_plugin/csrc/lcpp/`: llama.cpp b11211 (d7fb90e8, MIT)
  ggml-cuda files, byte-identical, with `VENDORED.md` (file list, sha256, why
  `mmq.cu`/`mmid.cu` are not vendored). Nothing vendored is edited.
- `plugin/vllm_gguf_plugin/csrc/lcpp_shim.cu`: all adaptation.
  - ggml-base stubs; errors throw instead of `abort()`.
  - `ggml_cuda_info()` from `cudaGetDeviceProperties`, filled lazily on the
    first op call.
  - A context that borrows torch's current stream.
  - A torch-tensor pool. Every allocation is zero-filled and gets a zeroed
    128×`block_q8_1_mmq` (18 KiB) tail, per Maxwell-Lyu f1d38ffdd0.
  - A copy of the non-MoE q8_1 branch of upstream `ggml_cuda_mul_mat_q`.
  - Ops `_C_gguf::lcpp_mul_mat_vec_q` (1..8 rows) and `lcpp_mul_mat_q` (any
    rows), signature `(W uint8[rows,bytes], X [n,K] f32/f16/bf16, type, row) ->
    [n,row]` in X's dtype.
  - Guards (dtype, 2-D, supported type, row range, K = X cols, K % 512, W
    inner stride 1, W row stride a multiple of the block size, 16-B alignment,
    X inner stride 1, MMVQ ≤ 8 rows) all run before any launch. The ops are
    registered for CPU too, where they run the guards and then reject.
- Build: `VLLM_GGUF_BUILD_LCPP=1` in `plugin/setup.py`. The default build is
  unchanged.
- Runtime: `VLLM_GGUF_LCPP=1` (default off, so e2b8ad5 behaviour is unchanged).
  - `linear.py`: for Q2_K/Q4_K/Q6_K/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS,
    ≤8 rows go to lcpp MMVQ and >8 to lcpp MMQ. IQ1_M keeps the old path.
  - Mixed-type fused layers: each shard is stored contiguously inside its
    padded region, so `_shard_weight` returns a view and the per-forward
    `.contiguous()` copy is gone.
  - A row-strided view can't do this job. The padded byte stride is not a
    multiple of the narrower shard's block size (2200/98, 1480/66, 2880/84),
    and the kernels index rows in blocks.
  - `diffusion_config.py` uses the same helper.
- `tests/cpu/test_lcpp_guards.py`: 42 guard cases in a `no_gpu` subprocess.

## Build (compile only)

```
source ~/gsq-vllm/tools/cuda-env.sh; export PATH=~/gsq-vllm/.venv/bin:$PATH
cd plugin && VLLM_GGUF_BUILD_LCPP=1 MAX_JOBS=1 python setup.py build_ext --inplace
```

Use an in-place build. Don't use `build-plugin.sh` from a worktree: it would
repoint the shared venv's editable install.

Measured under MemoryMax=3G, CPUQuota=200%, nice 19, MAX_JOBS=1:

- Clean build: 226 s wall.
- Per-TU compile: mmvq.cu 32 s; each MMQ instance 16–19 s; quantize 3 s.
- Peak cgroup memory: 0.90 GB.

The .so is 28.9 MB, sm_86 only. It NEEDs libcudart.so.13 plus torch, and it
resolves every ggml symbol.

`cublas_v2.h` (declarations only, never linked) comes from the venv's
`nvidia/cu13/include` via `-idirafter`.

## Checked on the CPU

- The .so loads under `no_gpu` and both ops register.
- `import vllm_gguf_plugin.ops` works with `VLLM_GGUF_LCPP=1` and with it off,
  and `torch.cuda.is_initialized()` stays False.
- `tests/cpu`: 458 passed, 54 xfailed.

## Untested: needs a GPU (sm86)

1. Correctness per type: `tests/gpu/test_kernel_parity.py` and
   `_guard_case.py`. They call `ops.ggml_mul_mat_*`; point them at
   `torch.ops._C_gguf.lcpp_*`, or run through `_fused_mul_mat_gguf` with
   `VLLM_GGUF_LCPP=1`.
   - Check MMQ at n = 1..7, where upstream's J_max tail is 0 and only the
     shim's 18 KiB tail protects the reads.
   - Check non-multiple-of-128 rows (the fallback tiles).
   - Run under compute-sanitizer (`GSQ_COMPUTE_SANITIZER`).
2. CUDA-graph capture and replay (`graph_replay` case). Pool tensors are
   stream-ordered torch allocations. `cudaFuncSetAttribute` runs on first use,
   so warm up before capture.
3. Speed: `bench/speed/` and per-layer microbenchmarks at 1/4/8/16/32/2048
   rows against e2b8ad5.
4. Parity: `bench/parity/` (KLD vs llama.cpp).
5. An end-to-end serve with `VLLM_GGUF_LCPP=1` (`scripts/serve-gsq.sh`),
   including the flat mixed-shard layout.
