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
  - A torch-tensor pool: uninitialised caching-allocator tensors, so scratch
    is reused across calls. MMQ adds a 128×`block_q8_1_mmq` (18 KiB) read tail
    after upstream's J_max tail and zeroes only those two tails, per
    Maxwell-Lyu f1d38ffdd0.
  - A copy of the non-MoE q8_1 branch of upstream `ggml_cuda_mul_mat_q`.
  - Ops `_C_gguf::lcpp_mul_mat_vec_q` (1..8 rows), `lcpp_mul_mat_q` (any
    rows), and three owned 1..8-row kernels: `lcpp_mul_mat_vec_iq3`
    (IQ3_S/IQ3_XXS, dp4a, `iq3_mul_mat_vec` in `lcpp_shim.cu`, phase 3 item
    5), `lcpp_mul_mat_vec_iq3_mma` (IQ3_S/IQ3_XXS, int8 tensor cores,
    `lcpp_owned_iq3_mma.cu`, K2) and `lcpp_mul_mat_vec_own` (Q4_K/IQ2_S,
    dp4a, `lcpp_owned_k4.cu`, K1). Signature `(W uint8[rows,bytes], X [n,K]
    f32/f16/bf16, type, row[, x_q8]) -> [n,row]` in X's dtype; the 1..8-row
    ops take an optional `x_q8`, X already quantized to MMVQ's q8_1 layout.
  - An owned fp32/fp16/bf16 → q8_1 quantizer (`quantize_x`, phase 3): the
    vendored quantizers' arithmetic and layouts line for line, reading X in
    its own dtype, so there is no input cast; its bytes equal the vendored
    ones on X.float() (`test_lcpp_quantize_vs_vendored`). MMVQ runs through
    upstream's q8_1 entry `ggml_cuda_op_mul_mat_vec_q`.
  - Guards (dtype, 2-D, supported type, row range, K = X cols, K % 512, W
    inner stride 1, W rows contiguous, 16-B alignment, X inner stride 1,
    MMVQ ≤ 8 rows) all run before any launch. The ops are
    registered for CPU too, where they run the guards and then reject.
- Build: `VLLM_GGUF_BUILD_LCPP=1` in `plugin/setup.py`. The default build is
  unchanged.
- Runtime: `VLLM_GGUF_LCPP=1` (default off, so e2b8ad5 behaviour is unchanged).
  - `linear.py` routing (`_lcpp_op`; n = activation rows, W rows = the
    weight's or shard run's rows). IQ1_M keeps the old path.

    | type | n = 1..5 | n = 6, 7 | n = 8 | n ≥ 9 | source |
    |---|---|---|---|---|---|
    | IQ3_S, IQ3_XXS | `lcpp_mul_mat_vec_iq3` (dp4a) | `lcpp_mul_mat_vec_iq3_mma` | `lcpp_mul_mat_vec_iq3_mma` | MMQ | phase3/item5, phase3/k2 |
    | Q4_K, W rows > 2048 | MMVQ at 1, 2; `lcpp_mul_mat_vec_own` from 3 | `lcpp_mul_mat_vec_own` | `lcpp_mul_mat_vec_own` | MMQ | opt/k1 |
    | IQ2_S, W rows > 2048 | `lcpp_mul_mat_vec_own` | `lcpp_mul_mat_vec_own` | `lcpp_mul_mat_vec_own` | MMQ | opt/k1 |
    | other Route L types; Q4_K/IQ2_S ≤ 2048 W rows | MMVQ | MMVQ | MMQ | MMQ | phase3 item 1 |

  - Fused layers quantize X once (`_quantize_x_q8_1`, opt-p) for all their
    shard runs whose op reads q8_1 (every op above but MMQ) and pass it as
    `x_q8`. The dp4a IQ3 kernel writes X's dtype; MMVQ, MMQ and the other two
    owned kernels write fp32 and the shim casts.
  - GGUF BF16/F16/F32 linears (GDN `in_proj_ba`) go through
    `GGUFUnquantizedLinearMethod`: ≤ 8 rows × ≤ 128 weight rows, no bias, as a
    batched gemv (`torch.bmm`), otherwise `F.linear` (opt-p). Independent of
    `VLLM_GGUF_LCPP`.
  - Mixed-type fused layers: each run of adjacent same-type shards is stored
    contiguously from the start of its first shard's region, so
    `_shard_weight` returns a view and the per-forward `.contiguous()` copy
    is gone.
  - A row-strided view can't do this job. The padded byte stride is not a
    multiple of the narrower shard's block size (2200/98, 1480/66, 2880/84),
    and the kernels index rows in blocks.
  - `diffusion_config.py` uses the same helper.
- `tests/cpu/test_lcpp_guards.py`: 42 guard cases in a `no_gpu` subprocess.

## Shim overhead

Kernel launches per op call, counted from the code (VERIFIED by reading, not
profiled). "16-bit X" means the fp16/bf16 activations vLLM passes.

| | before | after | phase 3 |
|---|---|---|---|
| MMVQ, 16-bit X | 5: cast X, memset whole q8, quantize_q8_1, mul_mat_vec_q, cast Y | 4: cast X, quantize_q8_1, mul_mat_vec_q, cast Y | 3: quantize_x, mul_mat_vec_q, cast Y |
| MMQ, 16-bit X | 5: cast X, memset whole q8, quantize_mmq_q8_1, mul_mat_q, cast Y; 7 with stream-k fixup (+ memset tmp_fixup, + fixup) | 5: cast X, memset tail only, quantize_mmq_q8_1, mul_mat_q, cast Y; 6 with stream-k fixup | 4: memset tail, quantize_x, mul_mat_q, cast Y; 5 with fixup |
| fp32 X | two fewer (no casts) | two fewer | one fewer (no cast Y) |

- Nothing is zero-filled except the MMQ tail. The quantizers write every byte
  of their q8 region, with zeros past `ne00`. The stream-k `tmp_fixup` is
  written by `mul_mat_q` before the fixup reads it. Upstream's ggml pool
  doesn't zero either. The MMQ tail memset is J_max+128 blocks (≤ 36 KiB),
  not the whole buffer.
- The output cast stays: MMVQ and MMQ write fp32 dst only (`float * dst` in
  both kernels' write-back), so a 16-bit Y needs vendored edits. Phase 3 took
  the input cast out with the owned quantizer instead (1.8 → 0.7 ms of casts
  per c=1 decode step). Fused layers run one product per run of adjacent
  same-type shards, e.g. GDN q/k/v (one attn_qkv tensor) + z: 2, not 4
  (433 → 356 GEMMs per target pass).

## MMVQ decode reuse (SASS)

**VERIFIED.** Checked with nvdisasm 13.3 on the sm_86 cubin of the built .so,
kernels `mul_mat_vec_q<IQ3_S, ncols_dst, false, false, false>`. `c[0x4][0x38]`
is the `iq3s_grid` relocation. Counts are for the main k-loop body of each
kernel:

| ncols_dst | rows/block | iq3s_grid loads | dp4a (IDP.4A) | LOP3 | y int loads | instructions |
|---|---|---|---|---|---|---|
| 1 | 1 | 8 | 8 | | | |
| 2 | 2 | 16 | 32 | | | |
| 4 | 2 | 16 | 64 | 128 | 32 | 477 |
| 8 | 2 | 16 | 128 | 128 | 64 | 639 |

- nvcc CSEs the IQ3_S decode across columns. Each iteration makes 8 grid
  loads per weight block (8 × 2 rows = 16), the same at ncols 2, 4 and 8.
- The qs/qh/signs/scales/d loads are the same: 14 U16 + 4 U8 at both 4 and 8.
- The sign unpack (LOP3 = 128) is also identical at 4 and 8.
- Only the per-column work grows: 8 y int loads + 1 ds load + 16 dp4a per
  column, about 40 instructions for each added column.
- This is legal because the loop has no stores and `vx` is `__restrict__`
  const.
- So "MMVQ re-decodes per column" (report 14 D4) is wrong for IQ3_S at
  b11211/nvcc 13. At 4–8 rows, the decode-once argument for MMQ-small or
  fastllm small-mmvq is weaker than assumed. Only a GPU measurement can say
  whether IQ3 MMVQ is gather-latency-bound (16 dependent L1 gathers per
  iteration).

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
