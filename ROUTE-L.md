# Route L: llama.cpp b11211 MMVQ/MMQ behind a shim

On main (built first on branch `route-l`), measured on a rented RTX 3090 in
phase 2, phase 3 and Integration 1: results and test counts in STATUS.md and
`cloud/results/`.

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
    weight's or shard run's rows, K = its columns; packed = the layer's IQ3
    runs were repacked at load, see below). Integration 2 table, pinned by
    `tests/cpu/test_lcpp_routing.py`:

    | type | n = 1..5 | n = 6, 7 | n = 8 | n = 9..32 | n ≥ 33 | source |
    |---|---|---|---|---|---|---|
    | IQ3_S, IQ3_XXS, packed | `lcpp_mul_mat_vec_iq3_mma_packed` | same | same | `lcpp_mul_mat_iq3_packed` (tiled) | `lcpp_mul_mat_iq3_packed` | phase3/r1, r2 |
    | IQ3_S, IQ3_XXS, not packed | `lcpp_mul_mat_vec_iq3` (dp4a) | `lcpp_mul_mat_vec_iq3_mma` | `lcpp_mul_mat_vec_iq3_mma` | MMQ | MMQ | phase3/item5, k2 |
    | Q4_K, W rows > 2048 | MMVQ at 1, 2; `lcpp_mul_mat_vec_own` from 3 | `lcpp_mul_mat_vec_own` | `lcpp_mul_mat_vec_own` | `lcpp_mul_mat_mma_k` | MMQ | opt/k1, opt/k3 |
    | IQ2_S, W rows > 2048 | `lcpp_mul_mat_vec_own` | `lcpp_mul_mat_vec_own` | `lcpp_mul_mat_vec_own` | `lcpp_mul_mat_mma_k` | MMQ | opt/k1, opt/k3 |
    | IQ4_XS, W rows > 2048 | MMVQ | MMVQ | MMQ | `lcpp_mul_mat_mma_k` at 9..16; at 17..32 only if W rows × K ≥ 12288 × 5120, else MMQ | MMQ | opt/k3 |
    | other Route L types; Q4_K/IQ2_S/IQ4_XS ≤ 2048 W rows | MMVQ | MMVQ | MMQ | MMQ | MMQ | phase3 item 1 |
    | IQ1_M (no MMQ upstream) | MMVQ | MMVQ | MMVQ | MMVQ in 8-row calls | stock dequantize + x @ W.T | opt-p2 |

  - Packing (`GGUFLinearMethod._pack_iq3`, `quantization/iq3_pack.py`, R1):
    with `VLLM_GGUF_LCPP=1`, after the padded weight is built, each IQ3 run
    of a layer is rewritten in place (same bytes, lossless) into the mma
    fragment order the packed kernels read, and `weight.iq3_packed` is set.
    All or nothing per layer: only if every IQ3 run has a multiple of 16 rows
    and 16-byte aligned bytes; otherwise the layer keeps the GGUF layout and
    the not-packed row applies. Methods that dequantize GGUF bytes
    (embedding, diffusion) set `pack_iq3 = False`.
  - A fused layer with several shard runs (mixed types) quantizes X once up
    front in `apply()` (`_quantize_x_q8_1`, opt-p) when any run's op reads
    q8_1 and passes it to those runs as `x_q8`. MMQ, `lcpp_mul_mat_mma_k` and
    `lcpp_mul_mat_iq3_packed` quantize X themselves in MMQ's layout
    (`_OWN_QUANTIZE_OPS`); every other op reads the shared q8_1. The dp4a IQ3
    kernel, mma_k and the tiled packed kernel write X's dtype; MMVQ, MMQ and
    the other owned kernels write fp32 and the shim casts.
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
- `tests/cpu/test_lcpp_guards.py`: 112 guard tests in a `no_gpu` subprocess;
  `tests/cpu/test_lcpp_routing.py`: the routing table above, every boundary, 134 cases (Integration 2).

## Shim overhead

Kernel launches per op call, counted from the code (VERIFIED by reading, not
profiled). "16-bit X" means the fp16/bf16 activations vLLM passes.

| | before | after | phase 3 | Integration 1 |
|---|---|---|---|---|
| MMVQ, 16-bit X | 5: cast X, memset whole q8, quantize_q8_1, mul_mat_vec_q, cast Y | 4: cast X, quantize_q8_1, mul_mat_vec_q, cast Y | 3: quantize_x, mul_mat_vec_q, cast Y | 2 or 3: [quantize_x], mul_mat_vec_q, cast Y |
| owned IQ3 dp4a, 16-bit X | | | 3: quantize_x, iq3_mul_mat_vec, cast Y | 1 or 2: [quantize_x], iq3_mul_mat_vec (writes 16-bit) |
| owned IQ3 mma, Q4_K/IQ2_S, 16-bit X | | | | 2 or 3: [quantize_x], kernel, cast Y |
| MMQ, 16-bit X | 5: cast X, memset whole q8, quantize_mmq_q8_1, mul_mat_q, cast Y; 7 with stream-k fixup (+ memset tmp_fixup, + fixup) | 5: cast X, memset tail only, quantize_mmq_q8_1, mul_mat_q, cast Y; 6 with stream-k fixup | 4: memset tail, quantize_x, mul_mat_q, cast Y; 5 with fixup | same as phase 3 |
| fp32 X | two fewer (no casts) | two fewer | one fewer (no cast Y) | one fewer where there is a cast Y |

[quantize_x] is skipped when the op gets `x_q8`: in a layer with several shard
runs, `apply()` quantizes X once before the runs and every q8_1-reading run
uses it. Single-run layers quantize inside the op.

- Nothing is zero-filled except the MMQ tail. The quantizers write every byte
  of their q8 region, with zeros past `ne00`. The stream-k `tmp_fixup` is
  written by `mul_mat_q` before the fixup reads it. Upstream's ggml pool
  doesn't zero either. The MMQ tail is J_max+128 blocks (≤ 36 KiB), not the
  whole buffer; since opt-p2 MMQ's quantize kernel zeroes it (one launch
  fewer per MMQ call than the phase 3 column).
- The output cast stays for MMVQ and MMQ: they write fp32 dst only (`float * dst` in
  both kernels' write-back), so a 16-bit Y needs vendored edits. The owned mma and
  Q4_K/IQ2_S kernels also write fp32 (16-bit output is a round-2 item). Phase 3 took
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

## Build

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

## Verification

- CPU: the .so loads under `no_gpu` and all five ops register without
  initialising CUDA (`tests/cpu/test_lcpp_guards.py`); `import
  vllm_gguf_plugin.ops` works with `VLLM_GGUF_LCPP=1` and without.
- GPU (rented sm86): kernel parity per type and row count against CPU models
  and vendored MMVQ, CUDA-graph capture/replay, guard cases, compute-sanitizer
  memcheck/initcheck, logit parity vs llama.cpp (phase 2), speed ladders.
  Counts and logs per stage: STATUS.md, `cloud/results/{phase2,phase3,opt-p,integration-1}`.
