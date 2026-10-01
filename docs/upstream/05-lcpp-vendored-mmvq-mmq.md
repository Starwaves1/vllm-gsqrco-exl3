# 05: llama.cpp b11211 MMVQ / MMQ behind a shim, opt-in (RFC)

Branch: `upstream/05-lcpp-vendored-mmvq-mmq` (4 commits, on `upstream/04-kernel-parity-harness`).
Depends on 04 for its tests (the CPU reference models); on 03 only through the stacking (the guard
harness files it extends).

**Title:** [RFC][Kernel] Opt-in llama.cpp b11211 MMVQ / MMQ kernels behind a shim (VLLM_GGUF_LCPP)

This is the large PR of the series (about 21k vendored lines, unmodified, and ~600 owned). It should
start with an RFC issue; proposed text below.

## Motivation

The plugin's CUDA kernels are llama.cpp b2899's. For IQ types there is no MMQ, so above the MMVQ row
limit every forward dequantizes the whole weight to bf16 and runs cuBLAS, and MMVQ re-reads the
activations for every two weight rows. For an IQ-heavy GGUF this dominates: a 27B Qwen3.5-architecture
model quantized to an IQ3_S-based mix (~82 % of weight bytes in IQ types) decodes at 32.4 tok/s on an
RTX 3090 with the stock kernels (MTP k=3), and prefills 8k tokens at 357 tok/s.

llama.cpp b11211 has MMQ (int8 tensor cores) for every one of these types and a much newer MMVQ.
Vendoring those files unmodified, behind one owned shim file and a default-off flag, gets the plugin
the same kernels and numerics as llama.cpp for these types without touching the default path.

## What changed

1. `csrc/lcpp/`: byte-identical copies of the llama.cpp b11211 (d7fb90e8) files MMVQ, MMQ and the q8_1
   quantizer need (found with `nvcc -M`), MMQ instances for nine types, llama.cpp's MIT LICENSE, and
   `VENDORED.md` with the source commit, every file's sha256 and why `mmq.cu` is not vendored. Excluded
   from clang-format and typos; LICENSE and VENDORED.md ship in the sdist.
2. `csrc/lcpp_shim.cu` (all adaptation; no vendored file is edited): the ggml-base / ggml-cuda.cu
   symbols the kernels link against (type sizes; error hooks that throw instead of `abort()`; device
   info from `cudaGetDeviceProperties`, filled on the first call); a context borrowing torch's current
   stream; a pool of uninitialised caching-allocator tensors (scratch reused, CUDA-graph safe); a copy
   of the non-MoE q8_1 branch of `ggml_cuda_mul_mat_q`; input checks before any launch; the ops
   `lcpp_mul_mat_vec_q` (1..8 rows) and `lcpp_mul_mat_q` (any rows), `(W, X, type, row) -> [n, row]` in
   X's dtype, also registered for CPU (checks only). Built only with `VLLM_GGUF_BUILD_LCPP=1`
   (setup.py, not for ROCm); used only with `VLLM_GGUF_LCPP=1` (ops.py refuses the flag without the
   build).
3. `quantization/linear.py`: under the flag, the nine types go to MMVQ below 8 rows and MMQ from 8.
   Mixed-type fused layers (e.g. gate Q4_K + up IQ4_XS) no longer copy each narrower shard per forward
   (`weight[start:end, :offset].contiguous()`): each run of adjacent same-type shards is stored
   contiguously when the padded weight is built and passed as a view, one product per run. A strided
   view cannot do this, because the padded row stride is not a multiple of the narrower shard's block
   size (2200/98, 1480/66, 2880/84 bytes). Without the flag nothing changes.
4. Tests: `test_lcpp_kernels.py` (CUDA), `test_lcpp_routing.py` and `test_lcpp_guards_cpu.py` (CPU),
   the lcpp ops added to the bad-input harness of 03, and the llama.cpp D2S6 reference model for Q2_K
   MMQ in `kernel_refs.py`.

### The MMQ read tail

`ggml_cuda_mul_mat_q` sizes its q8_1 buffer with a read tail of `ggml_cuda_mmq_get_J_max(..., ne11)`
blocks, which is 0 below 8 columns, while the kernel reads whole 8..128-column tiles. llama.cpp itself
sends at most 8 columns to MMVQ, so it is rarely exposed there; this plugin calls MMQ from 8 rows and
the tests call it at 1..7. The shim adds 128 `block_q8_1_mmq` (18 KiB) after upstream's tail and zeroes
both, the same fix Maxwell-Lyu's bridge made (f1d38ffdd0) after seeing IMAs and NaNs from the
uninitialised bytes. A ready-to-file llama.cpp issue with file:line references is in
`llamacpp-mmq-tail-issue.md`.

## How tested

- Build with `VLLM_GGUF_BUILD_LCPP=1 python setup.py build_ext --inplace` (CUDA 13.0, sm_86): clean.
  Clean build 226 s at one job under a 3 GB memory cap (mmvq.cu 32 s, each MMQ instance 16-19 s,
  peak 0.9 GB); the .so is 28.9 MB for sm_86. The default build is unchanged.
- `pytest tests --ignore=tests/test_kernels.py --ignore=tests/test_gguf_generation.py`: 265 passed, 9 skipped, 16 xfailed on vLLM 0.27.1 and vLLM main
- pre-commit hooks: clean. Vendored files: `sha256sum -c` against VENDORED.md and `cmp` against a
  llama.cpp checkout at d7fb90e8.
- GPU, on the development branch (RTX 3090, the 27B model's own tensors): 464 lcpp parity tests on all
  nine types, MMVQ 1..8 and MMQ 1..2048 rows, bf16 / fp16, odd row counts and poisoned scratch, with the
  calibrated tolerances unchanged (worst error to the closest CPU model 2.9e-3 bf16, 4e-4 fp16); CUDA-
  graph capture and replay bit-exact (81 of 81); every bad-input case rejected with no device fault
  (64 of 64, memcheck clean); MMQ at 1..7 rows clean under compute-sanitizer memcheck and initcheck with
  one cudaMalloc per tensor; mixed-shard layers bit-exact through `apply()`. The tests in this PR are
  those tests ported to sample-GGUF weights; in this form they were collected and their harness run on
  the CPU with the ops replaced by the reference models, not yet on a GPU.
- Serving: the model loads and serves with MTP; logit KLD against llama.cpp b11211 CUDA over 11 prompts
  (1k..120k tokens) 0.028 overall (stock kernels 0.040), top-1 agreement 97.9 %.

### Speed (RTX 3090, 350 W, vLLM 0.27.1, MTP k=3, T=0, 8 real prompts per concurrency)

| | decode c=1 | decode c=2 | prefill 8k | prefill 180k |
| --- | --- | --- | --- | --- |
| stock kernels (e2b8ad5) | 32.4 tok/s | 38.0 | 357 tok/s | 281 |
| this PR (MMQ from 9 rows) | 77.8 | 117.0 | 1036 | 589 |
| this PR (MMQ from 8 rows, as submitted) | 77.8 | 135.5 | | |

The "as submitted" row adds the 8-row threshold; one product per same-type run then cut GEMM launches
per target pass from 433 to 356 (ms per engine step -0.8 % / -1.2 % at c=1 / c=2).

## Risks and what a reviewer will ask

- **Size and maintenance.** ~21k vendored lines. They are never edited: updating means re-copying the
  list from a new tag and re-running the parity tests. The shim is the only coupling, to b11211's
  internal ggml-cuda interfaces (`ggml_cuda_op_mul_mat_vec_q`, `mul_mat_q_case`, `mmq_args`, the pool
  and context types), which do change between llama.cpp releases.
- **Build time.** A few minutes more at one job; opt-in only.
- **Numerics.** Results differ from the stock kernels (MMQ's stream-k order, llama.cpp's q8_1 details,
  the D2S6 Q2_K layout); they match llama.cpp's own CUDA backend's behaviour. The absolute logit gate
  we set (KLD <= 1e-3 against llama.cpp CUDA) fails for the stock kernels too.
- **Scope.** CUDA only, dense linears only (no MoE); IQ1_M is not routed here (no MMQ upstream).
- **Alignment.** The ops require 16-byte aligned W. A same-type run that starts at an odd row of a padded
  weight whose row stride is 8 mod 16 bytes (IQ3_S 2200, IQ2_S 1480, Q6_K 4200 bytes per 5120 values)
  would be rejected with a clear error; no layer of the models we ran hits that.
- **Related work.** Maxwell-Lyu's `codex-ggml-source-bridge` branch (Maxwell-Lyu/vllm-gguf-plugin)
  bridges the same ggml-cuda sources, the whole tree including MoE and FP4, restructured often. This
  PR takes the narrow slice for dense MMVQ / MMQ and borrows its read-tail fix. We would like to
  converge rather than compete: one vendored tree and one shim that both efforts use.

## Proposed RFC issue

> **[RFC] Optional llama.cpp matmul kernels for GGUF (vendored, behind a shim)**
>
> The plugin's CUDA kernels come from llama.cpp b2899. IQ types have no MMQ there, so above the MMVQ
> row limit every forward dequantizes the whole weight and runs cuBLAS; on an IQ3-heavy 27B GGUF on an
> RTX 3090 that is 357 tok/s of 8k prefill and 32 tok/s of MTP decode.
>
> Proposal: vendor, byte-identical, the llama.cpp ggml-cuda files for MMVQ, MMQ and the q8_1
> quantizer at a pinned tag (b11211 now; ~21k lines, sha256-listed), adapt them in one owned shim
> (`csrc/lcpp_shim.cu`: device info, a context on torch's stream, a torch-allocator pool, input
> checks), expose them as `_C_gguf.lcpp_*` ops, and route to them from `linear.py` only when the
> extension is built with `VLLM_GGUF_BUILD_LCPP=1` and `VLLM_GGUF_LCPP=1` is set. Defaults unchanged.
>
> Measured on that model: decode 32.4 -> 77.8 tok/s (c=1), 38.0 -> 135.5 (c=2), prefill 357 -> 1036
> tok/s at 8k. With owned kernels built on the same shim (follow-up PRs, each optional) the same
> model reaches 110 / 193 / 348 / 541 tok/s at c=1/2/4/8 and 1248 tok/s of 8k prefill.
>
> Questions for maintainers:
>
> 1. Is vendoring llama.cpp sources acceptable here, and at which granularity (these files only, or a
>    wider slice shared with other efforts such as the codex-ggml-source-bridge branch)?
> 2. Build flag vs. a separate optional extension module?
> 3. Tag bumps: who re-runs the GPU parity suite, on which hardware?
> 4. CI: the CPU tests (routing, input checks, host dequantization) run anywhere; the CUDA tests need a
>    GPU runner.
>
> A draft series is ready: vendored files + shim + routing (1 PR), then an owned q8_1 quantizer, owned
> IQ3 / Q4_K / IQ2_S / IQ4_XS kernels, a load-time IQ3 repack, and small plumbing changes, each its own
> PR with tests and numbers.
