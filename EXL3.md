# EXL3: exllamav3 trellis models on vLLM through a second plugin

Phase 0 (CPU only) on branch `exl3`, 2026-09-30. Design: `docs/adr/0002-exl3-via-plugin.md`
(proposed) and `docs/exl3-feasibility.md`. Model: `erlidev/Swift-1.5-Qwen3.8-27B-EXL3@SC_3.50bpw_H4_V6`
(revision 041dc382), with `turboderp/Qwen3.8-27B-exl3@3.50bpw` (8351c54e, phase 0's first
checkpoint) as the A/B; see "Model" below. The package builds, registers, loads both checkpoints'
names and shapes on the meta device, and passes its CPU tests. No kernel has run: every number the plugin would produce is
untested. Labels as elsewhere: VERIFIED (checked here), DOCUMENTED, INFERRED.

## What is here

| path | what |
|---|---|
| `plugin-exl3/` | package `vllm_exl3_plugin` (Apache-2.0), entry point `vllm.general_plugins: exl3` |
| `plugin-exl3/vllm_exl3_plugin/plugin.py` | `register()`: `register_quantization_config("exl3")`, nothing else, no monkeypatches |
| `.../format.py` | torch-free checkpoint facts: tensor suffixes, codebook multipliers, K from the tile width, config block parser |
| `.../quantization/config.py` | `EXL3Config`: reads `config.json` `quantization_config`, picks the method per layer |
| `.../quantization/linear.py` | `EXL3LinearMethod`: per-shard parts, loader, warmup, `apply` |
| `.../ops.py` | loads `_C_exl3`, routing `_exl3_op`, the vLLM custom op `_exl3_linear` + fake impl |
| `.../weights_adapter/qwen3_5.py` | Qwen3.5/3.8 name mapping notes, the unquantized-module rule, draft head names |
| `.../csrc/exl3/` | exllamav3 d3739fd (v1.5.3, MIT), 94 files byte-identical, `VENDORED.md` (sha256), `LICENSE` |
| `.../csrc/exl3_shim.cu` | all adaptation: ops, guards, capture guard, warmup, CPU registration |
| `tools/exl3_draft_head.py` | writes the pruned MTP draft head as bf16 (GPU, once per checkpoint) |
| `tools/exl3_meta_dry_run.py` | vLLM's real model + loader on the meta device against the checkpoint's metadata |
| `bench/parity/exl3_logits.py` | exllamav3-native logit dump for parity (not run yet) |
| `hf-config/Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6/`, `hf-config/Qwen3.8-27B-exl3-3.50bpw/` | each checkpoint's metadata: `config.json`, tokenizer files, safetensors index, shard headers (`safetensors_headers.json`, read by HTTP range), `PROVENANCE.json`; erlidev's also `quantization_config.json` (per-tensor bits) |
| `tests/cpu/test_exl3_*.py` | CPU tests (below) |

## Model

`erlidev/Swift-1.5-Qwen3.8-27B-EXL3@SC_3.50bpw_H4_V6` (041dc382, 14.67 GB): Swift 1.5 (the
GGUF route's finetune), self-calibrated, `sc_optimize` per-tensor recipe, `Qwen3_5ForConditionalGeneration`,
`mtp_num_hidden_layers` 1, codebook mul1. A/B: `turboderp/Qwen3.8-27B-exl3@3.50bpw` (base model).
Box jobs and `scripts/env.sh` default to erlidev's; `--alt` on a job (and `00-prep.sh model --alt`)
uses turboderp's, with `-alt` run/result dirs. VERIFIED on the fetched metadata:

- Tensors: 3,080 vs 2,426, all of it the vision tower (987 vs 333). The 2,093 text tensors
  (2,054 + 39 `mtp.*`) have the same names, dtypes and shapes; only trellis widths (bits) differ
  (216 of 409). erlidev's tower: q/k/v_proj, proj, fc1, fc2 and the merger's two linears as EXL3
  K6 (164 modules, fp16 biases and norms), fc1 padded 4304 -> 4352, plus the bf16 fused
  `attn.qkv` still stored. vLLM's tower wants fused bf16 qkv and 4304 rows: not loadable, not
  needed (text-only, Loading above).
- Bits (`quantization_config.json`, 401 text modules, matches the headers): decoder K2 x4 (layers
  0-1 gate/up), K3 x171, K4 x193, K5 x32 (k/v_proj); lm_head K4 (`head_bits` 4); MTP K4 x8
  (`mtp_bits` 4); vision K6 (`vision_bits` 6, skipped). No half-integer K. All integer K 1..8 are
  compiled; K2 and the K4 head are new against turboderp (K3-6), so `tests/gpu/exl3_cases.py`'s
  erlidev table has K2-up, K5-kproj and K4-lmhead cases.
- Format version: erlidev wrote it with exllamav3 1.5.0, turboderp with 1.4.2, the kernels are
  1.5.3 (d3739fd). `LinearEXL3`'s storage did not change from 1.4.2 to 1.5.3 for integer K
  (trellis int16 [k/16, n/16, 16K], fp16 suh/svh, int32 mul1, multiplier 0x83DCD12D in every
  entry); 1.5.1 only added half-integer K (16K+8 tiles), which neither uses. Same
  `calibration`/`out_scales`/`tensor_storage` fields. No shim needed.
- Size (text-only, what loads): 14.10 GB (13.13 GiB: embedding bf16 2.54 GB, lm_head 0.64,
  MTP 0.21, decoder 10.71) + the 0.42 GB draft head = 14.52 GB, 0.94 GB under the 15.46 GB weight
  budget for 200k at gpu-util 0.94. turboderp: 14.42 + 0.42 = 14.84 GB (0.62 under; its head is
  K6). The MTP draft's transient own copy of lm_head + embedding at load still needs checking.
- Chat template: erlidev's `chat_template.jinja` is byte-identical to turboderp's and to Swift
  1.5's stock one (c3cf9e34), tokenizer files too. It is never served: `serve-exl3.sh` passes
  production's template (qwen-sharp froggeric v22.1, d1f22a89, sha asserted), as the GGUF route
  does. Stock vs production: stock defaults to effort xhigh and injects the xhigh instruction;
  production defaults to medium, adds a terse-answer system block, `<|think_*|>` toggles, the
  developer role, a brief-`<think>` tool-call example, an optional JSON tool format, tool-error
  nudges, and tolerates a missing user turn. Both emit XML tool calls (`<tool_call><function=..>
  <parameter=..>`, `qwen3_coder`) and open the generation with `<think>\n` (`qwen3`): no parser
  change.

## Vendored files

exllamav3 `d3739fd393337b1ff4d6c2a342b12f0c87a9592f`, `exllamav3/exllamav3_ext/`: the `#include`
closure of the dense-linear entry points plus every `.cu` needed to link. 94 files, 381 KB,
60 compiled: `quant/{exl3_gemm, exl3_gemv, exl3_gemv_int8, exl3_kernel_map, exl3_devctx,
coop_autotune, hadamard, reconstruct, frac}.cu`, `hgemm.cu`, `hgemm_f16acc.cu`, `graph.cu`,
and 48 instance units in `quant/comp_units/` (K 1..8 x {3inst, mcg, mul1}, half-integer K,
GEMV and int8-GEMV instances). The list with hashes is `csrc/exl3/VENDORED.md`; `cmp` against a
checkout shows no differences (VERIFIED).

Three things the feasibility survey did not list (VERIFIED while building):

- `graph.cu` (exllamav3's own CUDA graph recorder) calls `CudaDrv::instance()` from
  `cuda_drv.cpp`, which dlopens libcuda. It is not vendored. The shim defines a stub that throws,
  since the shim never passes a `Graph` and the recorder never runs. `graph.cu` is compiled only
  because `exl3_gemm.cu` references `Graph` members.
- `quant/reconstruct.cuh` and `quant/hadamard.cuh` are not in the closure (no vendored file
  includes them). The shim declares the three functions it calls.
- The int8-activation GEMV (`EXL3_INT8_GEMV`) is **on by default** at d3739fd
  (`exl3_gemv_int8_mode()` returns 2 when the variable is unset), not opt-in as the survey says.
  It takes mul1 tensors with K <= 5 on Ampere at 1-2 rows, which is exactly our decode path. It
  changes the numerics (about 0.9 % of output RMS per its comment), `cudaMalloc`s a 16 MB
  workspace on first use, and the vendored comment calls it not graph-capturable. The shim sets
  `EXL3_INT8_GEMV=0` when the library loads (`test_int8_gemv_switched_off`). The parity reference
  must set it too.

## Ops (`torch.ops._C_exl3`)

| op | what |
|---|---|
| `exl3_gemm(x, trellis, suh, svh, mcg, mul1, out_fp32)` | vendored `exl3_gemm`: the QTIP GEMV where its 3090-tuned heuristic picks it, else the cooperative kernel, autotuned per shape. x fp16 `[m, k]` (the kernels' only activation dtype); output fp32 (`c_fp32`) or fp16. `EXL3LinearMethod.apply` casts bf16 x to fp16 once per layer, asks for fp32 output with a bf16 model (no fp16 rounding or overflow on the output side; exllamav3 itself defaults to fp16 output) and casts once after concatenating the parts |
| `exl3_dequant(trellis, suh, svh, mcg, mul1, n_start, n_count, had)` | fp16 `W[k, n_count]` (y = x @ W) for 128-aligned columns: `had=True` the original basis (`reconstruct_had_slice`, both Hadamards and suh/svh folded in), `had=False` the rotated basis (`reconstruct_slice`) |
| `exl3_had_r_128(x, pre_scale?, post_scale?, scale)` | vendored `had_r_128` on fp16 rows, into a new tensor |
| `exl3_hgemm(a, b)` | vendored `hgemm_recon`: fp16-accumulate MMA where the one-time rate probe enables it (a 3090 should, 2x), else cuBLAS |
| `exl3_warmup(trellis, suh, svh, mcg, mul1, rows, out_fp32)` | the one-time host work for one weight shape, see below |

Guards run before any CUDA call: trellis 3-D int16 contiguous 16-byte aligned, tile width 16*K
(K 1..8) or 16*K+8 (K 1..3, mul1 only), k and n multiples of 128, mcg/mul1 exclusive; x 2-D
fp16/bf16 contiguous aligned with k columns; suh/svh 1-D fp16 of size k/n, contiguous, aligned;
dequant ranges 128-aligned and inside n; row counts >= 1. The ops are registered for CPU too,
where the guards run and the call then fails with "must be CUDA tensors" (the lcpp_shim pattern).

**Warmup and capture.** The vendored code does host work on first use: `DevCtx` `cudaMalloc`s
its lock buffer and the cuBLAS workspace, `hgemm_f16acc` runs a timed rate probe
(`cudaEventSynchronize`), and each (shape, row bucket) runs an autotune session that times
candidates with syncs and writes `~/.cache/exllamav3/autotune/` (`EXLLAMAV3_TUNE_CACHE`
overrides). None of this may happen inside a CUDA graph capture. `exl3_warmup` does all of it for
one weight shape: `exl3_gemm` at row counts 1, 2, 4, 8, 16 (the vendored choices depend on m only
through these buckets: the GEMV's m == 1 mode, the GEMV limit of 8 rows, and the autotune key
`min(pow2(max(m, 2)), 16)`), plus one `hgemm_recon` at 384 x k x 4096, large enough for the
f16acc kernel's own one-time attribute set (`worthwhile()` wants >= 384 rows and a block per SM).
It records the shape and throws if the current stream is capturing. `exl3_gemm` and `exl3_hgemm` throw if the stream is
capturing and the shape was not warmed. `EXL3LinearMethod.process_weights_after_loading` calls
`exl3_warmup` once per distinct shape when the weights are on CUDA, which is during model load,
before vLLM's memory profiling and graph capture.

`EXL3_GEMM_H_ACC` (fp16 MMA accumulation, sm_86 only, "max observed error ~1 % of output RMS at
k=4096" per the source) is a compile-time vendored choice and stays on.

## Routing (`ops._exl3_op`, pinned by `tests/cpu/test_exl3_routing.py`)

n = activation rows of one product.

| n | route | what |
|---|---|---|
| 1..144 | `exl3_gemm` | vendored kernel; above 16 rows it re-streams the weight once per 16 rows |
| 17..144 | `MULTI_ROW_OP` if set | hook for the multi-row kernel; `None` by default, so these stay on `exl3_gemm` |
| 145..1023 | `recon_hgemm` | Hadamard x with suh, rotated dequant, `exl3_hgemm`, Hadamard y with svh |
| >= 1024 | `recon_had_hgemm` | original-basis dequant, `exl3_hgemm` on raw x |

This is exllamav3's own dispatch (`LinearEXL3.forward`, `reconstruct_hgemm`): threshold 144,
fused reconstruct from 1024 rows, dequant in 32768-column slices. Differences: the fp32 output
above, and the slices are concatenated instead of written into one output. That only matters
above 32768 outputs, i.e. the lm_head at more than 144 rows. Production's argv and the parity
harness never send that (logits are computed for sampled positions only), but a
`prompt_logprobs` request does, with whole prompt chunks: at 2048 rows about 1 GB of fp16
slices plus the concatenated copy, outside vLLM's profiled budget. Fix if it matters: let
`exl3_hgemm` write into a slice of one preallocated output, as exllamav3 does. vLLM's graph capture sizes (up to 48 in production)
all route to `exl3_gemm` (`test_capture_sizes_stay_on_gemm`), so captured graphs hold no dequant
buffers.

The hook: set `ops.MULTI_ROW_OP` to the name of a `torch.ops._C_exl3` op with `exl3_gemm`'s
signature and it takes rows `MULTI_ROW_MIN..MULTI_ROW_MAX` (17..144). That is the MTP verify pass
at c >= 5 with k=3 (4 rows per sequence). First candidate: trellis-serve's Marlin-EXL3 (MIT,
m-tiles to 64 rows, flat cost 1..16 rows per its README). It is accepted only after a bit-exact
decode check against `exl3_dequant`. The range will likely move down to 9 once measured.

## Loading

- Detection: `config.json` has `quantization_config.quant_method: "exl3"`, so vLLM's normal
  detection picks the registered config. `EXL3Config.from_config` parses bits, head_bits,
  mtp_bits and the codebook (`mul1` here; absent means 3INST; `mcg`). Per-tensor K comes from
  each trellis' last dimension at load. `get_min_capability` 80.
- Methods: every `LinearBase` and the `ParallelLMHead` get `EXL3LinearMethod`, except the modules
  the checkpoint stores unquantized (`weights_adapter/qwen3_5.is_unquantized_module`): GDN
  `in_proj_ba` (fp16 `in_proj_a`/`in_proj_b`, 48 outputs, never quantized by exllamav3), a bf16
  vision tower (turboderp's) and the pruned `draft_lm_head`. An EXL3 vision tower (`vision_bits`
  in the config, erlidev's V6) is refused at construction (`NotImplementedError` naming the
  text-only flag) unless every image/video limit is 0, when vLLM does not build the tower. Those get vLLM's unquantized methods. The input embedding stays
  bf16 on the GPU (2.37 GiB; no host-pinned path in phase 0).
- Names: exllamav3 writes the HF names with `.weight` replaced by `.trellis/.suh/.svh/.mul1`.
  vLLM's own mappers do the rest unchanged (`Qwen3_5ForConditionalGeneration`, `Qwen3_5Model`'s
  stacking, `Qwen3_5MTP`'s `mtp.` remap). The stacked name keeps the suffix
  (`.q_proj.trellis` -> `.qkv_proj.trellis`), so `EXL3LinearMethod` registers placeholder params
  `trellis`, `suh`, `svh`, `mul1` with its own `weight_loader`, which stores each checkpoint tensor
  under the shard id the mapper attached (`None`, `"q"/"k"/"v"`, `0/1`, or the GDN `in_proj_qkv`
  tuple `(0, 1, 2)`). No vLLM loader is patched (yeasah's `handles_fused_shards` patch is not
  needed; VERIFIED on the meta device). The codebook tensor's value is checked against the
  multiplier the kernels are compiled with.
- Parts: EXL3 tensors of one fused layer cannot be concatenated (K and suh differ per tensor),
  so `process_weights_after_loading` checks that the parts cover the output shards exactly and
  match the partition sizes, then replaces the placeholders with `exl3_{trellis,suh,svh}_{i}` in
  shard order. `apply` runs one routed `_exl3_linear` per part and concatenates: qkv 3 products,
  gate_up 2, GDN in_proj_qkvz 2: 401 EXL3 products per target pass (exllamav3's `exl3_mgemm`
  can do one launch per fused layer for equal-K parts, a GPU-phase option).
- Tensor parallelism: refused (`NotImplementedError`) in phase 0.
- Vision: production is text-only. Its argv passes `--limit-mm-per-prompt` twice; the last,
  `{"image":0,"video":0}`, wins, with `--enable-mm-embeds` (VERIFIED with vLLM's own parser; phase
  0's dry run had read the first, image 4). With every limit 0, vLLM main's `_mark_tower_model`
  puts a `StageMissingLayer` in place of `visual` and `AutoWeightsLoader` skips `visual.*`, so a
  checkpoint's vision tensors load nowhere and raise nothing (the GGUF plugin's dry run does the
  same). Precomputed image embeddings still work through `--enable-mm-embeds`.
- Meta dry run (VERIFIED, `tools/exl3_meta_dry_run.py [--alt]`, production's text-only argv), both
  checkpoints the same apart from the vision count: main model 2,054 tensors fed (401 EXL3), 0
  params missing, 257 EXL3 layers with 401 parts, 48 in_proj_ba + 48 conv1d + embedding
  unquantized; MTP draft 45 fed (9 EXL3), 0 missing, 6 EXL3 layers with 9 parts,
  `draft_lm_head` [40960, 5120] bf16 unquantized; shared `lm_head.*` and the embedding; vision
  skipped 987 (erlidev) / 333 (turboderp); unmapped 0. K: erlidev {2, 3, 4, 5} main, {4} MTP;
  turboderp {3, 4, 5, 6} and {4, 6}. Peak RSS 1.07 GB. `MM_IMAGES=4` builds the tower:
  turboderp's bf16 tower loads (2,387 fed, 110 vision modules unquantized), erlidev's is refused.
- The MTP draft loads its own copy of `lm_head` (0.89 GiB) and the embedding (2.37 GiB) from the
  checkpoint before vLLM shares the target's and drops them, as with production's W4A16. Check
  the load peak on the GPU.

## Draft head (`tools/exl3_draft_head.py`)

Production's overlay builds `mtp.draft_lm_head` over the ids in `mtp_draft_vocab_ids.pt` (40,960)
and loads `mtp.draft_lm_head.weight` from the checkpoint. An EXL3 lm_head cannot be row-sliced:
the output Hadamard mixes each block of 128 rows. Chosen design: pre-write the rows once. The
tool dequantizes the lm_head to its original basis with `exl3_dequant(had=True)` in 32768-column
slices (skipping slices without ids), keeps the id rows, and writes `mtp_draft_head.safetensors`
(`mtp.draft_lm_head.weight`, bf16, 0.39 GiB), adds it to the safetensors index (vLLM's default
loader reads only indexed files; the index is replaced, never written through a symlink, the
original kept as `.orig`), and saves the sorted ids. At serve time the draft head is a plain
unquantized `ParallelLMHead`: no runtime dequant, no load-order dependency on the target's head.
The rejected alternative (dequantize inside vLLM at load) needs the target's lm_head loaded
first and a custom loader path for no gain. Rows are the fp16 dequant rounded to bf16; they only
change what the drafter proposes. Tested on a synthetic checkpoint with a CPU stand-in for the
dequant op; the real run is GPU phase 1.

    python tools/exl3_draft_head.py MODEL_DIR --ids ~/qwen38-27b-rtx3090/prepare/draft_vocab_ids.json

## Build

    source tools/cuda-env.sh; export PATH=.venv-main/bin:$PATH MAX_JOBS=2
    cd plugin-exl3 && VLLM_EXL3_BUILD=1 python setup.py build_ext --inplace       # from a worktree
    GSQ_VENV=.venv-main GSQ_PLUGIN=plugin-exl3 tools/capped tools/build-plugin.sh  # editable install, main checkout only

Without `VLLM_EXL3_BUILD=1` the package is pure Python (config, loading, routing and most CPU
tests work; the ops are needed at runtime). Measured here (VERIFIED), clean build, CUDA 13.0,
torch 2.13 cu130, `TORCH_CUDA_ARCH_LIST=8.6`, under MemoryMax=6G, CPUQuota=200 %, nice 19,
MAX_JOBS=2: **450 s wall**, 891 s summed over 61 translation units, slowest `reconstruct.cu` 53 s,
the shim 47 s, `exl3_gemv.cu` 43 s, each instance unit 7-21 s. `.so` 42.7 MB, sm_86 only,
compiled and linked, never run. All 60 vendored `.cu` are built; compiling only the K/codebook
instances the checkpoint uses would need a shim-owned kernel map and is not worth it at this
build time.

## Tests (CPU, `GSQ_LIGHT=1 GSQ_VENV=.venv-main tools/capped tools/pytest tests/cpu -k exl3`: 167 passed)

| file | what | count |
|---|---|---|
| `test_exl3_guards.py` | ops registered under `no_gpu` without CUDA init; int8 GEMV off; every guard, in a subprocess | 58 |
| `test_exl3_config.py` | both fetched `config.json`s, codebooks, K from tile width, the entry point, method per layer, the EXL3 vision tower refused unless skipped | 42 |
| `test_exl3_mapping.py` | per checkpoint: format facts on the index/headers (409 text EXL3 modules; K per role), the unquantized split, the meta dry run; the two differ only in vision and bits; erlidev's `quantization_config.json` vs the headers | 8 |
| `test_exl3_routing.py` | the routing table at every boundary, the hook, both dequant paths reproducing x @ W against CPU stand-ins | 25 |
| `test_exl3_linear.py` | loader and parts: q/k/v out of order, GDN tuple shard, gate_up, missing shards/scales, bad shapes, codebook check, copy semantics, TP refusal, apply's concatenation | 16 |
| `test_exl3_draft_head.py` | the draft head tool on a synthetic checkpoint | 7 |
| `test_exl3_parity_skeleton.py` | the parity script's chunk plan | 8 |
| `test_exl3_gpu_cases.py` | the GPU kernel cases (`tests/gpu/exl3_cases.py`, one table per checkpoint, picked by `EXL3_MODEL`'s dir name) against each checkpoint's headers, every text K covered | 3 |

## Untested (everything numeric)

No op has run on a GPU. Not known: whether the vendored kernels are correct under the shim
(argument order, the fp32 output mode on bf16 input, the dequant slices), whether warmup covers
every first-use path (the capture guard should catch a miss), graph capture and replay, logit
parity, speed, VRAM fit, MTP acceptance with the uncalibrated 4-bit MTP layer, fp16 activation
overflow (bf16 values above 65504 become inf), the draft head tool on the real lm_head, and
load time. The name mapping and shapes are checked; the values are not.

## GPU phase plan

Harnesses as for GGUF: jobs go through the rented box's `gpuq` queue (`dash/`); `bench/speed/run.sh`,
`bench/parity/`, `bench/soak.sh`; `tests/gpu` with `GSQ_ALLOW_GPU=1`. Judge on ms/step at c=1/2/4/8.

| phase | work | how | GPU-h |
|---|---|---|---|
| 1 | kernel parity and plumbing | new `tests/gpu/test_exl3_kernels.py`: for each (K, codebook) in the checkpoint and m in 1..17, 48, 145, 1024: `exl3_dequant(had=True)` vs exllamav3's own `reconstruct_had_slice` bit-exact (reference venv), `exl3_gemm` vs x @ dequant in fp64 (tolerance from exllamav3's spread), both dequant routes vs the gemm route; capture/replay of `exl3_gemm` after warmup, and the guard firing without it; compute-sanitizer memcheck/initcheck on each op; then `tools/exl3_draft_head.py` on the real checkpoint and a serve smoke with MTP (new `scripts/serve-exl3.sh` on production's argv) | 3 |
| 2 | logit parity | exllamav3 reference venv; `bench/parity/exl3_logits.py` (finish the skeleton) at `EXL3_HGEMM_F16ACC=0` and `1` to measure exllamav3's own spread; `vllm_logprobs.py` taught an EXL3 model dir; `compare.py` with a relative gate (inside the spread), as Route L; log per-linear max abs activation for the fp16 overflow risk; MTP acceptance (`bench/speed/mtp_acceptance.py`); `tests/gpu/test_fit_200k.py` | 4 |
| 3 | speed ladder | `bench/speed/run.sh` against production W4A16 and GGUF Route L, both on the main argv; per-op ms/step profile; decide int8 GEMV (probably stays off) and `exl3_mgemm` for fused layers | 4 |
| 4 | multi-row kernel | vendor trellis-serve's Marlin-EXL3 as a second source behind `MULTI_ROW_OP`; bit-exact decode vs `exl3_dequant`; route 9..64 rows where it wins; optimization loop on Opus | 6-12 |
| 5 | optional Swift quant | exllamav3 convert at about 3.3-3.5 bpw, plain then SC | 2-6 |
| 6 | soak | `bench/soak.sh` 24 h at c=2 with graphs | 24 |

## GPU phase 1 box, part A (prep, CPU only, 2026-09-30)

Rented RTX 3090 (350 W, 64 cores) while its GPU ran the GGUF 24 h soak. Scripts:
`cloud/results/exl3/box-scripts/` (`00-prep.sh` stages, `lib.sh` + one gpuq job per `0N-*.sh`).
VERIFIED on the box:

- `/workspace/venv-main` = `env/prod-main-freeze.txt` (216 pins) + the `2a0fe5e1e1` overlay (38
  files; all 3184 tracked files checked). 123 distributions are hard links to the 0.27.1 venv's
  identical files (`venv-seed.py`, every file checked against its RECORD sha256), 93 downloaded:
  24 s, +3.2 GB; overlay 59 s. `import vllm` under `CUDA_VISIBLE_DEVICES=""`: 0.30.1rc1.dev285.
- Both plugins editable in that venv, CUDA 13.0.88 pip toolchain (`tools/setup-cuda-toolchain.sh`;
  the box's `/usr/local/cuda` is 12.8), sm86, 8 jobs beside the soak: `_C_gguf` with
  `VLLM_GGUF_BUILD_LCPP=1` 56 s (33.8 MB, 15 ops), `_C_exl3` 127 s (42.7 MB, 5 ops); vendored
  sha256 OK for both; freeze = `env/gsq-main-freeze.txt` + the EXL3 plugin; CUDA never initialized.
  Fix found here: the box has no system cuBLAS headers, so `plugin-exl3/setup.py` now falls back to
  the nvidia wheel's include dir, as `plugin/setup.py` does (ms4 builds from `/usr/include`).
- Parity reference `/workspace/venv-exl3ref`: exllamav3 d3739fd (1.5.3) from
  `/workspace/ref/exllamav3`, `exllamav3_ext` built for sm86 in 340 s (133.7 MB), torch 2.13.0+cu130
  and its libraries hard-linked from venv-main (same files), `EXL3_INT8_GEMV=0` set at interpreter
  start by a `.pth`. No flash-attn needed (`attn_mode` "flash_attn" is exllamav3's own kernel).
- `bench/parity/prompts.lock.json` is keyed by vLLM version: the corpus is the venv's vLLM source,
  so main has its own fingerprint (`20d770b6...`, identical on ms4 and the box). Without this,
  `prompts.py` refused to write on main.
- Not done: the checkpoint download (14.31 GiB for turboderp's; erlidev's is 13.66 GiB). The disk had 8 GB free (151 GB: 56 GB models,
  28 GB the soak's fs KV tier, 18 GB old runs, 16 GB torch compile cache). `00-prep.sh postsoak`
  deletes the soak's leftover KV state once the soak job has finished, with
  `CONFIRM_DELETE_SOAK_KV=1`; then `00-prep.sh model` (about 3 min at 96 MiB/s).

GPU jobs (not run), in order, one gpuq job each:
`run-job.sh NAME` is the gpuq entry point: it records each job's status in
`/workspace/logs/exl3/status/`, skips 03 and 06 when 02 failed and 05 when 04 failed, and lets
every other job run whatever came before (`postsoak` and `model` run first).

| job | what | est. |
|---|---|---|
| `01-kernel-parity.sh` | `tests/gpu/exl3_ref_dump.py` (exllamav3 on 6 checkpoint tensors, K 3/4/5/6), then `tests/gpu/test_exl3_kernels.py` (549 cases: dequant bit-exact, routed gemm vs fp64 inside exllamav3's error, routes, graph replay, fresh-process capture/guard cases), compute-sanitizer memcheck + initcheck on 71 of them | 1-1.5 h |
| `02-draft-head.sh` | `tools/exl3_draft_head.py` on the checkpoint with production's 40,960 ids; rows checked against the EXL3 lm_head | 5 min |
| `03-smoke.sh` | `scripts/serve-exl3.sh` on production's main argv (MTP k=5 schedule, fp8 KV, 200k): chat/reasoning/tool smoke, draft rows, MTP counters, VRAM, KV tokens | 15-20 min |
| `04-parity-ref.sh` | `prompts.py` + `bench/parity/exl3_logits.py`: exllamav3 logits for the 11 sequences (to 120k), then its own spread (`EXL3_HGEMM_F16ACC=0`) | 20-40 min |
| `05-parity-vllm.sh` | `vllm_logprobs.py --model` (EXL3 dir) at bf16 then fp8 KV, `compare.py` against 04 | 1.5-2.5 h |
| `06-ladder.sh` | `bench/speed/run.sh exl3 --start`: c=1/2/4/8 cohorts, prefill 8k/64k/180k, clocks | 1-1.5 h |
| `07-fit.sh` | `tests/gpu/test_fit_200k.py` against a serve-exl3 server at gpu-util 0.94, fp8 KV | 20-30 min |

exllamav3's context on 24 GB (INFERRED, not run): about 10.8 GiB of weights on the GPU (its
embedding stays on the CPU), fp16 KV 64 KiB/token for the 16 attention layers (7.3 GiB at
120k), one recurrent slot (0.15 GiB), 0.25 GB of logits per chunk: about 19-20 GiB at 120k, so
all 11 sequences should fit. `exl3_logits.py` records an out-of-memory sequence and goes on.
