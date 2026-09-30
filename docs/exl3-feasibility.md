# EXL3 on vLLM via a plugin: feasibility survey

2026-09-30. Read-only research: no GPU, no builds, nothing run against prod. Labels: VERIFIED (read in
source, file headers or HF metadata by me), DOCUMENTED (stated by a third party, not checked), INFERRED
(my reasoning, not measured). Target: one RTX 3090 (sm_86), Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`,
48 GDN + 16 full-attention layers, head_dim 256, 1 MTP layer), vLLM main 0.30.1rc1 + `qwen38/main` overlay,
MTP, fp8 KV, prefix caching, KV offload, CUDA graphs (PIECEWISE on main, capture up to 48), 200k context,
weights around 12 GB.

Verdict: go. Loading and serving EXL3 needs no vLLM patch on our target, the kernels vendor cleanly (MIT,
about 94 files), and ready-made Qwen3.8-27B quants exist, MTP layer included. The catch is the multi-row
decode regime. exllamav3's dense GEMM re-streams the whole weight once per 16 activation rows, which is
exactly the MTP verify pass at c>=4. Plan on an owned or second-source kernel for 9..64 rows, as Route L
needed for GGUF.

## 1. exllamav3 today

Pinned: `turboderp-org/exllamav3` master `d3739fd393337b1ff4d6c2a342b12f0c87a9592f` (v1.5.3, 2026-09-27).
MIT license (VERIFIED, GitHub API). A `dev` branch moves daily; pin master tags only.

**Format.** VERIFIED from `exllamav3/modules/quant/exl3.py` and the safetensors headers of
`turboderp/Qwen3.8-27B-exl3@3.50bpw`. Each quantized linear is:

| tensor | dtype | shape |
|---|---|---|
| `<key>.trellis` | int16 | `[in/16, out/16, 16*K]`, 16*K+8 for half-integer K (mul1 only) |
| `<key>.suh` | fp16 | `[in]`, input sign/scale folded with the input Hadamard |
| `<key>.svh` | fp16 | `[out]`, output sign/scale folded with the output Hadamard |
| `<key>.mul1` or `.mcg` | int32 0-dim | codebook flag (mul1 is the default codebook) |
| `<key>.bias` | fp16 | optional |

K (bits) is per tensor. The 3.50bpw quant is a mix of K=3 (80 tensors) and K=4 (145) with a K=6 `lm_head`.
Hadamard blocks are 128 wide on both sides, so in/out must be multiples of 128 (the converter pads).
Model-level metadata is `quantization_config` in `config.json` (`quant_method: exl3`, bits, head_bits,
mtp_bits, codebook) plus a per-tensor `quantization_config.json` (629 KB).

**Kernel dispatch** (VERIFIED, `LinearEXL3.forward`, `libtorch/linear.cpp`, `quant/exl3_gemm.cu`):

- rows <= 144 (`AUTO_RECONSTRUCT_THRESHOLD`): `exl3_gemm`, a cooperative-launch kernel
  (`cudaLaunchCooperativeKernel`, `grid.sync`) that decodes the trellis in registers and runs fp16 MMA.
  Input Hadamard is done in-kernel into an `A_had` scratch and the output Hadamard in the epilogue.
  - m <= 8 and K = 4 (mul1): a QTIP-style GEMV path (`exl3_gemv.cu`) when a shape heuristic tuned on a
    3090 says it wins (15-60% at n <= 4096 per the source comment). Other K fall through.
  - Optional int8-activation GEMV (`EXL3_INT8_GEMV=1`, m <= 2, not graph-capturable). Experimental.
  - **m > 16: the kernel loops `while (size_m_ > 0)` in 16-row chunks** (`exl3_gemm_kernel.cuh`), and
    each chunk is a full pass over the weight. 17-32 rows cost about 2x one row and 33-48 rows about 3x.
    This is the weak regime for us (section 5).
  - First call per shape bucket (m bucketed to 1, 2, 4, 8, 16+) runs an autotuner that times candidate
    tile shapes with `cudaEventSynchronize`/`cudaStreamSynchronize` and caches to
    `~/.cache/exllamav3/autotune/` (`EXLLAMAV3_TUNE_CACHE` overrides).
- rows > 144: dequantize to a dense fp16 weight (`reconstruct`, or `reconstruct_had_slice` with both
  Hadamards folded in at rows >= 1024), then `hgemm_recon`. On GeForce parts that is an owned
  fp16-accumulate MMA kernel (`hgemm_f16acc.cu`, 32-term fp16 partials flushed to fp32), enabled by a
  one-time rate probe that finds fp16-acc HMMA at least 1.5x the fp32-acc rate. That holds on a 3090 (2x).
  Otherwise cuBLAS with fp32 compute.

**Numerics** (VERIFIED). Activations must be fp16 (`TORCH_CHECK_DTYPE(A, kHalf)`), output fp16 or fp32.
On sm_86 only, the dense GEMM accumulates MMA in fp16 and folds into fp32 once per k-slice
(`EXL3_GEMM_H_ACC`). The source comment gives "max observed error ~1% of output RMS at k=4096" for a 14%
decode gain. Accumulation runs over K (5120 to 17408), not over context, so 200k context does not make
this worse. The real risk is the bf16 to fp16 cast of activations (overflow above 65504). yeasah measured
max |x| = 3.7e3 on a small model (DOCUMENTED). Not measured on Qwen3.8.

**CUDA archs.** Builds for the local GPU's capability via `TORCH_CUDA_ARCH_LIST` (`util/arch_list.py`),
sm_75 and up (`arch.cuh`). sm_86 has its own paths: fp16 accumulate, an inline-PTX codebook decode, and
Ampere branches in the kernel map and GEMV heuristics (VERIFIED). Every yeasah number is from sm_120, so
our 3090 would be the first vLLM sm_86 measurement of these paths.

**Standalone use.** The full extension (`exllamav3_ext`, about 230 files) includes attention, GDN,
sampling and MoE, plus `bindings.cpp` into the Python package. The dense-linear subset is separable. I
computed its include closure from `exl3_gemm.cu`, `exl3_gemv*.cu`, `exl3_kernel_map.cu`,
`exl3_devctx.cu`, `coop_autotune.cu`, `hadamard.cu`, `reconstruct.cu`, `frac.cu`, `hgemm*.cu`, `graph.cu`
and the comp units: **94 files, 390 KB, 57 of them template-instance units** (VERIFIED). Only
`graph.cu/.cuh` touch pybind (exllamav3's own graph recorder; we pass `graph = nullptr`). No dependency on
exllamav3's Python model code. Host-side state: `DevCtx` does `cudaMalloc` for workspace/locks on first
use, the autotuner syncs on a cache miss, and the f16acc probe syncs once (VERIFIED). None of these may
happen inside a capture.

**Published 3090 numbers** (none from vLLM):

| engine | quant | spec | c=1 decode | source |
|---|---|---|---|---|
| exllamav3 community branch 355c6ee | 4.00bpw | none | 42.8 tok/s | [r0b0tlab/qwen38-exl3-dflash2](https://github.com/r0b0tlab/qwen38-exl3-dflash2), DOCUMENTED |
| same | 4.00bpw | MTP | 116.3 tok/s (GSM8K, accept len 4.12) | same |
| same | 4.00bpw | DFlash2 | 162.9 tok/s; 25.3 tok/s at 150k depth, prefill 594 tok/s there | same |
| SGLang 0.5.20 + sglang-exl3 (Marlin-EXL3 kernels) | 3.00bpw | MTP k=3 | 96.2-98.8 prose, 141-143 code | [0xSero/trellis-serve](https://github.com/0xSero/trellis-serve), [local-ai-registry#83](https://github.com/0xSero/local-ai-registry/pull/83), DOCUMENTED |
| same, AWQ-INT4 | 4 bit | none | 45 tok/s | trellis-serve cuda/README, DOCUMENTED |

For scale, our GGUF path on vLLM does 110 tok/s at c=1 with MTP and 19.6 ms/step without MTP (README).
The prompts differ, so these are not head-to-head. No batched (c>1) EXL3 numbers exist for a dense 27B
on a 3090.

## 2. Quantizer and existing quants for Qwen3.8

- exllamav3 has `Qwen3_5ForConditionalGeneration`/`Qwen3_5ForCausalLM` (dense and MoE, with vision) in
  `architecture/qwen3_5.py`, GDN layers via `modules/gated_delta_net.py`, and an MTP model in
  `architecture/qwen3_5_mtp.py` (VERIFIED). `Qwen3NextModel` is there too. Qwen3.8 uses the Qwen3.5
  class name (VERIFIED from our local `config.json`).
- GDN: `in_proj_qkv`, `in_proj_z`, `out_proj` are quantized. `in_proj_a`/`in_proj_b` stay fp16 (`qmap =
  None`, 48 outputs, not a multiple of 128). VERIFIED in the source and the checkpoint (0.022 GiB each).
- MTP: `--mtp_bits` defaults to 4, uncalibrated. The MTP layer, `mtp.fc` included, is in the
  checkpoint as EXL3 (VERIFIED: `mtp.layers.0.*.trellis`, `mtp_bits: 4`). No bf16 draft needed.
- Embedding stays bf16 (2.37 GiB). The vision tower is bf16 (0.86 GiB) or 6-bit in "V" variants.

Existing quants (VERIFIED via the HF API):

| repo@revision | bits | safetensors | text-only GiB incl. bf16 embed | without embed |
|---|---|---|---|---|
| `turboderp/Qwen3.8-27B-exl3@3.50bpw` (8351c54e) | 3.5 avg (K3/K4), head 6, mtp 4 | 15.34 GB | 13.43 | 11.06 |
| `turboderp/Qwen3.8-27B-exl3@SC_3.00bpw_H4` (86b95530) | 3.0 self-calibrated, head 4 | 13.45 GB | 11.67 | 9.30 |
| `turboderp/Qwen3.8-27B-exl3@3.00bpw` | 3.0, head 6 | | about 11.9 | about 9.5 |
| `GestaltLabs/Qwen3.8-27B-EXL3-11.5GB` | 2.87 decoder, head 4 | 11.47 GB | | |
| `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` | 3.5 | 15.34 GB | | |
| `Honkware/Swift-Qwen3.8-27b-exl3-4.0bpw` | 4.0 (Swift finetune) | 16.35 GB | | |

turboderp's repo also has 2.0-6.0 plain and 1.4-6.0 SC branches (25 in total), with KLD/PPL-vs-VRAM
plots. Nobody has published a Swift quant at 3-3.5 bpw. For Swift (our GGUF's finetune, MTP acceptance
0.650) we would quantize ourselves. Cost: exllamav3 streams layer by layer, "a couple of minutes for
smaller models, up to a few hours for 70B+" on one consumer GPU (README, DOCUMENTED). For 27B on a 3090,
INFERRED 1.5-3 GPU-h for a plain quant, more for an SC recipe (`sc_trace.py` + `sc_optimize.py`). It
fits in 24 GB. Disk: the bf16 source (54 GB) plus a work dir the size of the output. Default
calibration is 250 rows x 2048 tokens from a bundled c4/code mix.

## 3. The community plugins

**yeasah/vllm-exl3-plugin** (Apache-2.0, created 2026-08-03, last push 2026-09-28, 1 star; HEAD
`1a73cefe66` "bump submodule to 0.29"). A well-documented single-author project: 66 CPU-runnable tests, a
`bench/` gate that pins token ids, logprobs and resident bytes per model (`qwen3.8-27B-3.0bpw-blockq-MTP-fp8`
among them), and notes on graphs, TP, MoE and embeddings. Supports dense, MoE, TP 1/2/4/8, quantized
lm_head, block-quantized embeddings, MTP. Kernels come from pip-installing exllamav3 from a submodule
(their fork, which carries an sm_90+ barrier fix). Nothing is vendored. All measurements are on an RTX
5070 Ti (sm_120).

Why it patches vLLM (fork branch `appliance/v0.28.0`, VERIFIED from `patches.md` and commit
`yeasah/vllm@8694dbfff`):

| patch | what | needed for us? |
|---|---|---|
| `handles_fused_shards` (`linear.py` `weight_loader_v2`, `parameter.py`) | `MergedColumnParallelLinear`/`QKVParallelLinear.weight_loader_v2` check `type(param) in (RowvLLMParameter, BasevLLMParameter)` by exact type, so a subclass param holding a fused tensor (`in_proj_qkv` -> `in_proj_qkvz` shards `(0,1,2)`) falls into the generic narrow path. "Qwen3.5 will not load without it." | **No.** Our GGUF plugin already avoids it by installing its own `weight_loader` in `create_weights` that accepts tuple shard ids and stores per-shard tensors (`params.py`, `_gguf_weight_type_loader_v2`). That works on main (compat doc). VERIFIED for GGUF, INFERRED for EXL3 |
| `ReplicatedLinear` weight_loader_v2 | Transformers-backend models only | No (native Qwen3.5) |
| embed `quant_config` default | 86 model files do not pass `quant_config` to `VocabParallelEmbedding` | No. `Qwen3_5Model` passes it on main (VERIFIED, `qwen3_5.py`), and our overlay carries `9e334ccca1` for the MTP embed |
| logit softcap alias | MuseGlimmer | No |
| TurboQuant KV commits | their KV experiments | No |

yeasah also confirms, and the source agrees, that `quant_method: exl3` in `config.json` is picked up by
vLLM's normal detection, so the GGUF plugin's EngineArgs/config-parser/loader monkeypatches have no EXL3
counterpart. That makes an EXL3 plugin smaller than the GGUF one.

Collaborator? Yes, likely. Same license, same design instincts (measured notes, dependency-bump gate),
Qwen3.8-27B + MTP already exercised. Differences: they depend on the whole exllamav3 package and a vLLM
fork, and have no Ampere data. Our sm_86 numbers and the patch-free fused-shard loader would be useful
to them. I would not fork their repo. Our harnesses and prod stack are different. Offer findings and
possibly the multi-row kernel upstream.

**vcruz305/vllm-exl3** (AGPL-3.0-only since recently, Apache-2.0 before; 25 stars, 12 forks; HEAD
`223e246ff8`, v0.5.0 dev). Routed-MoE first, DGX Spark GB10 first (GLM-5.3-Flash, DeepSeek-V4.1,
Qwen3.8-Flash-Next). Dense EXL3 is a side path ("declared dense EXL3 tensors"). Needs "vLLM fork
runtimes" per recipe. Its checked-in patches (`tools/patch_vllm_qwen4_exp/`: PLE embedding, MTP lm_head,
vision split) are Qwen4Exp-specific. The other models depend on model classes that exist only in fork
runtimes. Kernels: exllamav3 (their fork, for `exl3_moe_mixedk`) plus own csrc (fat GEMM, MoE). CI is
CPU-only without vLLM installed, per their compat doc. Not relevant to a dense 27B on a 3090, and AGPL
code cannot go into an Apache repo. Read-only reference, no collaboration.

**Third find: 0xSero/trellis-serve** (MIT, created 2026-09-28, one commit `1ace59c4b4`, 2 stars). SGLang
plugin for the 3090 with **Marlin-template EXL3 kernels**: vLLM's Marlin skeleton with the int4 dequant
swapped for trellis decode, K = 3/4/5/6, mul1 and mcg, load-time repack (K=4 a word permutation, K=6 a
lossless bit re-layout, K=3/5 as stored). m-tiles run to 64 rows (`thread_m_blocks` up to 4). One launch
set covers a fused group with per-shard `suh`. bf16 in/out, with the cast inside the Hadamard launches.
"Flat cost from 1 to 16 rows." Decoded weights are claimed bit-identical to `exllamav3_ext.reconstruct`,
with a four-level lossless check (DOCUMENTED). Also: embedding in pinned host memory (`SGLANG_EXL3_EMBED_HOST`),
and an MTP head restricted to 32k hot tokens. Very new and untested outside its author's recipes, but
it is the only public kernel aimed at our exact weak regime (9..64 rows on sm_86), and MIT.

## 4. What a vLLM main quantization plugin must implement

VERIFIED against `venv-main` vLLM sources and our GGUF plugin, which already runs on main
(`cloud/results/vllm-main-compat.md`).

- `register_quantization_config("exl3")` on a `QuantizationConfig` subclass: `get_name`,
  `get_supported_act_dtypes` (fp16, bf16; cast at the op boundary), `get_min_capability` 80,
  `get_config_filenames`, `from_config` (reads `config.json` `quantization_config`),
  `get_quant_method(layer, prefix)`. It sits behind main's `resolve_quant_method()`, which returns
  `get_quant_method()` unchanged when online quant is off.
- `LinearBase` -> `EXL3LinearMethod(LinearMethodBase)`. `create_weights` registers placeholder params
  (`trellis`, `suh`, `svh`, codebook flag) with **our own `weight_loader`** that takes int, str or
  tuple shard ids and keeps one entry per checkpoint tensor (the GGUF `data_container` pattern).
  `process_weights_after_loading` builds runs, trims padding and allocates scratch. `apply` routes by row
  count to custom ops registered with `direct_register_custom_op` and fake impls, for torch.compile.
- Fused layers: `qkv_proj` (q, k, v), `gate_up_proj`, GDN `in_proj_qkvz` (the checkpoint's `in_proj_qkv`
  covers shards 0-2, `in_proj_z` is shard 3, per `Qwen3_5Model.hf_to_vllm_mapper`), `in_proj_ba`. EXL3
  tensors of one fused layer cannot be concatenated. K differs per tensor, and `suh` differs even at
  equal K because the quantizer folds per-tensor RMS into it (yeasah, DOCUMENTED; consistent with the
  headers). So each fused layer is N GEMMs, or one grouped launch with per-shard `suh` (trellis-serve).
  `in_proj_ba` is fp16 in the file and goes to an unquantized method. The GGUF plugin's batched-gemv
  `GGUFUnquantizedLinearMethod` is the model.
- `ParallelLMHead` -> an EXL3 lm_head method (K=6 trellis, 248320 outputs). `VocabParallelEmbedding`
  -> unquantized bf16 by default. Candidates to save 2.37 GiB: pinned-host table + UVA gather, or a
  load-time int8 per-row transcode (owned).
- Weight loading: standard safetensors with the default loader. No custom model loader or config parser.
  `mul1`/`mcg` are 0-dim int32 tensors, and vLLM treats a registered-but-unloaded param as fatal, so
  register the flag param the codebook in `quantization_config` implies (yeasah, DOCUMENTED). The `visual.*`
  tensors must be skipped when the tower is not built; check this in the meta dry run.
- MTP: `qwen3_5_mtp.py` passes `quant_config` to `fc` (except for modelopt_fp4) and to the MTP decoder
  layer (VERIFIED), so the 4-bit MTP layer loads through the same method. The overlay's vocab-truncated
  `draft_lm_head` (40,960 rows, `mtp_draft_vocab_ids.pt`) is the one hard spot. The output Hadamard
  mixes blocks of 128 outputs, so rows of an EXL3 head cannot be gathered in quantized form. Plan: a
  one-off tool reconstructs `lm_head` to fp16, gathers the 40,960 rows and writes
  `mtp.draft_lm_head.weight` bf16 (0.39 GiB) beside the checkpoint. Loaded unquantized. The GGUF path
  gets the same head by a lossless row slice. The full 6-bit head is the fallback (0.89 GiB read per
  draft step).
- The main-branch MTP draft-config break found for GGUF (`create_speculative_config` pointing the draft
  at `.gguf`) does not apply. An EXL3 model is a normal HF directory.

## 5. Design proposal

**Layout.** A second package in the same repo, not a second method in `vllm_gguf_plugin`. The GGUF
plugin stays a clean fork of `vllm-project/vllm-gguf-plugin` for upstreaming, and the builds stay
independent.

```
plugin-exl3/
  setup.py                         VLLM_EXL3_BUILD=1 builds _C_exl3 (sm_86 only)
  vllm_exl3_plugin/
    plugin.py                      register(): quant config only, no monkeypatches
    quantization/{config,linear,lm_head,embedding,unquantized}.py
    format.py                      checkpoint reader: K per tensor, codebook, padding (torch-free)
    ops.py                         torch.ops._C_exl3 wrappers + fake impls
    csrc/exllamav3/                vendored, byte-identical, VENDORED.md (commit, sha256)
    csrc/exl3_shim.cu              all adaptation: ops, scratch, stream, guards, warmup
tools/exl3_draft_head.py           builds mtp.draft_lm_head from the EXL3 lm_head
bench/parity/exl3_logits.py        exllamav3-native logits dump (reference)
tests/cpu/test_exl3_*.py, tests/gpu/test_exl3_*.py
```

**Vendored files** (exllamav3 `d3739fd`, unmodified): `quant/{exl3_gemm,exl3_gemv,exl3_gemv_int8,
exl3_kernel_map,exl3_devctx,coop_autotune,hadamard,reconstruct,frac}.cu` with their `.cuh`,
`quant/{exl3_gemm_inner,exl3_gemm_kernel,exl3_gemv_kernel,exl3_gemv_int8_kernel,exl3_dq,codebook,
bits_k,hadamard_inner,quantize}.cuh`, `quant/comp_units/exl3_comp_unit_*`, `exl3_gemv_*_inst*`,
`hgemm.cu`, `hgemm_f16acc.cu`, `graph.cu/.cuh`, `ptx.cuh`, `arch.cuh`, `compat.cuh`, `util.cuh`,
`util.h`, `cuda_drv.h`. 94 files. Build time is unknown. The 57 instance units are the bulk. Compile
only the K/codebook instances the checkpoint uses (K 3/4/6, mul1) via the shim's own instance list if
the full set is too slow under the caps.

**Owned pieces.**

1. Loader: per-tensor runs for fused layers, padding trim, codebook flags, skip lists (`in_proj_a/b`,
   norms, conv1d stay unquantized).
2. Shim ops: `exl3_mm(x, trellis, suh, svh, K, cb, out)` with the bf16<->fp16 casts at the boundary,
   caller-owned `A_had` scratch sized at load (fixed addresses for graphs), fp16 or fp32 C.
   `exl3_reconstruct` + `hgemm_recon` for prefill, tiled to about 32 MiB slices.
3. Routing table, pinned by a CPU test like `test_lcpp_routing.py`. Rows 1..16: vendored `exl3_gemm`
   (GEMV where its heuristic picks it). Rows 17..144: vendored at first, then the multi-row kernel.
   Rows > 144: reconstruct + f16acc hgemm. The threshold must stay above the largest capture size (48),
   else reconstruct buffers get pinned in the graph pool (yeasah, DOCUMENTED).
4. Warmup and guards: prime `DevCtx`, the f16acc probe and the autotune cache for every (shape, m-bucket)
   in `process_weights_after_loading`, with `EXLLAMAV3_TUNE_CACHE` under the run dir. The shim throws if
   an autotune miss or first-use `cudaMalloc` happens while `cudaStreamIsCapturing`. Input guards as in
   Route L (dtype, contiguity, K % 128, alignment).
5. Multi-row decode kernel (9..64 rows) for the MTP verify pass. First candidate: vendor trellis-serve's
   Marlin-EXL3 (MIT, derived from vLLM Marlin, Apache-2.0) as a second vendored source. Accept it only
   after a level-1 bit-exact decode check against the vendored `reconstruct`. Owned kernel if it loses.
6. Embedding: bf16 first. Then host-pinned or int8, whichever measures better.
7. Draft head tool (section 4).

**Parity reference.** exllamav3 itself running the same checkpoint (its `Qwen3_5` architecture) on the
same token ids, dumping logits: `bench/parity/exl3_logits.py`, fed by the existing `prompts.lock.json`
and scored by `compare.py`. It needs its own venv with the full exllamav3 build. Gates: kernel level,
decoded weights bit-exact against `reconstruct` and layer output against an fp64 reference, as tight as
exllamav3's own kernel. Logit level: mean KLD and top-1 with a relative gate, as in Route L. vLLM runs
attention, GDN and the residual in bf16 while exllamav3 uses fp16, so bit-level logit parity is not
expected. First measure exllamav3's own spread (fp16 vs fp32-acc via `EXL3_*` knobs), then gate on
"inside that spread".

**Speed references.** Prod W4A16 on main (re-baselined per the compat doc, since k=5 schedule, 16 seqs
and PIECEWISE changed) and the GGUF Route L path on main. Judge on ms/step at c=1/2/4/8 (4/8/16/32
rows at k=3, 6/12/24 at prod's k=5), plus prefill at 8k/64k/180k and KV tokens at 200k.

**Rough expectation** (INFERRED, not measured). At 3.5bpw the per-pass weight bytes are 10.8 GiB
(decoder 9.92 + head 0.89) against the GGUF's ~12.4. The byte floor is about 12.4 ms per pass at
936 GB/s. At 1..16 rows, exllamav3's kernel should land near the GGUF path. At 17..48 rows it pays 2-3
weight passes and loses clearly, which is where GGUF was already weakest. Prefill: fp16-accumulate HMMA
is 2x the fp32-acc rate that Marlin uses, and reconstruct adds about 10% per 2048-row chunk, so prefill
should match or beat W4A16. Fit: 11.06 GiB + 0.39 draft head at 3.5bpw with a host-side embedding, or
13.43 + 0.39 with bf16 embed on GPU. With the embed on GPU, the 3.5bpw quant is about 1 GiB over the
GGUF's 12.45 GiB and gives up roughly 20k tokens of KV.

**Risks.**

- Multi-row decode (above). Without item 5, c=4/8 with MTP is likely slower than both references.
- fp16 activations: a bf16 value above 65504 becomes inf. Log per-linear max |x| during parity. If it
  bites, use a per-call scale in the shim.
- fp16-accumulate numerics on sm_86 (1% of RMS per the source). Measurable, and switchable only by
  rebuilding with `EXL3_GEMM_H_ACC` 0. That would need a vendored edit or a `-D` override (the macro is
  keyed on `__CUDA_ARCH__`, so a clean override needs one guarded line).
- CUDA graphs: cooperative launches captured fine under vLLM PIECEWISE and FULL on sm_120 (yeasah,
  DOCUMENTED). On sm_86 the GEMV and int8 paths differ, so it has to be re-verified. The int8 GEMV must
  stay off.
- Host syncs: autotune, f16acc probe and `DevCtx` `cudaMalloc` all run once. Prime them at load and
  guard against capture. `graph.cu` calls `cudaDeviceSynchronize`, but only in exllamav3's own capture,
  which we never call.
- A shared `locks` buffer per device: fine with one model stream. Watch the KV offload streams (they do
  no GEMMs, INFERRED).
- MTP acceptance: the checkpoint's MTP layer is uncalibrated 4-bit. Acceptance could land below the
  GGUF's 0.650 (Swift). Measure it early. It decides tok/s more than any kernel.
- Quality at equal bytes vs GSQ-RCO IQ3_S: not known. turboderp publishes KLD for his quants. We should
  put both on one KLD-vs-bf16 scale before deciding which format prod serves.

**Phased plan** (engineer days are INFERRED, GPU on a rented 3090 through `gpuq`):

| phase | work | days | GPU-h |
|---|---|---|---|
| 0 | CPU: package skeleton, format reader, config/linear/lm_head methods, weight loader with tuple shards, vendoring + VENDORED.md, shim build under caps, guard and routing tests, `meta_dry_run` against the 3.50bpw header (incl. MTP and draft head names) | 2-3 | 0 |
| 1 | kernel parity (bit-exact decode, fp64 layer check) per K and row count, graph capture/replay, compute-sanitizer, serve smoke with MTP | 1 | 3 |
| 2 | exllamav3 reference venv + `exl3_logits.py`, logit parity, MTP acceptance, `test_fit_200k` | 1-2 | 4 |
| 3 | speed ladder on the main argv vs W4A16 and GGUF (both re-baselined), per-op ms/step profile | 1 | 4 |
| 4 | multi-row kernel: vendor Marlin-EXL3, parity, route 9..64 rows; optimization loop on Opus | 3-5 | 6-12 |
| 5 | optional: Swift quant at about 3.3-3.5 bpw (plain, then SC) | 0.5 | 2-6 |
| 6 | 24 h soak at c=2 with graphs, dashboard | 0.5 | 24 |

Total: about 9-13 engineer days, 17-29 GPU-h before the soak, plus 24 h of soak.

## Sources

- exllamav3 at `d3739fd`: `modules/quant/exl3.py`, `exllamav3_ext/libtorch/linear.cpp`,
  `quant/exl3_gemm.cu`, `quant/exl3_gemm_inner.cuh`, `quant/exl3_gemm_kernel.cuh`, `quant/exl3_gemv.cu`,
  `quant/coop_autotune.cu`, `hgemm_f16acc.cu`, `arch.cuh`, `util/arch_list.py`, `architecture/qwen3_5*.py`,
  `modules/gated_delta_net.py`, `conversion/convert_model.py`, `doc/convert.md`.
- HF: `turboderp/Qwen3.8-27B-exl3` (refs, `config.json`, safetensors headers read by HTTP range).
- https://github.com/yeasah/vllm-exl3-plugin (README, `patches.md`, `docs/kernels.md`,
  `docs/format-and-loading.md`, `docs/exllamav3-arch.md`); https://github.com/yeasah/vllm/commit/8694dbfff
- https://github.com/vcruz305/vllm-exl3 (README, `docs/VLLM_COMPATIBILITY.md`, LICENSE)
- https://github.com/0xSero/trellis-serve (README, `cuda/README.md`, `kernels/marlin.py`, `csrc/exl3_marlin.cu`)
- https://github.com/0xSero/local-ai-registry/pull/83, https://github.com/r0b0tlab/qwen38-exl3-dflash2
- vLLM main in `~/qwen38-27b-rtx3090/venv-main`: `models/qwen3_5.py`, `models/qwen3_5_mtp.py`.
- Prior: vLLM issues #19896 and #3203 (EXL2/EXL3 requests, stale-closed, no maintainer objection).
