---
status: accepted
---

# Serve EXL3 through a second plugin package with vendored exllamav3 kernels

Accepted 2026-10-04 after GPU phases 1-3 (`cloud/results/exl3/REPORT.md`). Evidence for the choice:
`docs/exl3-feasibility.md`. What changed from the proposal is under "Outcome".

We want EXL3 (exllamav3's trellis format) on the production vLLM with everything Route L keeps working:
MTP, CUDA graphs, fp8 KV, prefix caching, KV offload and priority scheduling. vLLM must stay the server
and unpatched. On main, EXL3 needs no vLLM patch for Qwen3.8. The format is plain safetensors with
`quant_method: exl3` in `config.json`, so vLLM's normal detection and default loader work. The only
patch the community plugin needs for Qwen3.5 (`handles_fused_shards`) goes away with the GGUF plugin's
own tuple-shard `weight_loader`. We propose a second package, `plugin-exl3/vllm_exl3_plugin`, in this
repo. It registers quant method `exl3`, vendors the dense-linear subset of exllamav3 `d3739fd` (94
files, MIT) byte-identical behind one shim, `exl3_shim.cu`, and keeps all adaptation and owned kernels
beside it, the Route L pattern. First checkpoint: `turboderp/Qwen3.8-27B-exl3@3.50bpw`, which already
carries the MTP layer at 4 bits.

## Considered options

- **Depend on the exllamav3 pip package** (yeasah's approach). Builds the whole extension (about 230
  files: attention, GDN, sampling, MoE, pybind bindings) and pulls its Python deps. Version drift sits
  outside our pin. Rejected in favour of vendoring the closure we call.
- **Adopt yeasah/vllm-exl3-plugin** (Apache-2.0). Mature notes and tests, and Qwen3.8-27B + MTP already
  runs. But it needs a vLLM fork (v0.28/0.29), has no sm_86 data, and uses exllamav3's kernels as-is,
  with the same multi-row weakness. Keep as a reference and a likely collaborator. Not a base.
- **Adopt vcruz305/vllm-exl3.** AGPL-3.0-only, MoE and DGX-Spark first, fork runtimes. Rejected.
- **Add EXL3 as a second quant method inside `vllm_gguf_plugin`.** Couples two builds and muddies the
  GGUF fork's upstream diff. Rejected.
- **Transcode EXL3 to Marlin W4A16 or GGUF.** Lossy or larger. It throws away the reason to use EXL3.
  Rejected.

## Consequences

- vLLM gets no monkeypatches from this package. The GGUF plugin's EngineArgs and config-parser patches
  have no EXL3 counterpart.
- Row routing: 1..16 rows go to vendored `exl3_gemm` (GEMV where exllamav3's heuristic picks it),
  more than 144 rows to reconstruct + fp16-accumulate hgemm. 17..144 rows start on vendored `exl3_gemm`,
  which re-streams the weight per 16 rows. That is the MTP verify regime at c >= 4. A multi-row kernel
  is expected work, with trellis-serve's MIT Marlin-EXL3 as the first candidate and an owned kernel if
  it loses. It is accepted only on bit-exact decode against the vendored `reconstruct`.
- Autotune, DevCtx allocation and the f16acc rate probe are primed at load. The shim refuses to run
  them inside a CUDA graph capture.
- Activations cross into fp16 at the op boundary (bf16 residual kept). Overflow above 65504 is a
  watched risk, logged in parity runs.
- The vocab-truncated MTP draft head can't be sliced from a quantized EXL3 head, because the output
  Hadamard mixes 128 rows. A tool writes a bf16 `mtp.draft_lm_head` (0.39 GiB) beside the checkpoint.
- The parity reference is exllamav3 running the same checkpoint (`bench/parity/exl3_logits.py`), with
  a relative gate as in Route L. The speed references are prod W4A16 and GGUF Route L on the main argv,
  judged on ms/step.
- Estimated effort: 9-13 engineer days and 17-29 GPU-h, plus a 24 h soak.

## Outcome (2026-10-04)

- Checkpoint: erlidev's Swift-1.5 EXL3 quant (`SC_3.50bpw_H4_V6`), so Swift's MTP layer and acceptance
  come with it; no own quantization was needed. turboderp's base 3.50bpw stays the A/B reference.
- Routing: every K3/K4/K5 linear runs on `exl3_gemm_mr` from 1 to 384 rows (trellis-serve's Marlin-EXL3,
  vendored as the second source, plus the owned `exl3_marlin_h16.patch`: fp16 accumulation at 9..48
  rows); vendored `exl3_gemm` keeps K2, and above 384 rows the dequant + fp16 GEMM route. Parity is
  judged against exllamav3's own `exl3_gemm` error rather than bit-exact decode, which a different
  accumulation order cannot give; dequant stays bit-exact.
- The token embedding is page-locked in host memory by default (`EXL3_EMBED_HOST=1`, 2.37 GiB of VRAM;
  no ms/step change measured); that is what makes 200,000 tokens fit. The MTP draft head runs in fp8
  (`EXL3_DRAFT_FP8=1`).
- Still open: the 12 h soak and the other tiers (REPORT section 6); MTP must run at a fixed k on the
  current overlay (#50021's conv1d bound corrupts output when the schedule lowers k, for GSQ as well).

## Questions that were open for Garrett (answered by the outcome above)

- Serve turboderp's 3.50bpw (Qwen3.8 base, 13.4 GiB with bf16 embed on GPU) or quantize Swift ourselves
  (about 2-6 GPU-h) to keep Swift's MTP acceptance?
- Is a host-pinned embedding (saves 2.37 GiB of VRAM) acceptable in prod, or keep it in VRAM / int8?
