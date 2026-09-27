---
status: proposed
---

# Vendor llama.cpp b11211 kernels behind a shim (Route L)

The Swift GGUF is IQ-dominated (about 82% of weight bytes in IQ types), and the plugin's kernels are llama.cpp b2899: IQ types have no MMQ, so above the MMVQ threshold every forward dequantizes the whole weight to bf16 and runs cuBLAS, and MMVQ re-reads weights once per row. On the RTX 3090, int8 tensor-core MMA peaks at about 4× the bf16 rate, so any W×A16 route caps prefill well below an int8 MMQ. vLLM must stay the server (priority scheduling, KV tiers, the production argv). We decided to vendor llama.cpp b11211's MMVQ, MMQ and q8_1 quantize files unmodified into the plugin, adapt them with one owned shim file, and gate the new path behind a default-off flag, because it is the only source that covers every quant type in the file with tensor-core MMQ and uses the same numerics as the parity reference.

Status is proposed until Route L passes the Phase B kernel tests, parity, speed, fit and soak on the cloud box. It is compiled and CPU-checked only.

## Considered Options

- **Route A**: efschu's batched MMVQ plus vLLM PR #36226 IQ MMQ on the plugin's b2899 code. Small diff, but #36226 was closed unmerged, never compiled against this tree, and uses 4×32 dp4a tiles with no tensor cores, which are unlikely to beat dequant+cuBLAS at prefill. Kept compile-ready on branch `route-a` as a fallback.
- **Lossless transcode of IQ3_S/IQ3_XXS/IQ2_XXS into IQ4_XS containers**: fits (+1.77 GB) and is bit-exact, but published 3090 numbers (llama.cpp PR #8215) show IQ4_XS MMVQ slower per token than IQ3_S, so the extra bytes are expected to lose. Demoted.
- **Marlin, Humming, FLUTE/BitBLAS, EXL3/QTIP/QuIP#/AQLM**: all A16 (prefill capped by the bf16 rate); Marlin in vLLM main cannot express the IQ value sets losslessly, Humming's lossless encoding needs about 21 GB, FLUTE/BitBLAS are unmaintained, and the codebook formats would mean re-quantizing. Dead ends.
- **Maxwell-Lyu's `codex-ggml-source-bridge`**: the same idea, but WIP, restructured daily, and carrying the whole llama.cpp tree including MoE and FP4. Used as a template (its q8_1 scratch guard tail is required to avoid an IMA), not adopted.

## Consequences

- About 20k vendored lines, zero owned kernel code: all adaptation (device info, stream, scratch pool, ops, guards) lives in the shim. Updating means bumping the pinned tag and re-copying, never editing vendored files.
- The fp32 casts of activations and outputs remain, because b11211's quantizers and kernels take and write fp32 only. Removing them means owning a 16-bit quantizer or editing vendored code.
- IQ1_M (one tensor, 0.2% of bytes) has no MMQ upstream and stays on the old dequant path.
- The Route L build takes minutes instead of seconds (226 s clean under the caps).
- Decode at 2–8 rows is the one regime where Route L may not be optimal; the follow-ups (MMQ at small row counts, fastllm small-mmvq) stay inside Route L's ops and numerics.

Sources: `~/gsq-rco-handoff/reports/10-11-kernel-prior-art-survey.md`, `~/gsq-rco-handoff/reports/14-beyond-route-l.md`.
