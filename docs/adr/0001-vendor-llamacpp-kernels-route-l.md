---
status: accepted
---

# Vendor llama.cpp b11211 kernels behind a shim (Route L)

The Swift GGUF is IQ-dominated (about 82% of weight bytes in IQ types), and the plugin's kernels are llama.cpp b2899: IQ types have no MMQ, so above the MMVQ threshold every forward dequantizes the whole weight to bf16 and runs cuBLAS, and MMVQ re-reads weights once per row. On the RTX 3090, int8 tensor-core MMA peaks at about 4× the bf16 rate, so any W×A16 route caps prefill well below an int8 MMQ. vLLM must stay the server (priority scheduling, KV tiers, the production argv). We decided to vendor llama.cpp b11211's MMVQ, MMQ and q8_1 quantize files unmodified into the plugin, adapt them with one owned shim file, and gate the new path behind a default-off flag, because it is the only source that covers every quant type in the file with tensor-core MMQ and uses the same numerics as the parity reference.

Accepted by Garrett on 2026-09-28 as the direction: vendor llama.cpp matmul kernels into vLLM in a stable, tested way, and improve on them in owned code where measurements justify it. Since then, on a rented 3090 (STATUS.md, phases 2 and 3, Integration 1): kernel parity passes, the 200k fit holds, and the speed ladder has run. Logit parity vs llama.cpp does not meet its absolute gate (KLD <= 0.001, top-1 >= 0.99); it meets the relative one set in phase 1b (no worse than llama.cpp's own CPU-vs-CUDA spread). c=2 decode is still below the W4A16 baseline, and per engine step every concurrency is slower. No 24 h soak of the current build is recorded. Route A remains a fallback branch.

## Considered Options

- **Route A**: efschu's batched MMVQ plus vLLM PR #36226 IQ MMQ on the plugin's b2899 code. Small diff, but #36226 was closed unmerged, never compiled against this tree, and uses 4×32 dp4a tiles with no tensor cores, which are unlikely to beat dequant+cuBLAS at prefill. Kept compile-ready on branch `route-a` as a fallback.
- **Lossless transcode of IQ3_S/IQ3_XXS/IQ2_XXS into IQ4_XS containers**: fits (+1.77 GB) and is bit-exact, but published 3090 numbers (llama.cpp PR #8215) show IQ4_XS MMVQ slower per token than IQ3_S, so the extra bytes are expected to lose. Demoted.
- **Marlin, Humming, FLUTE/BitBLAS, EXL3/QTIP/QuIP#/AQLM**: all A16 (prefill capped by the bf16 rate); Marlin in vLLM main cannot express the IQ value sets losslessly, Humming's lossless encoding needs about 21 GB, FLUTE/BitBLAS are unmaintained, and the codebook formats would mean re-quantizing. Dead ends.
- **Maxwell-Lyu's `codex-ggml-source-bridge`**: the same idea, but WIP, restructured daily, and carrying the whole llama.cpp tree including MoE and FP4. Used as a template (its q8_1 scratch guard tail is required to avoid an IMA), not adopted.

## Consequences

- About 20k vendored lines, still unmodified: all adaptation (device info, stream, scratch pool, ops, guards) lives in the shim. Updating means bumping the pinned tag and re-copying, never editing vendored files. Owned code sits beside it where it measured faster: a 16-bit-input q8_1 quantizer (phase 3 item 4), the IQ3_S/IQ3_XXS dp4a kernel (phase 3 item 5, `lcpp_shim.cu`; 16-bit output from opt-p), the IQ3 int8 mma kernel (K2, `lcpp_owned_iq3_mma.cu`) and the Q4_K/IQ2_S dp4a kernel (K1, `lcpp_owned_k4.cu`), each tested against the vendored kernels.
- The input cast is gone (owned quantizer; a fused layer quantizes X once for all its runs that read q8_1, while MMQ runs quantize for themselves). The output cast remains for vendored MMVQ/MMQ, which write fp32 only, and for the mma and Q4_K/IQ2_S kernels; the dp4a IQ3 kernel writes 16-bit directly.
- IQ1_M (one tensor, 0.2% of bytes) has no MMQ upstream and stays on the old dequant path.
- The Route L build takes minutes instead of seconds (226 s clean under the caps).
- Decode at small row counts was the weak regime. After Integration 1, c=1 decode throughput is 1.07x the W4A16 baseline (from higher MTP acceptance; per step it is 1.10x slower), c=2..8 0.77-0.90x; target passes of 16-64 rows (c >= 4) still run on MMQ and are the next owned-kernel target. All of it stays inside Route L's ops and numerics.

Sources: `~/gsq-rco-handoff/reports/10-11-kernel-prior-art-survey.md`, `~/gsq-rco-handoff/reports/14-beyond-route-l.md`.
