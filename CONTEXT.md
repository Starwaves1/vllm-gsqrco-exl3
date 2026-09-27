# gsq-vllm

Serving the Swift 1.5 GSQ-RCO IQ3_S-mtp GGUF on Garrett's production vLLM stack, fast and correct enough to benchmark on DeepSWE like the other models. This file is the glossary; how things work lives in README.md, STATUS.md and `docs/adr/`.

## Language

### The model

**Swift GGUF**:
The one file under test: `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (Swift 1.5 finetune of Qwen3.8-27B, arch `qwen35`, embedded MTP head). Script and bench names shorten it to `gsq`.
_Avoid_: GSQ model (GSQ is the method), base Qwen3.8 (it is a finetune), "the GGUF" when a llama.cpp or other GGUF could be meant

**GSQ**:
Gumbel-softmax scalar quantization (arXiv 2604.18556), the method that refined the Swift GGUF's weights inside the GGUF format.
_Avoid_: using GSQ to name the Swift GGUF itself

**GSQ-native INT3**:
GSQ's own compressed-tensors output (e.g. `ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ`), run in vLLM through Humming. A different quant from the Swift GGUF, and the published one is base Qwen3.8 without MTP.
_Avoid_: GSQ checkpoint (ambiguous with the GGUF)

**RCO**:
The budget-constrained search (arXiv 2605.00649) that picks one quant type per tensor.
_Avoid_: calling RCO a format or a kernel

**Allocation profile**:
The per-tensor quant-type map, chosen by RCO, that the Swift GGUF is built with (IQ types hold about 82% of the bytes).
_Avoid_: tier, quant tier (tier means the KV tier here), mix

**Quant type**:
One ggml block format a tensor is stored in; the Swift GGUF uses 12 of them.
_Avoid_: dtype, bit-width

**IQ family**:
The codebook ("i-quant") quant types: IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS. They decode through lookup tables rather than affine scales.
_Avoid_: imatrix types (the imatrix is a calibration input, not a format)

**K-quants**:
The super-block quant types Q2_K, Q4_K, Q6_K. The output head is Q4_K and the MTP head Q6_K.
_Avoid_: standard types (that name covers the Q4_0/Q8_0 family too)

### The stacks

**Production stack**:
The live vLLM 0.27.1 install on the RTX 3090 serving `Starw1/Qwen3.8-27B-absolute-heresy-W4A16` (the heresy finetune, W4A16 AutoRound) behind ports 18080/18081. Read-only for this project.
_Avoid_: prod model = Swift, base Qwen3.8 W4A16 (both wrong: it is the heresy finetune), production W4A16 when Swift W4A16 is meant

**Isolated venv**:
This repo's `.venv`: a byte-for-byte copy of the production stack's packages plus gguf-py and the plugin. The only place the plugin exists.
_Avoid_: prod venv, gsq venv

**Baseline model**:
The W4A16 model the Swift GGUF's speed is compared against: the production model by default, or Swift W4A16 AutoRound when a quality-comparable baseline is needed. Always say which.
_Avoid_: production W4A16 without saying which checkpoint

**Plugin**:
`vllm-gguf-plugin`, the out-of-tree vLLM extension that loads GGUF files; here our fork of upstream `e2b8ad5`, split out as branch `swift-gsq-rco`.
_Avoid_: in-tree GGUF (removed from vLLM), the extension

**Adapter fix**:
Our one change to the plugin's `qwen3_5` weights adapter so a GGUF with a multimodal config but no mmproj serves as text-only.
_Avoid_: the plugin fix, the patch

**HF config dir**:
The config/tokenizer directory passed to vLLM alongside the Swift GGUF, built from Swift's HF files and verified against the GGUF metadata. It also names the model to vLLM.
_Avoid_: model dir (that is production's)

**KV tier namespace**:
The identity under which vLLM's filesystem KV tier stores blocks, keyed on the model path. The HF config dir must be unique so two models never share KV blocks.
_Avoid_: cache key, prefix cache (different layer)

### Decoding and kernels

**Decode**:
Steps that generate tokens for running requests; with MTP each step verifies several rows per request.
_Avoid_: generation, tg

**Prefill**:
Processing a prompt, arriving in chunks of up to 2048 rows.
_Avoid_: prompt processing, pp

**Rows per step**:
How many activation rows one quantized GEMM call processes (4 at concurrency 1 with MTP k=3, 8 at concurrency 2, 2048 for a prefill chunk). Distinct from a weight's output rows.
_Avoid_: tokens, columns, ncols, batch size

**MTP**:
Multi-token prediction: a draft head embedded in the Swift GGUF (`blk.64.nextn`) proposes tokens that the main model then verifies in one step.
_Avoid_: speculative decoding in general, EAGLE

**k**:
Number of tokens the MTP draft head proposes per step; production and all benchmarks use k=3.
_Avoid_: num speculative tokens, draft length

**Draft head**:
The MTP block plus the output head it reads through when drafting.
_Avoid_: draft model, nextn (that is the GGUF tensor prefix only)

**Acceptance**:
The fraction of drafted tokens the verify step keeps, per draft position.
_Avoid_: hit rate, acceptance length (a different metric)

**MMVQ**:
The quantized matrix-vector path: a few rows per step against quantized weights, activations quantized to q8_1.
_Avoid_: GEMV kernel, small-batch kernel

**MMQ**:
The quantized matrix-matrix path for many rows per step, int8 dot products on q8_1 activations.
_Avoid_: quantized GEMM (also covers MMVQ)

**Dequant+cuBLAS**:
The fallback that expands a whole weight to bf16 each forward and runs a dense matmul; what upstream uses for IQ types above the MMVQ threshold.
_Avoid_: Triton path (Triton serves only when the CUDA extension is absent), dequant path alone

**Triton fallback**:
The plugin's Triton dequant/GEMM kernels, which run only when the CUDA extension is disabled or missing. Never on the serving path.
_Avoid_: Triton path for the IQ large-batch route (that is dequant+cuBLAS)

### Routes

**Route A**:
Patch the plugin's own llama.cpp-b2899 kernels: efschu's batched MMVQ plus vLLM PR #36226's dp4a IQ MMQ plus guards. Branch `route-a`.
_Avoid_: Route A′ unless fastllm MMQ is included

**Route L**:
Vendor llama.cpp b11211's MMVQ/MMQ/quantize files unmodified behind a shim. The recommended route; see ADR 0001. Branch `route-l`.
_Avoid_: the bridge route (that is Maxwell-Lyu's), Route B

**Vendored kernels**:
The unmodified llama.cpp files copied into the plugin under Route L, updated only by bumping the pinned tag.
_Avoid_: our kernels, the port

**Shim**:
The single owned file that adapts vendored kernels to torch (device info, stream, scratch pool, ops, guards). Reserved for Route L.
_Avoid_: using "shim" for the no-GPU guard

**No-GPU guard**:
The import-first module that keeps CPU-only scripts from touching NVML or the GPU.
_Avoid_: no_gpu shim

### Evidence and done

**Parity**:
Logit agreement with llama.cpp b11211 on the same token ids: mean KLD ≤ 0.001 and top-1 ≥ 99%, including 100k+ contexts.
_Avoid_: accuracy, quality, correctness (broader: also covers kernel and dequant exactness)

**Speed**:
Decode tok/s at concurrency 1 and 2 against the baseline model, prefill ≥ 80% of it at 8k/64k/180k, MTP acceptance within 2 points of llama.cpp.
_Avoid_: performance (also covers fit)

**Fit**:
200k context with fp8 KV at gpu-memory-utilization ≤ 0.94.
_Avoid_: memory, VRAM check

**Soak**:
24 h at concurrency 2 with CUDA graphs: no illegal memory access, no restarts, no memory growth.
_Avoid_: stability test, burn-in

**Definition of done**:
The seven conditions in HANDOFF.md §2: drop-in, correct, fast, fits, stable, scientific, reproducible.
_Avoid_: acceptance criteria, done

**Phase A / Phase B**:
Phase A is CPU-only preparation; Phase B is everything that needs an sm86 GPU (kernel tests, parity, speed, fit, soak).
_Avoid_: step, stage

### Operations

**DeepSWE**:
The agentic coding benchmark all models are scored on; tasks have a 10,800 s time limit, so speed affects the score.
_Avoid_: SWE-bench

**Run 1**:
The DeepSWE run of the production model that holds the only GPU from 2026-09-27; no GPU use here until it ends.
_Avoid_: the bench, the job

**Cloud box**:
A rented sm86 GPU machine where Phase B can run, set up by `cloud/bootstrap.sh`.
_Avoid_: cloud GPU, remote, VM (the bench VM is a different machine)

**Capped job**:
Any build or test run through the resource wrapper (one heavy job at a time, memory and CPU caps, no GPU). The light cap (`GSQ_LIGHT=1`) is the smaller variant for quick tests.
_Avoid_: sandboxed, throttled
