# vllm-gsqrco-exl3

A vLLM quantization plugin that serves GSQ-RCO GGUF models on an RTX 3090 at production speed. It is a fork of [vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin), plus the build, test, benchmark, parity and soak harnesses used to measure it. vLLM itself is not forked.

Status: tested on one RTX 3090 with vLLM 0.27.1. EXL3 support is in progress (CPU-only so far, see below).

## What it does

GSQ-RCO models are GGUF files that mix IQ2, IQ3 and IQ4 types with K-quants. The stock plugin runs them, but slowly: its llama.cpp b2899 kernels have no tensor-core path for IQ types, so every prefill dequantizes whole weights to bf16. This fork adds Route L, a backend that keeps every vLLM feature we use working: MTP speculative decoding, CUDA graphs, prefix caching, CPU and filesystem KV offload, and priority scheduling.

Route L vendors llama.cpp b11211's MMVQ, MMQ and q8_1 quantize files byte for byte and calls them through one shim file, `lcpp_shim.cu`. Owned kernels replace them where a measurement showed a gain:

| owned piece | covers |
|---|---|
| q8_1 quantizer | reads fp32/fp16/bf16 activations directly, no input cast; zeroes MMQ's read tail |
| IQ3 dp4a decode kernel | IQ3_S, IQ3_XXS at 1-5 rows, unpacked layers |
| IQ3 int8-mma decode kernel | IQ3_S, IQ3_XXS at 6-8 rows, unpacked layers |
| lossless load-time IQ3 repack + packed int8-mma kernels | IQ3_S, IQ3_XXS at every row count; the decode kernel to 8 rows, a tiled kernel above |
| int8-mma kernel | Q4_K, IQ2_S, IQ4_XS at 9-32 rows, the MTP verify pass |
| dp4a kernel | Q4_K, IQ2_S at 1-8 rows |
| routing table | picks the op per type, row count and weight shape; pinned by `tests/cpu/test_lcpp_routing.py` |

Route L covers Q2_K, Q4_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS, and IQ1_M through MMVQ. Other GGUF types fall through to the stock plugin kernels. Route L is off by default. Without `VLLM_GGUF_LCPP=1` the plugin behaves like upstream `e2b8ad5`.

Tested models:

- [ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF), file `Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, 12.1 GB
- [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF), file `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, same tensor types and bytes as Swift

## Results

One RTX 3090 at its stock 350 W limit, Swift IQ3_S-mtp, fp8 KV, 200k max context, MTP k=3, CUDA graphs up to 32 tokens, gpu-memory-utilization 0.94. The reference is an AutoRound W4A16 build of the same base model on Marlin kernels, same box and same argv. Decode is greedy over 8 real prompts x 1024 tokens; ms/step is the time per engine step. Method and raw data are in [cloud/results/REPORT.md](cloud/results/REPORT.md).

| decode | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|
| this plugin, tok/s | 110.3 | 192.7 | 348.1 | 541.3 |
| this plugin, ms/step | 27.9 | 31.6 | 35.7 | 44.2 |
| W4A16 reference, tok/s | 94.1 | 194.4 | 345.1 | 505.4 |
| W4A16 reference, ms/step | 27.6 | 27.3 | 30.0 | 41.3 |
| ISTA base GGUF, tok/s | 104.8 | 186.9 | 333.1 | 528.8 |
| stock plugin kernels, tok/s | 32.4 | 38.0 | not run | not run |

| prefill, c=1, tok/s | 8k | 64k | 180k |
|---|---|---|---|
| this plugin | 1248 | 954 | 644 |
| W4A16 reference | 1108 | 868 | 603 |
| stock plugin kernels | 357 | 328 | 281 |

| fit at 200k context | this plugin | W4A16 reference | stock plugin |
|---|---|---|---|
| GPU KV cache, tokens | 253,906 | 207,812 | 245,312 |
| weights in VRAM | 12.45 GiB | 14.26 GiB | 12.29 GiB |

Without MTP, one row per sequence per step:

| ms/step, no MTP | c=1 | c=2 | c=4 | c=8 |
|---|---|---|---|---|
| this plugin | 19.6 | 20.4 | 22.7 | 26.9 |
| W4A16 reference | 20.6 | 22.3 | 22.8 | 22.8 |
| RedHatAI/Qwen3.8-27B-INT4 on stock vLLM | 22.6 | 24.2 | 24.8 | 24.9 |
| ratio, this plugin / W4A16 | 0.95 | 0.92 | 1.00 | 1.18 |

How to read this. Throughput with MTP is at or above the reference at every concurrency except c=2, where it is 0.9% lower, inside that point's run-to-run noise. Per engine step the plugin is 1% slower at c=1 and 7-19% slower at c=2 to 8. The tok/s lead comes from MTP acceptance, and acceptance depends on the model's MTP head: 0.650 for Swift, 0.628 for the ISTA base, 0.522 for the W4A16 reference. The per-step gap sits in the 4-32 row products of the MTP verify pass. Without MTP the plugin is faster per step than the reference at c=1 and c=2 and equal at c=4.

## Correctness

| check | result |
|---|---|
| kernel parity vs CPU references on real GGUF blocks, Route L on | 3920 pass, 183 skip, 0 fail |
| kernel parity, Route L off | 3822 pass, 281 skip, 0 fail |
| GPU input guards | 240 pass, 60 skip, 0 fail |
| compute-sanitizer memcheck + initcheck over every owned op | 168 cases, 0 errors |
| CPU suite, no GPU build needed | 564 pass, 54 xfail for a known 1-ulp fp16 K-quant dequant difference |
| CPU guards + routing table + IQ3 repack, with the Route L build | 259 pass |
| vendored llama.cpp files | sha256 match `VENDORED.md` |

Logit parity against llama.cpp b11211 CUDA on the same token ids, 11 prompts from 1k to 120k tokens, bf16 KV, no spec decode:

| | mean KLD | top-1 agreement |
|---|---|---|
| this plugin, all positions | 0.0249 | 98.18% |
| this plugin, prompts ≥ 100k | 0.0064 | 98.26% |
| stock plugin kernels, all positions | 0.0404 | |
| seq_000-005, mean of per-prompt KLD: this plugin | 0.043 | |
| seq_000-005: llama.cpp's own CPU vs CUDA backends | 0.120 | |

The absolute target of KLD ≤ 0.001 and top-1 ≥ 99% fails, for the stock kernels too. On the six prompts where llama.cpp's CPU-vs-CUDA spread was measured, the plugin is below that spread on every one. A few high-entropy positions carry the means. MTP greedy acceptance is 0.6684 against llama.cpp's 0.6732.

Stability. A 24 h soak at c=2 with CUDA graphs is in progress. A 1.19 h partial run on the same build served 621 requests with 0 faults and 0 restarts. GPU memory stayed flat after the first minute. Data is in `cloud/results/soak/`.

## EXL3 (in progress, CPU-only so far)

A second package, `plugin-exl3/` (`vllm_exl3_plugin`), serves [exllamav3](https://github.com/turboderp-org/exllamav3) EXL3 checkpoints on unpatched vLLM main. It registers quant method `exl3` and nothing else: an EXL3 checkpoint is a normal HF directory, so vLLM's own detection and safetensors loader do the rest. The dense-linear kernels are vendored byte for byte from exllamav3 v1.5.3 (94 files, MIT) behind one shim, `exl3_shim.cu`, the Route L pattern. First target: [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) at 3.50bpw.

Done on the CPU: the package, the shim (compiles and links for sm_86, never run), loading the checkpoint's names and shapes into vLLM's Qwen3.8 model and MTP draft on the meta device (nothing unmapped, nothing missing), the row routing, the pruned draft-head tool, and 153 CPU tests. Nothing numeric is tested yet: kernel parity, logit parity against exllamav3, speed, fit and soak are the GPU phases. Design and plan: [EXL3.md](EXL3.md), [docs/adr/0002-exl3-via-plugin.md](docs/adr/0002-exl3-via-plugin.md), [docs/exl3-feasibility.md](docs/exl3-feasibility.md).

## Install and build

Requirements: Linux, an sm_86 GPU, a driver that supports CUDA 13, [uv](https://docs.astral.sh/uv/). The CUDA toolkit comes from pip wheels. The build needs no system packages and no sudo.

```sh
git clone https://github.com/Starwaves1/vllm-gsqrco-exl3 && cd vllm-gsqrco-exl3
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r env/prod-freeze.txt    # vLLM 0.27.1, torch 2.13.0 cu130
uv pip install --python .venv/bin/python --no-deps \
  "gguf @ git+https://github.com/ggml-org/llama.cpp@b11211#subdirectory=gguf-py"
tools/setup-cuda-toolchain.sh                   # nvcc 13.0 into build/cu130, matching torch's cudart
VLLM_GGUF_BUILD_LCPP=1 tools/build-plugin.sh    # editable install, sm_86; a few minutes
```

Without `VLLM_GGUF_BUILD_LCPP=1` the build skips Route L and takes about 20 s. `tools/setup-cuda-toolchain.sh` also patches its private CUDA 13.0 header copy on glibc 2.41 or newer.

The published numbers also used a small vLLM 0.27.1 patch set, [Starwaves1/vllm@ba05ffab](https://github.com/Starwaves1/vllm/commit/ba05ffababdcf89ada26b5d34845e04901e2ddf3). It adds size-capped filesystem KV tiers and a row-pruned MTP draft head that reads `mtp_draft_vocab_ids.pt` from the HF config dir. On stock vLLM 0.27.1, set `MTP_DRAFT_VOCAB=0` and use a CPU-only KV tier. That combination was not benchmarked, and KV capacity will differ.

## Run

vLLM needs an HF config dir next to the GGUF. Both tested models have one in `hf-config/`. Check yours against the GGUF with:

```sh
.venv/bin/python tools/make_hf_config.py verify --gguf /models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --out hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp \
  --template hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp/chat_template.jinja
```

Give every model its own HF config dir. The plugin sets `model_config.model` to that dir, and vLLM's filesystem KV tier namespaces its keys on it, so two models sharing one dir would read each other's KV blocks.

```sh
export VLLM_GGUF_LCPP=1 VLLM_PLUGINS=gguf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp
.venv/bin/vllm serve /models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
  --hf-config-path $CFG --tokenizer $CFG --served-model-name swift-gsq \
  --gpu-memory-utilization 0.94 --max-model-len 200000 --max-num-seqs 8 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 --mamba-cache-mode align \
  --max-num-batched-tokens 2048 --long-prefill-token-threshold 128 --no-async-scheduling \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}' \
  --compilation-config '{"max_cudagraph_capture_size":32,"custom_ops":["+rms_norm","+silu_and_mul"]}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching --enable-cumem-allocator --scheduling-policy priority \
  --kv-offloading-size 16
```

The GGUF has no vision projector, so image and video limits must be 0, or pass `--language-model-only`.

## Tests and harnesses

```sh
tools/pytest tests/cpu                                            # CPU only, never touches the GPU
VLLM_GGUF_LCPP=1 GSQ_ALLOW_GPU=1 GSQ_GGUF=/models/X.gguf tools/pytest tests/gpu -v
```

`tools/pytest` keeps pytest in `build/pytest`, outside the venv. CPU tests import `tools/no_gpu.py` first. The 112 Route L guard tests in `tests/cpu` skip unless the plugin was built with `VLLM_GGUF_BUILD_LCPP=1`. GPU tests skip unless `GSQ_ALLOW_GPU=1`. `tools/capped` is an optional wrapper that caps memory and CPU for builds and tests on a shared host.

The harnesses were written for one setup. Paths and ports are environment overrides in `scripts/env.sh`.

| harness | what |
|---|---|
| `scripts/serve-gsq.sh`, `scripts/serve-baseline.sh` | serve the GGUF or the W4A16 reference on the same argv, from `env/prod-serve-argv.txt` |
| `bench/speed/run.sh` | decode ladder at c=1/2/4/8, salted prefill ladder at 8k/64k/180k, MTP acceptance from `/metrics`, GPU clocks; the decode cohort script it wraps is not public yet |
| `bench/parity/` | llama.cpp b11211 logit dumps vs vLLM logprobs on the same ids, KLD and top-1 per position |
| `bench/micro/gemm.py` | per-op GEMM microbenchmarks |
| `bench/soak.sh`, `bench/soak_load.py` | long soak: mixed chat, tool calls, long prompts up to 120k, aborted streams, random priorities; logs liveness, memory and device faults every 60 s |
| `cloud/bootstrap.sh` | sets up a rented sm_86 box and runs tests, parity, speed and an optional soak |
| `dash/` | optional status page for a rented box: GPU, memory, and a file-based GPU job queue polled over ssh; the box address comes from the gitignored `dash/data/box.json` |

## Repo layout

| path | what |
|---|---|
| `plugin/` | the plugin: git subtree of vllm-gguf-plugin `e2b8ad5` plus this project's commits. `git subtree split --prefix=plugin` gives a standalone plugin history |
| `plugin/vllm_gguf_plugin/csrc/lcpp/` | vendored llama.cpp b11211 files, unmodified; `VENDORED.md` lists them with sha256 |
| `plugin/vllm_gguf_plugin/csrc/lcpp_shim.cu`, `lcpp_owned_*.cu` | the shim and the owned kernels |
| `plugin/vllm_gguf_plugin/quantization/` | routing in `linear.py`, IQ3 repack in `iq3_pack.py` |
| `plugin-exl3/` | the EXL3 plugin (in progress); vendored exllamav3 files in `vllm_exl3_plugin/csrc/exl3/` with `VENDORED.md` |
| `hf-config/` | HF config dirs for the tested GGUFs, and the EXL3 checkpoint's metadata, with `PROVENANCE.json` |
| `tests/cpu`, `tests/gpu` | test suites |
| `bench/`, `scripts/`, `cloud/` | harnesses; `cloud/results/` holds every measurement |
| `tools/` | build helpers, HF config builder, reference dequantizers |
| `env/` | tested package set and reference argv |

Docs:

- [cloud/results/REPORT.md](cloud/results/REPORT.md), the benchmark report: method, profiles, per-kernel contributions, what was tried and dropped
- [STATUS.md](STATUS.md), the dated engineering log
- [docs/adr/0001-vendor-llamacpp-kernels-route-l.md](docs/adr/0001-vendor-llamacpp-kernels-route-l.md), why Route L
- [ROUTE-L.md](ROUTE-L.md), how the shim, routing and repack work
- [EXL3.md](EXL3.md), the EXL3 plugin: design, vendored files, ops, routing, GPU plan
- [CONTEXT.md](CONTEXT.md), project glossary

## Upstream notes

No PRs are open yet. The candidates, in order:

1. llama.cpp. `ggml_cuda_mul_mat_q` sizes MMQ's q8_1 read tail from `ggml_cuda_mmq_get_J_max`, which rounds `ne11` down to a multiple of 8. Below 8 activation columns the tail is zero, but the kernel still loads a full tile, so it reads past the allocation. llama.cpp's own dispatch sends ≤ 8 columns to MMVQ, so only direct MMQ callers hit it. The shim adds its own tail and zeroes it. The proposed fix sizes the tail from the largest tile the kernel can pick. Details are in REPORT.md section 14a.
2. vllm-gguf-plugin fixes that stand alone: staging unsharded weights from host memory, which avoids an OOM at MTP draft load, and serving a multimodal-config Qwen3.5 GGUF without a projector as text only.
3. vllm-gguf-plugin: Route L as an opt-in backend, with the vendored files, the shim, the routing, the build flag and the tests.
4. The owned kernels, one at a time with microbenchmarks and parity tests. The IQ3 repack and packed kernels come first.

Draft-head vocab pruning and the routing thresholds stay local. They were tuned on one 3090.

## Credits and licenses

This repository is Apache-2.0, see [LICENSE](LICENSE).

| source | license | used for |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) b11211 | MIT | vendored MMVQ, MMQ and quantize files, listed in `plugin/vllm_gguf_plugin/csrc/lcpp/VENDORED.md`; its LICENSE is copied beside them |
| [vllm-project/vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin) | Apache-2.0 | the base of `plugin/` |
| [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) | Apache-2.0 | ideas only, read and not copied: the decode-once IQ3 slice layout and sign step, and quantizing BF16 activations to q8_1 without an fp32 cast |
| Maxwell-Lyu's [codex-ggml-source-bridge](https://github.com/Maxwell-Lyu/vllm-gguf-plugin/tree/codex-ggml-source-bridge) branch | plugin fork | the zeroed MMQ read-tail workaround; used as a template for the shim |
| [ztxz16/fastllm](https://github.com/ztxz16/fastllm) | Apache-2.0 | prior art for small-batch GGUF decode kernels and a minimal ggml shim |

GSQ-RCO quantization: GSQ is Gumbel-Softmax Quantization ([arXiv:2604.18556](https://arxiv.org/abs/2604.18556)); RCO is Riemannian Constrained Optimization ([arXiv:2605.00649](https://arxiv.org/abs/2605.00649)). Thanks to ISTA-DASLab and ukisai for publishing the GGUFs.
