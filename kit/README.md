# Validation kit

This kit measures the GSQ-RCO GGUF and EXL3 vLLM plugins in this repository on your NVIDIA card. It writes one
folder of results that we merge into the per-architecture dispatch tables and the PR's hardware matrix. You need
no context beyond this page. One command does everything, and nothing outside this checkout and its work
directory is modified.

## What you need

- Linux x86_64 and an NVIDIA GPU (tested: RTX 3070 sm86; written for sm75, sm86, sm89 and sm120)
- NVIDIA driver 580 or newer (CUDA 13) for every tier. A CUDA 12.x driver runs tier 1 only, and also needs a
  local CUDA 12.x toolkit (`CUDA_HOME`).
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`), git, curl, python3, about 25 GB of disk
  (venv 9 GB, builds 2 GB, synthetic weights 3.5 GB, tier 2 models 6 GB; tier 3 another 27 GB)
- No sudo, no system packages. The CUDA compiler comes from pip wheels.

| tier | card | time | what it measures |
|---|---|---|---|
| bootstrap | any | 15-40 min, mostly compiling | venv with the pinned wheels (`env/gsq-main-freeze.txt`: torch 2.13.0, vLLM main d28795f), both plugins compiled for your card's compute capability only, vendored-source sha256 check |
| 1 | >= 4 GB | 15-30 min | kernel parity (GGUF Route L and EXL3, synthetic weights at the Qwen3.8-27B matrix shapes); a micro-benchmark of every candidate kernel per (quant type, rows 1..128, shape) against the vendored llama.cpp MMVQ/MMQ; CUDA-graph replay bit-exactness; efficiency against the card's measured copy bandwidth |
| 2 | >= 8 GB | ~25 min | Qwen3.5-4B served end to end: GGUF Q4_K_M with MTP k=3, decode ladder c=1/2/4/8 (ms/step, tok/s), a 200-request corruption check; the same for a Qwen3.5-4B EXL3 quant (no MTP head exists for it) |
| 3 | 24 GB | ~1.5 h | the same for the 27B production models (Swift-1.5-Qwen3.8-27B GSQ-RCO IQ3_S-mtp GGUF and its EXL3 3.50 bpw) |

## Run

```sh
git clone https://github.com/Starwaves1/vllm-gsqrco-exl3 && cd vllm-gsqrco-exl3
git checkout validation-kit            # or the commit you were given
kit/run.sh bootstrap                   # once; safe to rerun
kit/run.sh tier1                       # then tier2 (8 GB+), tier3 (24 GB)
```

`kit/run.sh all` runs bootstrap, tier 1, tier 2, and tier 3 when the card has 24 GB. Useful options:

| option | meaning |
|---|---|
| `--gpu N` | which card (nvidia-smi index; default 0). The kit sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=N`, checks that CUDA sees the card nvidia-smi names, and never touches another GPU. |
| `--name PREFIX` | results folder prefix, default `<hostname>-<card>` (e.g. `mybox-rtx3070`) |
| `--max-gpu-gb G` | the most GPU memory the kit may use (default 85 % of the card). Tier 1 caps its processes with `torch.cuda.set_per_process_memory_fraction`; tiers 2/3 turn it into vLLM's `--gpu-memory-utilization`. |
| `--allow-capped` | see "Power limit" below |
| `--idle-mib N`, `--wait S` | a measurement starts only when the card shows <= N MiB used (default 1024) and <= 10 % utilization; the kit waits up to S seconds (default 600), then stops |
| `--parity-k EXPR` | pytest `-k` filter for the parity tests, if a full run is too long |

Long runs survive a closed terminal with `nohup kit/run.sh tier1 > kit-tier1.log 2>&1 &`.

Environment: `KIT_WORK` (venv, builds, synthetic weights; default `kit/.work`), `HF_HOME` (model downloads;
default `kit/.work/hf`), `KIT_PORT` (vLLM port for tiers 2/3, default 18200), `MAX_JOBS` (parallel compiles,
default 4; each needs about 3 GB RAM), `UV` (path to uv).

## Power limit and clocks

Before every measurement the kit records the card's power limit, default limit, clocks, temperature, memory and
utilization (`gpucheck-*.json`). It samples clocks, power and clock-event reasons every second while measuring
(`clocks-*.csv`). If the power limit is below the card's default, the kit refuses to produce numbers, because they
would describe your power setting rather than the kernels. Restore the default limit, or pass `--allow-capped`:
the run then completes, and the summary labels every number as capped.

## Older and newer cards

- **Turing (sm75, RTX 20xx):** no bf16. The kit serves with `--dtype float16` and runs the micro-benchmark on fp16
  activations. There is no fp8 KV cache (`--kv-cache-dtype` stays auto) and no FlashAttention (sm80+), and the
  prebuilt FlashInfer kernels cover sm80+ only, so tiers 2/3 use `--attention-backend TRITON_ATTN`. The int8
  tensor-core kernels (Route L's owned IQ3/mma_k kernels) and the EXL3 kernels may not compile for sm75. The
  bootstrap then records the failure and builds what it can (Route L without them, or the stock GGUF kernels; EXL3
  phase 1 only, or none). The affected tests and benchmarks are reported as `unsupported`, not as failures.
- **Ada (sm89) and Blackwell (sm120):** fp8 KV cache as on Ampere. Nothing else changes.
- **CUDA 12 driver:** tier 1 only, as above. torch 2.13.0+cu129 from the PyTorch index; vLLM's Python code is
  installed without its CUDA 13 deps, which the plugins' kernel tests do not need.

All tiers use stock vLLM at the pinned commit. The published 27B numbers in the top-level README also used a small
vLLM patch set (filesystem KV tiers, a pruned MTP draft head), so tier 3 numbers are not directly comparable to
them.

## What comes out

`kit/results/<prefix>-<YYYY-MM-DD>/`:

| file | content |
|---|---|
| `summary.md` | the tables to paste into a PR: card, driver, power limit, clocks; parity counts; the GGUF routing winner per (type, rows) and the whole-model GEMM time per forward vs the copy-bandwidth floor; the EXL3 route winner per (bits, rows); the tier 2/3 ladders and corruption counts |
| `results.json` | everything machine-readable: `cc` ("8.6"), `tiers` (`{"1": "pass"/"partial"/"fail"}`), card, driver, CUDA, torch/vLLM versions and the repo commit, every power check and clock summary, every micro-benchmark cell, parity counts per test |
| `tier1/` | `gguf_micro.tsv` (every cell), `parity-*.xml` (junit) and logs, `exl3_micro/` |
| `tier2/`, `tier3/` | per model: `argv.txt`, `ladder.json`, `corruption.jsonl`, gzipped server logs and responses |
| `steps.json`, `bootstrap.json`, `env.json`, `gpucheck-*.json`, `clocks-*.csv` | the raw records |

`partial` means some steps were unsupported on the card or failed. `steps.json` says which, and why.

Then send the folder back: [RETURN.md](RETURN.md).

## Files

| file | role |
|---|---|
| `run.sh` | the entry point: bootstrap, the power/idle gate, tiers |
| `synth.py` | synthetic GGUF and EXL3 weights at the 27B shapes (random blocks with finite scales), so tier 1 needs no download |
| `micro_gguf.py` | the GGUF kernel micro-benchmark |
| `ladder.py` | the decode ladder client (tiers 2/3) |
| `report.py` | environment record, power gate, `results.json` + `summary.md` (`python3 kit/report.py selftest DIR` checks it on fake data) |
| `kit_guard.py` | pytest plugin that caps the test process's GPU memory |
| `data/swift-27b-gguf-tensors.tsv` | the 27B GGUF's tensor list (name, type, rows, K): the shapes and their counts |

The tests and benchmarks themselves are the repository's own: `tests/gpu/test_kernel_parity.py`,
`tests/gpu/test_exl3_kernels.py`, `tests/gpu/test_exl3_mr.py`, `bench/micro/exl3_mr.py`, `bench/corruption_check.py`.
