# Status: Swift GSQ-RCO IQ3_S-mtp GGUF on production vLLM (Phase A, CPU only)

Handoff snapshot, 2026-09-27. Labels: VERIFIED (checked in source or by running it here), DOCUMENTED (read in docs), INFERRED (reasoned, not checked).

Scope change from Garrett during the run: **deliverable 5 (kernel porting: multi-column MMVQ, IQ MMQ, dispatch) is paused until a prior-art survey comes back.** No kernel code was written, so there is no WIP kernel branch.

## Branches

| Branch | What |
|---|---|
| `main` | everything: `plugin/` (git subtree of vllm-project/vllm-gguf-plugin at e2b8ad5 plus our commits), tools, HF config dir, env records, this file |
| `swift-gsq-rco` | the plugin fork itself, `git subtree split --prefix=plugin`: upstream history through e2b8ad5, plus our adapter commit (7794689). Kept as a clean split because Garrett intends to send it upstream later; don't push it or open a PR right now. Regenerate after new plugin commits: `git subtree split --prefix=plugin -b swift-gsq-rco` |

Remote `plugin-upstream` has `pushurl = no_push`. Nothing was pushed anywhere.

## Done

1. **Isolated venv `.venv` = production's package set** (VERIFIED)
   - How it was built, and how to rebuild it:
     1. `uv venv --python ~/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 .venv`
     2. `uv pip install --python .venv/bin/python --link-mode=copy --no-deps --offline -r env/prod-freeze.txt`. `env/prod-freeze.txt` is production's `uv pip freeze`: 202 pins, all from PyPI, vllm 0.27.1 manylinux wheel, torch 2.13.0+cu130. Everything came from the uv cache in 14 s. **`--link-mode=copy` is required:** production's venv files are hard links into the uv cache, so a hard-linked venv would share inodes with production, and any in-place write would change production's files.
     3. Overlay Garrett's patched vLLM with his own tool, pointed at our venv:
        `SITE_PACKAGES=.venv/lib/python3.12/site-packages STATE_FILE=<tmp>/state BACKUP_ROOT=<tmp>/bk ~/qwen38-27b-rtx3090/scripts/deploy-vllm.sh --init v0.27.1`, then `... deploy-vllm.sh ba05ffababdc`. The fresh wheel matched v0.27.1 on all 2,755 tracked files. After deploy, `--verify` passes on all 2,769 files at ba05ffab, the commit production has deployed. Ignore the restart line the script prints: nothing was restarted.
     4. Copy production's top-level `site-packages/build_backend.py`. Both flashinfer-python and torch-c-dlpack-ext ship this file, so install order decides which copy wins. Nothing imports it at runtime.
     5. `uv pip install --no-deps --link-mode=copy ~/llama.cpp-b11211/gguf-py` (gguf 0.19.0 at tag b11211), then build the plugin (item 2).
   - Evidence:
     - `diff env/prod-freeze.txt <(uv pip freeze --python .venv/bin/python)` shows only `gguf @ file://…/gguf-py` and `-e file://…/plugin` (saved as `env/gsq-freeze.txt`).
     - `rsync -rcni --exclude __pycache__ --exclude '*.pyc' <prod site-packages>/ .venv/…/site-packages/` (full checksums) differs only in:
       - 41 `*.dist-info/RECORD` files, where the only differing lines are entry-point script hashes (the venv path is in the shebang);
       - `build_backend.py`, since fixed by copying production's;
       - production's stray `vllm/v1/worker/gpu/model_runner.py.orig` (a known Aug 21 patch backup).
     - `stat` link count is 1 (real copies).
   - To re-verify: rerun the freeze diff and the `rsync -rcn` above, and run `deploy-vllm.sh --verify` with the same env overrides.
2. **Plugin built from source, sm86 only** (VERIFIED). Commands: `tools/setup-cuda-toolchain.sh`, then `tools/capped tools/build-plugin.sh` (editable install, about 20 s).
   - The venv's own `nvidia/cu13` is unusable for building: it pairs nvcc 13.3.73 with cudart 13.0.96 headers, and CCCL `#error`s on that ("CUDA compiler and CUDA toolkit headers are incompatible").
   - So `build/cu130` holds a matched CUDA 13.0 toolchain (nvcc/crt/nvvm 13.0.88, cccl 13.0.85, runtime 13.0.96). These are the pins of the `cuda-toolkit` metapackage already in production's venv, and they match torch.version.cuda 13.0.
   - CUDA 13.0 headers clash with this host's glibc 2.43, which declares `rsqrt/rsqrtf` noexcept. The script adds `noexcept(true)` to those two declarations in our private header copy, only when glibc ≥ 2.41; CUDA ≥ 13.1 does the same with `_NV_RSQRT_SPECIFIER`. Host compiler is g++-13; `-lcudart` is satisfied by a `lib/libcudart.so` symlink.
   - Result: `_C_gguf.abi3.so` contains only a `sm_86` cubin and NEEDs only `libcudart.so.13` (no CUDA 12/13 runtime mix). All 9 cudart symbols it imports resolve against the venv's libcudart 13.0.96. All 6 ops register via `torch.ops.load_library` with `torch.cuda.is_initialized() == False`.
3. **Adapter fix** (`plugin/…/weights_adapter/qwen3_5.py`, commit 33cbfdd / 7794689 on the fork; VERIFIED in source, gate unit-checked).
   - Before: a multimodal config without mmproj raised. Even without the raise, the text prefix would have been `model.`, which `Qwen3VLForConditionalGeneration.hf_to_vllm_mapper` (qwen3_vl.py:1731, which maps `model.language_model.` to `language_model.model.`) does not map. Neither would the quant config's layouts and unquantized modules (model_loader/utils.py:309-314).
   - Now: the prefix follows `vision_config`, and `build_name_map` requires image and video limits of 0, or `--language-model-only`. Then vLLM's `_mark_tower_model` (interfaces.py:269-306) replaces the vision tower with `StageMissingLayer` instead of leaving it silently uninitialized. The plugin's loader has no "all weights loaded" check.
   - The multimodal config is kept because vLLM 0.27.1 only creates the MTP draft for `model_type qwen3_5` (config/speculative.py:516).
   - Checked: the gate accepts limits of 0 and `--language-model-only`, and rejects the defaults, `image>0`, and a missing multimodal config.
4. **HF config dir** `hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp/`, built and verified by `tools/make_hf_config.py build|verify` (35 checks pass; VERIFIED).
   - Contents: Swift W4A16 AutoRound's `config.json` minus `quantization_config`; Swift's tokenizer, processor and generation files byte-identical; `chat_template.jinja` = production's `qwen-sharp` template (froggeric v22.1). transformers prefers that file over tokenizer_config's entry (tokenization_utils_base.py:1783-1799). `PROVENANCE.json` holds sha256 of all sources and outputs.
   - Checks: every dimension equals the GGUF metadata (layers, heads, GDN dims, rope/mrope, layer_types incl. the MTP block, vocab, context); all 248,077 HF token strings equal the GGUF's; the 243 extra GGUF ids are `[PAD…]`; merges are identical; the pre-tokenizer regex equals llama.cpp's `qwen35` (llama-vocab.cpp:392-397); eos/bos match; AutoTokenizer loads production's template.
   - Keep this dir path unique to this model: the fs KV-tier namespace hashes `model_config.model`, which the plugin sets to this dir.
5. **CPU-safety tooling:**
   - `tools/capped`: the resource wrapper.
   - `tools/no_gpu.py`: import it first in any Python that touches torch or vLLM. It blocks NVML/driver dlopen through ctypes, so vLLM resolves `UnspecifiedPlatform` and nothing talks to the GPU.
   - `tools/pytest`: pytest living in `build/pytest`, outside the venv.

## Half-done

- **CPU tensor-mapping dry run** (deliverable 3, second half): not written. Plan:
  - Run `GGUFModelLoader.load_model` (the plugin's real loader) with vLLM's real `Qwen3_5ForConditionalGeneration`, and separately `Qwen3_5MTP` for `blk.64`, on the meta device, feeding memory-mapped GGUF tensors through the adapter.
  - Assert: no unmapped names (851 main + 15 MTP = 866); every non-vision parameter loaded; GGUF logical shapes equal each shard's partition size; the 48 `linear_attn.out_proj` layers got the GDN layout.
  - Needs:
    - a platform stub (subclass `vllm.platforms.cuda.NonNvmlCudaPlatform`, overriding `get_device_capability`→8.6, `get_device_name`, `get_device_total_memory`; set `vllm.platforms._current_platform` before building configs);
    - a gloo world of size 1 (`init_distributed_environment` + `initialize_model_parallel(1,1)`);
    - `device_config.device = meta`;
    - `vllm.plugins.load_general_plugins()`;
    - running under `tools/capped` (6 GB), not light.
  - The GDN V-row permutes copy about 650 MB of packed rows transiently; everything else stays memory-mapped.
  - Watch production's local `draft_lm_head` patch (qwen3_5_mtp.py:91-105). The GGUF has no `mtp.draft_lm_head`, so that head must stay disabled.
- **Dequant fixtures + GDN round-trip** (deliverable 4): the agent stopped before writing files. Its read-only findings:
  - VERIFIED: the venv's gguf-py equals b11211's.
  - VERIFIED: `libggml-base.so` is CPU-only (no FMA flags), so a ctypes bit-exact cross-check vs gguf-py is feasible.
  - VERIFIED stored GDN shapes: `attn_qkv` 10240 rows (Q 2048, K 2048, V 6144); `attn_gate` 6144; `ssm_alpha/beta` BF16 [48,5120]; `ssm_conv1d` F32 [10240,4]; `ssm_a`/`ssm_dt.bias` [48]; `ssm_norm` [128]. All `ssm_out` are quantized, so the `input_to_gguf` activation-permute path is the one used.
  - VERIFIED from `conversion/qwen.py`: the converter adds +1 to every `*norm.weight` except `linear_attn.norm.weight`, and the plugin's −1 sets cover exactly that set (read, not yet tested).
  - VERIFIED: e2b8ad5 fixed the CUDA table in `csrc/gguf/ggml-common.h`, not a Triton table as the original brief said. The Triton tables are built from gguf-py at runtime, and production has no gguf package, so this venv's 0.19.0 supplies them.
  - INFERRED: the CUDA `iq3xs_grid` equals 4× b11211's `iq3s_grid`, with `(0.5+s)*0.5` compensating. `iq1s_grid_gpu` is uint64 but used truncated to 32 bits. Both still need a test.
  - How to resume: fake the converter with `Qwen3_5TextModel.__new__` plus hparams, `tensor_map=gguf.get_tensor_name_map(QWEN35, 65)`, `fuse_qkv=False`, `fuse_gate_up_exps=False`. Fixture `.npz` keys were fixed as `raw, ref, ggml_type, tensor, rows, shape`, named `tests/fixtures/dequant/<TYPE>__<tensor>__r<row>.npz`, plus `manifest.json`.
- **README.md** (end state, phases, build and test): not written. This file stands in for it.

## Not started

- Deliverable 5, all of it (paused).
- Deliverable 4 files: `tools/ggml_ref.py`, `tools/make_dequant_fixtures.py`, `tests/cpu/test_{dequant_fixtures,kernel_tables,gdn_roundtrip}.py` (findings above).
- Deliverable 6, all harness files: `tests/gpu/`, `bench/parity/`, `bench/speed/`, `cloud/bootstrap.sh`, `scripts/serve-{gguf,llamacpp}.sh`. The agent stopped while it was still reading source. What it found, all VERIFIED:
  - **q8_1 activation quantization:** float `d = amax/127`, `q = roundf(x/d)`, with `half(d)` and `half(sum x)` stored (`csrc/gguf/gguf_kernel.cu:32-67`). The build uses `--use_fast_math` (`setup.py:42`), so a reference must allow rare ±1 flips in q.
  - **No contiguity, stride or alignment checks** on W or X anywhere; the kernels use `data_ptr()` only (`gguf_kernel.cu:98, 118-285`). Guards are needed.
  - **The min term differs by path:**
    - MMVQ Q4_K (`vecdotq.cuh:328-351`), MMQ Q2_K (`:232-262`) and IQ1_M (`:1707-1750`) use the sum of the quantized q;
    - MMQ Q4_K/Q5_K (`:353-378`) use `half(sum x)`.
  - **Triton fallback:** `ops.ggml_mul_mat_a8` sends IQ types to Triton (`triton/gemm/iq_quant/iq3_s.py`). That path doesn't quantize activations and rounds the weights to the activation dtype, so bf16 is rounded twice.
  - **F32/BF16:** `ggml_dequantize` on these types raises in Triton. They load through `weight_utils.py:194-198` instead.
  - **Production's start script** (`single-user/start_qwen_vision.sh`) sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `PYTHONHASHSEED=0` and an API key from `api_key.txt`, so the bench needs `--api-key`. The argv passes `--limit-mm-per-prompt` twice; the last one (`image 0, video 0`) wins.
- **Resume plan for the harnesses:**
  - Put shared defaults in `scripts/env.sh`: paths, a default port of 18090, and a refusal to use 18080/18081 without an explicit flag.
  - GPU use is opt-in via `GSQ_ALLOW_GPU=1`; without it, conftest imports `no_gpu` and the GPU tests skip.
  - Tight kernel check: accept the result if it matches either the q8_1 emulation (q-sum model, or x-sum for MMQ Q4_K/Q5_K) or the model with weights rounded to the activation dtype. The tolerance covers fp32 accumulation, output rounding, and the flips.
  - Loose check against full precision.
  - Run the out-of-bounds and misalignment tests in subprocesses, because CUDA errors are sticky.
  - Parity: a small C++ tool on b11211's libllama that dumps full logits only at flagged positions (e.g. the last 256-512 tokens of each 8k/32k/100k+ sequence), and the same token ids fed to vLLM with logits captured in-process, run without spec decode. Full logits for every position of a 100k sequence would be ~50 GB.
  - Bench: salt every prefill prompt so the prefix cache and KV tiers can't hit. Read MTP acceptance from vLLM `/metrics` and from llama-server's draft stats.

## Findings for the kernel-route decision

- VERIFIED, plugin CUDA code: `csrc/gguf` is llama.cpp **b2899** (May 2024).
  - MMVQ launches one grid-y slice per token, so an n-token batch reads the weights n times.
  - For IQ types, `linear.py:38-54` uses MMVQ up to 8 tokens (16 when rows ≤ 5120). Above that, since IQ types aren't in `MMQ_QUANT_TYPES`, it runs `ggml_dequantize` of the whole matrix every forward, then `x @ W.T`.
  - The plugin also ships Triton IQ GEMM kernels (`triton/gemm/iq_quant`), but `linear.py` never routes IQ types to them.
- VERIFIED, mixed quant types inside fused layers are common in this GGUF: gate/up (e.g. blk.0 IQ2_XS + IQ2_XXS), GDN qkv/z (IQ3_S + IQ3_XXS), and full-attention q/k/v (Q2_K + Q4_K). `GGUFLinearMethod` pads all shards to the widest row (wasted VRAM). `apply()` then runs `weight[start:end, :offset].contiguous()` every forward, which copies each narrower shard.
- VERIFIED, llama.cpp b11211 `ggml-cuda` has MMQ (int8 MMA, stream-k) for 9 of the 10 quantized types here. IQ1_M is the exception, and it appears only in `blk.13.ffn_gate`. b11211 also has multi-column MMVQ for all of them. Vendoring those files behind a small shim (host launcher + pool) is the low-risk route if the survey finds nothing better. Its q8_1 activation quantization would also match llama.cpp numerics.
- VERIFIED, `--language-model-only` does more than set the limits to 0. It enables a fused QK-norm/RoPE/gate path in Qwen3.5 attention (qwen3_next.py:315-329) that production doesn't use. It's a speed knob to A/B on both W4A16 and GGUF; the serve script mirrors production's `--limit-mm-per-prompt` instead.
- VERIFIED, production serves `Qwen3.8-27B-W4A16-AutoRound-fast` (base Qwen3.8), not Swift. The Swift tokenizer differs from the base one in its pre-tokenizer regex (`[\p{L}\p{M}]+`).

## Resource caps and interference rules used

- Every heavy command ran under `tools/capped`: a `flock` on `/tmp/gsq-heavy.lock` (one at a time), a wait until MemAvailable ≥ 8 GB, `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 -p CPUQuota=200%`, `nice 19`, `ionice -c3`, `CUDA_VISIBLE_DEVICES=""`, `MAX_JOBS=2`. The user cgroup delegates cpu/memory, which I checked in `/sys/fs/cgroup`. `GSQ_LIGHT=1` means 2 GB, 1 CPU, no lock.
- Never touched: production venv, unit, drop-ins, model dirs (read only), the running vLLM (pid 130752, only its `/proc` cmdline was read), ports 18080/18081, VM 192.168.1.84, Proxmox. No sudo, no pushes.
- The GGUF was only memory-mapped and its header read, never loaded whole.
- /tmp is tmpfs (RAM). Scratch was kept tiny and deleted. `build/` holds only `cu130/` (271 MB toolchain) and `pytest/` (3 MB); keep both, the build needs them. `.venv/` is 7.9 GB.

## Open questions for Garrett

- The kernel route (waiting on the prior-art survey).
- Cloud baseline: production's base `Qwen3.8-27B-W4A16-AutoRound-fast`, or `Swift-1.5-Qwen3.8-27B-W4A16-AutoRound`, which has the same Swift weights as the GGUF? The speed is the same architecture either way. Only the Swift checkpoint gives a quality-comparable baseline.
- GGUF download access for a cloud box (HF token?).

## Gotchas

- vLLM's platform detection calls NVML during `import vllm`. Scripts must `import no_gpu` before torch and vLLM. Early in this run, one NVML probe happened (a single `nvmlInit` via `vllm/platforms/cuda.py`, like one nvidia-smi query) before the guard existed. No kernel was loaded.
- `import torch` dlopens libcuda lazily. That's harmless with `CUDA_VISIBLE_DEVICES=""`, so the guard asserts on NVML and `torch.cuda.is_initialized()` instead.
- `uv pip install --target` without `--no-deps` shadows venv packages through PYTHONPATH, which is why `tools/pytest` installs pytest, pluggy and iniconfig only.
- The GGUF HF repo API answered 200 without auth on 2026-09-27, but its README says it's private/gated. Check download access before relying on a cloud bootstrap. `Starwaves1/vllm` @ ba05ffab and llama.cpp `b11211` (d7fb90e8) are publicly readable.
