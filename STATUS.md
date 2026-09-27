# Status: Swift GSQ-RCO IQ3_S-mtp GGUF on production vLLM (Phase A, CPU only)

## Current state (2026-09-27 late)

CPU suite: 416 pass / 54 xfail (the xfails are the K-quant CUDA dequant fp16 deviation, not on this model's hot path). Meta dry run passes (866/866 tensors mapped). Phase B harnesses are written but untested. Kernel survey: `reports/10-11-kernel-prior-art-survey.md` in the handoff dir. Route L is recommended over Route A; both are being prepared compile-only on branches `route-l` / `route-a`. GPU work is blocked until DeepSWE run 1 ends (~2026-10-01).

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
- VERIFIED, production serves `Starw1/Qwen3.8-27B-absolute-heresy-W4A16` (local dir `Qwen3.8-27B-W4A16-AutoRound-fast`, `base_model: MuXodious/Qwen3.8-27B-absolute-heresy`), not Swift. The Swift tokenizer differs from production's in its pre-tokenizer regex (`[\p{L}\p{M}]+`).

## Resource caps and interference rules used

- Every heavy command ran under `tools/capped`: a `flock` on `/tmp/gsq-heavy.lock` (one at a time), a wait until MemAvailable ≥ 8 GB, `systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=0 -p CPUQuota=200%`, `nice 19`, `ionice -c3`, `CUDA_VISIBLE_DEVICES=""`, `MAX_JOBS=2`. The user cgroup delegates cpu/memory, which I checked in `/sys/fs/cgroup`. `GSQ_LIGHT=1` means 2 GB, 1 CPU, no lock.
- Never touched: production venv, unit, drop-ins, model dirs (read only), the running vLLM (pid 130752, only its `/proc` cmdline was read), ports 18080/18081, VM 192.168.1.84, Proxmox. No sudo, no pushes.
- The GGUF was only memory-mapped and its header read, never loaded whole.
- /tmp is tmpfs (RAM). Scratch was kept tiny and deleted. `build/` holds only `cu130/` (271 MB toolchain) and `pytest/` (3 MB); keep both, the build needs them. `.venv/` is 7.9 GB.

## Open questions for Garrett

- The kernel route (waiting on the prior-art survey).
- Cloud baseline: production's `Starw1/Qwen3.8-27B-absolute-heresy-W4A16`, or `Swift-1.5-Qwen3.8-27B-W4A16-AutoRound`, which has the same Swift weights as the GGUF? The speed is the same architecture either way. Only the Swift checkpoint gives a quality-comparable baseline.
- GGUF download access for a cloud box (HF token?).

## Gotchas

- vLLM's platform detection calls NVML during `import vllm`. Scripts must `import no_gpu` before torch and vLLM. Early in this run, one NVML probe happened (a single `nvmlInit` via `vllm/platforms/cuda.py`, like one nvidia-smi query) before the guard existed. No kernel was loaded.
- `import torch` dlopens libcuda lazily. That's harmless with `CUDA_VISIBLE_DEVICES=""`, so the guard asserts on NVML and `torch.cuda.is_initialized()` instead.
- `uv pip install --target` without `--no-deps` shadows venv packages through PYTHONPATH, which is why `tools/pytest` installs pytest, pluggy and iniconfig only.
- The GGUF HF repo API answered 200 without auth on 2026-09-27, but its README says it's private/gated. Check download access before relying on a cloud bootstrap. `Starwaves1/vllm` @ ba05ffab and llama.cpp `b11211` (d7fb90e8) are publicly readable.

## 2026-09-27 evening: verification checklist

HANDOFF.md §8, worked read-only (no GPU, nothing restarted). Captured 20:20 UTC.

| # | Item | Result |
|---|---|---|
| 1 | GGUF checksum | SKIPPED re-hash (page-cache pressure; already verified). SHA256SUMS line = `9aecf1cd…677e5`, same as HANDOFF; size 12,120,016,896 matches. |
| 2 | Prod argv vs `production-vllm-server-info.json` | VERIFIED: `/proc/130752/cmdline` split on NUL equals JSON `argv` element for element (53 args). Process started 2026-09-25 23:22 EDT. The 4 model-file sha256s and the chat template sha256 match; deploy repo HEAD `2138d1ae8d`; prod venv has `vllm-0.27.1` and `torch-2.13.0` dist-info; GPU 595.91.07, 23,614/24,576 MiB. No drift. |
| 2a | Prod model identity (r03 vs r13) | VERIFIED r03: `-AutoRound-fast/README.md` has `base_model: MuXodious/Qwen3.8-27B-absolute-heresy` and serves as `Starw1/Qwen3.8-27B-absolute-heresy-W4A16`. Prod is the heresy finetune, not base Qwen3.8 and not Swift (Findings wording corrected). |
| 3 | Run 1 live | VERIFIED via `curl http://<bench VM>:8765/api/status`: job `pi-qwen3.8-27b-full113-k1-20260927-175853`, state running, concurrency 2, 1/113 done (1 failed, 0 timeouts), 2 trials in agent phase at ~28 tok/s. |
| 3a | Priority scheduling | VERIFIED in argv (`--scheduling-policy priority`). The dashboard does not expose priority. `/metrics`: `num_requests_running 1`, `waiting 0`, `num_preemptions_total 3`. |
| 4 | Adapter commit 33cbfdd / 7794689 | VERIFIED same patch (only the `plugin/` path prefix differs); `swift-gsq-rco` = upstream `e2b8ad5` + this one commit, 1 file, +23/-8; `git subtree split` still yields 7794689. Logic is right: the mm_proj-present paths are unchanged, and the new gate relies on `MultiModalConfig.get_limit_per_prompt`, which returns 0 under `language_model_only` (config/multimodal.py:350). Upstreamable in substance, with two gaps: no regression test is committed (the gate check was ad hoc; the plugin has no qwen3_5 adapter tests), and the commit message references local STATUS.md and vLLM 0.27.1 line refs. The error now fires in `build_name_map` rather than `patch_hf_config` (later, still clear). |
| 5a | vLLM PR #36226 | VERIFIED: `.diff` touches `csrc/quantization/gguf/{gguf_kernel.cu,mmq.cuh,vecdotq.cuh}`, the vLLM test file and `vllm/.../quantization/gguf.py`. With paths remapped to `vllm_gguf_plugin/csrc/gguf/`, `git apply --check` passes on e2b8ad5, and also on e2b8ad5 plus efschu's csrc. The PR does **not** touch `ggml-common.h`; it only indexes `iq3xs_grid`, so applying it to the plugin keeps e2b8ad5's fixed table automatically. Its dispatch change is in vLLM's `gguf.py` and has to be ported by hand to the plugin's `quantization/linear.py`. Not compiled. |
| 5b | efschu `qwen35-support` | VERIFIED: head `789d132`, 22 behind / 14 ahead of upstream `e2b8ad5` (merge-base `acf0e6d`). `mmvq.cuh` has `mul_mat_vec_q<…, ncols_dst>` launched with 1/2/4/8, and IQ3_S goes through the same `mmvq_launch`. `git merge-tree` onto e2b8ad5: all csrc merges clean; conflicts only in Python (config_parser, loader, quantization/{config,linear,params}, weights_adapter/__init__). It has no `ggml-common.h` change, so rebasing picks up the table fix. |
| 6 | localweights "29 vs 79" | SKIPPED: left to the kernel-survey agent. |
| 7 | Secrets | VERIFIED clean for this repo (excluding .venv/build) and the handoff dir: no hits for HF/OpenAI/GitHub/AWS tokens, private keys, `password=`, `api_key=`, Bearer, or emails; the git history (`git log --all -p`) has none either. The prod vLLM process has no `VLLM_API_KEY` in its environment, so no literal key could be checked. The only author is `Starwaves1 <…@users.noreply.github.com>`. For an upstream PR from `swift-gsq-rco`, a grep for `garrett`, `Starwaves`, `192.168` finds nothing. `main` (not for upstream) has the LAN IP of the bench VM in STATUS.md and `/home/garrett` paths in env/, PROVENANCE.json and make_hf_config.py. |

Repo reproducibility:
- `.gitignore` covers `.venv/ build/ __pycache__ *.pyc *.so *.egg-info /runs/`. No committed binaries: the largest tracked files are tokenizer.json (12.5 MB), vocab.json and merges.txt (text).
- Fork pinned: the b98f4a9 trailer `git-subtree-split: e2b8ad53…`; prod vLLM overlay `ba05ffab`; gguf-py b11211; CUDA wheel pins in `tools/setup-cuda-toolchain.sh`.
- Build recipe: README.md (commit 30dd497) and item 1 above. It is not one command yet: the venv and overlay steps are manual (DoD 7 gap).
- Remote `plugin-upstream`: `pushurl = no_push`, verified in `git config`. `git fetch` of efschu (FETCH_HEAD only, no remote added) and of plugin-upstream (still `e2b8ad5`) was done for 5b.

## 2026-09-27 evening: Phase B harnesses

Commits `b16541f` (harnesses) and `10d6774` (bootstrap pins `b16541f`). **Nothing here has run on a GPU.** Everything was written and checked on the CPU only.

| Entry point | What |
|---|---|
| `scripts/env.sh` | Shared settings and guards. Port 18090. 18080/18081 are refused with no override. `GSQ_ALLOW_GPU=1` gate. HF config dir must be `hf-config/<name>`, and its chat template must match production's sha256. The fs tier root must not be production's `/mnt/kvcache/tier` or anything under it. Production counts as live if :18080/:18081 listen or a production-venv vLLM runs. |
| `scripts/serve-gsq.sh`, `serve-baseline.sh` | Production's argv from `env/prod-serve-argv.txt`. They differ only in venv, model, `--hf-config-path`/`--tokenizer` (gsq), port, host 127.0.0.1, `--chat-template` (repo copy, same bytes) and fs tier `root_dir`/`max_bytes`. Optional: `GSQ_CPU_TIER_BYTES`, `GSQ_MAX_MODEL_LEN`. Plugin on/off via `VLLM_PLUGINS`. The argv diff is printed before launch; `--dry-run` stops there. Both refuse to start while production is live (`GSQ_ALLOW_LOCAL_BASELINE=1` / `GSQ_ALLOW_BESIDE_PROD=1` override). |
| `tests/gpu/` | Kernel parity per quant type on real GGUF rows (dequant, MMVQ 1–16 tokens, MMQ 16–512, production routing on whole tensors). OOB/contiguity/alignment/CUDA-graph cases, each in a subprocess (optional compute-sanitizer). 200k fit. MTP acceptance. 326 tests; all skip without `GSQ_ALLOW_GPU=1`. Shared constants are in `gsq_gpu.py`, not conftest, because `tests/cpu` has its own conftest. |
| `bench/parity/` | `prompts.py`: 11 salted sequences, 1k–120k tokens (two ≥100k), built from the pinned venv's vLLM source. Fingerprint `eb7ac4b4…` is locked in `prompts.lock.json`. `llama_logits.cpp` + `build.sh` (b11211 libllama; `--build-llama` for a cloud box). `vllm_logprobs.py`: in-process, full-vocab logprob probes via prefix cache. `compare.py`: mean KLD and top-1, overall and ≥100k only. `run.sh` ties them together. |
| `bench/speed/` | `run.sh gsq\|baseline [--start]`: runs production's `bench/run_benchmarks.sh single` twice, verbatim and sha256-pinned, from a staging dir with a `venv` symlink to `.venv`, so production's venv is never executed. Adds a salted prefill ladder: 8k/64k at c1+c2, 180k at c1. Records MTP per-position acceptance from `/metrics`. `mtp_acceptance.py` (vLLM `/metrics` or llama-server `timings.draft_n*`) and `serve-llamacpp.sh` (:18091) give the llama.cpp reference. |
| `bench/soak.sh`, `soak_load.py` | 24 h at c2 with production's CUDA-graph argv. Mix: chat, tool calls, long new and prefix-hit prompts up to 120k, aborted streams, greedy, random priorities. Every 60 s it logs liveness, GPU MiB of the server's session, RSS and gauges, and greps the log for faults. A dead server is not restarted unless `GSQ_SOAK_RESTART=1`. Report thresholds: 256 MiB GPU / 1 GiB RSS growth after the first hour. |
| `cloud/bootstrap.sh` | For an sm86 box: preflight checks, clone (URL or `git bundle`) at the pin, venv from `env/prod-freeze.txt`, production's `deploy-vllm.sh` overlay of `ba05ffab`, gguf-py b11211, `tools/build-plugin.sh`, freeze check, GGUF (sha256-checked). Then the llama.cpp build, llama MTP reference, tests, parity (bf16 KV, then fp8 for information), speed for gsq and baseline, optional `--soak`, then tar + rsync. `--baseline prod` (default) uses HF `Starw1/Qwen3.8-27B-absolute-heresy-W4A16`, sha-checked against production's config, quantization config and index. `--baseline swift` needs `GSQ_BASELINE_SRC` (the `-prepared` dir; not on HF) and runs with `--max-model-len -1`. `--dry-run` prints every command. |

How it was validated (all on the CPU):
- `bash -n` and shellcheck (from the shellcheck-py wheel, installed to /tmp and since deleted) on every script: clean.
- `py_compile` on every .py.
- `serve-*.sh --dry-run`. Every guard was triggered on purpose: ports 18080/18081, production's model dir as the HF config dir, production's tier root and a subdir of it, an unwritable tier root, no GPU gate, and baseline while production is live.
- `bootstrap.sh --dry-run` end to end, plus its argument errors.
- `g++ -fsyntax-only -Wall -Wextra` of `llama_logits.cpp` against b11211 headers: clean.
- `prompts.py --dry-run` twice gave the same fingerprint.
- The MTP `/metrics` parser was checked on synthetic counters.
- `soak_load.py report` on a synthetic 25 h run: passes, and fails on an injected IMA line.
- Reference math on real rows, VERIFIED: the Q4_K min-term split (zeroed quant bits → `dmin·m`) is block-constant and reproduces q81 exactly, and the q8_1 rounding is half-away-from-zero.
- All Python ran under `GSQ_LIGHT=1 tools/capped` with `no_gpu` imported. The one exception was a first bytecode-only `py_compile` pass, later redone capped.

Assumptions to confirm on the first GPU run (they are also TODOs in the code):
- Plugin API: `ops.ggml_dequantize(W_uint8[rows, bytes], type, rows, cols, dtype)` accepts float32 output. `ggml_mul_mat_vec_a8` / `ggml_mul_mat_a8(W, X, type, rows)` return `[n, rows]` in X's dtype. `_fused_mul_mat_gguf(x, W, type)` is importable from `quantization/linear.py`.
- Tolerances: TIGHT 4e-3 (bf16) / 1.5e-3 (fp16) against the best of q81/xsum/wround; LOOSE 5e-2 against full precision. On the CPU, the x-sum model already sits 1.5e-2 from full on Q4_K. Dequant fp16/bf16 is compared with atol 0 and may need 1 ulp.
- Guard tests are expected to fail at e2b8ad5: no checks exist. They are the acceptance test for the guards. Record which cases fault.
- Serving: `--hf-config-path` + `--tokenizer` on the hf-config dir is enough for the plugin's model_config.model/tokenizer. `LLM(max_logprobs=-1)` with `logprobs=-1` returns every vocab id in 0.27.1, and the probes hit the prefix cache in align mode.
- The KV capacity log lines match `kv_cache_utils.py:2389-2391`.
- llama-server b11211 reports `timings.draft_n`/`draft_n_accepted` with `--spec-type draft-mtp`.
- Cloud:
  - llama.cpp's CUDA build needs a system nvcc + cuBLAS; `build/cu130` has no cuBLAS dev files.
  - The deploy repo commit `2138d1ae` must be on GitHub (not checked).
  - `deploy-vllm.sh --init v0.27.1` must work on a fresh wheel, as it did here.

Defaults I chose, all visible in the argv diff:
- Host 127.0.0.1.
- A local test API key (`gsq-local-test`), never production's.
- fs tier at `/mnt/kvcache/gsq-tier`, capped at 100 GB. On this host it shares the drive with production's 300 GB tier; delete it after use.
- Parity runs vLLM with bf16 KV against llama.cpp f16 KV, so it measures the weights and kernels. fp8 is a separate informational run.
- Parity dumps are about 3 GB per engine and are deleted unless `KEEP=1`.
- The production environment variables come from this file's "Production's start script" notes, not from reading production's process environment.

STATUS.md itself is not committed by me: it held other agents' uncommitted sections.

## 2026-09-27 evening: CPU tests

Commit `27deeb5`. No GPU use: every run imported `no_gpu`, and the dry run also blocks torch CUDA init (it did catch one attempt, in `QwenGatedDeltaNetAttention` via `current_platform.current_device()`, now stubbed to meta). Run: `GSQ_LIGHT=1 tools/capped tools/pytest tests/cpu` → **416 passed, 54 xfailed**, 13 s, max RSS 1.2 GB.

- **Fixtures** (`tools/make_dequant_fixtures.py`, `tests/fixtures/dequant/`, 1.4 MB, committed): 84 real rows, 10 types × up to 3 tensors × 3 rows, schema as fixed above. Every gguf-py ref equals b11211 `libggml-base` `to_float` bit for bit (`tools/ggml_ref.py`, ctypes).
- **Plugin dequant vs gguf-py** (`test_dequant_fixtures.py`):
  - Triton kernels, run with `TRITON_INTERPRET=1` on CPU tensors: fp32 output bit-exact on all 84 rows, all 10 types. No bf16 check: the interpreter truncates fp32→bf16, compiled Triton rounds to nearest even.
  - CUDA `dequantize.cuh`, compiled for the host by `tools/cuda_dequant_host.py` (the unmodified source; `<<<>>>` launches rewritten to serial loops; `half` = `_Float16`, one RNE rounding per intrinsic). All 7 IQ types: bit-exact in fp32, and bf16 output = RNE(ref). Equality with the real sm86 build is INFERRED (fast-math, but there are no mul+add pairs or denormals on these paths).
  - **Finding: K-quant CUDA dequant is not bit-exact.** vLLM's b2899 copy does Q2_K/Q4_K/Q6_K in fp16 (`__hmul`/`__hsub` on `half`, and Q6_K's `__int2half_rn(sc*q)` rounds products above 2048), while ggml computes in fp32. In bf16 output, 16% (Q2_K), 28% (Q4_K) and 7% (Q6_K) of elements differ from RNE(ggml) by 1 ulp; max error ≤ 3.7e-3 of the row's absmax. Marked strict xfail, plus a bound test at 2⁻⁹·absmax. Impact on this model: small. linear.py sends K-quants to MMVQ/MMQ, not dequant, and the embedding is IQ2_S. `tests/gpu` dequant at atol 0 will fail for K-quants for this reason.
- **Kernel tables** (`test_kernel_tables.py`, 18 pass): the plugin's CUDA `iq2xxs/iq2xs/iq2s/iq3xxs` grids, `ksigns*`, `kmask`, `kvalues_iq4nl` equal b11211. `iq3xs_grid` = exactly 4 × b11211 `iq3s_grid` (all 512 entries; max 60 fits int8). `iq1s_grid_gpu` (uint64) fits 32 bits, equals b11211's uint32 table, and nibble-decodes to gguf-py's IQ1_S grid. The Triton tables equal gguf-py.
- **GDN round trip** (`test_gdn_roundtrip.py`, 4 pass): b11211 `Qwen3_5TextModel.modify_tensors` (via `__new__`, Swift's GDN dims) → plugin name map + `transform_weights` gives back every HF tensor exactly (A_log within 1e-6): qkv/z/a/b/conv1d/dt_bias/out_proj, the +1/−1 norms (not `linear_attn.norm`), q/k norms, and the MTP enorm/hnorm/shared_head norm/fc. Quantized out_proj stays GGML-tiled, and `input_to_gguf(x) @ W_gguf.T == x @ W_hf.T` exactly.
- **Meta dry run** (`tools/meta_dry_run.py`; 3 GB cap, peak RSS 2.0 GB, ~2 min): **PASS.** Real `GGUFModelLoader.load_model` + vLLM `Qwen3_5ForConditionalGeneration`, then `Qwen3_5MTP`. The platform is a NonNvmlCudaPlatform stub (sm86), with gloo world size 1, `load_general_plugins()`, `MTP_DRAFT_VOCAB=0` and fp8 KV + MTP k=3 config. The weight iterator yields meta tensors of the stored shape and dtype, so only the header is read (the plugin's own iterator does `torch.tensor(memmap)`, a full copy per tensor).
  - 866 = 851 main + 15 MTP; none unmapped, none overlapping.
  - Main: 917/917 params loaded. MTP: 17/19 (the missing `embed_tokens`/`lm_head` are shared from the target by design).
  - 258 GGUF layers, 498 shards: every stored logical (rows, cols) equals the partition size. GDN `in_proj_qkvz` loads as shards 0–3 (qkv split into q/k/v); 91 layers mix quant types across shards.
  - 48 `out_proj` layers have `GGUFHeadTilingLayout(3, 128)`.
  - One dry-run-only patch: `params._store_gguf_weight_type` calls `.item()` after moving to `param.device`, which fails on meta, so the script reads the value on the CPU first. That's harmless on a GPU (one sync per tensor).
- MemAvailable stayed ≥ 9.68 GB throughout. Scratch in /tmp removed. `tools/pytest` now passes `-p no:cacheprovider` (no `.pytest_cache` in the repo).
