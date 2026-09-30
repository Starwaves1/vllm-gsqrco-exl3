# gsq-vllm plugin vs production's vLLM main: compatibility assessment

2026-09-30, read-only (no GPU, no builds, nothing executed against either venv). Plugin at gsq-vllm
main 3302e32. "old" = this repo's `.venv` (vLLM 0.27.1 + overlay ba05ffab, deploy 2138d1ae8d);
"new" = `~/qwen38-27b-rtx3090/venv-main`. Labels: VERIFIED (read in source/proc/journal),
INFERRED (reasoned from source, not run).

## 1. What production runs now (VERIFIED)

- **vLLM**: `0.30.1rc1.dev285+gd28795f1a`, upstream wheel
  `https://wheels.vllm.ai/d28795f1a7af4e3ce2530d4f0bdaec4ecbede693/vllm-0.30.1rc1.dev285%2Bgd28795f1a-cp38-abi3-manylinux_2_28_x86_64.whl`
  (from `direct_url.json`), installed by uv into `~/qwen38-27b-rtx3090/venv-main`. `venv -> venv-main` is a
  symlink (cutover 2026-09-30 16:42); `venv-0271/` holds the old one.
- **Patch overlay**: no longer `patches/*.patch`. `scripts/deploy-vllm.sh` copies the files that differ
  from the wheel's base out of `~/vllm` branch `qwen38/main` (BASE_REF d28795f1a7). Deployed commit
  `2a0fe5e1e1` ("write-back: keep up to 256 MB of chunk jobs in flight", #7): 25 commits over upstream,
  38 files under `vllm/`. It carries the two plugin-relevant patches: `9e334ccca1` quant_config for the
  embedding table (main + MTP) and `ff58c3014d` the vocab-truncated MTP draft head ("syv patch",
  `qwen3_5_mtp.py`, still keyed on `model_config.model/mtp_draft_vocab_ids.pt`). Not carried over from
  `qwen38/v0.27.1` (upstream has them, or dropped): the #5 "GDN state only at kept chunks" knob, the
  #51812/#53046/#57104/#52805/#52771 backports, KVarN, DFlash2, the hybrid-KV/SW-promote pair.
- **Deploy repo**: branch `vllm-main`, HEAD `e09dee5` (it moved from 966f6c4 during this read). 18 commits
  after 2138d1ae8d, all deploy plumbing: `deploy/override.conf.{0271,vllm-main}`,
  `scripts/cutover-vllm-main.sh`, `scripts/deploy-vllm.sh` (venv-main + BASE_REF/EXPECT_VERSION),
  launchers (`VLLM_USE_V2_MODEL_RUNNER=0`, `SPEC_SCHEDULE`, `SPEC=none`), `verify.sh`.
  `patches/` and `patches-local/` are unchanged.
- **Live argv** (systemd MainPID 348189, started 17:12) vs `env/prod-serve-argv.txt`:

  | arg | env/prod-serve-argv.txt | live |
  |---|---|---|
  | interpreter | `venv/bin/python` | `venv-main/bin/python` (script is still `venv/bin/vllm`) |
  | model | `Qwen3.8-27B-W4A16-AutoRound-fast` | `Qwen3.8-27B-TT709-W4A16-AutoRound-fast` (drop-in `MODEL=`; an earlier run today served `-fast`) |
  | `--max-num-seqs` | 8 | 16 |
  | `--speculative-config` | mtp, k=3 | mtp, k=5, `num_speculative_tokens_per_batch_size` `[[1,4,5],[5,8,3],[9,16,2]]` |
  | `--compilation-config` | `max_cudagraph_capture_size` 32 | 48 |
  | `--kv-transfer-config` | has `"mamba_keep_every_n_chunks":8` | key removed (main's retention default) |
  | env | – | `VLLM_USE_V2_MODEL_RUNNER=0` (new); `PYTHONHASHSEED=0` and `VLLM_USE_FLASHINFER_SAMPLER=0` as before |

  Everything else matches. Effects in the journal: the dynamic k schedule forces
  `cudagraph_mode` FULL_AND_PIECEWISE -> **PIECEWISE** on the V1 runner; KV 203,112 tokens (1.02x at 200k).
- **Attention**: FLASHINFER (chosen from FLASHINFER, TRITON_ATTN), `decode_backend=flashinfer-native`, fp8
  e4m3 KV, sm86; GDN prefill Triton/FLA and GDN decode Triton; ViT FLASH_ATTN; "MTP drafter uses a
  40960-token draft head"; pageable FlashInfer plan buffers on.
- **FlashInfer**: `flashinfer-python 0.7.0` + prebuilt `flashinfer-jit-cache(-sm80)==0.7.0+cu130` (old venv
  had 0.6.16.post3 without a jit cache). This host's FlashInfer JIT is broken (no CUDA 13.0 toolkit), so any
  kernel outside the prebuilt set kills the engine. The jit-cache version must equal flashinfer-python.
- **torch/CUDA**: unchanged. Both venvs have `torch 2.13.0+cu130` with the same git_version (cf30153c),
  `nvidia-cuda-runtime 13.0.96`, `triton 3.7.1`, `transformers 5.15.1`. The pip nvcc/nvvm packages moved
  13.3 -> 13.4, but the plugin builds with its own `build/cu130` toolchain (nvcc 13.0.88).
- **venv-main lacks** `gguf` (0.19.0 from llama.cpp b11211 gguf-py) and `vllm-gguf-plugin`.

## 2. Plugin API surface vs main

| API the plugin uses | Where | Status on main |
|---|---|---|
| `register_quantization_config`, `QuantizationConfig` | `plugin.py`, `quantization/config.py` | OK. New class attrs `requires_device_loading`, `supports_pre_processed_weights`, `online_quantization_config=None`; layers now call `resolve_quant_method()`, which returns `get_quant_method()` unchanged when online quant is off |
| `LinearMethodBase`, `UnquantizedLinearMethod`, `register_weight_loader_v2_supported_method`, `WEIGHT_LOADER_V2_SUPPORTED` | `quantization/linear.py` | OK. `UnquantizedLinearMethod.__init__` now sets `self._gemm_impl`; `GGUFUnquantizedLinearMethod` inherits `__init__`, so `super().apply` works |
| `BasevLLMParameter`, `weight_loader_v2` | `quantization/params.py` | OK (docstrings only) |
| `VocabParallelEmbedding`/`ParallelLMHead`/`UnquantizedEmbeddingMethod` | `quantization/vocal_embeds.py`, `config.py` | OK. New `quant_method=`/`parallel_group=` kwargs; the embedding check is now `not isinstance(self, ParallelLMHead)`; the tp=1 path is unchanged |
| `direct_register_custom_op` | `linear.py`, `vocal_embeds.py`, `fused_moe.py` | Identical. No `torch.ops.vllm` name collisions |
| `torch.ops._C_gguf` via `load_general_plugins` | `ops.py` | OK. The `.so` NEEDs only libc10/libtorch_cpu/libtorch_cuda/libcudart.so.13 (no vLLM `_C`) |
| `BaseModelLoader`, `initialize_model`, `process_weights_after_loading` | `loader.py` | OK. `load_model` is overridden (upstream's new `create_model` is unused). New `maybe_retie_word_embeddings` is a no-op (Swift is untied) and `update_param_tp_status` is harmless |
| `WeightsMapper.apply_list`, `apply_vllm_mapper` | `quantization/config.py` | OK. vLLM now passes `get_rename_mapper()` (drops `None` maps), so the `strict=True` zip over `linear_layouts` stays safe although `Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper` gained `"mtp.": None` |
| `Qwen3_5ForConditionalGeneration` names / `packed_modules_mapping` | `weights_adapter/qwen3_5.py` | OK. Qwen3VL prefix mapper unchanged; `mtp.*` is now dropped by the mapper instead of `skip_prefixes`; `lm_head` is built then `tie_weights`; MTP draft override also accepts `qwen3_5_text` |
| `ModelConfig.multimodal_config`, `MultiModalConfig.get_limit_per_prompt` | `weights_adapter/qwen3_5.py:167` | OK (field still exists) |
| `EngineArgs.create_model_config` wrapper, `maybe_override_with_speculators` (both modules) | `plugin.py` | OK. Same fields and signatures; the call site is unchanged |
| `EngineArgs.create_speculative_config` wrapper + MTP draft config | `plugin.py:_patch_engine_args` | **BREAKS (INFERRED, high confidence)**, see below |
| `ConfigParserBase`/`HFConfigParser`, `list_filtered_repo_files`, `Qwen3_5Config` | `config_parser.py`, `gguf_utils.py` | OK (same signatures) |
| `FusedMoEMethodBase` and friends | `quantization/fused_moe.py` | Imports resolve; abstract methods unchanged. MoE is untested here and not needed for the dense 27B |
| OffloadingConnector / TieringOffloadingSpec args | harness argv only | OK. Unknown `extra_config` keys go through `.get`; the stale `mamba_keep_every_n_chunks` is ignored. The fs-tier on-disk layout changed upstream (the cutover wiped prod's tier), so wipe `/mnt/kvcache/gsq-tier` before the first main run |
| `--hf-config-path` | `scripts/serve-gsq.sh`, `plugin.py` | OK for the target (`ModelConfig` reads `hf_config_path or model`) |

**The break: MTP draft config source.** In 0.27.1, `SpeculativeConfig.__post_init__` set the MTP draft
`model = target_model_config.model`, which the plugin had already rewritten to the hf-config dir. The
plugin then set `draft_model_config.model_weights = <gguf>`. On main it is
`model = target_model_config.model_weights or target_model_config.model`, and the plugin sets
`model_weights` to the `.gguf`, so:

1. The draft `ModelConfig(model=<.gguf>, config_format="gguf")` runs `GGUFConfigParser` on
   `Path(gguf).parent`. That is `~/qwen38-27b-rtx3090/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/`, which
   has no `config.json`, so `PreTrainedConfig.get_config_dict` raises inside
   `original_create_speculative_config`, before the wrapper's fix-up runs. Serving with MTP would fail at
   engine-config time.
2. If a config.json were present there, both the overlay's syv patch and the plugin's `_draft_vocab_ids()`
   (`weights_adapter/qwen3_5.py:399`) would look for `<file>.gguf/mtp_draft_vocab_ids.pt`. The pruned
   40,960-row draft head would silently switch off: the draft would use the full 248k-row head, slower MTP
   with no error.

Minimal fix (proposed, not applied), in `plugin.py` `create_speculative_config`: while the original runs,
blank `target_model_config.model_weights` for a GGUF target with no configured draft model, then restore
it. This reproduces 0.27.1 exactly (draft `model` = hf-config dir). The existing line then sets the draft's
`model_weights` to the `.gguf`. `model_weights` is read only in that MTP branch (`config/speculative.py:1151`),
and 0.27.1 never reads it, so one code path serves both versions. About 6 lines, plus a CPU unit test in
`plugin/tests/test_plugin.py` asserting `draft_model_config.model == hf_config_path` and
`draft_model_config.model_weights == <gguf>`.

## 3. Verdict

- **Code changes**: `plugin/vllm_gguf_plugin/plugin.py` (above) only. No other plugin API is broken for the
  dense Qwen3.5 path.
- **Harness/env changes**: `env/prod-serve-argv.txt` (refresh from the live cmdline), `scripts/env.sh`
  (export `VLLM_USE_V2_MODEL_RUNNER=0`; main defaults to V2 and the qwen38/main stack targets V1),
  `cloud/bootstrap.sh` (DEPLOY_COMMIT e09dee5, VLLM_COMMIT 2a0fe5e1e1, `--init d28795f1a7` from the wheel
  instead of v0.27.1, flashinfer jit-cache), `env/gsq-freeze.txt` (new set). CONTEXT.md's "Production
  stack = 0.27.1" line becomes stale.
- **Rebuild**: not needed for ABI (identical torch build, and the `.so` links only torch/cudart 13). The local
  `plugin/vllm_gguf_plugin/_C_gguf.abi3.so` (Sep 27, no `lcpp_*` symbols) predates Route L/Integration 2
  csrc (5dfb559, Sep 29). Any local deploy needs `VLLM_GGUF_BUILD_LCPP=1 tools/build-plugin.sh` under
  `tools/capped` regardless of the vLLM version.
- **Tests that cover it**: `tools/meta_dry_run.py` (runs `EngineArgs(...speculative_config mtp).create_engine_config()`
  and real loader on meta tensors; catches break 1, **not** break 2 because it forces `MTP_DRAFT_VOCAB=0`),
  `plugin/tests/test_plugin.py` (registration, weight_loader_v2, parser, speculator probe; add the draft test),
  `tests/cpu/*` (kernel tables/IQ3 pack/LCPP routing; no vLLM API). GPU: `tests/gpu/test_mtp_acceptance.py`
  (draft head + acceptance, catches break 2), `test_fit_200k.py` (KV fit changes with 16 seqs/k=5),
  `test_kernel_guards.py`/`test_kernel_parity.py` (torch-only; expected unchanged).
- **Re-baseline**: the REPORT.md numbers are k=3, max 8 seqs, FULL_AND_PIECEWISE on 0.27.1. Prod on main is k=5
  with a per-batch schedule, 16 seqs, PIECEWISE. Re-measure both GGUF and the W4A16 baseline (ms/step) on
  main before comparing.
- **Effort**: about 0.5 day CPU-side (fix + test, venv recreate, plugin rebuild at 2 jobs, meta dry run, CPU
  tests). Then a GPU window of about 0.5 to 1 day (smoke with MTP, acceptance, fit_200k, bench re-baseline).
  Soak separately.

### Recommended order

1. Generate `env/prod-main-freeze.txt` read-only (below) and refresh `env/prod-serve-argv.txt` from
   `/proc/<MainPID>/cmdline`. Record the env deltas.
2. Plugin fix + unit test in `plugin.py` / `plugin/tests/test_plugin.py` (runs in the old venv too).
3. Create a new isolated venv `.venv-main` (keep `.venv` for 0.27.1 reproducibility; `GSQ_VENV` already
   switches it): same packages as prod's venv-main, overlay via `deploy-vllm.sh`, gguf-py, plugin built with
   LCPP.
4. CPU: `tests/cpu`, `plugin/tests`, then `tools/meta_dry_run.py`, once as-is and once with
   `MTP_DRAFT_VOCAB=1` to check the draft map includes `mtp.draft_lm_head.weight`.
5. GPU (Garrett's go or a rented 3090): wipe the gsq fs tier, serve-gsq smoke with MTP, test_mtp_acceptance,
   test_fit_200k, then re-bench GGUF and baseline on the main argv. Then the soak.

### Freeze and venv steps (proposed; nothing below was run)

```bash
# read-only freeze of prod (uv queries the interpreter; imports nothing from vllm; no GPU)
uv pip freeze --python ~/qwen38-27b-rtx3090/venv-main/bin/python > env/prod-main-freeze.txt
# no-python cross-check from dist-info names
ls ~/qwen38-27b-rtx3090/venv-main/lib/python3.12/site-packages | sed -n 's/\.dist-info$//p' | sort

# isolated venv (under tools/capped)
uv venv --python 3.12.13 .venv-main
uv pip install --python .venv-main/bin/python --no-deps -r env/prod-main-freeze.txt \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130   # vllm comes via its wheels.vllm.ai URL in the freeze
VLLM_REPO=~/vllm SITE_PACKAGES=$PWD/.venv-main/lib/python3.12/site-packages \
  STATE_FILE=/tmp/gsq-main-state BACKUP_ROOT=/tmp/gsq-main-bk \
  ~/qwen38-27b-rtx3090/scripts/deploy-vllm.sh --init d28795f1a7   # then: deploy 2a0fe5e1e1, then --verify
uv pip install --python .venv-main/bin/python --no-deps <llama.cpp b11211>/gguf-py
GSQ_VENV=.venv-main VLLM_GGUF_BUILD_LCPP=1 tools/build-plugin.sh   # build-plugin.sh hardcodes .venv: parameterize it
uv pip freeze --python .venv-main/bin/python > env/gsq-main-freeze.txt   # = prod-main-freeze + gguf + plugin
```
