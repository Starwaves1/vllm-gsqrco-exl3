# gsq-vllm plugin on production's vLLM main: CPU-side preparation

2026-09-30, on the production host, no GPU. Follows steps 1-4 of the "recommended order" in
`vllm-main-compat.md`. "old" = `.venv` (vLLM 0.27.1 + overlay ba05ffab), "main" = `.venv-main` (vLLM
0.30.1rc1.dev285+gd28795f1a + overlay 2a0fe5e1e1, production's current stack). All python ran with
`tools/no_gpu.py` imported first, under `tools/capped` (MemoryMax 6G or 2G, CPUQuota 200% or 100%, nice 19,
idle IO, MemAvailable >= 9 GB). The production server, its venv, the deploy repo and `~/vllm` were only read.

## 1. Production state recorded (read-only)

- `env/prod-main-freeze.txt`: `uv pip freeze` of `venv-main/bin/python`, 216 pins. It matches the dist-info
  names in site-packages one to one.
- `env/prod-main-serve-argv.txt`: `/proc/348189/cmdline` (MainPID of qwen-vllm.service, started 17:12:10,
  found via port 18081).
- `env/prod-main-provenance.txt`: deploy repo `vllm-main` e09dee5, wheel URL, overlay base d28795f1a7, overlay
  2a0fe5e1e1 (38 files under `vllm/`), launcher env (`VLLM_USE_V2_MODEL_RUNNER=0` etc.).
- The 0.27.1 files (`prod-freeze.txt`, `prod-serve-argv.txt`, `gsq-freeze.txt`) are unchanged.

## 2. What changed

- **`plugin/vllm_gguf_plugin/plugin.py`** (the fix from the assessment). The `create_speculative_config`
  wrapper blanks `target_model_config.model_weights` while vLLM builds the speculative config, then restores
  it in a `finally`. This applies only to a GGUF target with no configured draft model. On main the MTP draft
  now gets `model = <hf-config dir>` as on 0.27.1, and the existing line still sets the draft's
  `model_weights` to the `.gguf`. 0.27.1 never reads `model_weights` there, so one code path serves both.
- **`plugin/tests/test_plugin.py`**: `test_mtp_draft_config_comes_from_hf_config_dir` mocks
  `SpeculativeConfig` with main's rule (`model_weights or model`). Without the fix it fails with
  `'/tmp/model.gguf' == '/tmp/hf-config'`; with the fix it passes on both venvs.
- **`plugin/setup.py`**: with `VLLM_GGUF_BUILD_LCPP=1`, the llama.cpp sources were absolute paths. The editable
  install (`tools/build-plugin.sh`, the README command) compiled all 17 objects and then failed with
  "setup script specifies an absolute path". Only `setup.py build_ext --inplace`, used on the rented box,
  worked. The sources are now relative; include dirs stay absolute.
- **`tools/build-plugin.sh`, `tools/pytest`**: take the venv from `GSQ_VENV` (default `.venv`, as before).
- **`tools/meta_dry_run.py`**:
  - Engine args now follow `env/prod-main-serve-argv.txt`: MTP k=5, `draft_sample_method` probabilistic,
    `num_speculative_tokens_per_batch_size` [[1,4,5],[5,8,3],[9,16,2]], `max_num_seqs` 16,
    `max_num_batched_tokens` 2048, cudagraph capture 48 with production's custom_ops, fp8 KV, fp16 mamba
    state, prefix caching, `mamba_cache_mode` align, priority scheduling, long-prefill threshold 128, and
    `VLLM_USE_V2_MODEL_RUNNER=0`. It no longer uses `enforce_eager`. The KV connector and `max_model_len`
    200000 are not reproduced (4096 is used).
  - It no longer forces `MTP_DRAFT_VOCAB=0`. When the variable is unset, the pruned draft head is on, as in
    production. It then expects MTP 16 tensors (`output.weight` also feeds `mtp.draft_lm_head`, 61,440 rows)
    and an overlap of exactly `{output.weight}`, and it checks `draft.model == hf-config dir` and
    `draft.model_weights == .gguf`.
  - With the head on, the MTP `lm_head` is an empty GGUF placeholder shared from the target, so it counts as
    shared. The shape check reports non-2D shards instead of crashing.
- `.gitignore` gains `.venv-main`. `CONTEXT.md` no longer calls production "0.27.1".

## 3. `.venv-main`

- `uv venv` with CPython 3.12.13, then `uv pip install --link-mode=copy --no-deps --offline
  --extra-index-url https://flashinfer.ai/whl/cu130 --index-strategy unsafe-best-match -r
  env/prod-main-freeze.txt`. Every package came from the uv cache in 19 s, with no download. The FlashInfer
  jit-cache wheels are cached under that index, so plain `--offline` does not find them.
- `uv pip freeze` of the new venv is identical to `prod-main-freeze.txt`. `build_backend.py` is the same as
  production's, and the files are real copies (link count 1).
- Overlay: production's `scripts/deploy-vllm.sh`, run with `SITE_PACKAGES=.venv-main/...` and `STATE_FILE`
  and `BACKUP_ROOT` in /tmp. `--init d28795f1a7` verified the fresh wheel on 3182 tracked files. The deploy
  of 2a0fe5e1e1 wrote 38 files, and `--verify` passed on all 3184. The script prints a restart line; it was
  ignored and nothing was restarted. A checksum `rsync -rcn` against production's site-packages differs only
  in 43 `dist-info/RECORD` files (the venv path in entry-point shebangs), and neither side has extra files.
- gguf-py from `~/llama.cpp-b11211/gguf-py` (0.19.0). `env/gsq-main-freeze.txt` = prod-main-freeze plus gguf
  and the plugin.
- Plugin: `GSQ_VENV=.venv-main VLLM_GGUF_BUILD_LCPP=1 tools/capped tools/build-plugin.sh` took 133 s at
  2 jobs. It is an editable install whose finder maps to `plugin/vllm_gguf_plugin`. The `.so` is built in
  place (33.8 MB, 16 cubins, all sm_86). NEEDED: libc10, libtorch_cpu, libtorch_cuda, libcudart.so.13.
  No file under `.venv/` changed.
- Under no_gpu, all 15 `_C_gguf` ops register in both venvs (6 base + 9 Route L), and torch never
  initializes CUDA.
- Size: `.venv-main` 8.3 GB.

The in-place `.so` is shared: the old `.venv` also has an editable install of `plugin/`, so it now loads the
same Route L build instead of the stale Sep 27 one, which had no `lcpp_*` ops. torch is the same build in
both venvs.

## 4. Test results

| | old `.venv`, stale Sep 27 `.so` | old `.venv`, new `.so` | `.venv-main` |
|---|---|---|---|
| `tests/cpu` | 564 passed, 54 xfailed, **112 errors** | 676 passed, 54 xfailed | 676 passed, 54 xfailed |
| `plugin/tests` CPU set | 87 passed, 1 skipped | 87 passed, 1 skipped | 87 passed, 1 skipped |
| meta dry run, draft head on | PASS | PASS | PASS |
| meta dry run, `MTP_DRAFT_VOCAB=0` | PASS | PASS | PASS |

- The `plugin/tests` CPU set is the one the box scripts run: `test_plugin.py`, `diffusion/`,
  `test_gemma4_adapter.py`, `test_gguf_utils.py`, `test_ggml_common_tables.py`, with `-p no_gpu`. The skip is
  `vllm_omni`, which is not installed.
- The 112 errors are the Route L guard tests hitting a `.so` without `lcpp_*` ops. The README says they skip
  in that case, but they only skip when the `.so` is missing.
- Dry run, both venvs, head on:
  - Config: k=5 with the schedule, 16 seqs, capture up to 48; cudagraph FULL_AND_PIECEWISE becomes PIECEWISE,
    as in production's journal. The draft model is the hf-config dir and its `model_weights` is the `.gguf`.
  - main: 851 tensors mapped, 917/917 params loaded, 258 layers / 498 shards with no shape errors, 48 GDN
    layouts.
  - MTP: 16 tensors mapped, params 22, loaded 19 (+3 shared), 0 missing, 0 shape errors. That includes
    `draft_lm_head` at 61,440 rows.
  - Totals: 866 tensors, 0 unmapped, overlap = {output.weight}.
  - Peak RSS 1.9 GB (main) and 2.0 GB (old); about 130 s.
- Head off: MTP 15 tensors, 17/20 loaded (+3 shared), overlap 0, PASS.
- The fix is load-bearing: on `.venv-main` without it, the dry run fails in `create_engine_config` with
  pydantic `ValidationError: Unrecognized model in .../Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF`. That is break 1
  of the assessment. Break 2 (draft ids looked up next to the `.gguf`) cannot happen now: the draft
  `model` is the hf-config dir, and the head-on dry run loads `draft_lm_head` from there.

## 5. Left for the GPU phase

1. Harness switch for main:
   - `GSQ_VENV=.venv-main`, `GSQ_PROD_ARGV=env/prod-main-serve-argv.txt`, and `VLLM_USE_V2_MODEL_RUNNER=0` in
     the serve environment. `scripts/env.sh` does not export it yet, because on 0.27.1 the unset default is
     left alone.
   - `cloud/bootstrap.sh`: DEPLOY_COMMIT e09dee5, VLLM_COMMIT 2a0fe5e1e1, `--init d28795f1a7`, the prod-main
     freeze and the FlashInfer index.
   - Wipe `/mnt/kvcache/gsq-tier` first (the fs-tier layout changed).
2. Smoke: `serve-gsq` with MTP on main (chat, reasoning, tool call). The log should show the 61,440-row draft
   head.
3. `tests/gpu/test_mtp_acceptance.py` and `test_fit_200k.py` under the new argv (k=5 schedule, 16 seqs,
   capture 48, PIECEWISE); the KV fit changes with 16 seqs.
4. Re-benchmark in ms/step, GGUF against production's W4A16 baseline on main with the same argv. REPORT.md's
   numbers are 0.27.1, k=3, 8 seqs, FULL_AND_PIECEWISE. Production now serves
   `Qwen3.8-27B-TT709-W4A16-AutoRound-fast` with a 40,960-row draft list, while the GGUF uses Swift's
   61,440-row list.
5. Soak.
