# 12: MTP draft config from the HF config source on vLLM >= 0.29

Branch: `upstream/12-mtp-draft-config` (2 commits, on e2b8ad5). No dependency.

**Title:** [BugFix] Build the MTP draft config from the HF config source on vLLM >= 0.29

## Motivation

For `method == "mtp"` without a draft model, vLLM >= 0.29 (vllm#42079; seen at 0.30.1rc1.dev285,
`vllm/config/speculative.py`) sets the draft's `model` to `target_model_config.model_weights` when that
is set, to keep runai_streamer's weight source:

```python
self.model = (
    self.target_model_config.model_weights
    or self.target_model_config.model
)
```

For a GGUF target the plugin sets `model_weights` to the `.gguf` path, so the draft config is read from
the `.gguf`: there is no config.json beside it and engine creation fails with "Unrecognized model".
vLLM 0.27.1 used `target_model_config.model`, the HF config source the plugin resolves.

## What changed (plugin variant, this branch)

- `plugin.py`: the `create_speculative_config` wrapper blanks the target's `model_weights` while vLLM
  builds the speculative config for a GGUF target with no configured draft model, and restores it in a
  `finally`; the existing line then points the draft's `model_weights` at the `.gguf`. One code path
  for both vLLM versions.
- `tests/test_plugin.py`: a test with `SpeculativeConfig` stubbed to the newer rule; it fails without
  the fix (the draft's model is the `.gguf`).

## How tested

- `pytest tests --ignore=tests/test_kernels.py --ignore=tests/test_gguf_generation.py`: 105 passed, 6
  skipped on vLLM 0.27.1 and vLLM main.
- Without the fix, a CPU meta-device engine build of a 27B Qwen3.5-architecture GGUF with MTP on vLLM
  main fails in `create_engine_config` ("Unrecognized model"); with it the build passes on 0.27.1 and
  main and the draft loads from the HF config directory with its weights from the `.gguf`. (Those
  builds had an unrelated out-of-tree vLLM patch set installed.) The same wrapper served the 27B GGUF with MTP on
  vLLM main in production on 2026-10-01.

## Alternative: fix it in vLLM core

The plugin variant works around a vLLM rule that conflates "where the weights are" with "where the
config is". A core fix would keep the draft's config source and weights source apart, for example:

> **[Bugfix] Speculative MTP: take the draft config from `model`, the weights from `model_weights`**
>
> For `method="mtp"` with no draft model, `SpeculativeConfig` sets the draft's `model` to
> `target_model_config.model_weights or target_model_config.model`. `model_weights` is a weights
> location (runai_streamer's original path, or a GGUF file for the GGUF plugin), not a config source: a
> GGUF target's draft then fails to build ("Unrecognized model"). Set the draft's `model` from
> `target_model_config.model` and pass `model_weights` through to the draft's model config, so the
> draft loads its weights from the same source as the target (what the runai_streamer change needed)
> while its config comes from the same place as the target's.

If that lands, this plugin change becomes unnecessary for new vLLM and harmless for old.
