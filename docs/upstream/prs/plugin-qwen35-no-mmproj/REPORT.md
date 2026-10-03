# [BugFix] qwen3_5: serve a multimodal-config GGUF without mm_proj as text only

Target: vllm-project/vllm-gguf-plugin. Branch `pr/plugin-qwen35-no-mmproj` (2 commits on e2b8ad5, upstream `main`
as of 2026-10-03): `upstream/02-qwen35-no-mmproj` with the review's code-comment fix, the error message naming
`--language-model-only`, and the commit message corrected. Description: `docs/upstream/02-qwen35-no-mmproj.md`.
Related upstream: #120 ("Serve Qwen3.5/3.6/3.8 text-only GGUF without an mm_proj", closed unmerged 2026-09-17).
It took the other route, rebuilding the model as its text architecture, and needed +201/-14 across 8 files,
because the architecture is chosen in the config parser, before the loader knows whether an mm_proj exists.
It has one CHANGES_REQUESTED review (2026-09-04): `Qwen3_5ForCausalLM` is not registered in vLLM 0.26.0, so
that route needs a minimum-version bump. This branch keeps `Qwen3_5ForConditionalGeneration`, which 0.26.0
already has, and 0.26.0 already skips the tower at limit 0. No open PR or issue covers this.

## Initial problem and concise proof

The official Qwen3.5 / 3.6 / 3.8 HF configs are multimodal (`vision_config`), and text-only GGUFs of these models
ship without `*mmproj*.gguf`. Serving one with its official config (`--tokenizer` / `--hf-config-path` pointing
at the HF repo) stops in `Qwen35GGUFAdapter.patch_hf_config`:

```
RuntimeError: Could not find mm_proj for multimodal Qwen3.5/3.6 GGUF. Place *mmproj*.gguf beside the backbone
or pass model_loader_extra_config={'mm_proj': ...}.
```

That is `weights_adapter/qwen3_5.py:210-215` at e2b8ad5, which raises on `has_vision and files.mm_proj is None`.
The branch's `tests/test_qwen35_adapter.py` has 6 cases. On e2b8ad5's adapter, 5 fail and 1 passes (the unchanged
text-config case); on the branch all 6 pass (re-run 2026-10-03). The CPU suite gives 110 passed, 6 skipped. The
fix has served the 27B GGUF on GPU since 2026-09-27 (development commit 7794689), with MTP on vLLM 0.27.1 and
vLLM main.

## Problem mechanism

- Deleting the raise is not enough. `build_name_map` picks the text-weight prefix from the presence of mm_proj
  (`is_multimodal = files.mm_proj is not None`, `qwen3_5.py:234`), so without one the text weights map to
  `model.*` (`qwen3_5.py:117`). The model class is still `Qwen3_5ForConditionalGeneration`, because the
  config is still multimodal. Its `hf_to_vllm_mapper` only translates `model.language_model.` to
  `language_model.model.` (vLLM `models/qwen3_vl.py:1819-1825`), so none of the text weights would load.
- With the raise deleted but no other change, the vision tower would also be built and left uninitialized,
  because no mm_proj weights exist for it.
- vLLM already skips building a tower when every modality it serves has a limit of 0: `_mark_tower_model`
  (`models/interfaces.py:337-372`) replaces it with a `StageMissingLayer`.
  `MultiModalConfig.get_limit_per_prompt` returns 0 for every modality under `--language-model-only`
  (`config/multimodal.py:514-527`).

Verified against e2b8ad5 and vLLM d28795f1a7. Two corrections to the branch description: vLLM main (since #50734,
2026-08-10) also builds the Qwen3.5 MTP draft from the text-only model types, so "converting to the text config
loses MTP" is true only up to 0.27.x. And the reason the text-architecture route is larger is where the
architecture gets chosen, as #120 shows. That route is not wrong.

## Fix mechanism

The adapter keeps the multimodal config and takes the text prefix from the config (`vision_config` present means
the `language_model` prefix) rather than from mm_proj. With no mm_proj, `build_name_map` requires image and video
limits of 0 (`--limit-mm-per-prompt '{"image": 0, "video": 0}'` or `--language-model-only`), so vLLM skips the
tower instead of leaving it uninitialized. Without those limits it raises, and the message now says how to serve
text only (the message names both flags). +23/-8 in `qwen3_5.py` (+93/-8 with the test).

Nothing else changes. A multimodal GGUF with an mm_proj, or a text config, takes the same path as before. The
only behaviour change is for an input that failed before: it still fails without the limits, with a message
that names the fix.

## Final check agent's concise opinion

Fable 5.1 /check, 2026-10-03: **ready with edits (two text corrections, one code comment; no logic change)**.
Red/green is real: the e2b8ad5 adapter fails 5 and passes 1, the branch passes 6/6, the suite gives 110 passed /
6 skipped, and ruff is clean. The cited lines check out. `Qwen3_5ForConditionalGeneration`'s mapper has no
translation for `model.*`. The tower becomes a `StageMissingLayer` only when the limits for all of image and video
are 0. The MTP correction is right: main derives the draft from `qwen3_5_text`, and v0.27.1 only from
`qwen3_5`/`qwen3_5_moe`. Smaller fixes do not work. The architecture is fixed in `config_parser.py:43`, before
`loader.py:164`. The model class reads `vision_config` at init. Limits changed at load time cannot reach the
frontend processor in the other process, so raising is correct. MoE, text config with mm_proj, and remote repos
are unchanged. v0.26.0 also has tower-skip and `language_model_only`, so this route works where #120's did not.
Edits applied: #120's CHANGES_REQUESTED review (0.26.0 lacks `Qwen3_5ForCausalLM`) is now cited; the stale MTP
code comment and the motivation were replaced (new branch `pr/plugin-qwen35-no-mmproj`); the
`interfaces.py:337-372` cite and the line count were fixed; the error message now names `--language-model-only`.

## Fable's comment

pending
