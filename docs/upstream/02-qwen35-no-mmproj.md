# 02: Serve a multimodal-config Qwen3.5 GGUF without mm_proj as text only

Branch: `pr/plugin-qwen35-no-mmproj` (2 commits, on e2b8ad5; supersedes `upstream/02-qwen35-no-mmproj`: code
comment, error message and commit message corrected). No dependency.

**Title:** [BugFix] qwen3_5: serve a multimodal-config GGUF without mm_proj as text only

## Motivation

The official Qwen3.5 / 3.6 HF configs are multimodal (`vision_config`), and text-only GGUFs of these
models ship without an `*mmproj*.gguf`. Serving such a GGUF with its HF config (`--hf-config-path` or
the tokenizer repo) failed in `patch_hf_config` with "Could not find mm_proj". Removing the raise alone
is not enough: `build_name_map` chose the text prefix from the presence of mm_proj, so the text weights
would map to `model.*`, which `Qwen3_5ForConditionalGeneration`'s `hf_to_vllm_mapper`
(`model.language_model.` to `language_model.model.`) does not translate.

Converting the config to the text-only one at load time is not an option either: the architecture was
chosen in the config parser before the loader saw the files, and `Qwen3_5ForConditionalGeneration` needs
`vision_config`. Switching to the text architecture means choosing it in the config parser (+201/-14 in
the closed #120), and `Qwen3_5ForCausalLM` is not registered in vLLM 0.26.0. (On vLLM <= 0.27.x a text
config would also lose the MTP draft; vLLM main derives it from the text model types too.)

## What changed

- `weights_adapter/qwen3_5.py`: keep the multimodal config; take the text prefix from the config
  (`vision_config` present means the `language_model` prefix); when there is no mm_proj, require image
  and video limits of 0 (`--limit-mm-per-prompt '{"image": 0, "video": 0}'` or `--language-model-only`),
  so the vision tower is never built (vLLM marks it missing) instead of being left uninitialized.
  Without those limits the error now says how to serve text only.
- `tests/test_qwen35_adapter.py` (new, CPU): the accepted case (limits 0: `language_model` prefix, the
  vision config kept), the rejected ones (image or video limit above 0, no multimodal config) and the
  unchanged text-config case. The GGUF reads are stubbed.

## How tested

- The regression test: 6 passed; 5 of them fail without the fix.
- `pytest tests --ignore=tests/test_kernels.py --ignore=tests/test_gguf_generation.py`: 110 passed, 6
  skipped on vLLM 0.27.1 and vLLM main.
- A CPU meta-device load of a 27B text-only Qwen3.5-architecture GGUF with the official multimodal
  config and `--limit-mm-per-prompt` 0 / `--language-model-only` maps every tensor and builds the MTP
  draft; the same model has served on GPU with this change since (with MTP).

## Risks

- A multimodal config without mm_proj and without the limits used to fail with one message and now
  fails with another; nothing that loaded before behaves differently.
