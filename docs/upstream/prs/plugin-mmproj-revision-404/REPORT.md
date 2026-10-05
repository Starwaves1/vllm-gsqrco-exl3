# [BugFix] Fetch the mm_proj at the GGUF repo's own revision (stub, candidate)

Target: vllm-project/vllm-gguf-plugin. Status: candidate, not diagnosed past the traceback; no fix, no test.

## Initial problem and concise proof

On vLLM main (d28795f1a7), the README's remote Qwen3.5 example (`unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M --tokenizer Qwen/Qwen3.5-4B`,
seen with `--speculative-config` mtp and the `plugin-mtp-draft-config` fix applied) fails in the target load:

```
httpx.HTTPStatusError: Client error '404 Not Found' for url
'https://huggingface.co/unsloth/Qwen3.5-4B-MTP-GGUF/resolve/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/mmproj-BF16.gguf'
```

`851bf6e8...` is the commit of `Qwen/Qwen3.5-4B` (the config repo), not of the GGUF repo. RTX 3090, 2026-10-05, job
`uprs-gpu-tests` (`out/b12-e2e_mtp.log` on the Vast box, since lost). Stack: `gpu/model_runner.py:379` -> plugin
`loader.py:207 load_model` -> `:162 _prepare_adapter` -> `:148 _prepare_model_files` -> `weight_utils.py:101 download_mmproj`.

## Problem mechanism (hypothesis, to verify)

- vLLM main's `ModelConfig.__post_init__` resolves `self.revision` to a commit of `self.model` (`config/model.py:592`,
  `resolve_revision`). The plugin's `create_model_config` wrapper sets `self.model` to the HF config source, so the pinned
  commit belongs to the config repo.
- `_prepare_model_files` passes `model_config.revision` to `download_mmproj(repo_id, revision=...)` for the GGUF repo
  (`loader.py:145-150` at e2b8ad5): the config repo's commit does not exist there.
- To check: whether the backbone download avoids this (it got past `_prepare_weights`), whether the no-MTP form fails the
  same way (expected; the check never ran), and whether `resolve_explicit_mm_proj` has the same problem.

## Fix mechanism

Not written. The obvious candidate is to resolve the mm_proj at the revision the user gave for the GGUF (or `None`), not at
`model_config.revision`.
