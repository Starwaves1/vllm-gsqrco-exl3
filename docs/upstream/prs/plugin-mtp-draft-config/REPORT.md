# [BugFix] Build the MTP draft config from the HF config source on vLLM >= 0.29

Target: vllm-project/vllm-gguf-plugin. Branch `upstream/12-mtp-draft-config` (2 commits on e2b8ad5, upstream
`main` as of 2026-10-03). Description: `docs/upstream/12-mtp-draft-config.md`.
Related upstream: the cause is vllm-project/vllm#42079 (merged 2026-08-21; v0.28.0 was cut before it, so v0.29.0 is the first release with it).
vllm#53214 (open) adds `SpeculativeConfig.hf_config_path`, and plugin #117 (open, draft) builds on it to remove
the workarounds for an explicitly configured GGUF draft. Neither covers the case here, an MTP draft with no draft
model configured. Plugin #141 (open, large kernel PR) edits the same wrapper for an explicit `.gguf` draft and
leaves this case as it is. Plugin #70 (open) would move the config source to `hf_config_path` and keep `model`
as the `.gguf`. With it, neither `model` nor `model_weights` is a config source, so the draft would need
vllm#53214's draft `hf_config_path` and this wrapper would no longer be enough. No open PR or issue reports this bug.

## Initial problem and concise proof

On vLLM >= 0.29, a GGUF target with `--speculative-config '{"method": "mtp", ...}'` and its config taken from
`--tokenizer` / `--hf-config-path` (how GGUF repos without a `config.json` are served) fails at engine creation. The plugin README's own MTP
example (README.md:68-70 at e2b8ad5: `vllm serve unsloth/Qwen3.5-4B-MTP-GGUF:Q4_K_M --tokenizer Qwen/Qwen3.5-4B
--speculative-config ...`) is this case: that repo has no `config.json`.
CPU meta-device engine build of the 27B GGUF on vLLM main without the fix (`cloud/results/vllm-main-compat-cpu.md`
section 4):

```
create_engine_config -> pydantic ValidationError: Unrecognized model in .../Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF
```

With the fix the same build passes on 0.27.1 and main. Production served this GGUF with MTP on vLLM main from
the development branch, which carries the same wrapper, on 2026-10-01. The branch test stubs
`SpeculativeConfig` with main's rule. It fails without the fix (`- /tmp/hf-config  + /tmp/model.gguf`) and
passes with it; the CPU suite gives 105 passed, 6 skipped (re-run 2026-10-03).

## Problem mechanism

- The plugin's `create_model_config` wrapper sets `model_weights` to the `.gguf` reference and `model` to the
  resolved HF config source (`plugin.py:67-75` at e2b8ad5).
- vLLM #42079 changed the MTP draft's source. With no draft model given, `SpeculativeConfig` now sets
  `self.model = target_model_config.model_weights or target_model_config.model` (`config/speculative.py:1150-1153`,
  same on main today) so that runai_streamer keeps its remote weights path. For a GGUF target that is the `.gguf`.
- The draft `ModelConfig(model=self.model, config_format=target.config_format, ...)` (`speculative.py:1283-1307`)
  sends the `.gguf` to the plugin's `GGUFConfigParser`. Its `_resolve_config_source` maps the `.gguf` to the file's
  directory or to the GGUF repo (`config_parser.py:61-67`), where there is no `config.json`, hence
  "Unrecognized model". On 0.27.1 the draft used `target_model_config.model`, the config source.
- The plugin's existing post-fix (`plugin.py:96-97`, set the draft's `model_weights` to the `.gguf`) runs after
  `SpeculativeConfig` is built, which is too late.

Verified against e2b8ad5, vLLM d28795f1a7 and vLLM main (fetched 2026-10-03; the lines are unchanged).

## Fix mechanism

The existing `create_speculative_config` wrapper blanks the target's `model_weights` while vLLM builds the
speculative config, then restores it in a `finally`. It does this only for a GGUF target with no draft model
configured. vLLM then takes the draft config from `target_model_config.model` (the HF config source) on every
version, and the existing line points the draft's `model_weights` at the `.gguf`. On 0.27.1 the change does
nothing, because that version never reads `model_weights` there. About 15 changed lines (+45/-8 with the test).

Why not something else: setting `speculative_config["model"]` to the config directory would also skip vLLM's
"align the draft's quantization with the target" step, which runs only when no draft model is given
(`speculative.py:1154-1157`). The cleaner fix is in vLLM core: take the draft config from `model` and pass
`model_weights` through. The branch description has that text. If it lands, this wrapper becomes a no-op.
Downside: none found. The temporary blanking is undone in a `finally`, and only that one line of
`SpeculativeConfig` reads `target_model_config.model_weights`.

## Final check agent's concise opinion

Fable 5.1 /check, 2026-10-03: **ready with edits (text only; code right and minimal)**. Verified every cited line
against e2b8ad5, d28795f1a7 and current main. `target_model_config.model_weights` is read exactly once in
`SpeculativeConfig` and nowhere else in `create_speculative_config`, which calls the wrapper with keywords, so the
signature change is safe; 0.27.1 used `target.model`. The test is red on e2b8ad5 and green on the branch, and the
suite reproduces 105 passed / 6 skipped. The `speculative_config["model"]` alternative is correctly rejected. No
downsides found: single-threaded, restored in `finally`, non-MTP methods never read `model_weights`. Edits applied:
the first release is v0.29.0, not v0.28.0 (v0.28.0 was cut before #42079); the README's own MTP example is the
failing case; the branch description no longer says "not served on GPU"; the #70 interplay is noted.

## Fable's comment

pending
