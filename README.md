# gsq-vllm

Work repo for serving the Swift 1.5 GSQ-RCO IQ3_S-mtp GGUF
(`Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`, 12.1 GB, 866 tensors, arch `qwen35`)
on the production vLLM 0.27.1 stack. Production itself stays untouched: the GGUF
support comes from a fork of `vllm-project/vllm-gguf-plugin` that is installed
only in this repo's own venv.

Current state and open questions: [STATUS.md](STATUS.md). Wider context:
`~/gsq-rco-handoff/HANDOFF.md` and its `reports/`.

## Layout

| Path | What |
|---|---|
| `plugin/` | git subtree of `vllm-project/vllm-gguf-plugin` at `e2b8ad5` (pinned by the `git-subtree-split` trailer of b98f4a9), plus our commits |
| branch `swift-gsq-rco` | `plugin/` split out as a standalone fork: upstream history + our commits only. Regenerate with `git subtree split --prefix=plugin -b swift-gsq-rco` |
| `hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp/` | HF config/tokenizer dir passed to vLLM alongside the GGUF. `PROVENANCE.json` has the sha256 of every source and output |
| `env/prod-freeze.txt` | production venv's `uv pip freeze` (202 pins) |
| `env/gsq-freeze.txt` | this venv's freeze: production + `gguf` + the editable plugin |
| `env/prod-serve-argv.txt` | production `vllm serve` argv, one arg per line |
| `tools/capped` | resource wrapper for every build or test (see Constraints) |
| `tools/no_gpu.py` | import first in any CPU-only script; blocks NVML/libcuda loading |
| `tools/setup-cuda-toolchain.sh`, `tools/cuda-env.sh` | matched CUDA 13.0 toolchain in `build/cu130` |
| `tools/build-plugin.sh` | editable build of `plugin/` into `.venv` |
| `tools/make_hf_config.py` | `build` / `verify` the HF config dir against the GGUF |
| `tools/pytest` | pytest kept in `build/pytest`, outside the venv |
| `.venv/`, `build/` | not tracked (7.9 GB venv, 271 MB toolchain) |

Remote `plugin-upstream` has `pushurl = no_push`.

## Build

1. Venv identical to production (real copies, never hard links, because
   production's venv hard-links into the uv cache):

   ```sh
   uv venv --python ~/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12 .venv
   uv pip install --python .venv/bin/python --link-mode=copy --no-deps --offline -r env/prod-freeze.txt
   ```

2. Overlay production's patched vLLM (`Starwaves1/vllm` @ `ba05ffab`) with the
   deploy repo's tool, redirected at this venv (it prints a restart hint; ignore it):

   ```sh
   SITE_PACKAGES=.venv/lib/python3.12/site-packages STATE_FILE=/tmp/x/state BACKUP_ROOT=/tmp/x/bk \
     ~/qwen38-27b-rtx3090/scripts/deploy-vllm.sh --init v0.27.1
   SITE_PACKAGES=... STATE_FILE=... BACKUP_ROOT=... ~/qwen38-27b-rtx3090/scripts/deploy-vllm.sh ba05ffababdc
   ```

   Then copy production's top-level `site-packages/build_backend.py` over ours.

3. gguf-py from llama.cpp b11211:
   `uv pip install --python .venv/bin/python --no-deps --link-mode=copy ~/llama.cpp-b11211/gguf-py`

4. Plugin, sm86 only, CUDA 13.0 (about 20 s):

   ```sh
   tools/setup-cuda-toolchain.sh
   tools/capped tools/build-plugin.sh
   ```

   The venv's own `nvidia/cu13` pairs nvcc 13.3 with cudart 13.0 headers and CCCL
   rejects it, hence `build/cu130`. The script also patches the private header copy
   for glibc >= 2.41 (`rsqrt/rsqrtf` noexcept).

Check: `diff env/prod-freeze.txt <(uv pip freeze --python .venv/bin/python)` shows only
the `gguf` and `-e plugin` lines, and `deploy-vllm.sh --verify` (same env overrides) passes.

## CPU tests

```sh
GSQ_LIGHT=1 tools/capped tools/pytest tests/cpu
```

Tests must `import no_gpu` before torch or vLLM. GPU tests (not written yet) are opt-in
with `GSQ_ALLOW_GPU=1` and are skipped otherwise.

## Enabling the plugin

The plugin exists only in `.venv`; production's venv has no `vllm_gguf_plugin`, so
production cannot load it. It registers through the `vllm.general_plugins` entry point
`gguf` and loads when vLLM is started from `.venv/bin/vllm` (restrict with
`VLLM_PLUGINS=gguf` if needed). A GGUF launch differs from production's argv
(`env/prod-serve-argv.txt`) only in the model path (the `.gguf` file) and the HF config
dir (`hf-config/...`). The `--limit-mm-per-prompt '{"image":0,"video":0}'` that
production already passes is required: the GGUF has no mmproj, and the adapter refuses
to load a multimodal config unless image and video limits are 0 or
`--language-model-only` is set.

The HF config dir must stay unique to this model: the fs KV tier namespaces on
`model_config.model`, and sharing production's model dir would let the two models
reuse each other's KV blocks.

The swap script (`scripts/serve-gguf.sh`, default port 18090, refuses 18080/18081
without an explicit flag) is not written yet; see STATUS.md.

## Phase B (GPU)

Needs the GPU to itself: after DeepSWE run 1 finishes, or on a rented sm86 card.

1. Kernel tests against gguf-py / q8_1 emulation, including misaligned and
   non-contiguous inputs, each in a subprocess (CUDA errors are sticky).
2. Load the GGUF with CUDA graphs, MTP k=3, fp8 KV at gpu-util <= 0.94, 200k context.
3. Logit parity against llama.cpp b11211 on the same token ids (mean KLD <= 0.001,
   top-1 >= 99%, contexts up to 100k+), spec decode off.
4. Speed against production W4A16 with the same script: decode at concurrency 1 and 2,
   prefill at 8k/64k/180k with salted prompts, MTP acceptance from `/metrics`.
5. 24 h soak at concurrency 2.

Kernel route (efschu batched MMVQ, vLLM PR #36226 IQ MMQ, or others) is still open;
see STATUS.md and the handoff reports.

## Constraints

- No GPU use while DeepSWE run 1 is running. Production vLLM (ports 18080/18081) is
  read-only: no restarts, no changes to its venv, unit, drop-ins or model dirs.
- Run builds and tests under `tools/capped`: one heavy job at a time, waits for
  MemAvailable >= 8 GB, MemoryMax 6G, no swap, 200% CPU, nice 19, idle IO,
  `CUDA_VISIBLE_DEVICES=""`, `MAX_JOBS=2`.
- CUDA comes from pip wheels only; no driver or system package changes, no sudo.
- Scratch in /tmp only. Nothing is pushed and no upstream PR is opened without
  Garrett's say-so. Keep plugin changes self-contained so `swift-gsq-rco` stays
  upstreamable.
