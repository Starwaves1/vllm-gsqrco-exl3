# 01: Copy unsharded and vocab GGUF weights from host memory

Branch: `upstream/01-host-staging-copy` (2 commits, on `plugin-upstream/main` e2b8ad5). No dependency.

**Title:** [BugFix] Copy unsharded and vocab GGUF weights to the device from host memory

## Motivation

With `--enable-cumem-allocator` (sleep mode) vLLM loads the target and the MTP draft into one cumem
"weights" pool. In that pool expandable segments are off and `empty_cache()` cannot release a pool in
use, so a freed segment can only be reused by an allocation that fits it.

`_store_gguf_loaded_weight` and `_gguf_embedding_weight_loader` moved each unsharded weight to the
device first (`clone().to(device)`) and then copied it into the materialized parameter. Every such
staging tensor left a freed segment behind. After loading a 27B Qwen3.5-architecture GGUF target on a
24 GB RTX 3090, 18.42 GiB were reserved for 12.04 GiB allocated, and the MTP draft loaded next ran out
of memory at 23.1 GiB: none of the freed segments could hold its 2.37 GiB vocab placeholders.

## What changed

- `quantization/params.py`: unsharded weights and vocab weights go host to parameter in one `copy_`,
  with no device temporary. Sharded weights still move to the device first: a single-shard parameter
  keeps the stored tensor as its data.
- `tests/test_plugin.py`: a CPU test that records `Tensor.to()` calls naming a device while an
  unsharded linear weight and a vocab weight load (none allowed) and a sharded one (one).

## How tested

| | before | after |
| --- | --- | --- |
| reserved after the target load | 18.42 GiB | 17.36 GiB |
| MTP draft load | OOM at 23.1 GiB | completes, peak 22.54 GiB reserved |

(RTX 3090 24 GB, vLLM 0.27.1, `--enable-cumem-allocator`, MTP k=3.)

- `pytest tests --ignore=tests/test_kernels.py --ignore=tests/test_gguf_generation.py`: 105 passed, 6
  skipped on vLLM 0.27.1 and on vLLM main (0.30.1rc1.dev285); the new test fails without the fix.
- pre-commit hooks (ruff, ruff-format, typos, clang-format, markdownlint): clean.
- The change has served every GPU run of the development branch since it was made (loads of the 27B
  GGUF with and without MTP, a 1.2 h soak).

## Risks

- One host-to-device copy per unsharded weight instead of a host-to-device copy plus a device-to-device
  one: less traffic, not more. Load time was not measured separately.
- Headroom after the fix is about 1 GiB at the draft-load peak on the 24 GB card; the sharded path
  (`_create_padded_weight_param`) still leaves about 5.3 GiB reserved-but-free, so a larger GGUF can
  still OOM there. That is a separate change.
