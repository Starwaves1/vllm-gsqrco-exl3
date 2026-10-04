# Vendored trellis-serve (Marlin-template EXL3 kernels)

Source: https://github.com/0xSero/trellis-serve commit
1ace59c4b43ca16a50fb6b7acf8b3fd7e2351f96 (the repository's only commit, 2026-09-28),
directory `cuda/csrc/` (the dense kernels; the MoE front is not vendored). License: MIT,
`LICENSE` (the repository root's, verbatim). Inside it, `exl3_marlin_template.h` and the
host launch logic in `exl3_marlin.cu` derive from vLLM's Marlin (Apache-2.0, Neural Magic /
vLLM project; `third_party/marlin/LICENSE`), the decode and Hadamard arithmetic from
exllamav3 1.5.1 (MIT, turboderp; `third_party/exllamav3/LICENSE`). `NOTICE` is the root
notice, `NOTICE.cuda` is `cuda/NOTICE`.

Files are byte-identical copies at their upstream paths relative to `cuda/csrc/` (the
two notices and the root license excepted, see the table). **Do not edit them**; all
adaptation lives in `../exl3_mr_shim.cu`. Check: `cmp` each file against a checkout at the
commit above. To update, re-copy the same list, recompute the hashes and rebuild.

What is here: the `#include "..."` closure of `exl3_marlin.cu` (host code, torch entry
points, its own pybind module) and `exl3_marlin_inst.h` (one kernel instantiation unit).
19 files, 300 KB. Not vendored: the MoE kernels (`exl3_marlin_moe*`; their setup also
names an `exl3_moe_orig_builder.cu` that the commit does not contain), `setup.py`,
`build.sh`, the Python packages.

Built (`plugin-exl3/setup.py`, `VLLM_EXL3_BUILD=1`) as a second extension
`vllm_exl3_plugin._C_exl3_mr`, separate from `_C_exl3` so the phase-1 library is
unchanged: `exl3_marlin.cu` + `../exl3_mr_shim.cu` + one generated instantiation unit per
(row family mb 0..4, codebook 2 = mul1, K 3/4/5), generated the way upstream's
`cuda/csrc/setup.py` does (`build/trellis_gen/inst_mb*_cb2_k*.cu` and
`trellis_families.h`), with upstream's nvcc flags and defaults
(`-static-global-template-stub=false`, `TRELLIS_WRAP_LOAD=1`, `TRELLIS_PROBES=0`,
`TRELLIS_SLOT_REDUCE=1`, `TRELLIS_MCG_SELFADD=1`, `TRELLIS_K3_IMAD_SHIFTS=0`). The
extension's `PyInit__C_exl3_mr` is upstream's `PYBIND11_MODULE(TORCH_EXTENSION_NAME)`, so
upstream's Python entry points (`repack_trellis`, `unpack_trellis`, `exl3_gemm_marlin`,
knobs) are on the module too; the plugin only calls the shim's torch ops.

Weight layout the kernels read (`repack_trellis`, VERIFIED in `exl3_marlin.cu`), from
exllamav3's int16 `[k/16, n/16, 16K]`:

| K | layout | transform |
|---|---|---|
| 3 | int32 `[k/16, n/64, 4, 24]` | none: the same bytes viewed as int32 |
| 5 | int32 `[k/16, n/64, 4, 40]` | none: the same bytes viewed as int32 |
| 4 | int32 `[k/16, n/64, 32, 4]` | word permutation: 32-bit word l of tile (i, 4g+j) moves to [i, g, l, j] (lossless, same size) |
| 6 | int32 `[k/16, n/64, 32, 4, 2]` | bit re-layout, 8 bits per weight resident (lossless, +33 %); not used here |
| 1, 2, 7, 8, half-integer | - | not supported; those tensors stay on exllamav3's kernels |

Codebooks: 3INST, MCG, MUL1 (template parameter); only MUL1 is compiled (both
checkpoints use it).

| file | upstream path | sha256 |
|---|---|---|
| `exl3_decode.cuh` | `cuda/csrc/exl3_decode.cuh` | b0291379fc62e42ea48ca4d5a793ba1da6bfd6d0a2bacc7e5662adbf00d0d748 |
| `exl3_had.cuh` | `cuda/csrc/exl3_had.cuh` | fb4d3b9a111951e2f7b3e62106c89964d4f41f444ded48c0abcaeea7517cc955 |
| `exl3_marlin.cu` | `cuda/csrc/exl3_marlin.cu` | 9d58f156c2635d77fa0e62914f2ccc12dbabdbe885d42e5748a1046e700f2ce2 |
| `exl3_marlin_inst.h` | `cuda/csrc/exl3_marlin_inst.h` | fa7cf4d038aa13e51b838e4087861d2004021d9373044a18046937f4a55b7583 |
| `exl3_marlin_kernels.h` | `cuda/csrc/exl3_marlin_kernels.h` | a5ebd6082993ba9f80bb3cab7710e532b7e4c79ec1b05abf9fa9462875b5398a |
| `exl3_marlin_template.h` | `cuda/csrc/exl3_marlin_template.h` | 4df0b19f244ee5797079b0ea7d951d478592e4ccacc97528b8856377ed7251c5 |
| `LICENSE` | `LICENSE` | cafea160e12f3e5e2c8692d58c5c5f8535b3348830ab6f337f138b8ee67d6b1c |
| `NOTICE` | `NOTICE` | 5794f5dc352000cf0550f625c4d2861b5dc77493ce5a9102b6b059627747e0a6 |
| `NOTICE.cuda` | `cuda/NOTICE` | 49034d71c248f30ce6fd919b5933c222eefeed49a378555de5cae5347c1ca966 |
| `third_party/compat.cuh` | `cuda/csrc/third_party/compat.cuh` | dd6038fa6eb8b28df56b184e358e5633421af76eeb40bad4abc522ce6ba54f57 |
| `third_party/exllamav3/codebook.cuh` | `cuda/csrc/third_party/exllamav3/codebook.cuh` | 0e3c63b323f8d3cc15c6a8f2e2b3816efafd71e149a99997b30b2f806375138a |
| `third_party/exllamav3/hadamard_inner.cuh` | `cuda/csrc/third_party/exllamav3/hadamard_inner.cuh` | 8d8e437aced88735e919563301ffac0e4a2aac28cc542ed3f738e1216ea0c36b |
| `third_party/exllamav3/LICENSE` | `cuda/csrc/third_party/exllamav3/LICENSE` | 27a32b6263fcd96c79d3beeecf221c4366780bdf15ad51986f48650bd7369bff |
| `third_party/marlin/core/scalar_type.hpp` | `cuda/csrc/third_party/marlin/core/scalar_type.hpp` | 8d3671759c43a0219f51e3a1f32c94ef5d70ecc99e264100f4015a781d94bb99 |
| `third_party/marlin/dequant.h` | `cuda/csrc/third_party/marlin/dequant.h` | 39c4640d2de39374ef5d96e7fa27701a675922113c3f5b05d45369321090e728 |
| `third_party/marlin/LICENSE` | `cuda/csrc/third_party/marlin/LICENSE` | c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4 |
| `third_party/marlin/marlin.cuh` | `cuda/csrc/third_party/marlin/marlin.cuh` | 8a129f1f3086d83c7f33ee8fdb38a0d68c0b167c99c9cef60394c3219ede86b7 |
| `third_party/marlin/marlin_dtypes.cuh` | `cuda/csrc/third_party/marlin/marlin_dtypes.cuh` | 15b90a65eadb200a3f7165e2e0d7d899f36d4993b20dbd89093ef2cea274da71 |
| `third_party/marlin/marlin_mma.h` | `cuda/csrc/third_party/marlin/marlin_mma.h` | bf863d252bfc468eaff42b2c1bda583c5e6ab97ceacf0ac248c164564ddcce9b |
