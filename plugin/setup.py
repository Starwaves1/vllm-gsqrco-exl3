# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

import tomllib
from setuptools import setup


def _package_version() -> str:
    project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    version = project["tool"]["vllm_gguf_plugin"]["base_version"]
    suffix = os.environ.get("VLLM_GGUF_PLUGIN_LOCAL_VERSION_SUFFIX")
    if not suffix:
        return version
    normalized_suffix = suffix if suffix.startswith("+") else f"+{suffix}"
    return f"{version}{normalized_suffix}"


def _should_build_extension() -> bool:
    packaging_commands = {"sdist", "egg_info", "dist_info"}
    return not any(command in packaging_commands for command in sys.argv[1:])


setup_kwargs: dict = {"version": _package_version()}

if _should_build_extension():
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    is_rocm = getattr(torch.version, "hip", None) is not None

    nvcc_args = [
        "-O3",
        "-std=c++17",
        # Exposes aoti_torch_get_current_cuda_stream in the AOTI shim.
        "-DUSE_CUDA",
    ]
    if not is_rocm:
        # hipcc (ROCm 7.x) rejects nvcc-only flags like --use_fast_math.
        nvcc_args.insert(2, "--use_fast_math")

    sources = [
        "vllm_gguf_plugin/csrc/torch_bindings.cpp",
        "vllm_gguf_plugin/csrc/gguf/gguf_kernel.cu",
    ]
    include_dirs = [
        "vllm_gguf_plugin/csrc",
        "vllm_gguf_plugin/csrc/gguf",
    ]
    # Route L: llama.cpp b11211 MMVQ/MMQ (csrc/lcpp, vendored unmodified) plus
    # csrc/lcpp_shim.cu, adding ops lcpp_mul_mat_vec_q / lcpp_mul_mat_q.
    # Opt-in; the default build is unchanged.
    if os.environ.get("VLLM_GGUF_BUILD_LCPP") == "1" and not is_rocm:
        lcpp = pathlib.Path("vllm_gguf_plugin/csrc/lcpp").resolve()
        cuda = lcpp / "ggml/src/ggml-cuda"
        sources += ["vllm_gguf_plugin/csrc/lcpp_shim.cu", "vllm_gguf_plugin/csrc/lcpp_owned_k4.cu",
                    "vllm_gguf_plugin/csrc/lcpp_owned_iq3_mma.cu"] + [
            str(cuda / f) for f in ["mmvq.cu", "quantize.cu"]
        ] + [
            str(cuda / "template-instances" / f"mmq-instance-{t}.cu")
            for t in ["iq2_s", "iq2_xs", "iq2_xxs", "iq3_s", "iq3_xxs",
                      "iq4_xs", "q2_k", "q4_k", "q6_k"]
        ]
        # lcpp dirs first: csrc/gguf has its own (older) ggml-common.h etc.
        # gguf_kernel.cu still gets its own via quoted same-directory lookup.
        # Absolute: torch's ninja build runs from build/temp*.
        include_dirs = [str(lcpp / "ggml/include"), str(lcpp / "ggml/src"),
                        str(cuda)] + [str(pathlib.Path(d).resolve()) for d in include_dirs]
        nvcc_args += [
            "-DNDEBUG",
            "--extended-lambda",
            # llama.cpp needs the native half/bf16 operators torch disables.
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ]
        # vendors/cuda.h includes cublas_v2.h (declarations only, never linked);
        # a pip-only toolkit may lack it, so fall back to the nvidia wheel's.
        try:
            import nvidia

            for base in nvidia.__path__:
                for inc in sorted(pathlib.Path(base).glob("*/include")):
                    if (inc / "cublas_v2.h").exists():
                        nvcc_args += ["-Xcompiler", f"-idirafter,{inc}"]
        except ImportError:
            pass

    setup_kwargs.update(
        ext_modules=[
            CUDAExtension(
                name="vllm_gguf_plugin._C_gguf",
                sources=sources,
                include_dirs=include_dirs,
                py_limited_api=True,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": nvcc_args,
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        options={"bdist_wheel": {"py_limited_api": "cp310"}},
    )

setup(**setup_kwargs)
