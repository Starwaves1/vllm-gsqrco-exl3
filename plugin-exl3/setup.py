# SPDX-License-Identifier: Apache-2.0

import os
import pathlib
import sys

import tomllib
from setuptools import setup


def _package_version() -> str:
    project = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
    return project["tool"]["vllm_exl3_plugin"]["base_version"]


def _should_build_extension() -> bool:
    packaging_commands = {"sdist", "egg_info", "dist_info"}
    if any(command in packaging_commands for command in sys.argv[1:]):
        return False
    # Opt-in: without it the package is pure Python (config, loader, routing and
    # the CPU tests work; the linear method needs the ops at runtime).
    return os.environ.get("VLLM_EXL3_BUILD") == "1"


setup_kwargs: dict = {"version": _package_version()}

if _should_build_extension():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    # exllamav3 d3739fd (csrc/exl3, vendored unmodified, see csrc/exl3/VENDORED.md):
    # every .cu of the dense-linear closure, plus csrc/exl3_shim.cu (all adaptation).
    # Same nvcc flags as exllamav3's util/cuda_flags.py (minus --compress-mode).
    csrc = pathlib.Path("vllm_exl3_plugin/csrc")
    vendored = sorted(str(p) for p in (csrc / "exl3").rglob("*.cu"))
    sources = [str(csrc / "exl3_shim.cu"), *vendored]
    nvcc_args = [
        "-O3",
        "--use_fast_math",
        "-Xcudafe", "--diag_suppress=177",
        "-Xcudafe", "--diag_suppress=20012",
    ]
    setup_kwargs.update(
        ext_modules=[
            CUDAExtension(
                name="vllm_exl3_plugin._C_exl3",
                sources=sources,
                # Absolute: torch's ninja build runs from build/temp*.
                include_dirs=[str((csrc / "exl3").resolve()), str(csrc.resolve())],
                extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_args},
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )

setup(**setup_kwargs)
