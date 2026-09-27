#!/bin/bash
# Build toolchain for the plugin's CUDA extension: CUDA 13.0, the same CUDA version
# torch 2.13.0+cu130 was built with and the same cudart (13.0.96) the venv loads.
# Installed into build/cu130 (outside the venv, so the venv's package set stays
# identical to production's). Pins come from the cuda-toolkit 13.0 metapackage
# that production's venv already carries.
#
# Why not the venv's own nvidia/cu13? It mixes nvcc 13.3.73 with cudart 13.0.96
# headers, and CCCL rejects that pairing ("CUDA compiler and CUDA toolkit headers
# are incompatible").
#
# glibc >= 2.41 declares rsqrt/rsqrtf (C23) as noexcept; CUDA 13.0's
# crt/math_functions.h declares them without it, so any <cmath> include fails.
# CUDA >= 13.1 fixed this (_NV_RSQRT_SPECIFIER). On such hosts we add the same
# specifier to our private copy of the header. Older glibc (Ubuntu 22.04/24.04) is
# left untouched.
#
# usage: tools/setup-cuda-toolchain.sh        (idempotent)
# then:  source tools/cuda-env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-$ROOT/.venv/bin/python}
DEST=$ROOT/build/cu130
T=$DEST/nvidia/cu13

if [ ! -x "$T/bin/nvcc" ]; then
  uv pip install --python "$PY" --target "$DEST" --link-mode=copy --no-deps \
    nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88 \
    nvidia-cuda-cccl==13.0.85 nvidia-cuda-runtime==13.0.96
fi
# torch.utils.cpp_extension links -lcudart from $CUDA_HOME/lib64
ln -sf libcudart.so.13 "$T/lib/libcudart.so"
ln -sfn lib "$T/lib64"

glibc=$(ldd --version | awk 'NR == 1 {print $NF}')   # awk drains the pipe: head -1 would SIGPIPE ldd under pipefail
H=$T/include/crt/math_functions.h
if [ "$(printf '%s\n2.41\n' "$glibc" | sort -V | head -1)" = 2.41 ] \
   && ! grep -q 'GSQ glibc>=2.41' "$H"; then
  sed -i \
    -e 's|^\(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ double  *rsqrt(double x)\);|\1 noexcept(true); /* GSQ glibc>=2.41 */|' \
    -e 's|^\(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ float  *rsqrtf(float x)\);|\1 noexcept(true); /* GSQ glibc>=2.41 */|' \
    "$H"
  n=$(grep -c 'GSQ glibc>=2.41' "$H")
  [ "$n" = 2 ] || { echo "rsqrt patch applied to $n declarations, expected 2" >&2; exit 1; }
  echo "patched rsqrt/rsqrtf declarations for glibc $glibc"
fi
"$T/bin/nvcc" --version | tail -1
