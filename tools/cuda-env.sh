# source me: CUDA 13.0 build environment for the plugin extension (sm86 only).
_GSQ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME=$_GSQ_ROOT/build/cu130/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.6}
# g++-13 when present (the host compiler llama.cpp b11211 was built with here)
if command -v g++-13 >/dev/null 2>&1; then
  export CC=gcc-13 CXX=g++-13 NVCC_CCBIN=g++-13
fi
unset _GSQ_ROOT
