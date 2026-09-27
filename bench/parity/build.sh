#!/bin/bash
# Build recipe for the llama.cpp side of the parity check. Not run on the workstation
# (a llama.cpp CUDA build does not fit the Phase A caps); the tool itself is one file.
#
#   bench/parity/build.sh                  build llama_logits against an existing b11211
#                                          build (LLAMA_DIR, default ~/llama.cpp-b11211)
#   bench/parity/build.sh --build-llama    first clone + build llama.cpp b11211 (CUDA, sm86)
#                                          into LLAMA_DIR (cloud box; needs nvcc + cuBLAS)
#   bench/parity/build.sh --syntax-only    compile-check llama_logits.cpp, no output
#
# Output: $GSQ_RUNS/parity-bin/llama_logits (rpath to LLAMA_DIR/build/bin).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/env.sh"
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp-b11211}
LLAMA_TAG=b11211
LLAMA_COMMIT=d7fb90e8e2494b2908934d956a3202fd60152ee0
CXX=${CXX:-$(command -v g++-13 || command -v g++)}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = --build-llama ]; then
  if [ ! -d "$LLAMA_DIR/.git" ]; then
    git clone --branch "$LLAMA_TAG" --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi
  [ "$(git -C "$LLAMA_DIR" rev-parse HEAD)" = "$LLAMA_COMMIT" ] || gsq_die "$LLAMA_DIR is not $LLAMA_TAG ($LLAMA_COMMIT)"
  command -v nvcc >/dev/null || gsq_die "llama.cpp CUDA build needs nvcc + cuBLAS dev files on PATH (system CUDA toolkit)"
  # Same configuration as the workstation's reference build (its CMakeCache.txt):
  # shared libs, CUDA sm86, FA on, CUDA graphs on, Release.
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
  cmake --build "$LLAMA_DIR/build" -j "${JOBS:-$(nproc)}" --target llama llama-server
  shift
fi

INC=(-I"$LLAMA_DIR/include" -I"$LLAMA_DIR/ggml/include")
if [ "${1:-}" = --syntax-only ]; then
  "$CXX" -std=c++17 -fsyntax-only -Wall -Wextra "${INC[@]}" "$HERE/llama_logits.cpp"
  echo "syntax ok"; exit 0
fi
[ -f "$LLAMA_DIR/build/bin/libllama.so" ] || gsq_die "no $LLAMA_DIR/build/bin/libllama.so (use --build-llama)"
OUT=$GSQ_RUNS/parity-bin; mkdir -p "$OUT"
"$CXX" -O2 -std=c++17 -Wall "${INC[@]}" "$HERE/llama_logits.cpp" \
  -L"$LLAMA_DIR/build/bin" -lllama -Wl,-rpath,"$LLAMA_DIR/build/bin" -o "$OUT/llama_logits"
echo "built $OUT/llama_logits"
