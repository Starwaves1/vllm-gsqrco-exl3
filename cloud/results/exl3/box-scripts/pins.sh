# shellcheck shell=bash
# shellcheck disable=SC2034  # sourced by 00-prep.sh and overlay-vllm.sh
# Pins for the EXL3 phase-1 box (same stack as requested-workloads/01: production's vLLM main).
VLLM_FORK_URL=https://github.com/Starwaves1/vllm.git
VLLM_BASE=d28795f1a7af4e3ce2530d4f0bdaec4ecbede693
VLLM_OVERLAY=2a0fe5e1e1d7199f853012dfededcbb696e3083a
VLLM_EXPECT_VERSION=0.30.1rc1.dev285+gd28795f1a
FLASHINFER_INDEX=https://flashinfer.ai/whl/cu130
PY_VERSION=3.12.13
LLAMA_TAG=b11211
LLAMA_COMMIT=d7fb90e8e2494b2908934d956a3202fd60152ee0
EXL3_URL=https://github.com/turboderp-org/exllamav3
EXL3_COMMIT=d3739fd393337b1ff4d6c2a342b12f0c87a9592f   # v1.5.3, the vendored commit
EXL3_REPO=turboderp/Qwen3.8-27B-exl3
EXL3_REV=8351c54ef11a63e38bca978655086d7998b8a7f9      # branch 3.50bpw (hf-config/.../PROVENANCE.json)
