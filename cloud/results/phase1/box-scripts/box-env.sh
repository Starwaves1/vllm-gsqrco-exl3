# source me on the box: phase-1 harness environment (mirrors cloud/bootstrap.sh exports)
export GSQ_GGUF=/workspace/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
export GSQ_BASELINE_MODEL=/workspace/models/Qwen3.8-27B-W4A16-AutoRound-fast
export GSQ_DEPLOY_REPO=/workspace/deploy GSQ_PROD_REPO=/workspace/deploy
export GSQ_KV_TIER_ROOT=/workspace/kvtier GSQ_KV_TIER_MAX_BYTES=30000000000   # 151 GB disk
export GSQ_CPU_TIER_BYTES=$((13 * 1073741824))                                 # /dev/shm is 15 GB
export GSQ_RUNS=/workspace/runs LLAMA_DIR=/workspace/llama.cpp GSQ_ALLOW_GPU=1
export PATH=/root/.local/bin:$PATH
# box only: a stopped vLLM leaves its 13 GiB CPU-tier mmap in /dev/shm (15 GB here); clear it between runs
box_clean_shm() { pgrep -f "VLLM::EngineCore|bin/vllm serve" >/dev/null && { echo "vLLM still running; shm left alone"; return 1; }; rm -f /dev/shm/vllm_offload_*.mmap; }
