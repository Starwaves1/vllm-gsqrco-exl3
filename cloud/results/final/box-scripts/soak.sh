#!/bin/bash
# one gpuq job: the 24 h soak (bench/soak.sh) on wt-final's plugin build. A wrapper because gpuq
# stores its command as "$*" (quoting is lost, so `bash -c "source ...; ..."` does not survive), and
# because without PYTHONPATH the shared venv imports its editable /workspace/gsq-vllm/plugin.
source /workspace/box-env.sh
export PYTHONPATH=/workspace/wt-final/plugin:/workspace/wt-final/tools GSQ_ALLOW_GPU=1 VLLM_GGUF_LCPP=1 GSQ_RUNS=/workspace/runs
cd /workspace/wt-final && exec bench/soak.sh gsq --hours 24 --conc 2
