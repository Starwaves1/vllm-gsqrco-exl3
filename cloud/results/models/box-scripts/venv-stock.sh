#!/bin/bash
# Stock vLLM 0.27.1 venv: production's package freeze from PyPI (vllm wheel unpatched), no
# deploy overlay, no gguf-py, no plugin.
set -eu
export PATH=/root/.local/bin:$PATH
V=/workspace/venv-stock
date -u +"start %FT%TZ"
uv venv --python 3.12 $V
uv pip install --python $V/bin/python --no-deps -r /workspace/runs/models/prod-freeze.txt
uv pip freeze --python $V/bin/python > /workspace/runs/models/venv-stock-freeze.txt
diff <(grep -v '^#' /workspace/runs/models/prod-freeze.txt | sort) <(sort /workspace/runs/models/venv-stock-freeze.txt) && echo FREEZE_EQUAL
$V/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, vllm.__file__, 'torch', torch.__version__, torch.version.cuda)"
du -sh $V; date -u +"end %FT%TZ"
