#!/bin/bash
# LOCAL: freeze the local worktree as /workspace/wt-p2-<tag> on the box (tracked files as they
# are now, no .git), link the shared venv / CUDA toolchain / pytest, and submit job.sh.
#   stage.sh TAG [VAR=value ...]   prints the gpuq job id
set -e
TAG=$1; shift
BOX="ssh -p ${BOX_PORT:?} root@${BOX_HOST:?}"
cd /tmp/gsq-wt-p2
git ls-files -z | rsync -a --from0 --files-from=- -e "ssh -p $BOX_PORT" ./ root@$BOX_HOST:/workspace/wt-p2-$TAG/ 2>/dev/null
$BOX "cd /workspace/wt-p2-$TAG && ln -sfn /workspace/gsq-vllm/.venv .venv && mkdir -p build && ln -sfn /workspace/gsq-vllm/build/cu130 build/cu130 && ln -sfn /workspace/gsq-vllm/build/pytest build/pytest && chmod +x cloud/results/opt-p2/box-scripts/*.sh && rm -f .build-rc && (TAG=$TAG WT=/workspace/wt-p2-$TAG nohup bash -c 'source cloud/results/opt-p2/box-scripts/lib.sh; build' > /dev/null 2>&1 &) && gpuq submit p2-$TAG --cwd /workspace/wt-p2-$TAG -- env $* bash cloud/results/opt-p2/box-scripts/${JOB:-job.sh} $TAG $ARG" 2>/dev/null | tail -1
