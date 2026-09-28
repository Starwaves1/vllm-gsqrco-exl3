#!/bin/bash
source /workspace/box-env.sh
cd /workspace/gsq-vllm
tools/pytest tests/gpu/test_kernel_guards.py -q -rA -k lcpp > /workspace/logs/p2/guards-lcpp.log 2>&1
echo "rc=$?" >> /workspace/logs/p2/guards-lcpp.log
