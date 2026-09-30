#!/bin/bash
# integration-2, one gpuq job: the stock-kernel tests only (Route L off; no Route L op called),
# run -rA on this worktree and on Integration 1's, so the per-test outcomes can be diffed.
source /workspace/wt-int2/cloud/results/integration-2/box-scripts/lib.sh
T=tests/gpu/test_kernel_parity.py K="test_dequantize or test_mmvq or test_mmq or test_routing_whole_tensor or test_unquantized_small_n"
date -u +"start %FT%TZ"
env -u VLLM_GGUF_LCPP tools/pytest $T -q -rA -k "$K" 2>&1 | grep -E "^(PASSED|SKIPPED|FAILED|ERROR)" | sed 's/ - .*//' | sort > $L/stock-int2.txt
( cd /workspace/wt-int; env -u VLLM_GGUF_LCPP PYTHONPATH=/workspace/wt-int/plugin:/workspace/wt-int/tools tools/pytest $T -q -rA -k "$K" ) 2>&1 \
  | grep -E "^(PASSED|SKIPPED|FAILED|ERROR)" | sed 's/ - .*//' | sort > $L/stock-int1.txt
echo "int2: $(cut -d' ' -f1 $L/stock-int2.txt | sort | uniq -c | tr '\n' ' ')"
echo "int1: $(cut -d' ' -f1 $L/stock-int1.txt | sort | uniq -c | tr '\n' ' ')"
diff $L/stock-int1.txt $L/stock-int2.txt > /dev/null && echo "per-test outcomes identical" || diff $L/stock-int1.txt $L/stock-int2.txt | head -20
date -u +"end %FT%TZ"
