#!/bin/bash
# integration-2, one gpuq job: the sanitizer.sh cases that failed with CUDA OOM (whole IQ3 tensors
# packed on the GPU by iq3_pack.pack under the sanitizer's allocation tracking; a failed test's
# traceback keeps its tensors alive, so later cases OOM too), each in its own process under
# memcheck and initcheck; and the peak torch memory of one such case without the sanitizer.
source /workspace/wt-int2/cloud/results/integration-2/box-scripts/lib.sh
date -u +"start %FT%TZ"
T=tests/gpu/test_kernel_parity.py
ids=$(tools/pytest $T --collect-only -q 2>/dev/null | grep -E "^$T::(test_routing_packed_whole_tensor\[.*-(1|8|9|33|128)\]|test_lcpp_packed_layer\[(1|8|128)-.*\])$")
echo "cases: $(echo "$ids" | wc -l)"
for tool in memcheck initcheck; do
  bad=0; for id in $ids; do
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 /usr/local/cuda/bin/compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 20 \
      tools/pytest -q $id > $L/san2.log 2>&1; rc=$?
    cat $L/san2.log >> $L/sanitizer2-$tool.log
    [ $rc = 0 ] || { bad=$((bad+1)); echo "$tool rc=$rc $id: $(grep -E 'ERROR SUMMARY|Error' $L/san2.log | tail -2 | tr '\n' ' ')"; }
  done; echo "$tool: $bad of $(echo "$ids" | wc -l) cases not clean"; grep -h "ERROR SUMMARY" $L/sanitizer2-$tool.log | sort | uniq -c
done
$GSQ_VENV/bin/python - <<'PY'
import os, torch, gguf, numpy as np
GGUF = os.environ["GSQ_GGUF"]
from vllm_gguf_plugin.quantization import iq3_pack
r = gguf.GGUFReader(str(GGUF))
t = max((t for t in r.tensors if t.tensor_type.name == "IQ3_S"), key=lambda t: t.data.nbytes)
w = torch.from_numpy(np.ascontiguousarray(t.data)).cuda()
torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
p = iq3_pack.pack(w, int(t.tensor_type)); torch.cuda.synchronize()
print(f"pack {t.name} {tuple(w.shape)} {w.numel()/1e6:.1f} MB: peak {(torch.cuda.max_memory_allocated()-base)/2**30:.2f} GiB above the input")
PY
date -u +"end %FT%TZ"
