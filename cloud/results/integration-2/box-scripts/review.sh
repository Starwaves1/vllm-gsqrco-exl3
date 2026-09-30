#!/bin/bash
# integration-2 review round 1, one gpuq job on the review fixes (linear.py's IQ1_M chunk condition,
# the packed decode op in the x_q8 parity test and x_q8 guards; shim edits are comments, no rebuild):
# CPU guards + routing, the parity tests through linear.py / x_q8, the packed op's x_q8 guards.
source /workspace/wt-int2/cloud/results/integration-2/box-scripts/lib.sh
date -u +"start %FT%TZ"
tools/pytest tests/cpu/test_lcpp_guards.py tests/cpu/test_lcpp_routing.py -q > $L/rv-cpu.log 2>&1; echo "cpu rc=$?: $(tail -1 $L/rv-cpu.log)"
tools/pytest tests/gpu/test_kernel_parity.py -q -rs -k "test_lcpp_x_q8 or iq1_m_chunks or routing_whole_tensor or routing_packed_whole_tensor or packed_layer or mixed_shard_layer or same_type_run" > $L/rv-parity.log 2>&1; echo "parity rc=$?: $(tail -1 $L/rv-parity.log)"
tools/pytest tests/gpu/test_kernel_guards.py -q -rs -k "x_q8 and lcpp_iq3_mma_packed" > $L/rv-guards.log 2>&1; echo "guards rc=$?: $(tail -1 $L/rv-guards.log)"
date -u +"end %FT%TZ"
