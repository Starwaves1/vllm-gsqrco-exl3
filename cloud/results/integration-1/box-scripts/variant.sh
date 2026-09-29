#!/bin/bash
# integration-1 parity attribution: vLLM side of seq_009/seq_010 on patched copies of this build's
# plugin (same .so; Python routing only, nothing committed):
#   nocoalesce  apply() runs each shard of a mixed-type layer alone (phase 3 item 4b off)
#   noown       and, only if the pos-119966 spike survives, K1's Q4_K/IQ2_S kernel off
#               (_OWN_MIN_ROWS = {}: Q4_K/IQ2_S back to MMVQ < 8 rows, MMQ from 8)
source /workspace/wt-int/cloud/results/integration-1/box-scripts/lib.sh
P=/workspace/runs/p1b-parity
date -u +"start %FT%TZ"; stopall
run() {  # $1 = variant, rest = python patch statements
  local V=$1 C=/workspace/int-var-$1 O=/workspace/runs/int1-var-$1; shift
  rm -rf $C $O; mkdir -p $C $O/vllm
  rsync -a --exclude build $WT/plugin/ $C/plugin/
  $GSQ_VENV/bin/python - $C/plugin/vllm_gguf_plugin/quantization/linear.py "$@" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
for pair in sys.argv[2:]:
    a, b = pair.split("|||")
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(p, "w").write(s)
PY
  PYTHONPATH=$C/plugin:$WT/tools $GSQ_VENV/bin/python bench/parity/vllm_logprobs.py -d $P/prompts -o $O/vllm \
    --gguf $GSQ_GGUF --only seq_009,seq_010 > $O/vllm.log 2>&1; echo "$V vllm rc=$?"
  $GSQ_VENV/bin/python $S/spike.py $P/prompts $P/llama $O/vllm $GSQ_HF_CONFIG $O/kld.json seq_009 seq_010 > $O/spike.txt 2>&1
  echo "== $V"; cat $O/spike.txt; rm -rf $O/vllm $C
}
SPLIT='        yield _shard_weight(weight, start, offsets[ids[-1]][1], size), weight_type|||        w = _shard_weight(weight, start, offsets[ids[-1]][1], size)
        for i in ids:  # int1 attribution variant: every shard alone (item 4b off)
            yield w[offsets[i][0] - start : offsets[i][1] - start], weight_type'
run nocoalesce "$SPLIT"
if $GSQ_VENV/bin/python -c "import json,sys; d=json.load(open('/workspace/runs/int1-var-nocoalesce/kld.json'))['seq_010']; sys.exit(0 if d['kld'][d['positions'].index(119966)] > 0.3 else 1)"; then
  run noown "$SPLIT" '_OWN_MIN_ROWS = {WeightType.Q4_K: 3, WeightType.IQ2_S: 1}|||_OWN_MIN_ROWS = {}'
fi
date -u +"end %FT%TZ"
