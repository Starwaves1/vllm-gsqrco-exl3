import gzip,json,glob,os,re,time
B='/home/garrett/tools/gq/cloud/results/prod-deepswe-20261011/'
A='/home/garrett/gsq-vllm/cloud/results/async-prod-20261012/'
def rd(f):
    try:
        for l in gzip.open(f,'rt'):
            try: yield json.loads(l)
            except Exception: pass
    except (EOFError,OSError): pass
def load(d,prefix='metrics'):
    out=[]
    for f in sorted(glob.glob(d+prefix+'-*.jsonl.gz')): out+=list(rd(f))
    return out
K=['vllm:num_requests_running','vllm:kv_cache_usage_perc','vllm:generation_tokens_total',
 'vllm:inter_token_latency_seconds_count','vllm:inter_token_latency_seconds_sum',
 'vllm:iteration_tokens_total_count','vllm:iteration_tokens_total_sum',
 'vllm:iteration_tokens_total_bucket{le="16.0",}',
 'vllm:spec_decode_num_accepted_tokens_total','vllm:spec_decode_num_draft_tokens_total','vllm:spec_decode_num_drafts_total',
 'vllm:prompt_tokens_by_source_total{source="local_compute"}']
def series(recs):
    return [(r['t'],[r['m'].get(k) for k in K]) for r in recs if all(r['m'].get(k) is not None for k in K[:3])]
