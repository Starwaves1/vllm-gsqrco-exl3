"""Draft-vocab corpus (item 3): the served model's own output token ids (reasoning included:
the drafter drafts those too) for corpus_prompts.py's prompts, 8 concurrent requests,
server default sampling, max 2048 tokens with thinking / 1024 without (production's
drafter/gen_data.py limits). Stops issuing after MINUTES.
usage: corpus_gen.py PROMPTS.jsonl OUT.jsonl MINUTES   (GSQ_URL, GSQ_API_KEY from env)"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
url, key = os.environ["GSQ_URL"] + "/v1/chat/completions", os.environ["GSQ_API_KEY"]
prompts = [json.loads(l) for l in open(sys.argv[1])]
stop_at = time.time() + 60 * float(sys.argv[3])
def one(p):
    if time.time() > stop_at: return None
    body = {"model": "qwen3.8-27b", "messages": p["messages"], "max_tokens": 2048 if p["think"] else 1024,
            "chat_template_kwargs": {"enable_thinking": p["think"]}, "return_token_ids": True}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        c = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
        return {"src": p["src"], "think": p["think"], "finish": c.get("finish_reason"), "output_ids": c.get("token_ids")}
    except Exception as e:
        return {"src": p["src"], "error": str(e)[:200]}
t, ntok = time.time(), 0
with ThreadPoolExecutor(8) as ex, open(sys.argv[2], "w") as f:
    for r in ex.map(one, prompts):
        if r is None: continue
        f.write(json.dumps(r) + "\n"); ntok += len(r.get("output_ids") or [])
print(f"corpus: {ntok} output tokens in {(time.time() - t) / 60:.1f} min")
