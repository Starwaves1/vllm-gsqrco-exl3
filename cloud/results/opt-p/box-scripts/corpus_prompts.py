"""Draft-vocab corpus prompts (item 3): production's drafter/collect_prompts.py sources at 1/5
of its counts (UltraChat, Magicoder, da-instruction, reasoning-v1, skolegpt, GSM8K train;
think on for 70% of reasoning/GSM8K prompts, 40% of the rest). None of the bench's 8 prompts.
usage: corpus_prompts.py OUT.jsonl"""
import json, random, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
R = random.Random(1234)
def rows(repo, f):
    p = hf_hub_download(repo, f, repo_type="dataset")
    if p.endswith(".jsonl"):
        return [json.loads(l) for l in open(p)]
    return pq.read_table(p).to_pylist()
out = []
def add(src, msgs, k):
    out.append({"src": src, "messages": msgs, "think": R.random() < k})
uc = rows("HuggingFaceH4/ultrachat_200k", "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"); R.shuffle(uc)
for r in [r for r in uc if r["messages"] and r["messages"][0]["role"] == "user" and len(r["messages"][0]["content"]) >= 20][:460]:
    add("ultrachat", [{"role": "user", "content": r["messages"][0]["content"]}], 0.4)
mc = rows("ise-uiuc/Magicoder-OSS-Instruct-75K", "data-oss_instruct-decontaminated.jsonl"); R.shuffle(mc)
for r in mc[:220]:
    add("magicoder", [{"role": "user", "content": r["problem"]}], 0.4)
da = rows("syvai/da-instruction", "data/train-00000-of-00001.parquet"); R.shuffle(da)
for r in [r for r in da if r["conversations"] and r["conversations"][0]["role"] == "user"][:220]:
    add("da-instruction", [{"role": "user", "content": r["conversations"][0]["content"]}], 0.4)
rv = rows("syvai/reasoning-v1", "data/train-00000-of-00003.parquet"); R.shuffle(rv)
for r in [r for r in rv if r["conversations"] and r["conversations"][0]["role"] == "user"][:160]:
    add("da-reasoning", [{"role": "user", "content": r["conversations"][0]["content"]}], 0.7)
sk = rows("kobprof/skolegpt-instruct", "data/train-00000-of-00001.parquet"); R.shuffle(sk)
for r in sk[:200]:
    m = ([{"role": "system", "content": r["system_prompt"]}] if r.get("system_prompt") else [])
    add("skolegpt", m + [{"role": "user", "content": r["question"]}], 0.4)
gs = rows("openai/gsm8k", "main/train-00000-of-00001.parquet"); R.shuffle(gs)
for r in gs[:100]:
    add("gsm8k", [{"role": "user", "content": r["question"]}], 0.7)
R.shuffle(out)
with open(sys.argv[1], "w") as f:
    for p in out: f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(len(out), "prompts,", sum(p["think"] for p in out), "think")
