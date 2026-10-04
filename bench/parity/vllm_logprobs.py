"""vLLM side of the parity check: full-vocab logprobs at the same positions llama.cpp dumps.

Runs vLLM in-process from this repo's .venv with the GGUF plugin, production's model
flags minus speculative decoding and KV offload (parity is about the weights and kernels,
not the drafter), and max_logprobs=-1. For each sequence it warms the prefix cache with
ids[:pos[0]+1], then probes every position p with prompt ids[:p+1], max_tokens=1,
logprobs=-1 (the distribution of the token after p). With --enable-prefix-caching and
--mamba-cache-mode align each probe recomputes only up to one block, not the whole prefix.

Writes OUT/NAME.vllm.f32 in the llama_logits format (magic LOG1, n_pos, n_vocab, pos[],
float32 rows), holding logprobs; tokens absent from the returned dict are -inf.

  GSQ_ALLOW_GPU=1 .venv/bin/python bench/parity/vllm_logprobs.py -d PROMPT_DIR -o OUT_DIR [--model EXL3_DIR]
      [--kv-cache-dtype auto|fp8] [--mamba-ssm-cache-dtype float16|float32] [--only seq_003,...]
  .venv/bin/python bench/parity/vllm_logprobs.py --dry-run -d PROMPT_DIR   (no GPU: prints plan)

Checked on an RTX 3090 (phase 1b, cloud/results/phase1b/parity): logprobs=-1 returns all
248,320 vocab ids per probe (no -inf rows), max_logprobs=-1 works with the GGUF model, and
the probes hit the prefix cache (block size 400; 288 probes on a 65k sequence take ~6 min).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
if os.environ.get("GSQ_ALLOW_GPU") != "1":
    import no_gpu  # noqa: F401

import numpy as np  # noqa: E402

MAGIC = 0x4C4F4731
HF_CONFIG = Path(os.environ.get("GSQ_HF_CONFIG") or ROOT / "hf-config" / "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp")


def write_rows(path: Path, pos: np.ndarray, rows: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.asarray([MAGIC, len(pos), rows.shape[1]], np.int32).tofile(f)
        pos.astype(np.int32).tofile(f)
        rows.astype(np.float32).tofile(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-d", "--prompts", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--gguf", default=os.environ.get("GSQ_GGUF"))
    ap.add_argument("--model", help="a plain HF model dir instead of --gguf (EXL3: served with the exl3 "
                                    "plugin, its own config and tokenizer)")
    ap.add_argument("--kv-cache-dtype", default="auto", help="auto = bf16 (reference); fp8 = production")
    ap.add_argument("--mamba-ssm-cache-dtype", default="float16", help="production: float16")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.94)
    ap.add_argument("--only", default="", help="comma-separated sequence names")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    man = json.loads((a.prompts / "manifest.json").read_text())
    seqs = [s for s in man["sequences"] if not a.only or s["name"] in a.only.split(",")]
    max_len = max(s["n_tokens"] for s in seqs) + 8
    probes = sum(s["n_pos"] for s in seqs)
    print(f"{len(seqs)} sequences, {probes} probes, max_model_len {max_len}, kv {a.kv_cache_dtype}")
    if a.dry_run:
        return
    if os.environ.get("GSQ_ALLOW_GPU") != "1":
        raise SystemExit("GPU use is opt-in: GSQ_ALLOW_GPU=1")
    if not (a.model or a.gguf) or not a.out:
        raise SystemExit("--gguf (or GSQ_GGUF) or --model, and -o are required")
    a.out.mkdir(parents=True, exist_ok=True)

    os.environ["VLLM_PLUGINS"] = "lora_filesystem_resolver,lora_hf_hub_resolver," + ("gguf,exl3" if a.model else "gguf")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.plugins import load_general_plugins

    load_general_plugins()  # the plugin patches EngineArgs before LLM() builds configs
    src = {"model": a.model} if a.model else {"model": a.gguf, "hf_config_path": str(HF_CONFIG), "tokenizer": str(HF_CONFIG)}
    llm = LLM(
        **src,
        served_model_name="qwen3.8-27b",
        max_model_len=max_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        max_num_seqs=8,
        kv_cache_dtype=a.kv_cache_dtype,
        mamba_ssm_cache_dtype=a.mamba_ssm_cache_dtype,
        max_num_batched_tokens=2048,
        limit_mm_per_prompt={"image": 0, "video": 0},
        compilation_config={"max_cudagraph_capture_size": 32, "custom_ops": ["+rms_norm", "+silu_and_mul"]},
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        max_logprobs=-1,
        async_scheduling=False,
    )
    vocab = llm.llm_engine.model_config.get_vocab_size()
    sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=-1, detokenize=False)

    for s in seqs:
        ids = np.fromfile(a.prompts / f"{s['name']}.ids", np.int32).tolist()
        pos = np.fromfile(a.prompts / f"{s['name']}.pos", np.int32)
        t0 = time.time()
        llm.generate(TokensPrompt(prompt_token_ids=ids[: int(pos[0]) + 1]), sp, use_tqdm=False)
        t_warm = time.time() - t0
        rows = np.full((len(pos), vocab), -np.inf, np.float32)
        t0 = time.time()
        for i, p in enumerate(pos):
            out = llm.generate(TokensPrompt(prompt_token_ids=ids[: int(p) + 1]), sp, use_tqdm=False)
            lp = out[0].outputs[0].logprobs[0]
            for tid, v in lp.items():
                rows[i, tid] = v.logprob
        write_rows(a.out / f"{s['name']}.vllm.f32", pos, rows)
        print(f"{s['name']}: {s['n_tokens']} tokens, warm-up {t_warm:.1f}s, "
              f"{len(pos)} probes {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
