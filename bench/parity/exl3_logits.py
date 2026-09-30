"""exllamav3 side of the EXL3 parity check: logits of the same EXL3 checkpoint from exllamav3
itself, at the positions prompts.py picks, in the llama_logits format, so compare.py can score
vLLM + the EXL3 plugin against it (exl3 as the reference).

Reference engine: exllamav3 v1.5.3 (d3739fd, the commit csrc/exl3 is vendored from) running
the same checkpoint with its own Qwen3_5 architecture (text component; no vision, no MTP), in
its own venv with the full exllamav3_ext build (GSQ_EXL3_VENV, /workspace/venv-exl3ref on the
box; never vLLM's venv). Its kernels are the ones the plugin vendors, but around them it runs
attention, GDN and the residual in fp16 where vLLM uses bf16, so logits are not bit-identical:
gate on "inside exllamav3's own spread" (EXL3.md, GPU phase 2), measured with this script at
two settings (EXL3_HGEMM_F16ACC=0/1, the vendored fp16-accumulate hgemm switch).
EXL3_INT8_GEMV is forced to 0, as the plugin's shim does.

One exllamav3 cache (fp16 KV, max_num_tokens = the longest sequence rounded up to 256, one
recurrent-state slot) is reused by every sequence; each sequence gets a fresh recurrent state.
The ids are fed in chunks (plan()): chunks without dumped positions go through
model.prefill() (no head), a chunk that ends in a run of dumped positions goes through
model.forward() with last_tokens_only = that run's length, so the head only computes rows
that are kept (at most --logits-rows per chunk: 256 rows x 248,320 vocab x fp32 = 254 MB).
Nothing after the last dumped position is computed. A sequence that runs out of memory is
recorded in OUT/exl3_logits.json and skipped (the context limit to document), the rest go on.

Writes OUT/NAME.exl3.f32: int32 magic 0x4C4F4731, n_pos, n_vocab, pos[n_pos],
float32 logits[n_pos][n_vocab] (row i = distribution after ids[:pos[i]+1]); and
OUT/exl3_logits.json (settings, per-sequence status, timings, peak memory).

  GSQ_ALLOW_GPU=1 $GSQ_EXL3_VENV/bin/python bench/parity/exl3_logits.py -m MODEL_DIR -d PROMPT_DIR -o OUT_DIR
  python bench/parity/exl3_logits.py --dry-run -d PROMPT_DIR     (no GPU: prints the plan)
  compare.py reads *.llama.f32 as the reference: link NAME.exl3.f32 -> NAME.llama.f32
  (cloud/results/exl3/box-scripts/05-parity-vllm.sh does), then
  python bench/parity/compare.py -d PROMPT_DIR -l REF_DIR -v VLLM_DIR --json parity.json
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
if os.environ.get("GSQ_ALLOW_GPU") != "1":
    import no_gpu  # noqa: F401

import numpy as np  # noqa: E402

MAGIC = 0x4C4F4731
CHUNK = 2048
LOGITS_ROWS = 256


def read_i32(path: Path) -> np.ndarray:
    return np.fromfile(path, np.int32)


def write_rows(path: Path, pos: np.ndarray, rows: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.asarray([MAGIC, len(pos), rows.shape[1]], np.int32).tofile(f)
        pos.astype(np.int32).tofile(f)
        rows.astype(np.float32).tofile(f)


def plan(n: int, pos, chunk: int, logits_rows: int = LOGITS_ROWS):
    """[(start, end, n_logits)]: consecutive chunks from 0 to the last dumped position + 1.
    n_logits = 0: prefill only (no dumped position inside). n_logits > 0: forward with the head
    on the chunk's last n_logits rows, which are exactly dumped positions (a run of consecutive
    ones), so every computed row is kept. No chunk is longer than `chunk`, no logits run longer
    than `logits_rows`."""
    want = sorted({int(p) for p in pos})
    if not want or want[0] < 0 or want[-1] >= n:
        raise ValueError("positions must lie in [0, n)")
    todo, out, s, i = set(want), [], 0, 0
    while i < len(want):
        p = want[i]
        if p + 1 - s > chunk:  # next dumped position is beyond this chunk: prefill a full chunk
            out.append((s, s + chunk, 0))
            s += chunk
            continue
        e = p + 1  # extend over the run of consecutive dumped positions, within the limits
        while e in todo and e - s < chunk and e - p < logits_rows:
            e += 1
        out.append((s, e, e - p))
        i += e - p
        s = e
    return out


def run(model_dir: str, prompts: Path, out: Path, names: list[str], chunk: int, logits_rows: int) -> None:
    os.environ["EXL3_INT8_GEMV"] = "0"
    import json
    import time

    import torch
    from exllamav3 import Cache, Config, Model
    from exllamav3.version import __version__ as exl3_version

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    n_ctx = max(len(read_i32(prompts / f"{n}.ids")) for n in names)
    cache = Cache(model, max_num_tokens=((n_ctx + 255) // 256) * 256, max_batch_size=1)
    t0 = time.time()
    model.load(progressbar=False)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"exllamav3": exl3_version, "torch": torch.__version__, "model": model_dir,
           "EXL3_HGEMM_F16ACC": os.environ.get("EXL3_HGEMM_F16ACC", "(unset: on where the probe enables it)"),
           "EXL3_INT8_GEMV": os.environ["EXL3_INT8_GEMV"], "cache_tokens": cache.max_num_tokens,
           "load_s": round(time.time() - t0, 1), "gpu_mib_after_load": torch.cuda.memory_allocated() >> 20,
           "chunk": chunk, "logits_rows": logits_rows, "sequences": {}}
    vocab = config.vocab_size
    for name in names:
        ids = read_i32(prompts / f"{name}.ids")
        pos = read_i32(prompts / f"{name}.pos")
        row_of = {int(p): i for i, p in enumerate(pos)}
        rows = np.zeros((len(pos), vocab), np.float32)
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        state = cache.get_new_state() if model.caps.get("recurrent_states", False) else None
        try:
            with torch.inference_mode():
                for s, e, n_logits in plan(len(ids), pos, chunk, logits_rows):
                    x = torch.from_numpy(ids[s:e].astype(np.int64)).unsqueeze(0)
                    params = {"attn_mode": "flash_attn", "cache": cache, "past_len": s,
                              "batch_shape": (1, cache.max_num_tokens)}
                    if state is not None:
                        params["recurrent_states"] = [state]
                    if not n_logits:
                        model.prefill(x, params)
                        continue
                    params["last_tokens_only"] = n_logits
                    logits = model.forward(x, params)[0, -n_logits:, :vocab].float().cpu().numpy()
                    for j, p in enumerate(range(e - n_logits, e)):
                        rows[row_of[p]] = logits[j]
            write_rows(out / f"{name}.exl3.f32", pos, rows)
            st = {"status": "ok"}
        except torch.OutOfMemoryError as ex:
            st = {"status": "oom", "error": str(ex).splitlines()[0][:300]}
        finally:
            if state is not None:
                cache.release_state(state)
        st.update(n_tokens=len(ids), n_pos=len(pos), seconds=round(time.time() - t0, 1),
                  peak_gpu_mib=torch.cuda.max_memory_allocated() >> 20)
        rec["sequences"][name] = st
        (out / "exl3_logits.json").write_text(json.dumps(rec, indent=1))
        print(f"{name}: {len(ids)} tokens, {len(pos)} rows, {st['status']}, {st['seconds']} s, "
              f"peak {st['peak_gpu_mib']} MiB", flush=True)
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", help="EXL3 model directory")
    ap.add_argument("-d", "--prompts", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--only", help="comma-separated sequence names")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--logits-rows", type=int, default=LOGITS_ROWS)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = (a.prompts / "manifest.txt").read_text().split()
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    if a.dry_run:
        for n in names:
            ids, pos = read_i32(a.prompts / f"{n}.ids"), read_i32(a.prompts / f"{n}.pos")
            c = plan(len(ids), pos, a.chunk, a.logits_rows)
            print(f"{n}: {len(ids)} tokens, {len(pos)} positions, {sum(not k for *_, k in c)} prefill "
                  f"+ {sum(bool(k) for *_, k in c)} logits chunks, {sum(k for *_, k in c)} logits rows")
        return
    if not (a.model and a.out) or os.environ.get("GSQ_ALLOW_GPU") != "1":
        raise SystemExit("needs -m, -o and GSQ_ALLOW_GPU=1")
    run(a.model, a.prompts, a.out, names, a.chunk, a.logits_rows)


if __name__ == "__main__":
    main()
