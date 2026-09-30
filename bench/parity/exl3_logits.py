"""exllamav3 side of the EXL3 parity check (SKELETON, not yet run): logits of the same EXL3
checkpoint from exllamav3 itself, at the positions prompts.py picks, in the llama_logits
format, so compare.py can score vLLM + the EXL3 plugin against it (exl3 as the reference).

Reference engine: exllamav3 v1.5.3 (d3739fd, the commit csrc/exl3 is vendored from) running
the same checkpoint with its own Qwen3_5 architecture, in its own venv with the full
exllamav3_ext build (GSQ_EXL3_VENV; it must not share .venv-main). Its kernels are the ones
the plugin vendors, but around them it runs attention, GDN and the residual in fp16 where vLLM
uses bf16, so logits are not bit-identical: gate on "inside exllamav3's own spread" (EXL3.md,
GPU phase 2), measured first with this script at two settings (EXL3_HGEMM_F16ACC=0/1).
EXL3_INT8_GEMV is forced to 0, as the plugin's shim does.

For each sequence: one exllamav3 cache + recurrent state, the ids fed in chunks (prefill up
to the first dumped position, then forward chunks that return logits), rows kept at
NAME.pos. Writes OUT/NAME.exl3.f32: int32 magic 0x4C4F4731, n_pos, n_vocab, pos[n_pos],
float32 logits[n_pos][n_vocab] (row i = distribution after ids[:pos[i]+1]).

  GSQ_ALLOW_GPU=1 $GSQ_EXL3_VENV/bin/python bench/parity/exl3_logits.py -m MODEL_DIR -d PROMPT_DIR -o OUT_DIR
  python bench/parity/exl3_logits.py --dry-run -d PROMPT_DIR     (no GPU: prints the plan)
  python bench/parity/compare.py -d PROMPT_DIR -l OUT_DIR/exl3 -v OUT_DIR/vllm   (after renaming
      *.exl3.f32 to *.llama.f32, or with compare.py taught the suffix: phase 2 work)

To verify on the GPU before trusting it (phase 2): the params dict below against
exllamav3's generator (job.py / generator.py pass the same keys), the recurrent-state
lifetime across chunks, logits dtype (fp16 or fp32 from the head), and that the prompt set's
longest sequence fits cache + logits for one chunk in 24 GB.
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


def read_i32(path: Path) -> np.ndarray:
    return np.fromfile(path, np.int32)


def write_rows(path: Path, pos: np.ndarray, rows: np.ndarray) -> None:
    with open(path, "wb") as f:
        np.asarray([MAGIC, len(pos), rows.shape[1]], np.int32).tofile(f)
        pos.astype(np.int32).tofile(f)
        rows.astype(np.float32).tofile(f)


def chunks(n: int, first_pos: int, chunk: int):
    """(start, end, wants_logits): prefill chunks cover ids[:first_pos], the rest are
    forward chunks whose logits hold dumped positions."""
    out, s = [], 0
    while s < n:
        e = min(s + chunk, n)
        if s < first_pos < e:  # split so every dumped position sits in a forward chunk
            e = first_pos
        out.append((s, e, e > first_pos))
        s = e
    return out


def run(model_dir: str, prompts: Path, out: Path, names: list[str], chunk: int) -> None:
    os.environ["EXL3_INT8_GEMV"] = "0"
    import torch
    from exllamav3 import Cache, Config, Model

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    n_ctx = max(len(read_i32(prompts / f"{n}.ids")) for n in names)
    cache = Cache(model, max_num_tokens=((n_ctx + 255) // 256) * 256)
    model.load(progressbar=True)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        ids = read_i32(prompts / f"{name}.ids")
        pos = read_i32(prompts / f"{name}.pos")
        want = {int(p): i for i, p in enumerate(pos)}
        rows = np.zeros((len(pos), config.vocab_size), np.float32)
        state = cache.get_new_state() if model.caps.get("recurrent_states", False) else None
        with torch.inference_mode():
            for s, e, logits_wanted in chunks(len(ids), int(pos[0]), chunk):
                x = torch.from_numpy(ids[s:e].astype(np.int64)).unsqueeze(0)
                params = {
                    "attn_mode": "flash_attn",
                    "cache": cache,
                    "past_len": s,
                    "batch_shape": (1, cache.max_num_tokens),
                    "recurrent_states": [state] if state is not None else None,
                }
                if not logits_wanted:
                    model.prefill(x, params)
                    continue
                logits = model.forward(x, params)[0, :, : config.vocab_size].float().cpu().numpy()
                for p in range(s, e):
                    if p in want:
                        rows[want[p]] = logits[p - s]
        if state is not None:
            cache.release_state(state)
        write_rows(out / f"{name}.exl3.f32", pos, rows)
        print(f"{name}: {len(ids)} tokens, {len(pos)} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", help="EXL3 model directory")
    ap.add_argument("-d", "--prompts", type=Path, required=True)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--only", help="comma-separated sequence names")
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    names = (a.prompts / "manifest.txt").read_text().split()
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
    if a.dry_run:
        for n in names:
            ids, pos = read_i32(a.prompts / f"{n}.ids"), read_i32(a.prompts / f"{n}.pos")
            c = chunks(len(ids), int(pos[0]), a.chunk)
            print(f"{n}: {len(ids)} tokens, {len(pos)} positions, {sum(not w for *_, w in c)} prefill "
                  f"+ {sum(w for *_, w in c)} logits chunks")
        return
    if not (a.model and a.out) or os.environ.get("GSQ_ALLOW_GPU") != "1":
        raise SystemExit("needs -m, -o and GSQ_ALLOW_GPU=1 (GPU phase 2)")
    run(a.model, a.prompts, a.out, names, a.chunk)


if __name__ == "__main__":
    main()
