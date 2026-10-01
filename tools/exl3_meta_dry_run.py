"""EXL3 tensor-mapping dry run on the meta device (CPU only, no GPU, no weights).

Builds vLLM's real Qwen3_5ForConditionalGeneration and its Qwen3_5MTP draft for
erlidev/Swift-1.5-Qwen3.8-27B-EXL3@SC_3.50bpw_H4_V6 (--alt: turboderp/Qwen3.8-27B-exl3@3.50bpw)
on device=meta, with the EXL3 plugin's quant config detected from config.json, and runs vLLM's
real DefaultModelLoader.load_model on both. Only metadata is read (hf-config/<checkpoint>:
config.json, the safetensors index and the shard headers fetched by HTTP range): the weights iterator yields meta tensors of each
stored name/shape/dtype, except the 0-dim codebook tensors (the kernels' mul1 multiplier,
which the plugin checks at load).

Checks:
  * every checkpoint tensor is consumed: both models get every tensor, as vLLM's loader does;
    the main model keeps what its mapper keeps, the MTP draft what its remap passes on (counted
    where it enters its loader, so a change in the overlay's filter fails here); unknown names
    raise in vLLM's loader; production's effective argv is text-only (its last
    --limit-mm-per-prompt is image 0 / video 0, with --enable-mm-embeds), so vLLM does not
    build the vision tower and its loader skips model.visual.*: counted as skipped, not
    consumed. MM_IMAGES=4 builds the tower (only a bf16 tower loads: turboderp's, not
    erlidev's 6-bit one);
  * every model parameter is reported loaded by load_weights (strict, as vLLM does for
    unquantized checkpoints; MTP: embed_tokens and lm_head are shared from the target later
    but load here too, since the checkpoint has them);
  * each EXL3 layer's parts cover its output shards exactly and match the partition sizes
    (EXL3LinearMethod.process_weights_after_loading raises otherwise); the number of parts
    equals the number of .trellis tensors each model was fed;
  * with the pruned draft head (mtp_draft_vocab_ids.pt present, as production's overlay
    builds it): mtp.draft_lm_head.weight [40960, 5120] bf16 loads into the unquantized head.

Guards as tools/meta_dry_run.py: no_gpu first, torch CUDA init blocked, a platform stub
reporting sm86, gloo world size 1. Engine args follow production's argv
(env/prod-main-serve-argv.txt: MTP k=5 schedule, 16 seqs, capture 48, images 0 + mm embeds)
except max_model_len 4096. MTP_DRAFT_VOCAB=0 turns the draft head off.

Run (light: about 2 GB): GSQ_LIGHT=1 tools/capped .venv-main/bin/python tools/exl3_meta_dry_run.py [--alt]
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_gpu  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "plugin-exl3"))
DRAFT_HEAD = os.environ.get("MTP_DRAFT_VOCAB", "1") != "0"
DRAFT_ROWS = 40960
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")  # as production's launcher
os.environ["VLLM_PLUGINS"] = ""  # no entry-point plugins; the EXL3 one is registered below
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch  # noqa: E402


def _blocked(*a, **k):
    raise RuntimeError("CUDA initialization blocked (meta dry run)")


torch.cuda._lazy_init = _blocked
torch._C._cuda_getDeviceCount = lambda: 0
torch.cuda.is_available = lambda: False

HF_DIR = os.path.join(ROOT, "hf-config", "Qwen3.8-27B-exl3-3.50bpw" if "--alt" in sys.argv[1:]
                      else "Swift-1.5-Qwen3.8-27B-exl3-SC_3.50bpw_H4_V6")
MM_IMAGES = int(os.environ.get("MM_IMAGES", "0"))
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
          "I16": torch.int16, "I32": torch.int32}


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) // 1024
    return -1


from vllm.platforms.cuda import NonNvmlCudaPlatform  # noqa: E402
from vllm.platforms.interface import DeviceCapability  # noqa: E402
import vllm.platforms  # noqa: E402


class StubSm86Platform(NonNvmlCudaPlatform):
    @classmethod
    def get_device_capability(cls, device_id=0):
        return DeviceCapability(major=8, minor=6)

    @classmethod
    def get_device_name(cls, device_id=0):
        return "NVIDIA GeForce RTX 3090"

    @classmethod
    def get_device_total_memory(cls, device_id=0):
        return 24 * 1024**3

    @classmethod
    def device_count(cls):
        return 1

    @classmethod
    def is_fully_connected(cls, physical_device_ids):
        return True

    @classmethod
    def log_warnings(cls):
        pass

    @classmethod
    def set_device(cls, device):
        pass

    @classmethod
    def manual_seed_all(cls, seed):
        pass

    @classmethod
    def current_device(cls):
        return torch.device("meta")


vllm.platforms._current_platform = StubSm86Platform()


def checkpoint_tensors(draft_head: bool) -> dict[str, tuple[str, list[int]]]:
    with open(os.path.join(HF_DIR, "safetensors_headers.json")) as f:
        headers = json.load(f)
    with open(os.path.join(HF_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    tensors = {}
    for shard, h in headers.items():
        for name, t in h["tensors"].items():
            assert index[name] == shard, (name, shard)
            tensors[name] = (t["dtype"], t["shape"])
    assert set(tensors) == set(index), "index and headers disagree"
    if draft_head:
        tensors["mtp.draft_lm_head.weight"] = ("BF16", [DRAFT_ROWS, 5120])
    return tensors


def meta_weights(tensors, codebook_mult):
    for name, (dtype, shape) in tensors.items():
        if name.endswith((".mul1", ".mcg")):  # the value is checked at load: a real scalar
            yield name, torch.tensor(codebook_mult, dtype=torch.int64).to(torch.int32)
        else:
            yield name, torch.empty(shape, dtype=DTYPES[dtype], device="meta")


def _count(weights, captured):
    captured["arrived"] = 0
    for item in weights:
        captured["arrived"] += 1
        yield item


def kept_by(label: str, name: str) -> bool:
    """Whether the main model's / MTP draft's load_weights keeps a checkpoint name (vLLM's
    own mapping: Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper drops mtp.*;
    Qwen3_5MTP.load_weights keeps mtp.*, embed_tokens and lm_head)."""
    if label == "main":
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

        return Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper.map_name(name) is not None
    return name.startswith("mtp.") or "embed_tokens" in name or "lm_head" in name


def main():
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.models.utils import AutoWeightsLoader

    import vllm_exl3_plugin
    from vllm_exl3_plugin.format import EXL3QuantConfig
    from vllm_exl3_plugin.quantization.linear import EXL3LinearMethod

    vllm_exl3_plugin.register()
    with open(os.path.join(HF_DIR, "config.json")) as f:
        qcfg = EXL3QuantConfig.from_dict(json.load(f)["quantization_config"])
    tensors = checkpoint_tensors(DRAFT_HEAD)
    vision = {n for n in tensors if n.startswith("model.visual.")}
    skipped = set() if MM_IMAGES else vision  # the tower is not built; the loader skips these
    orig_awl = AutoWeightsLoader.load_weights

    tmp = tempfile.mkdtemp(prefix="gsq-exl3-dryrun-")
    try:
        model_dir = os.path.join(tmp, "model")
        os.makedirs(model_dir)
        for f in os.listdir(HF_DIR):
            os.symlink(os.path.join(HF_DIR, f), os.path.join(model_dir, f))
        if DRAFT_HEAD:  # synthetic ids: only the count matters here
            torch.save(torch.arange(DRAFT_ROWS, dtype=torch.int64) * 6,
                       os.path.join(model_dir, "mtp_draft_vocab_ids.pt"))

        args = EngineArgs(
            model=model_dir, tokenizer=model_dir, skip_tokenizer_init=True,
            limit_mm_per_prompt={"image": MM_IMAGES, "video": 0}, enable_mm_embeds=True,
            max_model_len=4096,
            max_num_seqs=16, max_num_batched_tokens=2048, kv_cache_dtype="fp8",
            mamba_ssm_cache_dtype="float16", async_scheduling=False,
            enable_prefix_caching=True, mamba_cache_mode="align",
            speculative_config={
                "method": "mtp", "num_speculative_tokens": 5,
                "draft_sample_method": "probabilistic",
                "num_speculative_tokens_per_batch_size": [[1, 4, 5], [5, 8, 3], [9, 16, 2]],
            },
            compilation_config={"max_cudagraph_capture_size": 48,
                                "custom_ops": ["+rms_norm", "+silu_and_mul"]},
        )
        vllm_config = args.create_engine_config()
        quant_name = vllm_config.quant_config.get_name()
        print(f"config ok: arch={vllm_config.model_config.architectures} quant={quant_name} "
              f"({vllm_config.quant_config!r}) draft={vllm_config.speculative_config.draft_model_config.model} "
              f"draft_head={DRAFT_HEAD} maxrss={rss_mb()}MB", flush=True)
        vllm_config.device_config.device = torch.device("meta")
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(world_size=1, rank=0, local_rank=0, backend="gloo",
                                         distributed_init_method=f"file://{tmp}/pg")
            initialize_model_parallel(1, 1, backend="gloo")

        results = {}
        for label, model_config in [("main", vllm_config.model_config),
                                    ("mtp", vllm_config.speculative_config.draft_model_config)]:
            # both models get every tensor, as vLLM's loader does; each keeps its own
            fed = {n: t for n, t in tensors.items() if kept_by(label, n) and n not in skipped}
            captured = {}

            def weights(self, mc, model):
                return meta_weights(tensors, qcfg.codebook_mult)

            def awl(self, w, *a, _orig=orig_awl, _captured=captured, **k):
                if self.module is _captured.get("top"):  # the model's own top-level loader
                    w = _count(w, _captured)
                return _orig(self, w, *a, **k)

            orig_weights = DefaultModelLoader.get_all_weights
            DefaultModelLoader.get_all_weights = weights
            AutoWeightsLoader.load_weights = awl
            try:
                with set_current_vllm_config(vllm_config):
                    from vllm.model_executor.model_loader.utils import get_model_architecture

                    arch_cls, _ = get_model_architecture(model_config)
                    orig_lw = arch_cls.load_weights

                    def lw(self, w, _orig=orig_lw, _captured=captured):
                        _captured["top"] = self
                        out = _orig(self, w)
                        _captured["loaded"] = set(out) if out is not None else None
                        return out

                    arch_cls.load_weights = lw
                    try:
                        model = DefaultModelLoader(vllm_config.load_config).load_model(
                            vllm_config, model_config)
                    finally:
                        arch_cls.load_weights = orig_lw
            finally:
                DefaultModelLoader.get_all_weights = orig_weights
                AutoWeightsLoader.load_weights = orig_awl

            params = {n for n, _ in model.named_parameters()}
            loaded = captured.get("loaded") or set()
            exl3 = {n: m for n, m in model.named_modules()
                    if isinstance(getattr(m, "quant_method", None), EXL3LinearMethod)}
            parts = sum(m.exl3_num_parts for m in exl3.values())
            bits = sorted({b for m in exl3.values() for b in m.exl3_bits})
            fed_trellis = sum(n.endswith(".trellis") for n in fed)
            # loaded names are pre-process; EXL3 placeholders became exl3_*_{i} params
            missing = sorted(p for p in params - loaded if ".exl3_" not in p)
            placeholder_missing = sorted(
                f"{n}.{t}" for n, m in exl3.items() for t in m.exl3_placeholders
                if f"{n}.{t}" not in loaded)
            unquant = sorted(n for n, m in model.named_modules()
                             if type(getattr(m, "quant_method", None)).__name__
                             in ("UnquantizedLinearMethod", "UnquantizedEmbeddingMethod"))
            # MTP: what its own remap passed on must be exactly what kept_by says it keeps
            # (main: its mapper drops inside the loader, so every tensor arrives)
            want_in = len(fed) if label == "mtp" else len(tensors)
            results[label] = dict(fed=set(fed), arrived=captured.get("arrived", 0), want_in=want_in, n_params=len(params), n_loaded=len(loaded),
                                  missing=missing + placeholder_missing, exl3_layers=len(exl3),
                                  parts=parts, fed_trellis=fed_trellis, bits=bits,
                                  arch=type(model).__name__, unquant=unquant)
            r = results[label]
            print(f"[{label}] {r['arch']}: kept {len(fed)} tensors ({fed_trellis} EXL3; {r['arrived']} reached "
                  f"its loader, want {r['want_in']}), "
                  f"params {r['n_params']}, loaded {r['n_loaded']}, missing {len(r['missing'])} "
                  f"{r['missing'][:6]}, EXL3 layers {len(exl3)} with {parts} parts, K {bits}, "
                  f"unquantized modules {len(unquant)} (e.g. {[u for u in unquant if 'visual' not in u][:4]}), "
                  f"maxrss={rss_mb()}MB", flush=True)
            del model

        names = set(tensors)
        consumed = results["main"]["fed"] | results["mtp"]["fed"]
        unmapped = sorted(names - consumed - skipped)
        overlap = sorted(results["main"]["fed"] & results["mtp"]["fed"])
        print(f"checkpoint tensors {len(names)} (+draft head {int(DRAFT_HEAD)}): main {len(results['main']['fed'])}"
              f" + mtp {len(results['mtp']['fed'])}; shared {overlap}; vision skipped {len(skipped)}"
              f" of {len(vision)}; unmapped {len(unmapped)} {unmapped[:8]}")
        ok = (not unmapped
              and set(overlap) == {"lm_head.trellis", "lm_head.suh", "lm_head.svh", "lm_head.mul1",
                                   "model.language_model.embed_tokens.weight"}
              and all(not r["missing"] and r["parts"] == r["fed_trellis"] and r["arrived"] == r["want_in"]
                      for r in results.values())
              and (not DRAFT_HEAD or "model.draft_lm_head" in results["mtp"]["unquant"]))
        no_gpu.assert_no_gpu_libs()
        fds = [os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")
               if os.path.exists(f"/proc/self/fd/{fd}")]
        assert not [f for f in fds if f.startswith("/dev/nvidia")], fds
        print("DRY RUN", "PASS" if ok else "FAIL", f"peak RSS {rss_mb()}MB")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
