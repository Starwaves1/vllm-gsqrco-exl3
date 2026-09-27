"""CPU tensor-mapping dry run on the meta device (Phase A, no GPU).

Runs the plugin's real GGUFModelLoader.load_model with vLLM's real
Qwen3_5ForConditionalGeneration, then Qwen3_5MTP for blk.64, on device=meta.
Only the GGUF header/tensor-info is read: the weight iterator yields meta
tensors of the stored shape/dtype instead of copying data (the plugin's own
iterator does torch.tensor(memmap), i.e. materializes each tensor in RAM).

Checks:
  * every one of the 866 GGUF tensors is mapped (851 main + 15 MTP), no unmapped;
  * every non-vision model parameter is reported loaded by load_weights
    (MTP: except embed_tokens/lm_head, which vLLM shares from the target);
  * each stored GGUF tensor's logical shape (rows, cols) equals the vLLM
    layer's partition size for that shard (TP=1);
  * the 48 linear_attn.out_proj layers carry the GDN head-tiling layout.

Guards: tools/no_gpu.py (NVML/libcuda via ctypes blocked), torch CUDA init
blocked, platform = NonNvmlCudaPlatform stub reporting sm86, gloo world size 1,
production's syv draft_lm_head patch disabled (MTP_DRAFT_VOCAB=0).

Run (3 GB cap, only when MemAvailable >= 11 GB):
  flock /tmp/gsq-heavy.lock systemd-run --user --scope -q -p MemoryMax=3G \
    -p MemorySwapMax=0 -p CPUQuota=200% nice -n 19 ionice -c3 \
    .venv/bin/python tools/meta_dry_run.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import no_gpu  # noqa: E402

os.environ["MTP_DRAFT_VOCAB"] = "0"
os.environ.setdefault("VLLM_PLUGINS", "gguf")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch  # noqa: E402


def _blocked(*a, **k):
    raise RuntimeError("CUDA initialization blocked (meta dry run)")


torch.cuda._lazy_init = _blocked
torch._C._cuda_getDeviceCount = lambda: 0
torch.cuda.is_available = lambda: False

import gguf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF = os.path.expanduser(
    "~/qwen38-27b-rtx3090/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/"
    "Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
)
HF_DIR = os.path.join(ROOT, "hf-config/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp")


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) // 1024
    return -1


# ---------------------------------------------------------------- platform
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
        return torch.device("meta")  # modules that pin buffers to "the GPU" land on meta


vllm.platforms._current_platform = StubSm86Platform()


# ------------------------------------------------------ meta weight iterator
_UNQUANT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
SEEN: dict[str, tuple[str, tuple[int, ...], str]] = {}
STATS: dict = {}  # mapped name -> (gguf name, shape, type)


def meta_weights_iterator(gguf_files, gguf_to_hf_name_map=None):
    for gguf_file in gguf_files:
        reader = gguf.GGUFReader(gguf_file)
        for t in reader.tensors:
            if gguf_to_hf_name_map is not None:
                if t.name not in gguf_to_hf_name_map:
                    continue
                name = gguf_to_hf_name_map[t.name]
            else:
                name = t.name
            wt = t.tensor_type
            shape = tuple(int(x) for x in t.data.shape)  # memmap view: no data read
            if wt.name in _UNQUANT:
                if wt.name == "BF16" and t.data.dtype.itemsize == 1:
                    shape = (*shape[:-1], shape[-1] // 2)
                dtype = _UNQUANT[wt.name]
            else:
                yield name.replace("weight", "weight_type"), torch.tensor(wt)
                dtype = torch.uint8
            SEEN[name] = (t.name, shape, wt.name)
            yield name, torch.empty(shape, dtype=dtype, device="meta")
        del reader


def main():
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.engine.arg_utils import EngineArgs
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    import vllm_gguf_plugin.loader as ploader
    from vllm_gguf_plugin.quantization.layout import GGUFHeadTilingLayout
    from vllm_gguf_plugin.quantization.linear import GGUFLinearMethod

    ploader.gguf_quant_weights_iterator_multi = meta_weights_iterator
    patch_weight_type_store()

    tmp = tempfile.mkdtemp(prefix="gsq-tests-dryrun-")
    try:
        args = EngineArgs(
            model=GGUF, hf_config_path=HF_DIR, tokenizer=HF_DIR, skip_tokenizer_init=True,
            limit_mm_per_prompt={"image": 0, "video": 0}, max_model_len=4096,
            enforce_eager=True, kv_cache_dtype="fp8",
            speculative_config={"method": "mtp", "num_speculative_tokens": 3},
        )
        vllm_config = args.create_engine_config()
        vllm_config.device_config.device = torch.device("meta")
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1, rank=0, local_rank=0, backend="gloo",
                distributed_init_method=f"file://{tmp}/pg",
            )
            initialize_model_parallel(1, 1, backend="gloo")
        print(f"config ok: arch={vllm_config.model_config.architectures} "
              f"quant={vllm_config.quant_config.get_name()} maxrss={rss_mb()}MB", flush=True)

        all_names = {t.name for t in gguf.GGUFReader(GGUF).tensors}
        results = {}
        for label, model_config in [
            ("main", vllm_config.model_config),
            ("mtp", vllm_config.speculative_config.draft_model_config),
        ]:
            SEEN.clear()
            captured = {}
            loader = ploader.GGUFModelLoader(vllm_config.load_config)
            orig_pwal = ploader.process_weights_after_loading

            def check_then_process(model, mc, dev, _captured=captured):
                _captured["shape_errors"] = check_shapes(model, GGUFLinearMethod)
                return orig_pwal(model, mc, dev)

            ploader.process_weights_after_loading = check_then_process
            orig_lw_cls = None
            try:
                with set_current_vllm_config(vllm_config):
                    arch_cls = None
                    from vllm.model_executor.model_loader.utils import get_model_architecture
                    arch_cls, _ = get_model_architecture(model_config)
                    orig_lw_cls = arch_cls.load_weights

                    def lw(self, weights, _orig=orig_lw_cls, _captured=captured):
                        out = _orig(self, weights)
                        _captured["loaded"] = set(out) if out is not None else None
                        return out

                    arch_cls.load_weights = lw
                    model = loader.load_model(vllm_config, model_config)
            finally:
                ploader.process_weights_after_loading = orig_pwal
                if orig_lw_cls is not None:
                    arch_cls.load_weights = orig_lw_cls

            params = {n for n, _ in model.named_parameters()}
            loaded = captured.get("loaded") or set()
            # MTP draft: embed_tokens/lm_head are shared from the target after
            # loading (Qwen35MtpGGUFAdapter.extra_unquantized_modules), not loaded.
            shared = {"lm_head.weight", "model.embed_tokens.weight"} if label == "mtp" else set()
            missing = sorted(p for p in params - loaded - shared if "visual." not in p)
            layouts = {
                n: m.quant_method.layout for n, m in model.named_modules()
                if isinstance(getattr(m, "quant_method", None), GGUFLinearMethod)
                and m.quant_method.layout is not None
            }
            gdn_ok = all(
                n.endswith("linear_attn.out_proj") and lay == GGUFHeadTilingLayout(3, 128)
                for n, lay in layouts.items()
            )
            results[label] = dict(
                mapped=set(v[0] for v in SEEN.values()), n_params=len(params),
                n_loaded=len(loaded), missing=missing, shape_errors=captured["shape_errors"],
                n_layouts=len(layouts), gdn_ok=gdn_ok, arch=type(model).__name__,
            )
            print(f"[{label}] shape check: {STATS['layers']} GGUF layers, {STATS['shards']} shards, "
                  f"{STATS['mixed']} with mixed shard types, shard-id sets {sorted(map(str, STATS['shard_ids']))}")
            r = results[label]
            print(f"[{label}] {r['arch']}: gguf tensors fed {len(r['mapped'])}, params {r['n_params']}, "
                  f"loaded {r['n_loaded']}, missing {len(missing)} {missing[:8]}, "
                  f"shape errors {len(r['shape_errors'])} {r['shape_errors'][:8]}, "
                  f"GDN layouts {r['n_layouts']} ok={gdn_ok}, maxrss={rss_mb()}MB", flush=True)
            del model

        fed = results["main"]["mapped"] | results["mtp"]["mapped"]
        overlap = results["main"]["mapped"] & results["mtp"]["mapped"]
        unmapped = sorted(all_names - fed)
        print(f"GGUF tensors {len(all_names)}; main {len(results['main']['mapped'])} + "
              f"mtp {len(results['mtp']['mapped'])}; overlap {len(overlap)}; unmapped {len(unmapped)} {unmapped[:10]}")
        ok = (
            len(all_names) == 866 and not unmapped and not overlap
            and len(results["main"]["mapped"]) == 851 and len(results["mtp"]["mapped"]) == 15
            and all(not r["missing"] and not r["shape_errors"] for r in results.values())
            and results["main"]["n_layouts"] == 48 and results["main"]["gdn_ok"]
        )
        no_gpu.assert_no_gpu_libs()
        fds = [os.readlink(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")
               if os.path.exists(f"/proc/self/fd/{fd}")]
        assert not [f for f in fds if f.startswith("/dev/nvidia")], fds
        print("DRY RUN", "PASS" if ok else "FAIL", f"peak RSS {rss_mb()}MB")
        return 0 if ok else 1
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def patch_weight_type_store():
    """params._store_gguf_weight_type moves the scalar to param.device before
    .item(); on meta that raises. Same logic, but read the value on the CPU."""
    from vllm_gguf_plugin.quantization import params as P

    def store(param, loaded_weight, shard_id=None):
        weight_type = int(loaded_weight.reshape(-1)[0].item())
        num_elements = getattr(param, "num_elements", 1)
        if shard_id is None:
            P._materialize_parameter_data(param, (num_elements,), torch.uint8)
            param.weight_type = weight_type
            if param.data.numel() == 1:
                param.data.fill_(weight_type)
            else:
                param.data.zero_()
                param.data[0] = weight_type
            return
        param.shard_weight_type[shard_id] = weight_type
        if len(param.shard_weight_type) == 1:
            param.weight_type = weight_type
        if not isinstance(param, P.UninitializedParameter):
            if param.data.numel() == 0:
                param.data = torch.empty(num_elements, dtype=torch.uint8, device=param.device)
            param.data[P._gguf_shard_id_as_int(shard_id)] = weight_type

    P._store_gguf_weight_type = store


def check_shapes(model, GGUFLinearMethod):
    """Compare each stored shard's logical (rows, cols) to the layer's partition sizes."""
    errors = []
    STATS.update(shards=0, layers=0, mixed=0, shard_ids=set())
    for name, m in model.named_modules():
        if not isinstance(getattr(m, "quant_method", None), GGUFLinearMethod):
            continue
        w, wt = m.weight, m.weight_type
        if hasattr(m, "embedding_dim"):  # VocabParallelEmbedding / ParallelLMHead
            in_size, out_sizes = m.embedding_dim, [m.org_vocab_size]
        else:
            in_size = m.input_size_per_partition
            out_sizes = list(getattr(m, "output_partition_sizes", [m.output_size_per_partition]))
        order = {"q": 0, "k": 1, "v": 2}
        shards = list(zip(w.shard_id, w.data_container)) if w.shard_id else [(None, w.data)]
        if not w.shard_id and len(out_sizes) > 1:
            out_sizes = [sum(out_sizes)]
        STATS["layers"] += 1
        STATS["shard_ids"].add(tuple(w.shard_id))
        if len(set(wt.shard_weight_type.values())) > 1:
            STATS["mixed"] += 1
        for sid, data in shards:
            STATS["shards"] += 1
            qt = wt.shard_weight_type.get(sid, wt.weight_type) if sid is not None else wt.weight_type
            idx = 0 if sid is None else order.get(sid, sid)
            want_rows = out_sizes[idx]
            if qt in (0, 1, 30):  # F32, F16, BF16: stored unpacked
                rows, cols = data.shape[0], data.shape[1]
            else:
                block, tsize = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qt)]
                rows, cols = data.shape[0], data.shape[1] // tsize * block
            if (rows, cols) != (want_rows, in_size):
                errors.append((name, sid, (rows, cols), (want_rows, in_size)))
    return errors


if __name__ == "__main__":
    sys.exit(main())
