# DIAG ONLY (box /tmp): log CUDA memory around each GGUF load_model, record allocation stacks
import os
if os.environ.get("GSQ_MEMDIAG") == "1":
    try:
        import torch
        from vllm_gguf_plugin import loader as _L
        _orig = _L.GGUFModelLoader.load_model
        _orig_init = _L.initialize_model
        def _gib(): return f"alloc={torch.cuda.memory_allocated()/2**30:.2f}GiB reserved={torch.cuda.memory_reserved()/2**30:.2f}GiB"
        def _init(*a, **k):
            print(f"[memdiag] before initialize_model {k.get('model_config').architectures if k.get('model_config') else ''} {_gib()}", flush=True)
            m = _orig_init(*a, **k)
            print(f"[memdiag] after initialize_model {_gib()}", flush=True)
            return m
        _L.initialize_model = _init
        def _load(self, vllm_config, model_config, prefix=""):
            draft = "MTP" in str(model_config.architectures)
            print(f"[memdiag] load_model start arch={model_config.architectures} prefix={prefix!r} {_gib()}", flush=True)
            if draft:
                torch.cuda.memory._record_memory_history(max_entries=500000)
            try:
                m = _orig(self, vllm_config, model_config, prefix)
            except BaseException:
                if draft:
                    torch.cuda.memory._dump_snapshot("/workspace/runs/p1-diag-mem/oom-snapshot.pickle")
                    print(f"[memdiag] OOM snapshot dumped {_gib()}", flush=True)
                raise
            print(f"[memdiag] load_model end arch={model_config.architectures} {_gib()}", flush=True)
            if draft:
                torch.cuda.memory._dump_snapshot("/workspace/runs/p1-diag-mem/ok-snapshot.pickle")
            return m
        _L.GGUFModelLoader.load_model = _load
    except Exception as e:
        print("[memdiag] patch failed", e, flush=True)
