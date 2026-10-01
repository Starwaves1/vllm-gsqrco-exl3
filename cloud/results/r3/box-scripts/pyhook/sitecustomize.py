"""Round-3 diagnosis hook, loaded only when this directory is on PYTHONPATH (22-idle-profile.sh).

R3_PTRACE_ANY=1   prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY): lets py-spy (not an ancestor of
                  the server) attach under yama ptrace_scope=1 inside an unprivileged container.
R3_INSTR_DIR=DIR  wall/CPU timers around the engine loop, scheduler, model runner, KV connector,
                  tiering manager and CUDA event/stream syncs, recorded on the EngineCore's main
                  thread only. One JSON line per engine cycle (step start -> next step start, so
                  post_step and input-queue work belong to the step before) in
                  DIR/instr-<pid>.jsonl. Times are inclusive (nested wrapped calls count in both).
Cost: one perf_counter_ns pair and two dict updates per wrapped call, ~40 calls per step.
"""

import os
import sys

if os.environ.get("R3_PTRACE_ANY") == "1":
    try:
        import ctypes

        ctypes.CDLL(None, use_errno=True).prctl(0x59616D61, ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF), 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass

if os.environ.get("R3_INSTR_DIR"):
    import functools
    import importlib.abc
    import inspect
    import json
    import threading
    import time

    _pc = time.perf_counter_ns
    _ct = time.thread_time_ns
    TARGETS = {
        "vllm.v1.engine.core": {"EngineCore": ["post_step", "_process_aborts_queue"],
                                "EngineCoreProc": ["_process_input_queue", "_process_engine_step"]},
        "vllm.v1.core.sched.scheduler": {"Scheduler": [
            "schedule", "update_from_output", "get_grammar_bitmask", "update_draft_token_ids",
            "_update_after_schedule"]},
        "vllm.v1.worker.gpu_model_runner": {"GPUModelRunner": [
            "execute_model", "sample_tokens", "take_draft_token_ids", "_update_states", "_prepare_inputs",
            "_build_attention_metadata", "_model_forward", "propose_draft_token_ids", "_bookkeeping_sync",
            "_sample", "_determine_batch_execution_and_padding", "_preprocess",
            "_update_states_after_model_execute"]},
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector": {"OffloadingConnector": [
            "handle_preemptions", "start_load_kv", "wait_for_save", "get_finished", "build_connector_worker_meta",
            "get_num_new_matched_tokens", "update_state_after_alloc", "build_connector_meta",
            "update_connector_output", "request_finished", "request_finished_all_groups", "take_events",
            "get_kv_connector_stats", "get_transfer_results", "bind_connector_metadata",
            "clear_connector_metadata"]},
        "vllm.v1.kv_offload.tiering.manager": {"TieringOffloadingManager": [
            "on_schedule_end", "_writeback_step", "_process_finished_jobs", "_flush_pending_promotions",
            "_flush_pending_cascades", "lookup", "prepare_store", "touch", "complete_store"]},
        "torch.cuda.streams": {"Event": ["synchronize"], "Stream": ["synchronize"]},
    }
    _acc: dict = {}
    _cnt: dict = {}
    _st = {"tid": None, "t0": None, "c0": None, "rec": None, "n": 0}
    _out = os.path.join(os.environ["R3_INSTR_DIR"], f"instr-{os.getpid()}.jsonl")
    _f = None
    _patched: list = []
    _missing: list = []

    def _wrap(name, fn):
        @functools.wraps(fn)
        def w(*a, **k):
            if _st["tid"] != threading.get_ident():
                return fn(*a, **k)
            t = _pc()
            try:
                return fn(*a, **k)
            finally:
                d = _pc() - t
                _acc[name] = _acc.get(name, 0) + d
                _cnt[name] = _cnt.get(name, 0) + 1
        w.__r3__ = True
        return w

    def _emit(now_pc, now_ct):
        global _f
        r = _st["rec"]
        if r is None:
            return
        r["cycle_ns"] = now_pc - _st["t0"]
        r["cycle_cpu_ns"] = now_ct - _st["c0"]
        r["acc"] = dict(_acc)
        r["cnt"] = dict(_cnt)
        if _f is None:
            os.makedirs(os.path.dirname(_out), exist_ok=True)
            _f = open(_out, "a", buffering=1 << 16)
            _f.write(json.dumps({"meta": True, "pid": os.getpid(), "patched": _patched, "missing": _missing}) + "\n")
        _f.write(json.dumps(r) + "\n")
        _st["n"] += 1
        if _st["n"] % 64 == 0:
            _f.flush()

    def _wrap_step(fn):
        @functools.wraps(fn)
        def step(self, *a, **k):
            if _st["tid"] is None:
                _st["tid"] = threading.get_ident()
            if _st["tid"] != threading.get_ident():
                return fn(self, *a, **k)
            t, c = _pc(), _ct()
            _emit(t, c)
            _acc.clear()
            _cnt.clear()
            _st["t0"], _st["c0"] = t, c
            res = fn(self, *a, **k)
            try:
                nrun = len(self.scheduler.running)
            except Exception:  # noqa: BLE001
                nrun = -1
            _st["rec"] = {"t": time.time(), "step_ns": _pc() - t, "step_cpu_ns": _ct() - c, "nrun": nrun,
                          "exec": bool(res[1]) if isinstance(res, tuple) and len(res) == 2 else None}
            return res
        step.__r3__ = True
        return step

    def _patch(mod, spec):
        for cname, meths in spec.items():
            cls = getattr(mod, cname, None)
            if cls is None:
                _missing.append(f"{mod.__name__}.{cname}")
                continue
            if mod.__name__ == "vllm.v1.engine.core" and cname == "EngineCore":
                if not getattr(cls.step, "__r3__", False):
                    cls.step = _wrap_step(cls.step)
                    _patched.append("EngineCore.step")
            for m in meths:
                raw = inspect.getattr_static(cls, m, None)
                if raw is None:
                    _missing.append(f"{cname}.{m}")
                    continue
                if isinstance(raw, (staticmethod, classmethod)):
                    if getattr(raw.__func__, "__r3__", False):
                        continue
                    setattr(cls, m, type(raw)(_wrap(f"{cname}.{m}", raw.__func__)))
                elif callable(raw):
                    if getattr(raw, "__r3__", False):
                        continue
                    setattr(cls, m, _wrap(f"{cname}.{m}", raw))
                else:
                    _missing.append(f"{cname}.{m}")
                    continue
                _patched.append(f"{cname}.{m}")

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in TARGETS:
                return None
            for f in sys.meta_path:
                if f is self or not hasattr(f, "find_spec"):
                    continue
                spec = f.find_spec(fullname, path, target)
                if spec is not None:
                    break
            else:
                return None
            loader = spec.loader
            orig = loader.exec_module

            def exec_module(module, _orig=orig, _name=fullname):
                _orig(module)
                _patch(module, TARGETS[_name])
            loader.exec_module = exec_module
            return spec

    for _name, _spec in TARGETS.items():
        if _name in sys.modules:
            _patch(sys.modules[_name], _spec)
    sys.meta_path.insert(0, _Finder())
