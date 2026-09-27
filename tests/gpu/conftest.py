"""GPU test suite (Phase B). Opt-in: without GSQ_ALLOW_GPU=1 every test is skipped and
tools/no_gpu.py is imported before anything else, so collection never touches the GPU.

  GSQ_ALLOW_GPU=1 tools/pytest tests/gpu -v              (on a GPU box, repo root)
  tools/pytest tests/gpu --collect-only                  (anywhere; no GPU)

Environment: GSQ_GGUF (the .gguf), GSQ_URL / GSQ_SERVER_LOG (reuse a running serve-gsq.sh
instead of starting one), GSQ_ALLOW_BESIDE_PROD=1 (only with Garrett's go).
Heavy imports (torch, vllm, the plugin) happen inside fixtures and tests, never at
module level.
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from gsq_gpu import API_KEY, GGUF, GPU, PORT, ROOT, RUNS, URL  # noqa: E402,F401  (imports no_gpu when not GPU)


def _listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def pytest_configure(config):
    if PORT in (18080, 18081) or ":18080" in URL or ":18081" in URL:
        pytest.exit("refusing production ports 18080/18081", returncode=2)
    if GPU and (_listening(18080) or _listening(18081)) and os.environ.get("GSQ_ALLOW_BESIDE_PROD") != "1":
        pytest.exit("production is live on this host; GPU tests refused (GSQ_ALLOW_BESIDE_PROD=1 overrides)", returncode=2)


def pytest_collection_modifyitems(config, items):
    if GPU:
        return
    skip = pytest.mark.skip(reason="GPU tests are opt-in: GSQ_ALLOW_GPU=1")
    for it in items:
        it.add_marker(skip)


@pytest.fixture(scope="session")
def gguf_reader():
    import gguf

    if not GGUF.exists():
        pytest.skip(f"GGUF not found: {GGUF}")
    return gguf.GGUFReader(str(GGUF))


@pytest.fixture(scope="session")
def tensors_by_type(gguf_reader):
    """type name -> list of 2-D ReaderTensors (data: uint8 [rows, bytes_per_row])."""
    out: dict[str, list] = {}
    for t in gguf_reader.tensors:
        if len(t.shape) == 2:
            out.setdefault(t.tensor_type.name, []).append(t)
    return out


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


@pytest.fixture(scope="session")
def gsq_server():
    """(url, log path) of a serve-gsq.sh server: reuse GSQ_URL if healthy (then GSQ_SERVER_LOG
    must name its log), else start one and stop it at the end of the session."""
    if _healthy(URL):
        log = os.environ.get("GSQ_SERVER_LOG")
        yield URL, Path(log) if log else None
        return
    run = RUNS / time.strftime("%Y%m%d-%H%M%S-gputests")
    run.mkdir(parents=True, exist_ok=True)
    log = run / "server.log"
    env = dict(os.environ, GSQ_LOG=str(log))
    proc = subprocess.Popen([str(ROOT / "scripts/serve-gsq.sh")], env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.time() + 2400
        while not _healthy(URL):
            if proc.poll() is not None:
                pytest.fail(f"serve-gsq.sh exited with {proc.returncode}; see {log}")
            if time.time() > deadline:
                pytest.fail(f"server not healthy after 40 min; see {log}")
            time.sleep(5)
        yield URL, log
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
