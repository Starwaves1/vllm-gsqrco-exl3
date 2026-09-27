"""Shared settings for tests/gpu (a uniquely named module: tests/cpu has its own
conftest, so test modules must not `from conftest import`)."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "bench" / "parity"))
sys.path.insert(0, str(Path(__file__).parent))
GPU = os.environ.get("GSQ_ALLOW_GPU") == "1"
if not GPU:
    import no_gpu  # noqa: F401

GGUF = Path(os.environ.get("GSQ_GGUF") or next(
    (p for p in [ROOT / "models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf",
                 Path.home() / "qwen38-27b-rtx3090/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-GGUF/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"]
     if p.exists()), ROOT / "models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"))
PORT = int(os.environ.get("GSQ_PORT", "18090"))
URL = os.environ.get("GSQ_URL", f"http://127.0.0.1:{PORT}")
API_KEY = os.environ.get("GSQ_API_KEY", "gsq-local-test")
RUNS = Path(os.environ.get("GSQ_RUNS", ROOT / "runs"))

# The 10 quantized ggml types in the Swift IQ3_S-mtp GGUF (header read 2026-09-27),
# plus F32/BF16 which load unquantized (weight_utils.py) and have no kernel.
QUANT_TYPES = ["IQ1_M", "IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ3_S", "IQ3_XXS", "IQ4_XS", "Q2_K", "Q4_K", "Q6_K"]
