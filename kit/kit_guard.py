"""pytest plugin (-p kit_guard): cap this process's CUDA memory at KIT_MEM_FRACTION of the card, so a
test that outgrows a small card fails with an allocator error instead of pushing the card to OOM."""

import os


def pytest_configure(config):
    frac = os.environ.get("KIT_MEM_FRACTION")
    if frac and os.environ.get("GSQ_ALLOW_GPU") == "1":
        import torch

        torch.cuda.set_per_process_memory_fraction(float(frac))
