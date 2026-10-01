# SPDX-License-Identifier: Apache-2.0
"""EXL3 checkpoint format facts (exllamav3 d3739fd), torch-free.

A quantized linear `<key>` is stored as (modules/quant/exl3.py, LinearEXL3):
  <key>.trellis  int16 [in/16, out/16, 16*K] (16*K + 8 for half-integer K, mul1 only)
  <key>.suh      fp16 [in]   input signs/scales, applied before the input Hadamard
  <key>.svh      fp16 [out]  output signs/scales, applied after the output Hadamard
  <key>.mul1 | <key>.mcg   int32 0-dim codebook multiplier (absent: the 3INST codebook)
  <key>.bias     fp16 [out]  optional
K (bits per weight) is per tensor. The model-level block is config.json's
`quantization_config` (quant_method "exl3", bits, head_bits, mtp_bits, codebook, and
vision_bits when the vision tower is quantized too).
"""

from __future__ import annotations

from dataclasses import dataclass

QUANT_METHOD = "exl3"
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mul1", "mcg")
# Codebook multipliers the kernels are compiled with (exl3_lib/quantize.py)
MCG_MULT = 0xCBAC1FED
MUL1_MULT = 0x83DCD12D
CODEBOOKS = ("3inst", "mcg", "mul1")  # the kernels' cb index 0, 1, 2
HAD = 128  # Hadamard block, both sides: in and out are multiples of it


def bits_from_tile(width: int) -> float:
    """K from the trellis tile width (last dim), as the kernels derive it."""
    if width % 16 == 0 and 1 <= width // 16 <= 8:
        return float(width // 16)
    if width % 16 == 8 and 1 <= width // 16 <= 3:
        return width // 16 + 0.5
    raise ValueError(f"trellis tile width {width} is not 16*K (K 1..8) or 16*K+8 (K 1..3)")


def as_uint32(value: int) -> int:
    """The int32 codebook tensor's value as the uint32 multiplier it stores."""
    return value & 0xFFFFFFFF


@dataclass(frozen=True)
class EXL3QuantConfig:
    """config.json `quantization_config` of an exllamav3 checkpoint."""

    bits: float
    head_bits: float | None
    mtp_bits: float | None
    codebook: str
    version: str | None = None
    vision_bits: float | None = None  # set when the vision tower is EXL3 too ("V" variants)

    @classmethod
    def from_dict(cls, d: dict) -> EXL3QuantConfig:
        method = d.get("quant_method")
        if method != QUANT_METHOD:
            raise ValueError(f"quant_method is {method!r}, not {QUANT_METHOD!r}")
        # exllamav3 omits codebook for the default 3INST codebook
        codebook = d.get("codebook", "3inst")
        if codebook not in CODEBOOKS:
            raise ValueError(f"unknown EXL3 codebook {codebook!r} (known: {CODEBOOKS})")
        return cls(
            bits=float(d["bits"]),
            head_bits=float(d["head_bits"]) if d.get("head_bits") is not None else None,
            mtp_bits=float(d["mtp_bits"]) if d.get("mtp_bits") is not None else None,
            codebook=codebook,
            version=d.get("version"),
            vision_bits=float(d["vision_bits"]) if d.get("vision_bits") is not None else None,
        )

    @property
    def codebook_param(self) -> str | None:
        """Name of the 0-dim codebook tensor each quantized linear carries, if any."""
        return {"3inst": None, "mcg": "mcg", "mul1": "mul1"}[self.codebook]

    @property
    def codebook_mult(self) -> int | None:
        return {"3inst": None, "mcg": MCG_MULT, "mul1": MUL1_MULT}[self.codebook]

    @property
    def mcg(self) -> bool:
        return self.codebook == "mcg"

    @property
    def mul1(self) -> bool:
        return self.codebook == "mul1"


def quantized_modules(tensor_names) -> set[str]:
    """Checkpoint module keys stored as EXL3 (those with a .trellis tensor)."""
    return {n.removesuffix(".trellis") for n in tensor_names if n.endswith(".trellis")}
