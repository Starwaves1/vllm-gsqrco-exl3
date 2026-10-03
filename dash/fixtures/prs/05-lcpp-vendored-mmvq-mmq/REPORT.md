# 05: Vendored llama.cpp MMVQ / MMQ (FIXTURE)

## Initial problem and concise proof

- Decode at 1..8 rows is 1.6x slower than llama.cpp on the same weights.
- Measured with `bench/ladder.py`, 5 runs each.

## Problem mechanism (diagnosis, checked and verified)

The plugin's MMVQ kernels are an older copy of llama.cpp's and *miss* the newer tiling.

## Fix mechanism

Vendor llama.cpp b11211 behind a shim and a default-off flag.

## Final check agent's concise opinion

Not yet checked.

## Fable's comment

Not yet written.
