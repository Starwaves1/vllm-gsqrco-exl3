# Validation kit: sm86 (Ampere), measured on NVIDIA GeForce RTX 3070 (torn-gpu, 2026-10-04)

Results describe the generation (compute capability); the card is the provenance. `python3 kit/report.py agree` checks them against other cards of the same generation.

| | |
|---|---|
| card | NVIDIA GeForce RTX 3070, 8192 MiB, compute capability 8.6, PCI 00000000:04:00.0 |
| driver / CUDA | 595.84 / 13.4 |
| power limit | 220 W (default 220 W) |
| max clocks | SM 2100 MHz, mem 7001 MHz |
| software | torch 2.13.0 (CUDA 13.0), vLLM 0.30.1rc1.dev285+gd28795f1a, triton 3.7.1 |
| repo | https://github.com/Starwaves1/vllm-gsqrco-exl3 @ a09946cf08dd |

Clocks while busy (util >= 50 %):

| step | busy s | median SM MHz | median mem MHz | median / max W | max C | clock-event reasons |
|---|---|---|---|---|---|---|
| tier1-micro | 305 | 1800 | 6801 | 210.6 / 221.2 | 74 | 0x0000000000000004 x280 |
| tier1-parity | 57 | 1920 | 6801 | 89.6 / 148.0 | 65 | 0x0000000000000004 x4 |

Tiers: tier 1 **partial**

| step | status | note |
|---|---|---|
| tier1.parity-gguf | ok |  |
| tier1.parity-exl3 | failed | test failures, see parity-exl3.log |
| tier1.micro-gguf | ok |  |
| tier1.micro-exl3 | ok |  |

| build | status | note |
|---|---|---|
| gguf_route_l | ok |  |
| exl3 | ok |  |

## Tier 1: GGUF kernel parity (synthetic weights at the 27B shapes)

3922 passed, 0 failed, 0 errors, 183 skipped, 235.8 s

| test | passed | failed | errors | skipped |
|---|---|---|---|---|
| test_dequantize | 30 | 0 | 0 | 0 |
| test_iq3_pack_inplace_peak | 2 | 0 | 0 | 0 |
| test_iq3_pack_roundtrip | 2 | 0 | 0 | 0 |
| test_lcpp_graph_replay | 84 | 0 | 0 | 6 |
| test_lcpp_iq1_m_chunks | 3 | 0 | 0 | 0 |
| test_lcpp_iq3 | 912 | 0 | 0 | 96 |
| test_lcpp_iq3_graph_replay | 24 | 0 | 0 | 0 |
| test_lcpp_iq3_packed | 960 | 0 | 0 | 0 |
| test_lcpp_iq3_packed_graph_replay | 10 | 0 | 0 | 0 |
| test_lcpp_iq3_packed_tiled | 450 | 0 | 0 | 0 |
| test_lcpp_iq3_packed_tiled_graph_replay | 8 | 0 | 0 | 0 |
| test_lcpp_mixed_shard_layer | 24 | 0 | 0 | 0 |
| test_lcpp_mma_k | 333 | 0 | 0 | 45 |
| test_lcpp_mma_k_first_call_in_capture | 1 | 0 | 0 | 0 |
| test_lcpp_mma_k_graph_replay | 27 | 0 | 0 | 0 |
| test_lcpp_mma_k_whole_tensor | 18 | 0 | 0 | 0 |
| test_lcpp_mmq | 216 | 0 | 0 | 0 |
| test_lcpp_mmq_odd_rows | 18 | 0 | 0 | 0 |
| test_lcpp_mmvq | 160 | 0 | 0 | 0 |
| test_lcpp_packed_layer | 12 | 0 | 0 | 0 |
| test_lcpp_quantize_vs_vendored | 228 | 0 | 0 | 12 |
| test_lcpp_same_type_run | 40 | 0 | 0 | 8 |
| test_lcpp_x_q8 | 30 | 0 | 0 | 0 |
| test_lcpp_x_q8_graph_replay | 24 | 0 | 0 | 0 |
| test_mmq | 18 | 0 | 0 | 0 |
| test_mmvq | 120 | 0 | 0 | 0 |
| test_quantize_x_q8_1_mixed_route | 1 | 0 | 0 | 0 |
| test_routing_packed_whole_tensor | 18 | 0 | 0 | 0 |
| test_routing_whole_tensor | 144 | 0 | 0 | 16 |
| test_unquantized_small_n | 5 | 0 | 0 | 0 |

## Tier 1: EXL3 kernel parity (synthetic weights at the 27B shapes)

398 passed, 340 failed, 0 errors, 373 skipped, 98.3 s

| test | passed | failed | errors | skipped |
|---|---|---|---|---|
| test_bf16_io_same_bits | 0 | 29 | 0 | 6 |
| test_decode_exact | 6 | 1 | 0 | 0 |
| test_dequant_bitexact | 0 | 0 | 0 | 16 |
| test_deterministic | 0 | 7 | 0 | 0 |
| test_draft_head_fp8 | 0 | 3 | 0 | 0 |
| test_embed_host_gather | 0 | 4 | 0 | 0 |
| test_fp32_fp16_outputs_agree | 10 | 2 | 0 | 0 |
| test_gemm_bits_match_exllamav3 | 0 | 0 | 0 | 320 |
| test_gemm_mr_multi | 0 | 5 | 0 | 0 |
| test_gemm_vs_fp64 | 318 | 152 | 0 | 18 |
| test_graph_replay | 42 | 16 | 0 | 4 |
| test_linear_parts_same_bits | 0 | 4 | 0 | 0 |
| test_lm_head_many_rows_bounded | 0 | 2 | 0 | 0 |
| test_padded_rows_in_capture | 0 | 63 | 0 | 9 |
| test_routes_agree | 10 | 0 | 0 | 0 |
| test_routing_mr1 | 0 | 32 | 0 | 0 |
| test_routing_repacked | 0 | 20 | 0 | 0 |
| test_subprocess_case | 11 | 0 | 0 | 0 |
| test_unwarmed_capture_refused | 1 | 0 | 0 | 0 |

First failures:

- `test_gemm_vs_fp64[K4-lmhead-1024-fp32]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.89 GiB. GPU 0 has a total capacity of 7.66 GiB of which 1.43 GiB is free. Including non-PyTorch 
- `test_gemm_vs_fp64[K4-lmhead-1024-fp16]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.25 GiB. GPU 0 has a total capacity of 7.66 GiB of which 1.76 GiB is free. Including non-PyTorch 
- `test_fp32_fp16_outputs_agree[K3-down-16]`: AssertionError: {'finite': True, 'max_rel': 0.019941873516840798, 'ref_rms': 6.573942041342564, 'rel_rms': 0.004036329551672158}
assert 0.004036329551672158 <= 
- `test_fp32_fp16_outputs_agree[K3-down-48]`: AssertionError: {'finite': True, 'max_rel': 0.018686850654170566, 'ref_rms': 6.607081113336081, 'rel_rms': 0.00401733257224903}
assert 0.00401733257224903 <= 0.
- `test_decode_exact[K4-lmhead]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 608.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 997.75 MiB is free. Including non-PyTo
- `test_gemm_vs_fp64[K3-down-17-bf16model]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 680.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 797.75 MiB is free. Including non-PyTo
- `test_gemm_vs_fp64[K3-down-17-fp16model]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 680.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 797.75 MiB is free. Including non-PyTo
- `test_gemm_vs_fp64[K3-down-24-bf16model]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 680.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 793.75 MiB is free. Including non-PyTo
- `test_gemm_vs_fp64[K3-down-24-fp16model]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 680.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 793.75 MiB is free. Including non-PyTo
- `test_gemm_vs_fp64[K3-down-32-bf16model]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 680.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 793.75 MiB is free. Including non-PyTo
- `test_gemm_vs_fp64[K3-down-32-fp16model]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 680.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 793.75 MiB is free. Including non-PyTo
- `test_gemm_vs_fp64[K3-down-48-bf16model]`: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 680.00 MiB. GPU 0 has a total capacity of 7.66 GiB of which 771.75 MiB is free. Including non-PyTo

## Tier 1: GGUF routing winner per (type, rows) for sm86

Copy bandwidth (floor) 404 GB/s, matmul 41.1 TFLOPS, int8 40.2 TOPS; X bfloat16; Route L built. Cell: the fastest kernel summed over the type's 27B shapes (weighted by tensor count); `(+x%)` = how much slower the current routing is there.

| type | n=1 | n=2 | n=4 | n=6 | n=8 | n=12 | n=16 | n=24 | n=32 | n=48 | n=64 | n=128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IQ1_M | mmvq | mmvq | mmvq | mmvq | mmvq | mmvq/8 | mmvq/8 | mmvq/8 | mmvq/8 | s-dq+cublas | s-dq+cublas | s-dq+cublas |
| IQ2_S | own | own | own | own | mma_k (+4%) | mma_k | mma_k | mma_k | mma_k | mmq | mma_k (+5%) | mmq |
| IQ2_XS | mmvq | mmvq | mmvq | mmq (+11%) | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq |
| IQ2_XXS | mmvq | mmvq | mmq (+9%) | mmq (+20%) | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq |
| IQ3_S | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p (+5%) | iq3-mma-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p |
| IQ3_XXS | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p |
| IQ4_XS | mmvq | mmvq | mmvq | mma_k (+4%) | mma_k (+3%) | mma_k | mma_k | mma_k | mma_k | mmq | mmq | mmq |
| Q2_K | mmvq | mmvq | s-mmq (+25%) | mmq (+57%) | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq |
| Q4_K | mmvq | mmvq | mma_k | mma_k (+15%) | mma_k (+12%) | mma_k | mma_k | mma_k | mma_k | mmq | mmq | mmq |
| Q6_K | mmvq | mmvq | mmvq | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq |

Whole-model GEMM time per forward (all 27B linears incl. lm_head and the MTP layer), ms:

| rows | current routing | best per cell | vendored llama.cpp | copy-bandwidth floor | routing / floor |
|---|---|---|---|---|---|
| 1 | 30.768 | 30.649 | 31.946 | 28.813 | 1.07 |
| 2 | 31.459 | 31.455 | 34.661 | 28.813 | 1.09 |
| 4 | 33.456 | 32.955 | 42.231 | 28.813 | 1.16 |
| 6 | 37.058 | 34.931 | 48.107 | 28.813 | 1.29 |
| 8 | 37.195 | 36.066 | 49.781 | 28.813 | 1.29 |
| 12 | 38.619 | 37.856 | 54.503 | 28.813 | 1.34 |
| 16 | 39.348 | 38.605 | 53.785 | 28.813 | 1.37 |
| 24 | 44.541 | 44.285 | 57.408 | 28.813 | 1.55 |
| 32 | 47.804 | 47.446 | 64.685 | 28.813 | 1.66 |
| 48 | 59.207 | 59.191 | 72.753 | 28.813 | 2.05 |
| 64 | 70.908 | 70.476 | 82.212 | 28.813 | 2.46 |
| 128 | 114.927 | 114.927 | 129.586 | 28.813 | 3.99 |

CUDA-graph replay bit-identical to eager: 2466 of 2466 timed cells

## Tier 1: EXL3 route winner per (bits K, rows)

| K | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm |
| 3 | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr |
| 4 | gemm 4/mr 5 | mr 6/gemm 3 | mr 6/gemm 3 | gemm 4/mr 5 | mr 6/gemm 3 | mr 6/gemm 3 | mr 6/gemm 3 | mr | mr | mr | mr | mr |
| 5 | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr |

| rows | best route ms per forward | exl3_gemm (dequant above 144) ms | floor ms |
|---|---|---|---|
| 1 | 31.972 | 36.554 | 28.456 |
| 2 | 32.229 | 36.942 | 28.456 |
| 4 | 32.545 | 37.373 | 28.456 |
| 6 | 32.9 | 37.689 | 28.456 |
| 8 | 33.189 | 38.127 | 28.456 |
| 12 | 36.538 | 39.351 | 28.456 |
| 16 | 37.195 | 40.527 | 28.456 |
| 24 | 43.925 | 78.516 | 28.456 |
| 32 | 45.203 | 80.711 | 28.456 |
| 48 | 60.814 | 121.39 | 28.456 |
| 64 | 104.609 | 163.706 | 28.456 |
| 128 | 199.251 | 330.106 | 28.456 |
