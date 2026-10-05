# Validation kit: sm86 (Ampere), measured on NVIDIA GeForce RTX 3090 (5180ec32f349, 2026-10-05)

Results describe the generation (compute capability); the card is the provenance. `python3 kit/report.py agree` checks them against other cards of the same generation.

| | |
|---|---|
| card | NVIDIA GeForce RTX 3090, 24576 MiB, compute capability 8.6, PCI 00000000:01:00.0 |
| driver / CUDA | 580.173.02 / 13.0 |
| power limit | 350 W (default 420 W) **CAPPED, numbers labelled capped** |
| max clocks | SM 2100 MHz, mem 9751 MHz |
| software | torch 2.13.0 (CUDA 13.0), vLLM 0.30.1rc1.dev285+gd28795f1a, triton 3.7.1 |
| repo | https://github.com/Starwaves1/vllm-gsqrco-exl3.git @ e5e34678d01a (dirty) |

Clocks while busy (util >= 50 %):

| step | busy s | median SM MHz | median mem MHz | median / max W | max C | clock-event reasons |
|---|---|---|---|---|---|---|
| tier1-micro | 170 | 1695 | 9501 | 346.0 / 351.6 | 58 | 0x0000000000000004 x167 |
| tier1-parity | 49 | 1950 | 9501 | 189.7 / 347.7 | 43 | 0x0000000000000004 x4 |
| tier3-exl3-swift-27b-3.50bpw-k3 | 57 | 1755 | 9501 | 348.6 / 350.2 | 53 | 0x0000000000000004 x57 |

Tiers: tier 1 **pass**, tier 3 **partial**

| step | status | note |
|---|---|---|
| tier1.parity-gguf | ok |  |
| tier1.parity-exl3 | ok |  |
| tier1.micro-gguf | ok |  |
| tier1.micro-exl3 | ok |  |
| tier3.gguf-swift-27b-iq3s-k3 | failed | server did not start: (EngineCore pid=367123) ERROR 10-05 06:47:12 [core.py:1433]     raise ValueError(msg) |
| tier3.exl3-swift-27b-3.50bpw-k3 | ok |  |

| build | status | note |
|---|---|---|
| gguf_route_l | ok |  |
| exl3 | ok |  |

## Tier 1: GGUF kernel parity (synthetic weights at the 27B shapes)

3922 passed, 0 failed, 0 errors, 183 skipped, 129.7 s

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

738 passed, 0 failed, 0 errors, 373 skipped, 76.6 s

| test | passed | failed | errors | skipped |
|---|---|---|---|---|
| test_bf16_io_same_bits | 29 | 0 | 0 | 6 |
| test_decode_exact | 7 | 0 | 0 | 0 |
| test_dequant_bitexact | 0 | 0 | 0 | 16 |
| test_deterministic | 7 | 0 | 0 | 0 |
| test_draft_head_fp8 | 3 | 0 | 0 | 0 |
| test_embed_host_gather | 4 | 0 | 0 | 0 |
| test_fp32_fp16_outputs_agree | 12 | 0 | 0 | 0 |
| test_gemm_bits_match_exllamav3 | 0 | 0 | 0 | 320 |
| test_gemm_mr_multi | 5 | 0 | 0 | 0 |
| test_gemm_vs_fp64 | 470 | 0 | 0 | 18 |
| test_graph_replay | 58 | 0 | 0 | 4 |
| test_linear_parts_same_bits | 4 | 0 | 0 | 0 |
| test_lm_head_many_rows_bounded | 2 | 0 | 0 | 0 |
| test_padded_rows_in_capture | 63 | 0 | 0 | 9 |
| test_routes_agree | 10 | 0 | 0 | 0 |
| test_routing_mr1 | 32 | 0 | 0 | 0 |
| test_routing_repacked | 20 | 0 | 0 | 0 |
| test_subprocess_case | 11 | 0 | 0 | 0 |
| test_unwarmed_capture_refused | 1 | 0 | 0 | 0 |

## Tier 1: GGUF routing winner per (type, rows) for sm86

Copy bandwidth (floor) 842 GB/s, matmul 66.0 TFLOPS, int8 68.1 TOPS; X bfloat16; Route L built. Cell: the fastest kernel summed over the type's 27B shapes (weighted by tensor count); `(+x%)` = how much slower the current routing is there.

| type | n=1 | n=2 | n=4 | n=6 | n=8 | n=12 | n=16 | n=24 | n=32 | n=48 | n=64 | n=128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IQ1_M | mmvq | mmvq | mmvq | mmvq | mmvq | mmvq/8 | mmvq/8 | mmvq/8 | mmvq/8 | s-dq+cublas | s-dq+cublas | s-dq+cublas |
| IQ2_S | own | own | own | own | mma_k (+6%) | mma_k | mma_k | mma_k | mma_k | mmq | mmq | mmq |
| IQ2_XS | mmvq | mmvq | mmvq | mmq (+5%) | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq |
| IQ2_XXS | mmvq | mmvq | mmvq | mmq (+12%) | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq |
| IQ3_S | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p |
| IQ3_XXS | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-mma-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p | iq3-tiled-p |
| IQ4_XS | mmvq | mmvq | mmvq | mma_k (+11%) | mma_k (+5%) | mma_k | mma_k | mma_k | mma_k | mmq | mmq | mmq |
| Q2_K | mmvq | mmvq | mmq (+20%) | mmq (+52%) | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq |
| Q4_K | mmvq | mmvq | own | mma_k (+5%) | mma_k (+8%) | mma_k | mma_k | mma_k | mma_k | mmq | mmq | mmq |
| Q6_K | mmvq | mmvq | mmvq | mmvq | mmq | mmq | mmq | mmq | mmq | mmq | mmq | mmq |

Whole-model GEMM time per forward (all 27B linears incl. lm_head and the MTP layer), ms:

| rows | current routing | best per cell | vendored llama.cpp | copy-bandwidth floor | routing / floor |
|---|---|---|---|---|---|
| 1 | 16.439 | 16.338 | 17.615 | 13.828 | 1.19 |
| 2 | 17.068 | 17.063 | 19.606 | 13.828 | 1.23 |
| 4 | 18.973 | 18.811 | 26.084 | 13.828 | 1.37 |
| 6 | 21.949 | 20.657 | 31.451 | 13.828 | 1.59 |
| 8 | 22.081 | 21.466 | 32.475 | 13.828 | 1.60 |
| 12 | 23.661 | 23.451 | 36.378 | 13.828 | 1.71 |
| 16 | 23.572 | 23.232 | 34.767 | 13.828 | 1.70 |
| 24 | 27.958 | 27.877 | 36.92 | 13.828 | 2.02 |
| 32 | 29.773 | 29.641 | 41.297 | 13.828 | 2.15 |
| 48 | 36.556 | 36.503 | 44.849 | 13.828 | 2.64 |
| 64 | 43.521 | 43.45 | 50.957 | 13.828 | 3.15 |
| 128 | 72.809 | 72.521 | 80.278 | 13.828 | 5.27 |

CUDA-graph replay bit-identical to eager: 2466 of 2466 timed cells

## Tier 1: EXL3 route winner per (bits K, rows)

| K | 1 | 2 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm | gemm | dequant |
| 3 | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr |
| 4 | mr 8/gemm 1 | mr | mr 8/gemm 1 | mr 8/gemm 1 | mr 8/gemm 1 | mr | mr | mr | mr | mr | mr | mr |
| 5 | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr | mr |

| rows | best route ms per forward | exl3_gemm (dequant above 144) ms | floor ms |
|---|---|---|---|
| 1 | 18.343 | 22.829 | 13.65 |
| 2 | 18.696 | 23.616 | 13.65 |
| 4 | 18.874 | 24.161 | 13.65 |
| 6 | 18.997 | 24.62 | 13.65 |
| 8 | 19.113 | 25.022 | 13.65 |
| 12 | 22.003 | 26.039 | 13.65 |
| 16 | 22.417 | 27.488 | 13.65 |
| 24 | 26.365 | 51.336 | 13.65 |
| 32 | 27.272 | 54.289 | 13.65 |
| 48 | 38.373 | 80.825 | 13.65 |
| 64 | 63.746 | 106.641 | 13.65 |
| 128 | 118.86 | 208.182 | 13.65 |

## Tier3: exl3-swift-27b-3.50bpw-k3

exl3-swift-27b-3.50bpw-k3; MTP k=3, T=0, 256 tokens/request

| c | tok/s | ms/step | tok/step | mean TPOT ms |
|---|---|---|---|---|
| 1 | 101.0 | 27.12 | 2.74 | 9.90 |
| 2 | 180.0 | 27.30 | 2.46 | 11.11 |
| 4 | 300.9 | 32.51 | 2.45 | 13.29 |
| 8 | 481.5 | 41.07 | 2.47 | 16.62 |

Corruption check (exl3-swift-27b-3.50bpw-k3): 200 of 200 flagged, by kind {'early_eos': 0, 'repeat': 0, 'foreign': 0, 'bad_utf8': 0, 'error': 200}, T=0 token mismatches vs eager reference 0
