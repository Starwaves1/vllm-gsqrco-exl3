// EXL3 trellis window extraction + codebook decode, register form, K = 4.
//
// Derived from ExLlamaV3 v1.5.1 (MIT, Copyright (c) 2025 Turboderp):
//   exllamav3_ext/quant/exl3_gemv_kernel.cuh  (dq8_regs_4bits, decode_pair_cb2_dp4a_, decode8)
//   exllamav3_ext/quant/codebook.cuh          (decode_3inst_2, vendored unchanged in third_party/exllamav3)
//   exllamav3_ext/util.cuh                    (half4, half_uint16 / half2_uint32 unions)
// The instruction sequence that produces each fp16 weight is copied unchanged so that decoded values are
// bit-identical to exllamav3_ext.reconstruct. Do not "optimise" the arithmetic here.
#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

typedef struct __align__(8) half4
{
    half2 x;
    half2 y;
    __device__ half4() = default;
    __device__ half4(half2 x_, half2 y_) : x(x_), y(y_) {}
    __device__ half4(half h0, half h1, half h2, half h3) :
         x(__halves2half2(h0, h1)),
         y(__halves2half2(h2, h3)) {}
}
half4;

union half2_uint32
{
    uint32_t as_uint32;
    half2 as_half2;
    __device__ half2_uint32(uint32_t val) : as_uint32(val) {}
    __device__ half2_uint32(half2 val) : as_half2(val) {}
    __device__ half2_uint32() : as_uint32(0) {}
};

union half_uint16
{
    uint16_t as_uint16;
    half as_half;
    __device__ half_uint16(uint16_t val) : as_uint16(val) {}
    __device__ half_uint16(half val) : as_half(val) {}
    __device__ half_uint16() : as_uint16(0) {}
};

#include "third_party/exllamav3/codebook.cuh"

#define TRELLIS_FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define TRELLIS_BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))

namespace trellis_exl3 {

// cb 13 = MCG through a shared-memory table: lut[w] = decode_3inst<1>(w) for every 16-bit window, generated on the
// device by the codebook arithmetic itself (bit-exact by construction; trellis_build_mcg_lut in exl3_marlin_moe.cu).
// Two LDS.U16 + one pack per weight pair instead of two multiplies, two lop3, two shuffles and an hadd2.
#define TRELLIS_CB_MCG_LUT 13

#ifndef TRELLIS_MCG_SELFADD
  #define TRELLIS_MCG_SELFADD 1
#endif
#ifndef TRELLIS_K3_IMAD_SHIFTS
  #define TRELLIS_K3_IMAD_SHIFTS 0   // 0 | 3 | 6 of the K=3 window shifts as IMAD.HI on the FMA pipe (experiment)
#endif
// MCG pair with the same fp16 adds as decode_3inst_2<1> (bit-identical: value = fp16(lo16) + fp16(hi16) of the masked
// product) but gathered differently: one fp16 self-add per weight (HADD2 with an .H1_H1 source swizzle, value in the
// low half) and ONE permute per pair, instead of two permutes (lows / highs) and one HADD2. Two INT-pipe ops per pair
// become fp16-pipe ops.
__device__ __forceinline__ half2 decode_mcg_selfadd_2(uint32_t w0, uint32_t w1)
{
    uint32_t x0 = w0 * 0xCBAC1FEDu;
    uint32_t x1 = w1 * 0xCBAC1FEDu;
    asm ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
    asm ("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
    half2 h0 = half2_uint32(x0).as_half2;
    half2 h1 = half2_uint32(x1).as_half2;
    half v0 = __hadd(__low2half(h0), __high2half(h0));
    half v1 = __hadd(__low2half(h1), __high2half(h1));
    return __halves2half2(v0, v1);
}

template <typename FragB>
__device__ __forceinline__ void decode8_lut(const half* __restrict__ lut, uint32_t w0, uint32_t w1, uint32_t w2,
    uint32_t w3, uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    f0[0] = __halves2half2(lut[w0], lut[w1]);
    f0[1] = __halves2half2(lut[w2], lut[w3]);
    f1[0] = __halves2half2(lut[w4], lut[w5]);
    f1[1] = __halves2half2(lut[w6], lut[w7]);
}

// B fragment of the m16n8k16 MMA: 2 x half2 = 4 weights
template <typename FragB, int cb>
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    if constexpr (cb == 10 || cb == 11)
    {
        // TIMING PROBES: cb 10 = extraction + the 8 multiplies only; cb 11 = + the 8 dp4a byte sums (no pack/hfma2)
        uint32_t x[8] = {w0 * 0x83DCD12Du, w1 * 0x83DCD12Du, w2 * 0x83DCD12Du, w3 * 0x83DCD12Du,
                         w4 * 0x83DCD12Du, w5 * 0x83DCD12Du, w6 * 0x83DCD12Du, w7 * 0x83DCD12Du};
        if constexpr (cb == 11)
            for (int i = 0; i < 8; i++) x[i] = __dp4a(x[i], 0x01010101u, 0x6400u);
        f0[0] = half2_uint32((x[0] & 0xffff) | (x[1] << 16)).as_half2;
        f0[1] = half2_uint32((x[2] & 0xffff) | (x[3] << 16)).as_half2;
        f1[0] = half2_uint32((x[4] & 0xffff) | (x[5] << 16)).as_half2;
        f1[1] = half2_uint32((x[6] & 0xffff) | (x[7] << 16)).as_half2;
    }
    else if constexpr (cb == 12)
    {
        // TIMING PROBE: full MUL1 codebook arithmetic on un-extracted inputs (8 distinct multiplies, not CSE-able)
        f0[0] = decode_mul1_product_2(w0 * 0x83DCD12Du, w1 * 0x83DCD12Fu);
        f0[1] = decode_mul1_product_2(w0 * 0x83DCD131u, w1 * 0x83DCD133u);
        f1[0] = decode_mul1_product_2(w0 * 0x83DCD135u, w1 * 0x83DCD137u);
        f1[1] = decode_mul1_product_2(w0 * 0x83DCD139u, w1 * 0x83DCD13Bu);
    }
    else if constexpr (cb == 3 || cb == 5 || cb == 9)
    {
        // TIMING PROBES, never served (not in the default build: TRELLIS_CODEBOOKS=2,3,4): cb 3 = window extraction
        // kept, codebook arithmetic dropped (raw window bits as fp16); cb 4 = see dq8_regs_*: no extraction, no shuffle; cb 5 = cb 3 without the shuffle.
        f0[0] = half2_uint32(w0 | (w1 << 16)).as_half2;
        f0[1] = half2_uint32(w2 | (w3 << 16)).as_half2;
        f1[0] = half2_uint32(w4 | (w5 << 16)).as_half2;
        f1[1] = half2_uint32(w6 | (w7 << 16)).as_half2;
    }
    else if constexpr (cb == 2 || cb >= 6)   // cb 6, 7, 8, 13, 14, 15 = MUL1 with an experimental kernel structure (probe builds)
    {
        f0[0] = decode_mul1_product_2(w0 * 0x83DCD12Du, w1 * 0x83DCD12Du);
        f0[1] = decode_mul1_product_2(w2 * 0x83DCD12Du, w3 * 0x83DCD12Du);
        f1[0] = decode_mul1_product_2(w4 * 0x83DCD12Du, w5 * 0x83DCD12Du);
        f1[1] = decode_mul1_product_2(w6 * 0x83DCD12Du, w7 * 0x83DCD12Du);
    }
    else if constexpr (cb == 1 && TRELLIS_MCG_SELFADD)
    {
        f0[0] = decode_mcg_selfadd_2(w0, w1);
        f0[1] = decode_mcg_selfadd_2(w2, w3);
        f1[0] = decode_mcg_selfadd_2(w4, w5);
        f1[1] = decode_mcg_selfadd_2(w6, w7);
    }
    else
    {
        f0[0] = decode_3inst_2<cb>(w0, w1);
        f0[1] = decode_3inst_2<cb>(w2, w3);
        f1[0] = decode_3inst_2<cb>(w4, w5);
        f1[1] = decode_3inst_2<cb>(w6, w7);
    }
}

// K = 4: lane l of a 16x16 tile decodes sequence positions 8l .. 8l+7 from its own stream word `b` and the
// previous lane's word `a` (tail-biting: lane 0 uses lane 31). Same order as ExLlamaV3's dq8_aligned_4bits.
// f0 = positions 0..3 (tile column l/4), f1 = positions 4..7 (tile column l/4 + 8); k rows 2(l%4) + {0,1,8,9}:
// the tensor-core B fragment order, which is how EXL3 stores tiles.
template <typename FragB, int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1,
                                               const half* __restrict__ lut = nullptr)
{
    if constexpr (cb == 4)   // timing probe: no window extraction, no codebook
    {
        f0[0] = half2_uint32(b).as_half2; f0[1] = half2_uint32(a).as_half2;
        f1[0] = half2_uint32(b ^ a).as_half2; f1[1] = half2_uint32(b + a).as_half2;
        return;
    }
    if constexpr (cb == 12)  // timing probe: codebook without window extraction
    {
        decode8<FragB, cb>(b, a, 0, 0, 0, 0, 0, 0, f0, f1);
        return;
    }
    if constexpr (cb == 9)   // timing probe: half the extraction (funnel + 3 bfe + mask), no codebook
    {
        uint32_t s9, v0, v1, v2, v3;
        TRELLIS_FSHF_IMM(s9, b, a, 20);
        v3 = b & 0xffff;
        TRELLIS_BFE16_IMM(v2, b, 8);
        TRELLIS_BFE16_IMM(v1, b, 16);
        TRELLIS_BFE16_IMM(v0, s9, 8);
        decode8<FragB, cb>(v0, v0, v1, v1, v2, v2, v3, v3, f0, f1);
        return;
    }
    if constexpr (cb == 15)  // experiment: a second funnel (>> 4) makes windows 6, 4, 2 byte aligned: 11 ops instead of 12
    {
        uint32_t s4, s20, v0, v1, v2, v3, v4, v5, v6, v7;
        TRELLIS_FSHF_IMM(s4, b, a, 4);
        TRELLIS_FSHF_IMM(s20, b, a, 20);
        v7 = b & 0xffff;
        TRELLIS_BFE16_IMM(v5, b, 8);
        TRELLIS_BFE16_IMM(v3, b, 16);
        v6 = s4 & 0xffff;
        TRELLIS_BFE16_IMM(v4, s4, 8);
        TRELLIS_BFE16_IMM(v2, s4, 16);
        TRELLIS_BFE16_IMM(v1, s20, 4);
        TRELLIS_BFE16_IMM(v0, s20, 8);
        decode8<FragB, cb>(v0, v1, v2, v3, v4, v5, v6, v7, f0, f1);
        return;
    }
    if constexpr (cb == 14)  // experiment: every window = one permute; nibble offsets pre-aligned by a multiply
    {
        // (v >> 4) & 0xffff == bytes 2,3 of v * 2^12; (v >> 12) & 0xffff == bytes 2,3 of v * 2^4 (mod 2^32).
        // In the final code a bfe at a nibble offset is SHF.R + SGXT (two INT-unit ops), a permute is one and the
        // multiply runs on the (idle) multiplier.
        uint32_t s14, v0, v1, v2, v3, v4, v5, v6, v7;
        TRELLIS_FSHF_IMM(s14, b, a, 20);
        v7 = b & 0xffff;
        v6 = __byte_perm(b * 4096u, 0u, 0x7732);
        v5 = __byte_perm(b, 0u, 0x7721);
        v4 = __byte_perm(b * 16u, 0u, 0x7732);
        v3 = __byte_perm(b, 0u, 0x7732);
        v2 = s14 & 0xffff;
        v1 = __byte_perm(s14 * 4096u, 0u, 0x7732);
        v0 = __byte_perm(s14, 0u, 0x7721);
        decode8<FragB, cb>(v0, v1, v2, v3, v4, v5, v6, v7, f0, f1);
        return;
    }
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    if constexpr (cb == 6)   // experiment: plain shifts and masks instead of bfe
    {
        TRELLIS_FSHF_IMM(s, b, a, 20);
        w7 = b & 0xffff; w6 = (b >> 4) & 0xffff; w5 = (b >> 8) & 0xffff; w4 = (b >> 12) & 0xffff; w3 = b >> 16;
        w2 = s & 0xffff; w1 = (s >> 4) & 0xffff; w0 = (s >> 8) & 0xffff;
        decode8<FragB, cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
        return;
    }
    if constexpr (cb == 7)   // experiment: 64-bit field extracts, no funnel shift
    {
        uint64_t v = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
        uint64_t r;
  #define TRELLIS_BFE64(dst, off) asm("bfe.u64 %0, %1, " #off ", 16;" : "=l"(r) : "l"(v)); dst = (uint32_t)r
        TRELLIS_BFE64(w7, 0); TRELLIS_BFE64(w6, 4); TRELLIS_BFE64(w5, 8); TRELLIS_BFE64(w4, 12);
        TRELLIS_BFE64(w3, 16); TRELLIS_BFE64(w2, 20); TRELLIS_BFE64(w1, 24); TRELLIS_BFE64(w0, 28);
  #undef TRELLIS_BFE64
        decode8<FragB, cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
        return;
    }
    TRELLIS_FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    TRELLIS_BFE16_IMM(w6, b, 4);
    TRELLIS_BFE16_IMM(w5, b, 8);
    TRELLIS_BFE16_IMM(w4, b, 12);
    TRELLIS_BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    TRELLIS_BFE16_IMM(w1, s, 4);
    TRELLIS_BFE16_IMM(w0, s, 8);
    if constexpr (cb == TRELLIS_CB_MCG_LUT)
        decode8_lut<FragB>(lut, w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
    else
        decode8<FragB, cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// K = 3: a lane's 8 windows are the 16-bit fields ending at stream bits 24l + 3(p+1), p = 0..7, i.e. bits
// [24l - 13, 24l + 24) of the tail-biting 768-bit tile stream. They always sit in two consecutive tile words at a
// lane-constant alignment, so the tile is staged byte-exact (24 words) and every lane gathers its two words
// (k3_lane_words) and funnel-shifts them. Verbatim port of ExLlamaV3's dq8_regs_3bits (exl3_gemv_kernel.cuh) and
// its lane constants (exl3_gemv_kernel.cuh, bits == 3 branch); `a` is the HIGH word (tile word src_a), `b` the LOW
// word (src_b), fshift(b, a, s) = ((a << 32) | b) >> s, s in (0, 44]. Fragment order as for K = 4.
__device__ __forceinline__ uint32_t k3_fshift(uint32_t b, uint32_t a, int shift)
{
    uint64_t merged = ((uint64_t)a << 32) | (uint64_t)b;
    return (uint32_t)(merged >> shift);
}

// Per-lane constants: tile word indices of the two source words and the shift of the lowest window.
__device__ __forceinline__ void k3_lane_words(int lane, int& src_a, int& src_b, int& s2)
{
    int t_offset = lane << 3;
    int b1 = (t_offset + 257) * 3;
    int b2 = b1 + 21;
    int i0 = (b1 - 16) / 32;
    int i2 = (b2 - 1) / 32;
    s2 = (i2 + 1) * 32 - b2;
    src_a = i0 % 24;
    src_b = i2 % 24;
}

template <typename FragB, int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1,
                                               const half* __restrict__ lut = nullptr)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = k3_fshift(b, a, s2);
    w3 = k3_fshift(b, a, s2 + 12);
    if constexpr (TRELLIS_K3_IMAD_SHIFTS >= 6)
    {
        // Right shifts as IMAD.HI (x * 2^(32-n) >> 32 == x >> n, exact for unsigned): the FMA pipe instead of the
        // INT pipe, which is the critical one for MCG decode. 6 of the 8 windows.
        w6 = __umulhi(w7, 1u << 29); w5 = __umulhi(w7, 1u << 26); w4 = __umulhi(w7, 1u << 23);
        w2 = __umulhi(w3, 1u << 29); w1 = __umulhi(w3, 1u << 26); w0 = __umulhi(w3, 1u << 23);
    }
    else if constexpr (TRELLIS_K3_IMAD_SHIFTS >= 3)
    {
        w6 = __umulhi(w7, 1u << 29); w5 = w7 >> 6; w4 = __umulhi(w7, 1u << 23);
        w2 = w3 >> 3; w1 = __umulhi(w3, 1u << 26); w0 = w3 >> 9;
    }
    else
    {
        w6 = w7 >> 3; w5 = w7 >> 6; w4 = w7 >> 9;
        w2 = w3 >> 3; w1 = w3 >> 6; w0 = w3 >> 9;
    }
    if constexpr (cb == TRELLIS_CB_MCG_LUT)
        decode8_lut<FragB>(lut, w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                           w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
    else
        decode8<FragB, cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                           w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

// K5: K = 5: a lane's 8 windows are the 16-bit fields ending at stream bits 40l + 5(p+1), p = 0..7 (1280-bit
// tail-biting tile stream, 40 words). ExLlamaV3 decodes them as two dq4<5> groups (exl3_dq.cuh dq_dispatch, bits == 5:
// dq4<5, cb>(ptr, t_offset) and dq4<5, cb>(ptr, t_offset + 4), t_offset = 8 * lane); each group of four windows spans
// 31 bits and sits in two tile words (a = word i0, b = word i2) at a lane-constant alignment s2. The tile is staged
// byte-exact (40 words) and every lane gathers its four words (k5_lane_words: verbatim dq4 index arithmetic).
// Per lane and group: src_a = i0 % 40 (HIGH word), src_b = i2 % 40 (LOW word), s2 = the shift of the group's last window.
__device__ __forceinline__ void k5_lane_words(int lane, int& a0, int& b0, int& s0, int& a1, int& b1, int& s1)
{
    #pragma unroll
    for (int g = 0; g < 2; g++)
    {
        int t_offset = (lane << 3) + 4 * g;
        int bb0 = (t_offset + 257) * 5 - 16;   // start of first word
        int bb1 = bb0 + 3 * 5;                 // start of last word
        int bb2 = bb1 + 16;                    // end of last word
        int i0 = bb0 / 32;
        int i2 = (bb2 - 1) / 32;
        int s2 = (i2 + 1) * 32 - bb2;
        if (g == 0) { a0 = i0 % 40; b0 = i2 % 40; s0 = s2; }
        else        { a1 = i0 % 40; b1 = i2 % 40; s1 = s2; }
    }
}

// dq4<5> x 2 in register form. ExLlamaV3 computes each window as fshift(b, a, s2 + 5m) & 0xffff, m = 0..3; the four
// windows of a group lie in the 31 bits above s2 of the 64-bit (a:b), so they are taken from the one 32-bit funnel
// result v = (a:b) >> s2 as (v >> 5m) & 0xffff: the same bits (s2 + 5m + 16 <= s2 + 31 < s2 + 32). Group 0 gives
// w0..w3 (positions 8l .. 8l+3), group 1 w4..w7; decode8 pairs them exactly as dq4 fills frag0 / frag1.
template <typename FragB, int cb>
__device__ __forceinline__ void dq8_regs_5bits(uint32_t a0, uint32_t b0, int s0, uint32_t a1, uint32_t b1, int s1,
                                               FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w3 = k3_fshift(b0, a0, s0);
    w2 = w3 >> 5; w1 = w3 >> 10; w0 = w3 >> 15;
    w7 = k3_fshift(b1, a1, s1);
    w6 = w7 >> 5; w5 = w7 >> 10; w4 = w7 >> 15;
    decode8<FragB, cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                       w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

// K = 6: a lane's 8 windows end at stream bits 48l + 6(p+1), p = 0..7, and reach back to bit 48l - 10. The load-time
// repack (`repack_trellis`, K = 6 branch) stores per lane and tile the 64 stream bits [48l - 16, 48l + 48) (tail-biting)
// as two words hi:lo, so window p = (hi:lo >> 6(7-p)) & 0xffff. Same windows as ExLlamaV3's dq4<6> (exl3_dq.cuh),
// extracted with one funnel shift: lo holds windows 7,6,5; (hi:lo >> 18) holds 4,3,2; hi holds 1,0.
// Fragment order as for K = 4.
template <typename FragB, int cb>
__device__ __forceinline__ void dq8_regs_6bits(uint32_t hi, uint32_t lo, FragB& f0, FragB& f1)
{
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    TRELLIS_FSHF_IMM(s, lo, hi, 18);
    w7 = lo & 0xffff;
    TRELLIS_BFE16_IMM(w6, lo, 6);
    TRELLIS_BFE16_IMM(w5, lo, 12);
    w4 = s & 0xffff;
    TRELLIS_BFE16_IMM(w3, s, 6);
    TRELLIS_BFE16_IMM(w2, s, 12);
    TRELLIS_BFE16_IMM(w1, hi, 4);
    TRELLIS_BFE16_IMM(w0, hi, 10);
    decode8<FragB, cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

}  // namespace trellis_exl3
