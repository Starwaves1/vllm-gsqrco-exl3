// trellis-exl3: 128-point blockwise Hadamard with bf16 or fp16 at the boundary.
//
// The arithmetic is ExLlamaV3's had_hf_r_128_inner (third_party/exllamav3/hadamard_inner.cuh, MIT, Turboderp),
// statement for statement: fp16 pre-scale (__hmul2), fp32 4-point butterfly in the lane, 5 fp32 warp-shuffle rounds,
// fp32 * 1/sqrt(128), round-to-nearest to fp16, fp16 post-scale (__hmul2). The only additions are the dtype
// conversions at the two ends:
//   in_bf16:  bf16 -> fp32 (exact) -> fp16 round-to-nearest-even, overflow to +-inf, NaN stays NaN. This is the
//             value conversion torch's `x.to(torch.float16)` performs (and ExLlamaV3's own loader, stloader_cu.cu),
//             so `kernel(x_bf16)` == `kernel(x_bf16.to(float16))` bit for bit (checked in parity/marlin_parity.py).
//   out_bf16: fp16 -> fp32 (exact) -> bf16 round-to-nearest-even = torch's `y.to(torch.bfloat16)`; bf16 has the
//             wider exponent, so nothing overflows; fp16 inf/NaN stay inf/NaN.
// With both flags false this is had_hf_r_128_inner.
#pragma once
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "exl3_decode.cuh"  // half4
#include "third_party/exllamav3/hadamard_inner.cuh"

template <bool in_bf16, bool out_bf16, bool pre_scale, bool post_scale>
inline __device__ void trellis_had_r_128_inner(const half* __restrict__ input_ptr, half* __restrict__ output_ptr,
                                              const half* __restrict__ scale, const int scale_idx,
                                              const float r_scale)
{
    int t = threadIdx.x & 31;

    // Load
    half4 v;
    if constexpr (in_bf16)
    {
        const __nv_bfloat162* rd = ((const __nv_bfloat162*) input_ptr) + 2 * t;
        float2 f0 = __bfloat1622float2(rd[0]);
        float2 f1 = __bfloat1622float2(rd[1]);
        v.x = __floats2half2_rn(f0.x, f0.y);
        v.y = __floats2half2_rn(f1.x, f1.y);
    }
    else
        v = ((half4*) input_ptr)[t];

    // Pre scale
    if constexpr (pre_scale)
    {
        half4 scales = ((half4*) scale)[scale_idx + t];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // 4 element had
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;

    // 32 element had, warp shuffle
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);

    // Post scale
    if constexpr (post_scale)
    {
        half4 scales = ((half4*) scale)[scale_idx + t];
        v.x = __hmul2(v.x, scales.x);
        v.y = __hmul2(v.y, scales.y);
    }

    // Store
    if constexpr (out_bf16)
    {
        __nv_bfloat162* wr = ((__nv_bfloat162*) output_ptr) + 2 * t;
        wr[0] = __float22bfloat162_rn(__half22float2(v.x));
        wr[1] = __float22bfloat162_rn(__half22float2(v.y));
    }
    else
        ((half4*) output_ptr)[t] = v;
}

// Output transform of one lane on a register value (used inside the GEMM launch, exl3_marlin_template.h):
// v = lane t's 4 fp16 values of a 128-block -> Had128 -> * scales, optionally stored as bf16 bits.
// Same statements as the post_scale arm above, so the result equals the separate output launch bit for bit.
template <bool out_bf16>
__device__ __forceinline__ half4 trellis_had_out_reg(half4 v, const half4 scales, const float r_scale, const int t)
{
    float v0 = __half2float(__low2half(v.x));
    float v1 = __half2float(__high2half(v.x));
    float v2 = __half2float(__low2half(v.y));
    float v3 = __half2float(__high2half(v.y));
    float s0 = v0 + v1;
    float d0 = v0 - v1;
    float s1 = v2 + v3;
    float d1 = v2 - v3;
    float h0 = s0 + s1;
    float h1 = d0 + d1;
    float h2 = s0 - s1;
    float h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale);
    v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
    v.x = __hmul2(v.x, scales.x);
    v.y = __hmul2(v.y, scales.y);
    if constexpr (out_bf16)
    {
        half4 o;
        *reinterpret_cast<__nv_bfloat162*>(&o.x) = __float22bfloat162_rn(__half22float2(v.x));
        *reinterpret_cast<__nv_bfloat162*>(&o.y) = __float22bfloat162_rn(__half22float2(v.y));
        return o;
    }
    return v;
}
