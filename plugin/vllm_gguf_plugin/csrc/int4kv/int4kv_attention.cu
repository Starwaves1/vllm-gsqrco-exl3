// SPDX-License-Identifier: Apache-2.0
//
// Paged decode / MTP-verify attention over vLLM's int4_per_token_head KV cache
// (vllm/v1/attention/ops/int4_per_token_head.py). Same cache, same numerics:
//
//   slot (one token, one KV head) = [K 128 B | K scale f32 | V 128 B | V scale f32]
//   nibble codes u in 0..15, byte j = dims (2j, 2j+1) low/high, x = s * (u - zp),
//   zp = scale_bits & 0xF, s = scale_bits & ~0xF; K and V stored after the
//   randomized Hadamard rotation H*D (unnormalized, D = +-1 signs).
//
// One CTA = (KV split, KV head, sequence) and every query row of that head
// (GQA group x query tokens, <= 24 rows). Scores are computed transposed,
// S^T = K Q^T, so the query rows are the n8 dimension of mma.m16n8k16 and the
// 6 x 4 rows of a Qwen3.8 verify step fill three n8 tiles with no padding.
// Q is rotated in the prologue (256-point FWHT across a warp); the output is
// rotated back in the split-combine kernel.
//
// Head size 256 only (the layout maths below assumes it).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <Python.h>

#include <cmath>
#include <cstdint>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

namespace {

constexpr int kHead = 256;
constexpr int kSlot = 264;      // bytes per (token, KV head)
constexpr int kKScale = 128;    // byte offsets inside a slot
constexpr int kV = 132;
constexpr int kVScale = 260;
constexpr int kWarps = 8;
constexpr int kThreads = 32 * kWarps;
constexpr int kTile = 16 * kWarps;  // tokens per pipeline stage, one m16 tile per warp
constexpr int kSRow = 272;          // smem slot stride: 16 mod 32 B keeps K reads conflict-free
constexpr int kQStride = kHead + 8; // halves per Q row in smem (528 B, conflict-free ldmatrix)
constexpr int kMaxSplits = 64;

struct Params {
  const void* q;
  const uint8_t* kv;
  const int* block_table;
  const int* seq_lens;
  const int* query_start_loc;
  const float* signs;
  float* part_o;   // [tokens, q heads, splits, 256] unnormalized, rotated space
  float* part_ml;  // [tokens, q heads, splits, 2] running max (log2 domain), sum
  int64_t q_stride_tok, q_stride_head;                // elements
  int64_t kv_stride_blk, kv_stride_head, kv_stride_slot;  // bytes
  int bt_stride, block_size, num_q_heads, group, num_splits;
  float scale_log2;  // softmax scale * log2(e) / 16 (Q is rotated orthonormally)
};

// Tokens per split for one sequence, a whole number of tiles.
__device__ __forceinline__ int split_len(int seq_len, int num_splits) {
  const int per = (seq_len + num_splits - 1) / num_splits;
  return (per + kTile - 1) / kTile * kTile;
}

__device__ __forceinline__ void cp_async8(void* smem, const void* gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], const void* p) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(s));
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// The only place that turns stored codes into MMA operands. `w` holds eight
// nibbles; nibbles (k, k + 4) become half2 (u_k - zp_lo, u_{k+4} - zp_hi) in
// out[k]. `zp` is half2 (1024 + zp_lo, 1024 + zp_hi) as raw bits: OR-ing a
// nibble into fp16 1024.0 gives 1024 + u exactly, so one subtraction removes
// both the bias and the zero point.
__device__ __forceinline__ void dequant8(uint32_t w, uint32_t zp, uint32_t (&out)[4]) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const uint32_t biased = ((w >> (4 * k)) & 0x000F000Fu) | 0x64006400u;
    const __half2 d = __hsub2(*reinterpret_cast<const __half2*>(&biased),
                              *reinterpret_cast<const __half2*>(&zp));
    out[k] = *reinterpret_cast<const uint32_t*>(&d);
  }
}

__device__ __forceinline__ uint32_t zp_bias(uint32_t lo_scale_bits, uint32_t hi_scale_bits) {
  return 0x64006400u | (lo_scale_bits & 0xFu) | ((hi_scale_bits & 0xFu) << 16);
}
__device__ __forceinline__ float scale_of(uint32_t bits) { return __uint_as_float(bits & ~0xFu); }

// Unnormalized 256-point Walsh-Hadamard transform of a row held as 8
// consecutive elements per lane (element 8 * lane + j in x[j]).
__device__ __forceinline__ void fwht256(float (&x)[8], int lane) {
#pragma unroll
  for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (!(j & h)) {
        const float a = x[j], b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float o = __shfl_xor_sync(0xffffffffu, x[j], m);
      x[j] = (lane & m) ? o - x[j] : x[j] + o;
    }
  }
}

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ __half from_float(float v) { return __float2half_rn(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float v) { return __float2bfloat16_rn(v); }

// Column of physical rotated dim p in the permuted Q layout. The K fragment
// takes lane t's 32 bytes (dims 64t .. 64t+63) of a token row and splits each
// 32-bit word into the nibble pairs (k, k + 4); Q is stored in the matching
// order so a plain ldmatrix yields the B fragment.
__device__ __forceinline__ int q_column(int p) {
  const int t = p >> 6, i = (p >> 3) & 7, r = p & 7;
  const int step = 2 * i + ((r >> 1) & 1);
  return 16 * step + 8 * (r & 1) + 2 * t + (r >> 2);
}

template <typename T, int NT>
__global__ void __launch_bounds__(kThreads, 1) int4kv_attention_kernel(const Params p) {
  constexpr int kRows = 8 * NT;                      // query rows, padded
  constexpr int kPStride = NT == 1 ? 8 : 24;         // halves per P^T row
  extern __shared__ __align__(16) uint8_t smem[];
  uint8_t* kv_s = smem;                                               // 2 x kTile x kSRow
  __half* q_s = reinterpret_cast<__half*>(smem + 2 * kTile * kSRow);  // kRows x kQStride
  __half* p_s = q_s + kRows * kQStride;                               // kTile x kPStride
  float* red = reinterpret_cast<float*>(p_s + kTile * kPStride);      // kWarps x kRows

  const int split = blockIdx.x, kv_head = blockIdx.y, seq = blockIdx.z;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const int q0 = p.query_start_loc[seq];
  const int q_len = p.query_start_loc[seq + 1] - q0;
  if (q_len <= 0) return;
  const int rows = q_len * p.group;
  const int seq_len = p.seq_lens[seq];
  const int chunk = split_len(seq_len, p.num_splits);
  const int begin = split * chunk, end = min(seq_len, begin + chunk);

  // Partial-output row of query row r (token q0 + r / group, head kv_head * group + r % group).
  auto part_row = [&](int r) {
    return (static_cast<int64_t>(q0 + r / p.group) * p.num_q_heads + kv_head * p.group +
            r % p.group) * p.num_splits + split;
  };
  if (begin >= end) {
    if (threadIdx.x < rows) {
      p.part_ml[2 * part_row(threadIdx.x)] = -INFINITY;
      p.part_ml[2 * part_row(threadIdx.x) + 1] = 0.f;
    }
    return;
  }

  // Each warp copies its own 16-token group of a tile: 32 lanes x 8 B per
  // slot, plus the 33rd 8 B chunk of all 16 slots. A group never crosses a
  // cache block (tiles start at multiples of 16 and block_size % 16 == 0).
  const uint8_t* kv_head_base = p.kv + kv_head * p.kv_stride_head;
  const int* bt = p.block_table + static_cast<int64_t>(seq) * p.bt_stride;
  auto load_tile = [&](int tile0, int stage) {
    const int tok = tile0 + 16 * warp;
    if (tok >= end) return;
    const int n = min(16, end - tok);
    const uint8_t* src = kv_head_base + bt[tok / p.block_size] * p.kv_stride_blk +
                         (tok % p.block_size) * p.kv_stride_slot;
    uint8_t* dst = kv_s + (stage * kTile + 16 * warp) * kSRow;
    for (int j = 0; j < n; ++j) cp_async8(dst + j * kSRow + 8 * lane, src + j * p.kv_stride_slot + 8 * lane);
    if (lane < n) cp_async8(dst + lane * kSRow + 256, src + lane * p.kv_stride_slot + 256);
  };
  load_tile(begin, 0);
  cp_async_commit();

  // Prologue: rotate Q (D signs, FWHT, 1/16 so the rotation is orthonormal)
  // and store it in fp16 in the permuted column order.
  {
    float sign[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) sign[j] = p.signs[8 * lane + j];
    for (int r = warp; r < kRows; r += kWarps) {
      float x[8];
      if (r < rows) {
        const T* src = static_cast<const T*>(p.q) + (q0 + r / p.group) * p.q_stride_tok +
                       (kv_head * p.group + r % p.group) * p.q_stride_head + 8 * lane;
        const uint4 raw = *reinterpret_cast<const uint4*>(src);
        const T* v = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = to_float(v[j]) * sign[j];
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = 0.f;
      }
      fwht256(x, lane);
#pragma unroll
      for (int j = 0; j < 8; ++j) q_s[r * kQStride + q_column(8 * lane + j)] = __float2half_rn(x[j] * 0.0625f);
    }
  }

  // Per-thread query columns: 8n + 2t + c for n < NT, c < 2.
  float m_run[NT][2], l_run[NT][2];
  int limit[NT][2];  // exclusive causal bound on the token index
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = 8 * n + 2 * t + c;
      m_run[n][c] = -INFINITY;
      l_run[n][c] = 0.f;
      limit[n][c] = r < rows ? min(end, seq_len - q_len + 1 + r / p.group) : 0;
    }
  float o[4][NT][4] = {};  // O^T: 4 dim m-tiles x NT query n-tiles

  const int dims = warp % 4;    // PV: this warp's 64 output dims
  const int half_ = warp / 4;   // PV: this warp's 64 tokens of the tile
  const int n_tiles = (end - begin + kTile - 1) / kTile;
  for (int it = 0; it < n_tiles; ++it) {
    const int stage = it & 1, tile0 = begin + it * kTile;
    __syncthreads();  // stage ^ 1 is free (PV of it - 1 done); q_s ready
    if (it + 1 < n_tiles) load_tile(tile0 + kTile, stage ^ 1);
    cp_async_commit();
    cp_async_wait<1>();
    __syncwarp();

    // ---- S^T = K Q^T for this warp's 16 tokens ----
    const uint8_t* ks = kv_s + (stage * kTile + 16 * warp) * kSRow;
    uint32_t kw[2][8];  // rows g and g + 8: bytes 32t .. 32t + 31
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint4* row = reinterpret_cast<const uint4*>(ks + (g + 8 * h) * kSRow + 32 * t);
      const uint4 a = row[0], b = row[1];
      kw[h][0] = a.x; kw[h][1] = a.y; kw[h][2] = a.z; kw[h][3] = a.w;
      kw[h][4] = b.x; kw[h][5] = b.y; kw[h][6] = b.z; kw[h][7] = b.w;
    }
    const uint32_t ks_bits[2] = {*reinterpret_cast<const uint32_t*>(ks + g * kSRow + kKScale),
                                 *reinterpret_cast<const uint32_t*>(ks + (g + 8) * kSRow + kKScale)};
    const uint32_t kzp[2] = {zp_bias(ks_bits[0], ks_bits[0]), zp_bias(ks_bits[1], ks_bits[1])};
    float s[NT][4] = {};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      uint32_t e0[4], e1[4];
      dequant8(kw[0][i], kzp[0], e0);
      dequant8(kw[1][i], kzp[1], e1);
#pragma unroll
      for (int sub = 0; sub < 2; ++sub) {
        const uint32_t a[4] = {e0[2 * sub], e1[2 * sub], e0[2 * sub + 1], e1[2 * sub + 1]};
        const int col = 16 * (2 * i + sub);
        if constexpr (NT >= 2) {
          uint32_t b[4];
          ldmatrix_x4(b, q_s + (8 * (lane / 16) + lane % 8) * kQStride + col + 8 * ((lane / 8) % 2));
          mma16816(s[0], a, b[0], b[1]);
          mma16816(s[1], a, b[2], b[3]);
        }
        if constexpr (NT != 2) {
          constexpr int n = NT - 1;
          uint32_t b[2];
          ldmatrix_x2(b, q_s + (8 * n + lane % 8) * kQStride + col + 8 * ((lane / 8) % 2));
          mma16816(s[n], a, b[0], b[1]);
        }
      }
    }

    // ---- online softmax, max shared across warps ----
    const int tok[2] = {tile0 + 16 * warp + g, tile0 + 16 * warp + g + 8};
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        float mx = -INFINITY;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          float& v = s[n][2 * h + c];
          v = tok[h] < limit[n][c] ? v * scale_of(ks_bits[h]) * p.scale_log2 : -INFINITY;
          mx = fmaxf(mx, v);
        }
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 8));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
        if (g == 0) red[warp * kRows + 8 * n + 2 * t + c] = mx;
      }
    __syncthreads();

    const uint32_t vs_bits[2] = {*reinterpret_cast<const uint32_t*>(ks + g * kSRow + kVScale),
                                 *reinterpret_cast<const uint32_t*>(ks + (g + 8) * kSRow + kVScale)};
    float alpha[NT][2];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      float pv[2][2];
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int r = 8 * n + 2 * t + c;
        float mx = m_run[n][c];
#pragma unroll
        for (int w = 0; w < kWarps; ++w) mx = fmaxf(mx, red[w * kRows + r]);
        const float base = mx == -INFINITY ? 0.f : mx;
        alpha[n][c] = exp2f(m_run[n][c] - base);
        m_run[n][c] = mx;
        float sum = 0.f;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const float pr = exp2f(s[n][2 * h + c] - base);
          sum += pr;
          // P with the V scale folded in; masked slots may hold stale bytes.
          // An fp16 operand: a V scale past the fp16 range saturates, not inf.
          pv[h][c] = tok[h] < limit[n][c] ? fminf(pr * scale_of(vs_bits[h]), 65504.f) : 0.f;
        }
        l_run[n][c] = l_run[n][c] * alpha[n][c] + sum;
      }
#pragma unroll
      for (int h = 0; h < 2; ++h)
        *reinterpret_cast<__half2*>(p_s + (16 * warp + g + 8 * h) * kPStride + 8 * n + 2 * t) =
            __floats2half2_rn(pv[h][0], pv[h][1]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        o[j][n][0] *= alpha[n][0];
        o[j][n][1] *= alpha[n][1];
        o[j][n][2] *= alpha[n][0];
        o[j][n][3] *= alpha[n][1];
      }
    __syncthreads();

    // ---- O^T += V^T P^T: this warp's 64 dims x 64 tokens ----
    const uint8_t* vs = kv_s + (stage * kTile + 64 * half_) * kSRow;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const uint8_t* r0 = vs + (16 * kk + 2 * t) * kSRow;
      uint32_t v[4], z[4];
#pragma unroll
      for (int h = 0; h < 4; ++h) {
        const uint8_t* row = r0 + ((h & 1) + 8 * (h >> 1)) * kSRow;  // tokens 2t, 2t+1, 2t+8, 2t+9
        v[h] = *reinterpret_cast<const uint32_t*>(row + kV + 32 * dims + 4 * g);
        z[h] = *reinterpret_cast<const uint32_t*>(row + kVScale);
      }
      // Pair tokens (2t, 2t+1) byte-wise so each half2 holds one dim of both tokens.
      uint32_t e[4][4];
      dequant8(__byte_perm(v[0], v[1], 0x5410), zp_bias(z[0], z[1]), e[0]);  // dims 8g + 0..3
      dequant8(__byte_perm(v[0], v[1], 0x7632), zp_bias(z[0], z[1]), e[1]);  // dims 8g + 4..7
      dequant8(__byte_perm(v[2], v[3], 0x5410), zp_bias(z[2], z[3]), e[2]);
      dequant8(__byte_perm(v[2], v[3], 0x7632), zp_bias(z[2], z[3]), e[3]);
      uint32_t b[NT][2];
      const __half* pk = p_s + (64 * half_ + 16 * kk) * kPStride;
      if constexpr (NT >= 2) {
        uint32_t r[4];
        ldmatrix_x4_trans(r, pk + (lane % 8 + 8 * ((lane / 8) % 2)) * kPStride + 8 * (lane / 16));
        b[0][0] = r[0]; b[0][1] = r[1]; b[1][0] = r[2]; b[1][1] = r[3];
      }
      if constexpr (NT != 2) {
        constexpr int n = NT - 1;
        ldmatrix_x2_trans(b[n], pk + (lane % 8 + 8 * ((lane / 8) % 2)) * kPStride + 8 * n);
      }
      // m-tile j: row g = dim 8g + 2j, row g + 8 = dim 8g + 2j + 1.
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int lo = j / 2, sh = 2 * (j % 2);
        const uint32_t a[4] = {e[lo][sh], e[lo][sh + 1], e[lo + 2][sh], e[lo + 2][sh + 1]};
#pragma unroll
        for (int n = 0; n < NT; ++n) mma16816(o[j][n], a, b[n][0], b[n][1]);
      }
    }
  }

  // ---- epilogue: reduce l over lanes and warps, O over the two token halves ----
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      float l = l_run[n][c];
      l += __shfl_xor_sync(0xffffffffu, l, 4);
      l += __shfl_xor_sync(0xffffffffu, l, 8);
      l += __shfl_xor_sync(0xffffffffu, l, 16);
      l_run[n][c] = l;
    }
  __syncthreads();  // all PV reads of kv_s and red done
  if (g == 0)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int c = 0; c < 2; ++c) red[warp * kRows + 8 * n + 2 * t + c] = l_run[n][c];
  float* o_s = reinterpret_cast<float*>(kv_s);
  constexpr int kOPerThread = 4 * NT * 4;
  if (half_ == 1) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int k = 0; k < 4; ++k)
          o_s[((dims * kOPerThread) + (j * NT + n) * 4 + k) * 32 + lane] = o[j][n][k];
  }
  __syncthreads();
  if (half_ == 1) return;
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int k = 0; k < 4; ++k) o[j][n][k] += o_s[((dims * kOPerThread) + (j * NT + n) * 4 + k) * 32 + lane];
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = 8 * n + 2 * t + c;
      if (r >= rows) continue;
      const int64_t pr = part_row(r);
      float4* dst = reinterpret_cast<float4*>(p.part_o + pr * kHead + 64 * dims + 8 * g);
      dst[0] = make_float4(o[0][n][c], o[0][n][2 + c], o[1][n][c], o[1][n][2 + c]);
      dst[1] = make_float4(o[2][n][c], o[2][n][2 + c], o[3][n][c], o[3][n][2 + c]);
      if (warp == 0 && g == 0) {
        float l = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) l += red[w * kRows + r];
        p.part_ml[2 * pr] = m_run[n][c];
        p.part_ml[2 * pr + 1] = l;
      }
    }
}

struct CombineParams {
  const float* part_o;
  const float* part_ml;
  const float* signs;
  const int* query_start_loc;
  void* out;
  int64_t out_stride_tok, out_stride_head;
  int num_tokens, num_q_heads, num_splits, num_seqs;
};

// Merge the splits of one (token, query head) row, rotate back (D H / 256)
// and store. One warp per row, 8 dims per lane.
template <typename T>
__global__ void int4kv_combine_kernel(const CombineParams p) {
  const int row = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= p.num_tokens * p.num_q_heads) return;
  const int token = row / p.num_q_heads, head = row % p.num_q_heads;
  T* dst = static_cast<T*>(p.out) + token * p.out_stride_tok + head * p.out_stride_head + 8 * lane;
  float x[8] = {};
  if (token < p.query_start_loc[p.num_seqs]) {
    const float* ml = p.part_ml + 2 * static_cast<int64_t>(row) * p.num_splits;
    float mx = -INFINITY;
    for (int s = 0; s < p.num_splits; ++s) mx = fmaxf(mx, ml[2 * s]);
    float l = 0.f;
    for (int s = 0; s < p.num_splits; ++s) {
      if (ml[2 * s] == -INFINITY) continue;
      const float w = exp2f(ml[2 * s] - mx);
      l += w * ml[2 * s + 1];
      const float4* src = reinterpret_cast<const float4*>(
          p.part_o + (static_cast<int64_t>(row) * p.num_splits + s) * kHead + 8 * lane);
      const float4 a = src[0], b = src[1];
      x[0] += w * a.x; x[1] += w * a.y; x[2] += w * a.z; x[3] += w * a.w;
      x[4] += w * b.x; x[5] += w * b.y; x[6] += w * b.z; x[7] += w * b.w;
    }
    fwht256(x, lane);
    const float inv = l > 0.f ? 1.f / (l * kHead) : 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] *= p.signs[8 * lane + j] * inv;
  }
  uint4 packed;
  T* v = reinterpret_cast<T*>(&packed);
#pragma unroll
  for (int j = 0; j < 8; ++j) v[j] = from_float<T>(x[j]);
  *reinterpret_cast<uint4*>(dst) = packed;
}

// Fewest splits that fill the GPU to >= 95% in whole waves (one CTA per SM).
int choose_splits(int ctas_per_split, int num_sms) {
  int best = 1;
  double best_eff = 0.0;
  for (int s = 1; s <= kMaxSplits; ++s) {
    const int ctas = ctas_per_split * s;
    const int waves = (ctas + num_sms - 1) / num_sms;
    const double eff = static_cast<double>(ctas) / (waves * num_sms);
    if (eff >= 0.95) return s;
    if (eff > best_eff + 1e-9) best_eff = eff, best = s;
  }
  return best;
}

cudaStream_t current_stream(int32_t device) {
  void* s = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &s));
  return s != nullptr ? static_cast<cudaStream_t>(s) : cudaStreamLegacy;
}

template <typename T, int NT>
void launch(const Params& p, dim3 grid, cudaStream_t stream) {
  constexpr int kRows = 8 * NT, kPStride = NT == 1 ? 8 : 24;
  constexpr int smem = 2 * kTile * kSRow + 2 * kRows * kQStride + 2 * kTile * kPStride +
                       4 * kWarps * kRows;
  static_assert(2 * kTile * kSRow >= 4 * (4 * NT * 4) * 32 * 4, "O exchange must fit in the KV stages");
  STD_TORCH_CHECK(cudaFuncSetAttribute(int4kv_attention_kernel<T, NT>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem) == cudaSuccess,
                  "int4kv: cannot reserve shared memory");
  int4kv_attention_kernel<T, NT><<<grid, kThreads, smem, stream>>>(p);
}

}  // namespace

void int4kv_attention(Tensor out, Tensor q, Tensor kv_cache, Tensor block_table,
                      Tensor seq_lens, Tensor query_start_loc, Tensor rht_signs, double scale,
                      int64_t max_query_len) {
  STD_TORCH_CHECK(q.dim() == 3 && q.size(2) == kHead && q.stride(2) == 1,
                  "int4kv: q must be [tokens, heads, 256] with unit last stride");
  STD_TORCH_CHECK(out.dim() == 3 && out.size(0) == q.size(0) && out.size(1) == q.size(1) &&
                      out.size(2) == kHead && out.stride(2) == 1 && out.scalar_type() == q.scalar_type(),
                  "int4kv: out must match q");
  STD_TORCH_CHECK(q.scalar_type() == ScalarType::Half || q.scalar_type() == ScalarType::BFloat16,
                  "int4kv: q must be fp16 or bf16");
  STD_TORCH_CHECK(kv_cache.dim() == 4 && kv_cache.size(3) == kSlot && kv_cache.stride(3) == 1 &&
                      kv_cache.element_size() == 1,
                  "int4kv: kv_cache must be the packed uint8 [blocks, kv heads, block, 264] cache");
  STD_TORCH_CHECK(kv_cache.size(2) % 16 == 0, "int4kv: block size must be a multiple of 16");
  STD_TORCH_CHECK(block_table.scalar_type() == ScalarType::Int && seq_lens.scalar_type() == ScalarType::Int &&
                      query_start_loc.scalar_type() == ScalarType::Int && block_table.stride(1) == 1,
                  "int4kv: int32 metadata expected");
  STD_TORCH_CHECK(rht_signs.scalar_type() == ScalarType::Float && rht_signs.numel() == kHead,
                  "int4kv: rht_signs must be 256 fp32 signs");
  const int num_q_heads = static_cast<int>(q.size(1));
  const int num_kv_heads = static_cast<int>(kv_cache.size(1));
  STD_TORCH_CHECK(num_q_heads % num_kv_heads == 0, "int4kv: bad GQA ratio");
  const int group = num_q_heads / num_kv_heads;
  const int nt = static_cast<int>((group * max_query_len + 7) / 8);
  STD_TORCH_CHECK(nt >= 1 && nt <= 3, "int4kv: at most 24 query rows per KV head");
  const int num_seqs = static_cast<int>(seq_lens.size(0));
  const int num_tokens = static_cast<int>(q.size(0));
  if (num_tokens == 0 || num_seqs == 0) return;

  const int32_t device = q.get_device_index();
  const DeviceGuard guard(device);
  const cudaStream_t stream = current_stream(device);
  int num_sms = 0;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
  const int num_splits = choose_splits(num_seqs * num_kv_heads, num_sms);

  Tensor part_o = torch::stable::new_empty(q, {num_tokens, num_q_heads, num_splits, kHead}, ScalarType::Float);
  Tensor part_ml = torch::stable::new_empty(q, {num_tokens, num_q_heads, num_splits, 2}, ScalarType::Float);

  Params p{};
  p.q = q.data_ptr();
  p.kv = static_cast<const uint8_t*>(kv_cache.data_ptr());
  p.block_table = static_cast<const int*>(block_table.data_ptr());
  p.seq_lens = static_cast<const int*>(seq_lens.data_ptr());
  p.query_start_loc = static_cast<const int*>(query_start_loc.data_ptr());
  p.signs = static_cast<const float*>(rht_signs.data_ptr());
  p.part_o = static_cast<float*>(part_o.data_ptr());
  p.part_ml = static_cast<float*>(part_ml.data_ptr());
  p.q_stride_tok = q.stride(0);
  p.q_stride_head = q.stride(1);
  p.kv_stride_blk = kv_cache.stride(0);
  p.kv_stride_head = kv_cache.stride(1);
  p.kv_stride_slot = kv_cache.stride(2);
  p.bt_stride = static_cast<int>(block_table.stride(0));
  p.block_size = static_cast<int>(kv_cache.size(2));
  p.num_q_heads = num_q_heads;
  p.group = group;
  p.num_splits = num_splits;
  p.scale_log2 = static_cast<float>(scale * M_LOG2E / 16.0);

  const dim3 grid(num_splits, num_kv_heads, num_seqs);
  const bool bf16 = q.scalar_type() == ScalarType::BFloat16;
  switch (nt) {
    case 1: bf16 ? launch<__nv_bfloat16, 1>(p, grid, stream) : launch<__half, 1>(p, grid, stream); break;
    case 2: bf16 ? launch<__nv_bfloat16, 2>(p, grid, stream) : launch<__half, 2>(p, grid, stream); break;
    default: bf16 ? launch<__nv_bfloat16, 3>(p, grid, stream) : launch<__half, 3>(p, grid, stream); break;
  }

  CombineParams c{};
  c.part_o = p.part_o;
  c.part_ml = p.part_ml;
  c.signs = p.signs;
  c.query_start_loc = p.query_start_loc;
  c.out = out.data_ptr();
  c.out_stride_tok = out.stride(0);
  c.out_stride_head = out.stride(1);
  c.num_tokens = num_tokens;
  c.num_q_heads = num_q_heads;
  c.num_splits = num_splits;
  c.num_seqs = num_seqs;
  const int combine_blocks = (num_tokens * num_q_heads + 3) / 4;
  if (bf16)
    int4kv_combine_kernel<__nv_bfloat16><<<combine_blocks, 128, 0, stream>>>(c);
  else
    int4kv_combine_kernel<__half><<<combine_blocks, 128, 0, stream>>>(c);
  STD_TORCH_CHECK(cudaGetLastError() == cudaSuccess, "int4kv: launch failed");
}

STABLE_TORCH_LIBRARY(_C_int4kv, ops) {
  ops.def(
      "int4kv_attention(Tensor(a!) out, Tensor q, Tensor kv_cache, Tensor block_table, "
      "Tensor seq_lens, Tensor query_start_loc, Tensor rht_signs, float scale, "
      "int max_query_len) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C_int4kv, CUDA, ops) { ops.impl("int4kv_attention", TORCH_BOX(&int4kv_attention)); }

static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C_int4kv", nullptr, -1, nullptr};
extern "C" PyObject* PyInit__C_int4kv(void) { return PyModule_Create(&module_def); }
