// SPDX-License-Identifier: Apache-2.0
// Owned Q4_K / IQ4_XS / IQ2_S x q8_1 product for up to 64 activation rows on int8
// tensor cores (mma.sync m16n8k32 / m16n8k16 s8 x s8 -> s32, sm_80+): MTP decode at
// c = 4..16 (4 rows per sequence). Op lcpp_mul_mat_mma_k in lcpp_shim.cu;
// cloud/results/opt/k3 has the data and the variants tried.
//
// Same inputs as the vendored MMQ: GGUF blocks as stored, and X quantized by the
// shim's quantize_x into MMQ's block_q8_1_mmq layout (128 values of one column per
// block, the type's ds layout: Q4_K DS4, IQ4_XS / IQ2_S D4). What differs is the
// pipeline. MMQ copies a 128-row tile, decodes it into shared memory, then
// multiplies, one K step at a time, one CTA per SM; at 16-64 columns that leaves
// DRAM idle while the SM decodes (it reads 17408 x 5120 Q4_K at ~450 GB/s).
// Here:
//  - a tile is 64 weight rows (one 16-row mma tile per warp of 4) x all columns;
//  - each 256-value K step, the raw weight blocks of the tile's rows and the q8_1
//    blocks of every column are copied to shared memory with cp.async, 1-2 steps
//    ahead;
//  - each warp decodes its rows' blocks from shared memory straight into mma A
//    fragments (no decoded tile), and multiplies them with every column tile;
//  - the per-slice scaling is the vendored vec_dot's (below), accumulated in
//    registers over the CTA's K range;
//  - work is split into (tile, K step) units; each CTA takes a contiguous run of them
//    (stream-K), so every SM gets the same work whatever the tile count. A tile a CTA
//    covers whole is written in X's dtype directly; the pieces of a tile shared by
//    CTAs go to fp32 scratch, summed in CTA (K) order by a second kernel.
//
// Fragment layout (PTX m16n8k32 .s8): lane (g = lane/4, t = lane%4) holds A rows g
// and g+8 at k 4t..4t+3 (a0, a1) and 16+4t..16+4t+3 (a2, a3), B column g at the same
// k (b0, b1), C rows g / g+8 x columns 2t / 2t+1. A B word is one int of a column's
// q8_1 quants; an A word is the matching 4 weights:
//  - Q4_K: low / high nibbles of qs int 8j+t / 8j+4+t give sub-blocks 2j / 2j+1;
//  - IQ4_XS: int 4s+t of sub-block s, low nibbles -> a0, high -> a2 (kvalues lookup);
//  - IQ2_S: 16-value sub-scales, so two m16n8k16 per 32 values (MMQ does the same);
//    lane t takes word t%2 of grid entry t/2 (first 16 values) / 2+t/2 (last 16).
//
// Numerics: each slice's int32 sum is exact, and its term is the vendored MMQ
// vec_dot's expression on the same operands (ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma
// for Q4_K incl. its fp16 d*sc / dmin*m products, _q8_0_q8_1_mma for IQ4_XS,
// _q8_0_16_q8_1_mma for IQ2_S; x_df / x_dm as the load_tiles functions build them),
// added in K order. MMQ adds in the same order within a CTA; its stream-k split
// and ours differ, so results match MMQ to fp32 reordering, not bit for bit.
//
// Memory safety by construction: every copy is an exact granule of W or of the q8_1
// buffer. Q4_K blocks are 16-byte aligned (16-byte copies), IQ4_XS 8-byte (8-byte
// copies); an IQ2_S block (82 B) is copied as the 84-byte 4-aligned window around
// it, which with an even block count per row (K % 512 == 0, check_inputs) never
// leaves its row. Tail rows re-read row nrows-1 and are not stored. q8_1 columns
// past ncols are zero in shared memory and never read from global memory.

#include "common.cuh"
#include "mmq.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace {

constexpr int YW = sizeof(block_q8_1_mmq) / 4;   // 36 ints per column per 128 values (4 scale, 32 quant)
constexpr int YG = sizeof(block_q8_1_mmq) / 16;  // 9 16-byte granules per block_q8_1_mmq
constexpr int YQ = QK_K / QK8_1_MMQ;             // 2 block_q8_1_mmq per column per K step
constexpr int MAX_NT = 8;                        // 64 columns
static_assert(sizeof(block_q8_1_mmq) == 144, "block_q8_1_mmq layout");

// Staging, per weight row and K step (one block): WG granules of G bytes at offsets
// e * G of the block's window, WS ints per row (a stride that puts rows g = 0..7 on
// distinct banks for the fragment loads). TABLE: shared ints of lookup table.
template <ggml_type type> struct traits;
template <> struct traits<GGML_TYPE_Q4_K> {
  using block = block_q4_K;
  static constexpr int G = 16, WG = 9, WS = 36, TABLE = 0;
};
template <> struct traits<GGML_TYPE_IQ4_XS> {
  using block = block_iq4_xs;
  static constexpr int G = 8, WG = 17, WS = 36, TABLE = 4;  // kvalues_iq4nl
};
template <> struct traits<GGML_TYPE_IQ2_S> {  // window within the row only for an even block count (K % 512)
  using block = block_iq2_s;
  static constexpr int G = 4, WG = 21, WS = 21, TABLE = 2048;  // iq2s_grid as 2 x 1024 words
};

constexpr int WARPS = 4;
constexpr int THREADS = WARPS * WARP_SIZE;
constexpr int M = 16 * WARPS;  // weight rows per tile: one 16-row mma tile per warp

// Shared memory for NT 8-column tiles: STAGES steps (K steps) in flight, 3 where the CTA
// still fits twice per SM (<= 48 KB), else 2.
template <ggml_type type, int NT>
struct geometry {
  using tr = traits<type>;
  static constexpr int W_INTS = M * tr::WS;
  static constexpr int Y_INTS = YQ * 8 * NT * YW;
  static constexpr int STAGE_INTS = W_INTS + Y_INTS;
  static constexpr size_t smem(int stages) { return (size_t)(tr::TABLE + stages * STAGE_INTS) * 4; }
  static constexpr int STAGES = smem(3) <= 48 * 1024 ? 3 : 2;
  static constexpr size_t SMEM = smem(STAGES);
};

template <int G>
static __device__ __forceinline__ void cp_async(int* dst, const void* src) {
  const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  if constexpr (G == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src));
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(d), "l"(src), "n"(G));
  }
}
static __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N>
static __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

static __device__ __forceinline__ void mma_k32(int (&c)[4], const int (&a)[4], int b0, int b1) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
static __device__ __forceinline__ void mma_k16(int (&c)[4], int a0, int a1, int b0) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

// Negates the bytes of g flagged in the low nibble of bits. The IQ2_S grid has no
// zero byte, so (g ^ 0xFF) + 1 per byte never carries (lcpp_owned_k4.cu's own_negate).
static __device__ __forceinline__ int negate4(uint32_t g, uint32_t bits) {
  const uint32_t ones = ((bits & 0xF) * 0x00204081u) & 0x01010101u;
  return (int)((g ^ (ones * 0xFFu)) + ones);
}

// half2(lo, hi) of two small non-negative ints (< 1024), exactly: 0x6400 | v is 1024 + v.
static __device__ __forceinline__ half2 small_half2(uint32_t lo, uint32_t hi) {
  const uint32_t bits = (lo | hi << 16) | 0x64006400u;
  return __hsub2(*reinterpret_cast<const half2*>(&bits), make_half2(1024.0f, 1024.0f));
}

// float(c) for |c| < 2^22, exactly: the bits of 1.5 * 2^23 + c. I2F runs at a quarter of
// the FP32 rate on sm_86 and would be the product's limit at 64 columns. The slice sums
// stay far below 2^22 (at most 32 x 127 x 127).
static __device__ __forceinline__ float i2f(int c) {
  return __int_as_float(c + 0x4B400000) - 12582912.0f;
}

// Per 16-row tile and K step: header values of rows g (rows[0]) and g+8 (rows[1]), and
// per slice s (32 values) the A fragment and the row scales. boff: the block's byte
// offset in its staged window (IQ2_S).
template <ggml_type type> struct tile_rows;

template <> struct tile_rows<GGML_TYPE_Q4_K> {
  // x_dm of ggml_cuda_mmq_load_tiles_q4_K: dm * (1, -1) * (sc, m) in fp16, per sub-block
  half2 dm[2];
  int sc[2][2], mn[2][2];  // [row][sub-blocks 0-3 / 4-7]
  int qa[2], qb[2];
  __device__ __forceinline__ void init(const int* const (&rows)[2], int) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      dm[r] = __hmul2(*(const half2*)rows[r], make_half2(1.0f, -1.0f));
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        sc[r][k] = unpack_scales_q45_K(rows[r] + 1, k);
        mn[r][k] = unpack_scales_q45_K(rows[r] + 1, 2 + k);
      }
    }
  }
  // A fragment and, per row, (d * sc, -dmin * m)
  template <int s>
  __device__ __forceinline__ void slice(const int* const (&rows)[2], int t, int (&a)[4], float2 (&d)[2]) {
    if (s % 2 == 0) {  // qs ints 8j+t, 8j+4+t (j = s/2) hold sub-blocks 2j (low nibbles) and 2j+1 (high)
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        qa[r] = rows[r][4 + 4 * s + t];
        qb[r] = rows[r][8 + 4 * s + t];
      }
    }
    constexpr int sh = 4 * (s % 2), bs = 8 * (s % 4);
    a[0] = (qa[0] >> sh) & 0x0F0F0F0F;
    a[1] = (qa[1] >> sh) & 0x0F0F0F0F;
    a[2] = (qb[0] >> sh) & 0x0F0F0F0F;
    a[3] = (qb[1] >> sh) & 0x0F0F0F0F;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint32_t scw = (uint32_t)sc[r][s / 4], mnw = (uint32_t)mn[r][s / 4];
      d[r] = __half22float2(__hmul2(dm[r], small_half2((scw >> bs) & 0xFF, (mnw >> bs) & 0xFF)));
    }
  }
};

template <> struct tile_rows<GGML_TYPE_IQ4_XS> {
  // x_df of ggml_cuda_mmq_load_tiles_iq4_xs: d * (ls - 32) per sub-block
  uint32_t hs[2], ls[2];  // scales_h, scales_l
  float dd[2];
  __device__ __forceinline__ void init(const int* const (&rows)[2], int) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const uint32_t w0 = (uint32_t)rows[r][0];  // d | scales_h << 16
      dd[r] = __half2float(__ushort_as_half((unsigned short)(w0 & 0xFFFF)));
      hs[r] = w0 >> 16;
      ls[r] = (uint32_t)rows[r][1];
    }
  }
  // get_int_from_table_16 (vecdotq.cuh) on the table in registers (tab: the kernel's
  // copy): the values of the low (x) and high (y) nibbles of q4, in byte order
  static __device__ __forceinline__ int2 lookup16(uint32_t q4, const uint32_t (&tab)[4]) {
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t lo = __byte_perm(tab[0], tab[1], q4 >> (16 * i));
      const uint32_t hi = __byte_perm(tab[2], tab[3], q4 >> (16 * i));
      tmp[i] = __byte_perm(lo, hi, sel >> (16 * i));
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
  }
  template <int s>
  __device__ __forceinline__ void slice(const int* const (&rows)[2], const uint32_t (&tab)[4], int t, int (&a)[4],
                                        float (&d)[2]) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int2 v = lookup16(rows[r][2 + 4 * s + t], tab);
      a[r] = v.x;
      a[2 + r] = v.y;
      const int l = ((ls[r] >> (4 * s)) & 0x0F) | (((hs[r] >> (2 * s)) & 0x03) << 4);
      d[r] = dd[r] * (l - 32);
    }
  }
};

template <> struct tile_rows<GGML_TYPE_IQ2_S> {
  // block: d @0, qs @2, signs @34, qh @66, scales @74
  const uint8_t* p[2];
  float dd[2];
  __device__ __forceinline__ void init(const int* const (&rows)[2], int boff) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      p[r] = (const uint8_t*)rows[r] + boff;
      dd[r] = __half2float(*(const half*)p[r]);
    }
  }
  // a[0..1]: rows g, g+8 of values 0..15, a[2..3] of values 16..31 (one m16n8k16 each);
  // d[k][r]: x_df of ggml_cuda_mmq_load_tiles_iq2_s, ((ls_k * d + d/2) / 4)
  template <int s>
  __device__ __forceinline__ void slice(const uint32_t* table, int t, int (&a)[4], float (&d)[2][2]) {
    const int e = t / 2, w = t % 2;  // lane t: word w of grid entry e (first half) / 2+e (second)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        const int l = 2 * k + e;
        const int idx = p[r][2 + 4 * s + l] | ((p[r][66 + s] << (8 - 2 * l)) & 0x300);
        a[2 * k + r] = negate4(table[2 * idx + w], (uint32_t)p[r][34 + 4 * s + l] >> (4 * w));
      }
      const int l = p[r][74 + s];
      d[0][r] = ((l & 0x0F) * dd[r] + dd[r] / 2) / 4;
      d[1][r] = ((l >> 4) * dd[r] + dd[r] / 2) / 4;
    }
  }
};

// Adds one staged K step (8 slices) x NT column tiles into acc for the warp's 16 rows.
// y: the step's staged q8_1, [128-value block][column][36 ints].
template <ggml_type type, int NT>
static __device__ __forceinline__ void step_product(const int* const (&rows)[2], const int* __restrict__ y,
                                                    const uint32_t* __restrict__ table, const uint32_t (&tab)[4],
                                                    int boff, int g, int t, float (&acc)[NT][4]) {
  constexpr int NP = 8 * NT;
  tile_rows<type> tr;
  tr.init(rows, boff);
  auto slice = [&](auto sc) {
    constexpr int s = decltype(sc)::value;
    const int* ys = y + (s / 4) * NP * YW;
    int a[4];
    if constexpr (type == GGML_TYPE_Q4_K) {
      float2 d[2];
      tr.template slice<s>(rows, t, a, d);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int* yc = ys + nt * 8 * YW;
        const int b0 = yc[g * YW + 4 + 8 * (s % 4) + t], b1 = yc[g * YW + 8 + 8 * (s % 4) + t];
        const float2 e[2] = {__half22float2(*(const half2*)(yc + 2 * t * YW + s % 4)),
                             __half22float2(*(const half2*)(yc + (2 * t + 1) * YW + s % 4))};
        int c[4] = {0, 0, 0, 0};
        mma_k32(c, a, b0, b1);
#pragma unroll
        for (int l = 0; l < 4; ++l) {  // ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma
          acc[nt][l] += d[l / 2].x * e[l % 2].x * i2f(c[l]);
          acc[nt][l] += d[l / 2].y * e[l % 2].y;
        }
      }
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
      float d[2];
      tr.template slice<s>(rows, tab, t, a, d);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int* yc = ys + nt * 8 * YW;
        const int b0 = yc[g * YW + 4 + 8 * (s % 4) + t], b1 = yc[g * YW + 8 + 8 * (s % 4) + t];
        const float dB[2] = {((const float*)yc)[2 * t * YW + s % 4], ((const float*)yc)[(2 * t + 1) * YW + s % 4]};
        int c[4] = {0, 0, 0, 0};
        mma_k32(c, a, b0, b1);
#pragma unroll
        for (int l = 0; l < 4; ++l) {  // ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma
          acc[nt][l] += i2f(c[l]) * d[l / 2] * dB[l % 2];
        }
      }
    } else {  // IQ2_S
      float d[2][2];
      tr.template slice<s>(table, t, a, d);
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int* yc = ys + nt * 8 * YW;
        const int b0 = yc[g * YW + 4 + 8 * (s % 4) + t], b1 = yc[g * YW + 8 + 8 * (s % 4) + t];
        const float dB[2] = {((const float*)yc)[2 * t * YW + s % 4], ((const float*)yc)[(2 * t + 1) * YW + s % 4]};
        int c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
        mma_k16(c0, a[0], a[1], b0);
        mma_k16(c1, a[2], a[3], b1);
#pragma unroll
        for (int l = 0; l < 4; ++l) {  // ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma
          acc[nt][l] += dB[l % 2] * (i2f(c0[l]) * d[0][l / 2] + i2f(c1[l]) * d[1][l / 2]);
        }
      }
    }
  };
  slice(std::integral_constant<int, 0>{});
  slice(std::integral_constant<int, 1>{});
  slice(std::integral_constant<int, 2>{});
  slice(std::integral_constant<int, 3>{});
  slice(std::integral_constant<int, 4>{});
  slice(std::integral_constant<int, 5>{});
  slice(std::integral_constant<int, 6>{});
  slice(std::integral_constant<int, 7>{});
}

template <typename T> static __device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ half from_float<half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Work units: (tile, K step) in tile-major order, u = tile * nsteps + step; CTA c of G
// takes units [c * units / G, (c + 1) * units / G) (stream-K).
static __device__ __forceinline__ int first_unit(int c, int units, int nctas) {
  return (int)((int64_t)c * units / nctas);
}

// Where a stream-K CTA's piece of tile T goes in scratch: slot 2c if T is the CTA's first
// tile, else 2c + 1 (a CTA's pieces are its first and last tiles, the ones it shares).
static __device__ __forceinline__ int sk_slot(int cta, int tile, int first_tile) {
  return 2 * cta + (tile == first_tile ? 0 : 1);
}

// dst [ncols, nrows]. A tile the CTA covers whole: X's dtype into dst. Otherwise its
// piece in fp32 into part [slot][ncols][M].
template <ggml_type type, int NT, typename OutT>
static __global__ void __launch_bounds__(THREADS)
mma_k(const char* __restrict__ vx, const int4* __restrict__ vy, OutT* __restrict__ dst, float* __restrict__ part,
      const int nrows, const int ncols, const int64_t row_bytes, const int nsteps, const int units) {
  using tr = traits<type>;
  using geo = geometry<type, NT>;
  constexpr int NP = 8 * NT, G = tr::G, WG = tr::WG, WS = tr::WS, STAGES = geo::STAGES;
  constexpr int bsize = sizeof(typename tr::block);
  extern __shared__ int4 smem4[];
  int* const smem = reinterpret_cast<int*>(smem4);
  uint32_t* const table = reinterpret_cast<uint32_t*>(smem);
  int* const stage0 = smem + tr::TABLE;

  const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * WARP_SIZE + lane;
  const int g = lane / 4, t = lane % 4;
  const int ubegin = first_unit(blockIdx.x, units, gridDim.x), uend = first_unit(blockIdx.x + 1, units, gridDim.x);
  const int nu = uend - ubegin;

  if constexpr (type == GGML_TYPE_IQ2_S) {
    for (int i = tid; i < tr::TABLE / 2; i += THREADS) {
      table[2 * i + 0] = (uint32_t)iq2s_grid[i];
      table[2 * i + 1] = (uint32_t)(iq2s_grid[i] >> 32);
    }
  } else if constexpr (type == GGML_TYPE_IQ4_XS) {
    if (tid < 16) {
      ((int8_t*)table)[tid] = kvalues_iq4nl[tid];
    }
  }
  // q8_1 columns ncols..NP-1: zero in every stage, never copied over
  for (int i = tid; i < STAGES * YQ * (NP - ncols) * YW; i += THREADS) {
    const int per_q = (NP - ncols) * YW;
    const int sq = i / per_q, r = i - sq * per_q;
    stage0[(sq / YQ) * geo::STAGE_INTS + geo::W_INTS + (sq % YQ) * NP * YW + ncols * YW + r] = 0;
  }
  __syncthreads();
  uint32_t tab[4] = {0, 0, 0, 0};  // IQ4_XS: kvalues_iq4nl, in per-thread registers (an asm load:
  // nvcc would otherwise prove the words uniform and copy them from uniform registers
  // before every byte_perm)
  if constexpr (type == GGML_TYPE_IQ4_XS) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      asm("ld.shared.b32 %0, [%1];" : "=r"(tab[i]) : "r"((uint32_t)__cvta_generic_to_shared(table + i)));
    }
  }

  // This thread's copies, the same every step: weight granules tid + c * THREADS (tile
  // row i / WG, granule i % WG of the window), q8_1 granules tid + c * THREADS per block.
  constexpr int NWC = (M * WG + THREADS - 1) / THREADS, NYC = (NP * YG + THREADS - 1) / THREADS;
  int wrow[NWC], wdst[NWC], woff[NWC];  // woff: byte offset from the issue tile's first row
#pragma unroll
  for (int c = 0; c < NWC; ++c) {
    const int i = min(tid + c * THREADS, M * WG - 1), r = i / WG;
    wrow[c] = r;
    wdst[c] = r * WS + (i - r * WG) * (G / 4);
  }
  const int nyg = ncols * YG;  // q8_1 granules per 128-value block

  // The next unit to copy: (itile, ib) from ubegin on, and the tile's copy offsets (rows
  // past the end read row nrows-1; offsets stay below M * row_bytes, checked on the host).
  int itile = ubegin / nsteps, ib = ubegin - itile * nsteps;
  const char* wtile = nullptr;
  auto set_tile = [&] {
    wtile = vx + (int64_t)itile * M * row_bytes;
#pragma unroll
    for (int c = 0; c < NWC; ++c) {
      woff[c] = min(wrow[c], nrows - 1 - itile * M) * (int)row_bytes + (wdst[c] - wrow[c] * WS) * 4;
    }
  };
  set_tile();
  // Copies the next unit (block ib of tile itile) into stage st: the tile's weight rows and
  // the q8_1 of all columns.
  auto issue = [&](int st) {
    int* sw = stage0 + st * geo::STAGE_INTS;
    const char* wsrc = wtile + (((int64_t)ib * bsize) & ~(int64_t)(G - 1));
#pragma unroll
    for (int c = 0; c < NWC; ++c) {
      if (c < NWC - 1 || tid + c * THREADS < M * WG) {
        cp_async<G>(sw + wdst[c], wsrc + woff[c]);
      }
    }
    int* sy = sw + geo::W_INTS;
    const int4* ysrc = vy + (int64_t)ib * YQ * nyg;
#pragma unroll
    for (int c = 0; c < NYC; ++c) {
      const int i = tid + c * THREADS;
      if (i < nyg) {
#pragma unroll
        for (int q = 0; q < YQ; ++q) {
          cp_async<16>(sy + (q * NP * YG + i) * 4, ysrc + q * nyg + i);
        }
      }
    }
    if (++ib == nsteps) {
      ib = 0;
      ++itile;
      set_tile();
    }
  };

  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < nu) {
      issue(i);
    }
    cp_async_commit();
  }

  float acc[NT][4];
  auto zero = [&] {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        acc[nt][l] = 0.0f;
      }
    }
  };
  // Stores the accumulated tile: whole in X's dtype, a piece in fp32 scratch.
  auto flush = [&](int tile) {
    const bool whole = tile * nsteps >= ubegin && (tile + 1) * nsteps <= uend;
    const int first_tile = ubegin / nsteps;
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
      for (int l = 0; l < 4; ++l) {
        const int r = warp * 16 + g + 8 * (l / 2), row = tile * M + r, col = nt * 8 + 2 * t + l % 2;
        if (row >= nrows || col >= ncols) {
          continue;
        }
        if (whole) {
          dst[(int64_t)col * nrows + row] = from_float<OutT>(acc[nt][l]);
        } else {
          part[((int64_t)sk_slot(blockIdx.x, tile, first_tile) * ncols + col) * M + r] = acc[nt][l];
        }
      }
    }
  };

  int st = 0, pst = STAGES - 1;  // stage of the next step, stage the copy STAGES - 1 steps ahead goes to
  int i = 0;                      // steps done
  // one pass per tile the CTA's units touch: its steps [b0, b1), then the tile's store
  for (int tile = ubegin / nsteps; i < nu; ++tile) {
    const int b0 = i == 0 ? ubegin - tile * nsteps : 0, b1 = min(nsteps, uend - tile * nsteps);
    zero();
    for (int b = b0; b < b1; ++b, ++i) {
      cp_async_wait<STAGES - 2>();
      __syncthreads();  // step i landed for every thread; step i-1's stage is free
      if (i + STAGES - 1 < nu) {
        issue(pst);
      }
      cp_async_commit();
      const int* sw = stage0 + st * geo::STAGE_INTS;
      const int* rows[2] = {sw + (warp * 16 + g) * WS, sw + (warp * 16 + g + 8) * WS};
      step_product<type, NT>(rows, sw + geo::W_INTS, table, tab, (int)(((int64_t)b * bsize) & (G - 1)), g, t, acc);
      st = st + 1 == STAGES ? 0 : st + 1;
      pst = pst + 1 == STAGES ? 0 : pst + 1;
    }
    flush(tile);
  }
  cp_async_wait<0>();
}

// Per tile (blockIdx.x) that more than one CTA worked on: the sum of their pieces in CTA
// order (= K order), into dst.
template <typename OutT>
static __global__ void mma_k_fixup(const float* __restrict__ part, OutT* __restrict__ dst, const int nrows,
                                   const int ncols, const int nsteps, const int units, const int nctas) {
  const int tile = blockIdx.x;
  // the CTA whose units contain u: the largest c with c * units / nctas <= u
  auto owner = [&](int u) { return (int)(((int64_t)(u + 1) * nctas + units - 1) / units) - 1; };
  const int c0 = owner(tile * nsteps), c1 = owner((tile + 1) * nsteps - 1);
  if (c0 == c1) {
    return;  // one CTA covered the tile and wrote it
  }
  for (int i = threadIdx.x; i < ncols * M; i += blockDim.x) {
    const int col = i / M, r = i - col * M, row = tile * M + r;
    if (row >= nrows) {
      continue;
    }
    float s = 0.0f;
    for (int c = c0; c <= c1; ++c) {
      const int first_tile = first_unit(c, units, nctas) / nsteps;
      const float v = part[((int64_t)sk_slot(c, tile, first_tile) * ncols + col) * M + r];
      s = c == c0 ? v : s + v;
    }
    dst[(int64_t)col * nrows + row] = from_float<OutT>(s);
  }
}

// The launch of one product: CTAs and scratch.
struct plan {
  int units, nctas;
  bool pieces;  // some tile is shared by CTAs: scratch and the fixup kernel
  size_t smem, work_bytes;
};

// Resident CTAs per SM of the instance (the fewest over the output types), and on the
// first call per device the dynamic shared memory limit above 48 KB.
template <typename OutT>
static int occupancy(void (*kernel)(const char*, const int4*, OutT*, float*, int, int, int64_t, int, int), size_t smem) {
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int occ = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, THREADS, smem));
  return occ;
}
template <ggml_type type, int NT>
static int ctas_per_sm() {
  constexpr size_t smem = geometry<type, NT>::SMEM;
  static int occ[GGML_CUDA_MAX_DEVICES] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  GGML_ASSERT(dev < GGML_CUDA_MAX_DEVICES);
  if (occ[dev] == 0) {
    occ[dev] = std::max(1, std::min({occupancy(mma_k<type, NT, float>, smem), occupancy(mma_k<type, NT, half>, smem),
                                     occupancy(mma_k<type, NT, nv_bfloat16>, smem)}));
  }
  return occ[dev];
}

// Stream-K over every resident CTA slot (at least 4 units per CTA).
template <ggml_type type, int NT>
static plan make_plan(int nrows, int ncols, int nblocks, int nsm) {
  plan p{};
  p.smem = geometry<type, NT>::SMEM;
  p.units = (nrows + M - 1) / M * nblocks;
  p.nctas = std::max(1, std::min(ctas_per_sm<type, NT>() * nsm, p.units / 4));
  p.pieces = p.units % p.nctas != 0 || (p.units / p.nctas) % nblocks != 0;
  p.work_bytes = p.pieces ? (size_t)2 * p.nctas * ncols * M * sizeof(float) : 0;
  return p;
}

template <ggml_type type, int NT, typename OutT>
static void launch_t(const char* vx, const void* vy, void* dst, float* work, size_t work_bytes, int nrows, int ncols,
                     int64_t row_bytes, int nblocks, int nsm, cudaStream_t stream) {
  const plan p = make_plan<type, NT>(nrows, ncols, nblocks, nsm);
  GGML_ASSERT((int64_t)M * row_bytes <= INT_MAX);  // the kernel's per-thread copy offsets are int
  GGML_ASSERT(work_bytes >= p.work_bytes && (p.work_bytes == 0 || work != nullptr));
  mma_k<type, NT, OutT><<<p.nctas, dim3(WARP_SIZE, WARPS), p.smem, stream>>>(
      vx, (const int4*)vy, (OutT*)dst, work, nrows, ncols, row_bytes, nblocks, p.units);
  if (p.pieces) {
    mma_k_fixup<OutT><<<(nrows + M - 1) / M, 256, 0, stream>>>(work, (OutT*)dst, nrows, ncols, nblocks, p.units,
                                                               p.nctas);
  }
}

// The kernel instance for ncols: 2 / 4 / 8 column tiles.
template <ggml_type type, typename OutT>
static void launch_out(const char* vx, const void* vy, void* dst, float* work, size_t work_bytes, int nrows,
                       int ncols, int64_t row_bytes, int nblocks, int nsm, cudaStream_t stream) {
  if (ncols <= 16) {
    launch_t<type, 2, OutT>(vx, vy, dst, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream);
  } else if (ncols <= 32) {
    launch_t<type, 4, OutT>(vx, vy, dst, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream);
  } else {
    launch_t<type, 8, OutT>(vx, vy, dst, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream);
  }
}

template <ggml_type type>
static void launch_type(const char* vx, const void* vy, void* dst, int dst_kind, float* work, size_t work_bytes,
                        int nrows, int ncols, int64_t row_bytes, int nblocks, int nsm, cudaStream_t stream) {
  switch (dst_kind) {
    case 0: launch_out<type, float>(vx, vy, dst, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream); break;
    case 1: launch_out<type, half>(vx, vy, dst, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream); break;
    default: launch_out<type, nv_bfloat16>(vx, vy, dst, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream); break;
  }
}

template <ggml_type type>
static size_t work_bytes_type(int nrows, int ncols, int nblocks, int nsm) {
  if (ncols <= 16) {
    return make_plan<type, 2>(nrows, ncols, nblocks, nsm).work_bytes;
  }
  if (ncols <= 32) {
    return make_plan<type, 4>(nrows, ncols, nblocks, nsm).work_bytes;
  }
  return make_plan<type, 8>(nrows, ncols, nblocks, nsm).work_bytes;
}

}  // namespace

bool mma_k_supported(int type) {
  return type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_IQ2_S;
}

int mma_k_max_cols() { return 8 * MAX_NT; }

// fp32 scratch mma_k_cuda needs for this product (0: none).
size_t mma_k_work_bytes(int type, int nrows, int k, int ncols, int nsm) {
  const int nblocks = k / QK_K;
  switch (type) {
    case GGML_TYPE_Q4_K: return work_bytes_type<GGML_TYPE_Q4_K>(nrows, ncols, nblocks, nsm);
    case GGML_TYPE_IQ4_XS: return work_bytes_type<GGML_TYPE_IQ4_XS>(nrows, ncols, nblocks, nsm);
    case GGML_TYPE_IQ2_S: return work_bytes_type<GGML_TYPE_IQ2_S>(nrows, ncols, nblocks, nsm);
    default: GGML_ABORT("mma_k: type %d", type);
  }
}

// W [nrows, row_bytes] blocks of type (mma_k_supported), 16-byte aligned, contiguous
// rows, k % 512 == 0; vy: quantize_x's block_q8_1_mmq for ncols (1..64) columns in
// the type's ds layout; dst [ncols, nrows] fp32 / fp16 / bf16 (dst_kind 0 / 1 / 2);
// work: work_bytes (>= mma_k_work_bytes) of fp32 scratch, null if 0.
void mma_k_cuda(int type, const char* vx, const void* vy, void* dst, int dst_kind, float* work, size_t work_bytes,
                int nrows, int k, int64_t row_bytes, int ncols, int nsm, cudaStream_t stream) {
  GGML_ASSERT(ncols >= 1 && ncols <= 8 * MAX_NT && k % (2 * QK_K) == 0);
  const int nblocks = k / QK_K;
  switch (type) {
    case GGML_TYPE_Q4_K:
      launch_type<GGML_TYPE_Q4_K>(vx, vy, dst, dst_kind, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream);
      break;
    case GGML_TYPE_IQ4_XS:
      launch_type<GGML_TYPE_IQ4_XS>(vx, vy, dst, dst_kind, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream);
      break;
    case GGML_TYPE_IQ2_S:
      launch_type<GGML_TYPE_IQ2_S>(vx, vy, dst, dst_kind, work, work_bytes, nrows, ncols, row_bytes, nblocks, nsm, stream);
      break;
    default: GGML_ABORT("mma_k: type %d", type);
  }
}
