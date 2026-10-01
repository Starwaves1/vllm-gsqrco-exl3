// SPDX-License-Identifier: Apache-2.0
// IQ3_S / IQ3_XXS x q8_1 product for 1..8 activation rows on int8 tensor cores
// (mma.sync m16n8k32 s8 x s8 -> s32, sm_80+). Owned code, called from
// lcpp_shim.cu (op lcpp_mul_mat_vec_iq3_mma).
//
// One mma = 16 weight rows x one 32-value slice (one q8_1 block) x 8 activation
// columns (columns >= ncols are zero), so the per-column dp4a work of the dp4a
// kernel (iq3_mul_mat_vec in lcpp_shim.cu) is gone and the cost is flat in
// ncols.
//
// (Below the kernel on the GGUF bytes, the same kernel on the packed layout of
// quantization/iq3_pack.py, and a tiled kernel on the packed layout for any row
// count.)
//
// Layout: persistent CTAs of 4 warps; a CTA takes 16-row tiles, its warps split
// K by weight block (warp w: blocks w, w+4, ...). Per block a warp
//   - copies its 16 rows' block bytes into shared memory with coalesced 8-byte
//     loads and stages the block's 8 q8_1 slices of every activation column
//     (both fetched into registers one block ahead, while the previous block
//     computes),
//   - decodes straight into mma fragments: lane (g, t) holds rows g and g+8
//   and,
//     of every slice, values 8t..8t+7 (words 2t, 2t+1; the mma's k order is
//     permuted the same way on A and B, which leaves every sum unchanged),
//   - looks each 4-value word up in a signed grid table built once per CTA:
//     T[grid index | sign nibble] = the grid word with the flagged bytes
//     negated (IQ3_S 512 x 16 words = 32 KB, IQ3_XXS 256 x 16 = 16 KB), one
//     prmt + one shared load per word,
//   - one mma per slice, then the sub-scale and d_q8 per output element; d_w is
//     applied once per weight block. The 4 warps' sums are added in warp order.
//
// Numerics: each slice's int32 sum is exact (the same 32 int8 products as the
// vendored dp4a chain), the sub-scale is applied in integers exactly as in
// vendored vec_dot_iq3_*_q8_1 (IQ3_XXS: (ls*s + s/2)/2 == trunc(s*(2ls+1)/4)
// for every int s), so each slice term d_q8 * float(scaled sum) is exact up to
// one fp32 rounding. The fp32 order differs from MMVQ: a block's 8 slice terms
// are summed first and multiplied by d_w once (MMVQ multiplies each term by d_w
// * d_q8), then blocks in K order per warp, then the 4 warps in order.

#include "common.cuh"
#include "mmq.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int WARPS = 4;  // warps per CTA, splitting K
constexpr int THREADS = WARPS * WARP_SIZE;
constexpr int ROWS = 16;                         // weight rows per tile (mma M)
constexpr int SLICES = QK_K / QK8_1;             // q8_1 blocks per weight block
constexpr int Y_BLOCK = sizeof(block_q8_1) / 4;  // 9 words
constexpr int Y_COL = SLICES * Y_BLOCK;          // words per staged column
constexpr int W_ROW =
    120;  // staged bytes per weight row: the block's 8-byte window

template <ggml_type type>
struct traits;
template <>
struct traits<GGML_TYPE_IQ3_S> {
  using block = block_iq3_s;
  static constexpr int grid_size = 512;
  static constexpr int words =
      15;  // 8-byte words covering 110 B at an even offset
  static __device__ const uint32_t* grid() { return iq3s_grid; }
};
template <>
struct traits<GGML_TYPE_IQ3_XXS> {
  using block = block_iq3_xxs;
  static constexpr int grid_size = 256;
  static constexpr int words = 13;  // 98 B
  static __device__ const uint32_t* grid() { return iq3xxs_grid; }
};

static __device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b,
                                                uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

static __device__ __forceinline__ uint32_t lds16(const char* p) {
  return *(const uint16_t*)p;
}

// g with the bytes flagged in the low nibble of signs negated. The IQ3 grids
// have no zero byte, so (g ^ 0xFF) + 1 per byte never carries.
static __device__ __forceinline__ uint32_t negate(uint32_t g, uint32_t signs) {
  const uint32_t ones = (signs * 0x00204081u) & 0x01010101u;
  return (g ^ (ones * 0xFFu)) + ones;
}

// The slice's int32 sum with its sub-scale applied, as the vendored vec_dot.
template <ggml_type type>
static __device__ __forceinline__ int scaled(int sumi, int ls2p1) {
  const int y = sumi * ls2p1;
  // y / 4 rounded toward zero: + 3 when negative (|y| < 2^24, so bits 31:30 are
  // the sign)
  return type == GGML_TYPE_IQ3_S ? y : (y + (int)((uint32_t)y >> 30)) >> 2;
}

// Lane t's bytes of one row's block, read from the staged copy, and the
// per-block values derived from them. IQ3_S table index: qs | sign nibble << 8
// | qh << 12.
struct raw_s {
  uint32_t d, sc0, sc1, qh[4], q[8], sb[8];
};
static __device__ __forceinline__ void load(raw_s& r, const char* b, int t) {
  r.d = lds16(b);
  r.sc0 = lds16(b + offsetof(block_iq3_s, scales));
  r.sc1 = lds16(b + offsetof(block_iq3_s, scales) + 2);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    r.qh[i] = lds16(b + offsetof(block_iq3_s, qh) + 2 * i);
#pragma unroll
  for (int s = 0; s < SLICES; ++s) {
    r.q[s] = lds16(b + offsetof(block_iq3_s, qs) + 8 * s + 2 * t);
    r.sb[s] = *(const uint8_t*)(b + offsetof(block_iq3_s, signs) + 4 * s + t);
  }
}
struct row_s {
  float d;
  uint32_t sc, qa[2],
      qb[2];  // qh bit 2t / 2t+1 of slice s at bit 4 of byte s%4 of [s/4]
};
static __device__ __forceinline__ void prep(row_s& p, const raw_s& r, int t) {
  p.d = __half2float(__ushort_as_half((unsigned short)r.d));
  p.sc = r.sc0 | r.sc1 << 16;
  const uint32_t h0 = r.qh[0] | r.qh[1] << 16, h1 = r.qh[2] | r.qh[3] << 16;
  p.qa[0] = ((h0 >> (2 * t)) & 0x01010101u) << 4;
  p.qb[0] = ((h0 >> (2 * t + 1)) & 0x01010101u) << 4;
  p.qa[1] = ((h1 >> (2 * t)) & 0x01010101u) << 4;
  p.qb[1] = ((h1 >> (2 * t + 1)) & 0x01010101u) << 4;
}
// Words 2t, 2t+1 of slice s and 2*ls+1 (values of vec_dot_iq3_s_q8_1).
template <int s>
static __device__ __forceinline__ void slice(const row_s& p, const raw_s& r,
                                             int, const uint32_t* table,
                                             int& w0, int& w1, int& l) {
  constexpr uint32_t k = s % 4,
                     z = 8 | k;  // z: sign of a 0x00/0x10 byte = 0x00
  const uint32_t h0 =
      prmt(p.qa[s / 4], 0, z | k << 4 | z << 8 | z << 12);  // qh bit at 12
  const uint32_t h1 = prmt(p.qb[s / 4], 0, z | k << 4 | z << 8 | z << 12);
  w0 = table[(prmt(r.q[s], r.sb[s], 0x3240) & 0x0FFFu) |
             h0];  // qs0 | sign nibble 0 << 8
  w1 = table[prmt(r.q[s], r.sb[s] >> 4, 0x3241) |
             h1];  // qs1 | sign nibble 1 << 8
  l = 1 + 2 * ((p.sc >> (4 * s)) & 0xF);
}

// IQ3_XXS table index: q3 | sign nibble << 8.
struct raw_xxs {
  uint32_t d, q[8], a0[8], a1[8];
};
static __device__ __forceinline__ void load(raw_xxs& r, const char* b, int t) {
  r.d = lds16(b);
#pragma unroll
  for (int s = 0; s < SLICES; ++s) {
    r.q[s] = lds16(b + offsetof(block_iq3_xxs, qs) + 8 * s + 2 * t);
    r.a0[s] = lds16(b + offsetof(block_iq3_xxs, qs) + QK_K / 4 + 4 * s);
    r.a1[s] = lds16(b + offsetof(block_iq3_xxs, qs) + QK_K / 4 + 4 * s + 2);
  }
}
struct row_xxs {
  float d;
};
static __device__ __forceinline__ void prep(row_xxs& p, const raw_xxs& r, int) {
  p.d = __half2float(__ushort_as_half((unsigned short)r.d));
}
// Words 2t, 2t+1 of slice s and 2*ls+1 (values of vec_dot_iq3_xxs_q8_1).
template <int s>
static __device__ __forceinline__ void slice(const row_xxs&, const raw_xxs& r,
                                             int t, const uint32_t* table,
                                             int& w0, int& w1, int& l) {
  const uint32_t aux = r.a0[s] | r.a1[s] << 16;
  const uint32_t raw = (aux >> (7 * t)) & 0x7F;  // signs of words 2t, 2t+1
  const uint32_t full = raw | (__popc(raw) & 1) << 7;  // unpack_ksigns
  w0 = table[prmt(r.q[s], full, 0x3240) & 0x0FFFu];
  w1 = table[prmt(r.q[s], full >> 4, 0x3241)];
  l = 2 * (aux >> 28) + 1;
}

template <ggml_type type>
struct types;
template <>
struct types<GGML_TYPE_IQ3_S> {
  using raw = raw_s;
  using row = row_s;
};
template <>
struct types<GGML_TYPE_IQ3_XXS> {
  using raw = raw_xxs;
  using row = row_xxs;
};

// 48.5 KB with the IQ3_S table: 2 CTAs per SM.
struct smem {
  __align__(16) int y[WARPS]
                     [8 * Y_COL];  // staged q8_1 slices, [column][slice] blocks
  __align__(16) char w[WARPS][ROWS * W_ROW];  // staged weight windows; the
                                              // reduction reuses it
};

template <ggml_type type>
static __global__ void __launch_bounds__(THREADS)
    iq3_mma(const char* __restrict__ vx, const block_q8_1* __restrict__ vy,
            float* __restrict__ dst, const int nrows, const int nblocks,
            const int64_t row_bytes, const int ncols) {
  using tr = traits<type>;
  using raw_t = typename types<type>::raw;
  using row_t = typename types<type>::row;
  constexpr int nstage = (8 * 18 + WARP_SIZE - 1) /
                         WARP_SIZE;  // int4s per lane per block at ncols = 8
  constexpr int bsize = sizeof(typename tr::block);
  constexpr int nwin = ROWS * tr::words;  // 8-byte words per warp per block
  constexpr int nw = (nwin + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ uint32_t table[tr::grid_size * 16];
  extern __shared__ __align__(16) char dyn[];
  smem& sm = *reinterpret_cast<smem*>(dyn);

  const int lane = threadIdx.x, warp = threadIdx.y,
            tid = warp * WARP_SIZE + lane;
  for (int gi = tid; gi < tr::grid_size; gi += THREADS) {
    const uint32_t gv = tr::grid()[gi];
    const int base =
        type == GGML_TYPE_IQ3_S ? (gi & 0xFF) | (gi >> 8) << 12 : gi;
#pragma unroll
    for (int nib = 0; nib < 16; ++nib) table[base | nib << 8] = negate(gv, nib);
  }
  int* yt = sm.y[warp];
  for (int i = ncols * Y_COL + lane; i < 8 * Y_COL; i += WARP_SIZE)
    yt[i] = 0;  // columns >= ncols
  char* wt = sm.w[warp];
  __syncthreads();

  const int nby =
      nblocks * SLICES;  // q8_1 blocks per activation row (K % 512 == 0)
  const int g = lane / 4, t = lane % 4;
  const int ntiles = (nrows + ROWS - 1) / ROWS;
  // activation staging: copy lane + 32c is int4 e of column j's 288 B for the
  // block
  int ysrc[nstage], ydst[nstage];
#pragma unroll
  for (int c = 0; c < nstage; ++c) {
    const int i = min(lane + WARP_SIZE * c, ncols * 18 - 1), j = i / 18,
              e = i - 18 * j;
    ysrc[c] = j * (nby / 4) * 9 + e;
    ydst[c] = j * (Y_COL / 4) + e;
  }
  // weight staging: copy lane + 32c is 8-byte word wword of row wrow's window
  int wrow[nw], wword[nw];
#pragma unroll
  for (int c = 0; c < nw; ++c) {
    const int i = min(lane + WARP_SIZE * c, nwin - 1);
    wrow[c] = i / tr::words;
    wword[c] = i - wrow[c] * tr::words;
  }
  const int* yb = yt + g * Y_COL + 1 + 2 * t;  // qs words 2t, 2t+1 of column g
  const int* yd0 = yt + 2 * t * Y_COL;         // d_q8 of columns 2t, 2t+1
  const int* yd1 = yd0 + Y_COL;
  const int4* y4 = reinterpret_cast<const int4*>(vy);
  const char* wend = vx + (int64_t)nrows * row_bytes;

  uint2 wn[nw];
  int4 yn[nstage];
  // Next block's bytes into registers: 8-byte words from the 8-byte boundary at
  // or below the block's first byte (tail rows read row nrows-1). The window
  // can run up to 10 bytes past the block; past the end of W, words are read as
  // their low half or as 0. That drops no byte of the last block only because K
  // % 512 == 0 keeps W's size 0 or 4 mod 8 (last block at offset 2 or 6);
  // check_inputs enforces it.
  auto fetch = [&](int tl, int bl) {
#pragma unroll
    for (int c = 0; c < nw; ++c) {
      if (lane + WARP_SIZE * c < nwin) {
        const char* bp =
            vx + (int64_t)min(tl * ROWS + wrow[c], nrows - 1) * row_bytes +
            (int64_t)bl * bsize;
        const char* a =
            (const char*)((uintptr_t)bp & ~(uintptr_t)7) + 8 * wword[c];
        wn[c] = a + 8 <= wend
                    ? *(const uint2*)a
                    : make_uint2(a + 4 <= wend ? *(const uint32_t*)a : 0u, 0u);
      }
    }
#pragma unroll
    for (int c = 0; c < nstage; ++c) {
      if (lane + WARP_SIZE * c < ncols * 18) yn[c] = y4[18 * bl + ysrc[c]];
    }
  };

  int tile = blockIdx.x, b = warp;
  if (tile < ntiles && b < nblocks) fetch(tile, b);
  for (; tile < ntiles; tile += gridDim.x) {
    const int row0 = tile * ROWS;
    float acc[4] = {0.0f, 0.0f, 0.0f,
                    0.0f};  // (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)
    for (; b < nblocks; b += WARPS) {
#pragma unroll
      for (int c = 0; c < nstage; ++c) {
        if (lane + WARP_SIZE * c < ncols * 18)
          reinterpret_cast<int4*>(yt)[ydst[c]] = yn[c];
      }
#pragma unroll
      for (int c = 0; c < nw; ++c) {
        if (lane + WARP_SIZE * c < nwin)
          *(uint2*)(wt + wrow[c] * W_ROW + 8 * wword[c]) = wn[c];
      }
      {  // next: this tile's next block, else the next tile's first
        int nb = b + WARPS, nt = tile;
        if (nb >= nblocks) {
          nb = warp;
          nt = tile + gridDim.x;
        }
        if (nt < ntiles && nb < nblocks) fetch(nt, nb);
      }
      __syncwarp();
      // the block's offset in each row's window (vx is 16-byte aligned)
      const int off0 = (int)(((int64_t)min(row0 + g, nrows - 1) * row_bytes +
                              (int64_t)b * bsize) &
                             7);
      const int off1 =
          (int)(((int64_t)min(row0 + g + 8, nrows - 1) * row_bytes +
                 (int64_t)b * bsize) &
                7);
      raw_t r0, r1;
      load(r0, wt + g * W_ROW + off0, t);
      load(r1, wt + (g + 8) * W_ROW + off1, t);
      __syncwarp();
      row_t p0, p1;
      prep(p0, r0, t);
      prep(p1, r1, t);
      float bacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#define IQ3_MMA_STEP(s)                                                      \
  {                                                                          \
    int a0, a1, a2, a3, l0, l1;                                              \
    slice<s>(p0, r0, t, table, a0, a2, l0);                                  \
    slice<s>(p1, r1, t, table, a1, a3, l1);                                  \
    const int b0 = yb[(s) * Y_BLOCK], b1 = yb[(s) * Y_BLOCK + 1];            \
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;                                      \
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, " \
        "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"                      \
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)                             \
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));             \
    const float dq0 = __low2float(*(const half2*)(yd0 + (s) * Y_BLOCK));     \
    const float dq1 = __low2float(*(const half2*)(yd1 + (s) * Y_BLOCK));     \
    bacc[0] += dq0 * (float)scaled<type>(c0, l0);                            \
    bacc[1] += dq1 * (float)scaled<type>(c1, l0);                            \
    bacc[2] += dq0 * (float)scaled<type>(c2, l1);                            \
    bacc[3] += dq1 * (float)scaled<type>(c3, l1);                            \
  }
      IQ3_MMA_STEP(0);
      IQ3_MMA_STEP(1);
      IQ3_MMA_STEP(2);
      IQ3_MMA_STEP(3);
      IQ3_MMA_STEP(4);
      IQ3_MMA_STEP(5);
      IQ3_MMA_STEP(6);
      IQ3_MMA_STEP(7);
#undef IQ3_MMA_STEP
      acc[0] += p0.d * bacc[0];
      acc[1] += p0.d * bacc[1];
      acc[2] += p1.d * bacc[2];
      acc[3] += p1.d * bacc[3];
      __syncwarp();
    }
    b = warp;

    // the 4 warps' sums, added in warp order, through the staged-weight buffers
    float* red = (float*)wt;
#pragma unroll
    for (int e = 0; e < 4; ++e) red[e * WARP_SIZE + lane] = acc[e];
    __syncthreads();
    const int e = warp, l = lane;  // thread (e, l) finishes element e of lane l
    const int row = row0 + l / 4 + 8 * (e / 2), col = 2 * (l % 4) + e % 2;
    float sum = ((const float*)sm.w[0])[e * WARP_SIZE + l];
#pragma unroll
    for (int w = 1; w < WARPS; ++w)
      sum += ((const float*)sm.w[w])[e * WARP_SIZE + l];
    if (col < ncols && row < nrows) dst[(int64_t)col * nrows + row] = sum;
    __syncthreads();  // before the next tile's staging overwrites the buffers
  }
}

// Resident CTAs x SMs of kernel with `bytes` of dynamic shared memory, per
// device; set on the kernel's first (eager) call, with its dynamic shared
// memory limit. ctas: the kernel's own cache.
template <typename Kernel>
static int resident_ctas(int (&ctas)[16], Kernel kernel, size_t bytes) {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  GGML_ASSERT(dev < 16);
  if (ctas[dev] == 0) {
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    int occ = 0, sms = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel,
                                                             THREADS, bytes));
    CUDA_CHECK(
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    ctas[dev] = std::max(1, occ) * sms;
  }
  return ctas[dev];
}

template <ggml_type type>
void launch(const char* vx, const block_q8_1* y, float* dst, int nrows,
            int nblocks, int64_t row_bytes, int ncols, cudaStream_t stream) {
  constexpr size_t bytes = sizeof(smem);
  static int ctas[16] = {0};
  const int ntiles = (nrows + ROWS - 1) / ROWS;
  iq3_mma<type><<<std::min(ntiles, resident_ctas(ctas, iq3_mma<type>, bytes)),
                  dim3(WARP_SIZE, WARPS), bytes, stream>>>(
      vx, y, dst, nrows, nblocks, row_bytes, ncols);
}

// ---------------------------------------------------------------------------
// The same product on the packed layout (quantization/iq3_pack.py, applied at
// load by GGUFLinearMethod._pack_iq3), 1..32 activation rows. Per 16-row tile
// and weight block W holds one 16-byte aligned record with each lane's bytes in
// mma fragment order, so a lane fetches them with 6 (IQ3_S) / 7 (IQ3_XXS)
// coalesced loads straight into registers (no staging copy, no 2-byte shared
// loads), and a word's table index is one byte permute (IQ3_S: grid index byte
// + the 5 bits above it) or permute + mask (IQ3_XXS: the pack re-codes each
// pair's 7 sign bits so that the table rebuilds both 4th signs; no parity per
// word). Tiles, warps, mma, sub-scales and the fp32 order are the kernel
// above's: at 1..8 rows the two are bit-identical, and above 8 each output
// column is computed exactly as in an 8-row call.

struct pfrag {
  int4 q0, q1;  // grid-index bytes of rows g, g+8: byte 2s+e (of 16) = word
                // 2t+e of slice s
  uint4 h;      // IQ3_S: H0..H3; IQ3_XXS: B0, B1, B2 (bit 7 of each byte: B3's
                // bytes 2, 3)
  uint32_t h4;  // IQ3_S: H4; IQ3_XXS: B3's bytes 0, 1
  uint2 sc;     // sub-scale nibbles of rows g, g+8
  uint32_t d;   // half2: d of rows g, g+8
};

template <ggml_type type>
static __device__ __forceinline__ void pload(pfrag& f,
                                             const char* __restrict__ tb,
                                             int lane) {
  constexpr int fb = type == GGML_TYPE_IQ3_S
                         ? 1664
                         : 1472;  // lane bytes before the sub-scales and d
  f.q0 = *(const int4*)(tb + 16 * lane);
  f.q1 = *(const int4*)(tb + 512 + 16 * lane);
  if (type == GGML_TYPE_IQ3_S) {
    f.h = *(const uint4*)(tb + 1024 + 16 * lane);
    f.h4 = *(const uint32_t*)(tb + 1536 + 4 * lane);
  } else {
    const uint2 b01 = *(const uint2*)(tb + 1024 + 8 * lane);
    f.h =
        make_uint4(b01.x, b01.y, *(const uint32_t*)(tb + 1280 + 4 * lane), 0u);
    f.h4 = *(const uint16_t*)(tb + 1408 + 2 * lane);
  }
  f.sc = *(const uint2*)(tb + fb + 8 * (lane / 4));
  f.d = *(const uint32_t*)(tb + fb + 64 + 4 * (lane / 4));
}

template <int c>
static __device__ __forceinline__ uint32_t comp(const int4& v) {
  return (uint32_t)(c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w);
}

// Per-block words the table indices are cut from: IQ3_S X0..X7 (byte j of X[4R
// + c]: the 5 bits above the grid index of word 2t + (j & 1), slice 2c + (j >>
// 1), row g + 8R, bits 5-7 zero); IQ3_XXS B0..B3 (byte s % 4 of B[2R + s / 4]:
// slice s's re-coded sign byte, bit 7 don't-care) and B >> 3 (the second word's
// nibble in bits 0-3).
template <ggml_type type>
struct pwords {
  uint32_t x[8];
};
static __device__ __forceinline__ void pdecode(pwords<GGML_TYPE_IQ3_S>& w,
                                               const pfrag& f) {
  w.x[0] = f.h.x & 0x1F1F1F1Fu;
  w.x[1] = f.h.y & 0x1F1F1F1Fu;
  w.x[2] = f.h.z & 0x1F1F1F1Fu;
  w.x[3] = f.h.w & 0x1F1F1F1Fu;
  w.x[4] = f.h4 & 0x1F1F1F1Fu;
  w.x[5] = ((f.h.x >> 5) & 0x07070707u) | ((f.h.y >> 2) & 0x18181818u);
  w.x[6] = ((f.h.z >> 5) & 0x07070707u) | ((f.h.w >> 2) & 0x18181818u);
  w.x[7] = ((f.h4 >> 5) & 0x07070707u) | ((f.h.y >> 4) & 0x08080808u) |
           ((f.h.w >> 3) & 0x10101010u);
}
static __device__ __forceinline__ uint32_t
spare4(uint32_t v) {  // bit 7 of each byte, byte 0 first
  return (((v >> 7) & 0x01010101u) * 0x01020408u) >> 24;
}
static __device__ __forceinline__ void pdecode(pwords<GGML_TYPE_IQ3_XXS>& w,
                                               const pfrag& f) {
  const uint32_t sp = spare4(f.h.x) | spare4(f.h.y) << 4 | spare4(f.h.z) << 8 |
                      ((f.h4 >> 7) & 1u) << 12 | ((f.h4 >> 15) & 1u) << 13;
  w.x[0] = f.h.x;
  w.x[1] = f.h.y;
  w.x[2] = f.h.z;
  w.x[3] = f.h4 | (sp & 0x7Fu) << 16 | (sp >> 7) << 24;
#pragma unroll
  for (int i = 0; i < 4; ++i) w.x[4 + i] = w.x[i] >> 3;
}

// Slice s's A fragment: a0/a2 = row g words 2t/2t+1, a1/a3 = row g+8.
template <int s>
static __device__ __forceinline__ void pslice(const pwords<GGML_TYPE_IQ3_S>& w,
                                              const pfrag& f,
                                              const uint32_t* table, int& a0,
                                              int& a1, int& a2, int& a3) {
  constexpr int c = s / 2;
  constexpr uint32_t j0 = 2 * (s % 2), j1 = j0 + 1;
  // bytes: grid index, X byte j, then 0 (X's bit 7, replicated)
  constexpr uint32_t s0 =
      j0 | (4 + j0) << 4 | (0xC + j0) << 8 | (0xC + j0) << 12;
  constexpr uint32_t s1 =
      j1 | (4 + j1) << 4 | (0xC + j1) << 8 | (0xC + j1) << 12;
  const uint32_t qa = comp<c>(f.q0), qb = comp<c>(f.q1);
  a0 = table[prmt(qa, w.x[c], s0)];
  a2 = table[prmt(qa, w.x[c], s1)];
  a1 = table[prmt(qb, w.x[4 + c], s0)];
  a3 = table[prmt(qb, w.x[4 + c], s1)];
}
template <int s>
static __device__ __forceinline__ void pslice(
    const pwords<GGML_TYPE_IQ3_XXS>& w, const pfrag& f, const uint32_t* table,
    int& a0, int& a1, int& a2, int& a3) {
  constexpr int c = s / 2, k = s % 4, r = s / 4;
  constexpr uint32_t j0 = 2 * (s % 2), j1 = j0 + 1;
  constexpr uint32_t s0 = j0 | (4 + k) << 4,
                     s1 = j1 | (4 + k) << 4;  // bytes 2, 3 masked off
  const uint32_t qa = comp<c>(f.q0), qb = comp<c>(f.q1);
  a0 = table[prmt(qa, w.x[r], s0) & 0x0FFFu];
  a2 = table[4096 + (prmt(qa, w.x[4 + r], s1) & 0x0FFFu)];
  a1 = table[prmt(qb, w.x[2 + r], s0) & 0x0FFFu];
  a3 = table[4096 + (prmt(qb, w.x[6 + r], s1) & 0x0FFFu)];
}

// The signed grid table of the packed kernels (pslice's indices), PTABLE words.
// IQ3_S: T[q | sign nibble << 8 | qh << 12]; IQ3_XXS: T[q | n << 8] for a
// pair's first word (n = its 3 signs + the nibble's parity) and T[4096 + (q | n
// << 8)] for the second (n = the first's parity + its 3 signs); the 4th sign is
// the parity of the other 3 and n's parity bit.
constexpr int PTABLE = 8192;
template <ggml_type type>
static __device__ __forceinline__ void build_ptable(uint32_t* table, int tid,
                                                    int nthreads) {
  for (int i = tid; i < PTABLE; i += nthreads) {
    const uint32_t q = i & 0xFF, n = (i >> 8) & 0xF;
    uint32_t gv, signs;
    if (type == GGML_TYPE_IQ3_S) {
      gv = iq3s_grid[q | (i >> 12) << 8];
      signs = n;
    } else {
      gv = iq3xxs_grid[q];
      signs = i < 4096 ? (n & 7) | ((n >> 3) ^ (__popc(n & 7) & 1)) << 3
                       : (n >> 1) | ((n & 1) ^ (__popc(n >> 1) & 1)) << 3;
    }
    table[i] = negate(gv, signs);
  }
}

// NG groups of 8 activation columns (1..8 * NG columns): per slice the A
// fragment is cut once and used by NG mmas. Each output column is computed as
// at NG = 1, in the same order.
template <ggml_type type, int NG>
static __global__ void __launch_bounds__(THREADS)
    iq3_mma_packed(const char* __restrict__ vx,
                   const block_q8_1* __restrict__ vy, float* __restrict__ dst,
                   const int nrows, const int nblocks, const int ncols) {
  using tr = traits<type>;
  constexpr int ycols = 8 * NG;
  constexpr int nstage = (ycols * 18 + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int64_t tbytes = ROWS * sizeof(typename tr::block);
  __shared__ uint32_t table[PTABLE];
  extern __shared__ __align__(16) int
      dyn_p[];  // [WARPS][ycols * Y_COL] staging, [WARPS][NG * 128] sums

  const int lane = threadIdx.x, warp = threadIdx.y,
            tid = warp * WARP_SIZE + lane;
  build_ptable<type>(table, tid, THREADS);
  int* yt = dyn_p + warp * ycols * Y_COL;
  float* red = (float*)(dyn_p + WARPS * ycols * Y_COL);
  for (int i = ncols * Y_COL + lane; i < ycols * Y_COL; i += WARP_SIZE)
    yt[i] = 0;  // columns >= ncols
  __syncthreads();

  const int nby = nblocks * SLICES;
  const int g = lane / 4, t = lane % 4;
  const int ntiles = nrows / ROWS;
  int ysrc[nstage], ydst[nstage];
#pragma unroll
  for (int c = 0; c < nstage; ++c) {
    const int i = min(lane + WARP_SIZE * c, ncols * 18 - 1), j = i / 18,
              e = i - 18 * j;
    ysrc[c] = j * (nby / 4) * 9 + e;
    ydst[c] = j * (Y_COL / 4) + e;
  }
  const int* yb = yt + g * Y_COL + 1 + 2 * t;
  const int* yd0 = yt + 2 * t * Y_COL;
  const int* yd1 = yd0 + Y_COL;
  const int4* y4 = reinterpret_cast<const int4*>(vy);

  pfrag fn;
  int4 yn[nstage];
  auto fetch = [&](int tl, int bl) {
    pload<type>(fn, vx + ((int64_t)tl * nblocks + bl) * tbytes, lane);
#pragma unroll
    for (int c = 0; c < nstage; ++c) {
      if (lane + WARP_SIZE * c < ncols * 18) yn[c] = y4[18 * bl + ysrc[c]];
    }
  };

  int tile = blockIdx.x, b = warp;
  if (tile < ntiles && b < nblocks) fetch(tile, b);
  for (; tile < ntiles; tile += gridDim.x) {
    float acc[NG][4] = {};
    for (; b < nblocks; b += WARPS) {
#pragma unroll
      for (int c = 0; c < nstage; ++c) {
        if (lane + WARP_SIZE * c < ncols * 18)
          reinterpret_cast<int4*>(yt)[ydst[c]] = yn[c];
      }
      const pfrag f =
          fn;  // this block's weight words; fn takes the next block's
      {
        int nb = b + WARPS, nt = tile;
        if (nb >= nblocks) {
          nb = warp;
          nt = tile + gridDim.x;
        }
        if (nt < ntiles && nb < nblocks) fetch(nt, nb);
      }
      __syncwarp();
      pwords<type> w;
      pdecode(w, f);
      float bacc[NG][4] = {};
#define IQ3_MMA_PSTEP(s)                                                       \
  {                                                                            \
    int a0, a1, a2, a3;                                                        \
    pslice<s>(w, f, table, a0, a1, a2, a3);                                    \
    const int l0 = 1 + 2 * ((f.sc.x >> (4 * (s))) & 0xF),                      \
              l1 = 1 + 2 * ((f.sc.y >> (4 * (s))) & 0xF);                      \
    _Pragma("unroll") for (int j = 0; j < NG; ++j) {                           \
      const int b0 = yb[8 * j * Y_COL + (s) * Y_BLOCK],                        \
                b1 = yb[8 * j * Y_COL + (s) * Y_BLOCK + 1];                    \
      int c0 = 0, c1 = 0, c2 = 0, c3 = 0;                                      \
      asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, " \
          "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"                      \
          : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)                             \
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));             \
      const float dq0 =                                                        \
          __low2float(*(const half2*)(yd0 + 8 * j * Y_COL + (s) * Y_BLOCK));   \
      const float dq1 =                                                        \
          __low2float(*(const half2*)(yd1 + 8 * j * Y_COL + (s) * Y_BLOCK));   \
      bacc[j][0] += dq0 * (float)scaled<type>(c0, l0);                         \
      bacc[j][1] += dq1 * (float)scaled<type>(c1, l0);                         \
      bacc[j][2] += dq0 * (float)scaled<type>(c2, l1);                         \
      bacc[j][3] += dq1 * (float)scaled<type>(c3, l1);                         \
    }                                                                          \
  }
      IQ3_MMA_PSTEP(0);
      IQ3_MMA_PSTEP(1);
      IQ3_MMA_PSTEP(2);
      IQ3_MMA_PSTEP(3);
      IQ3_MMA_PSTEP(4);
      IQ3_MMA_PSTEP(5);
      IQ3_MMA_PSTEP(6);
      IQ3_MMA_PSTEP(7);
#undef IQ3_MMA_PSTEP
      const float2 dw = __half22float2(*(const half2*)&f.d);
#pragma unroll
      for (int j = 0; j < NG; ++j) {
        acc[j][0] += dw.x * bacc[j][0];
        acc[j][1] += dw.x * bacc[j][1];
        acc[j][2] += dw.y * bacc[j][2];
        acc[j][3] += dw.y * bacc[j][3];
      }
      __syncwarp();
    }
    b = warp;

    // the warps' sums, added in warp order
#pragma unroll
    for (int j = 0; j < NG; ++j) {
#pragma unroll
      for (int e = 0; e < 4; ++e)
        red[(warp * NG + j) * 128 + e * WARP_SIZE + lane] = acc[j][e];
    }
    __syncthreads();
    for (int i = tid; i < NG * 128; i += THREADS) {
      const int j = i / 128, e = (i / WARP_SIZE) % 4, l = i % WARP_SIZE;
      const int row = tile * ROWS + l / 4 + 8 * (e / 2),
                col = 8 * j + 2 * (l % 4) + e % 2;
      float sum = red[j * 128 + e * WARP_SIZE + l];
#pragma unroll
      for (int w = 1; w < WARPS; ++w)
        sum += red[(w * NG + j) * 128 + e * WARP_SIZE + l];
      if (col < ncols) dst[(int64_t)col * nrows + row] = sum;
    }
    __syncthreads();
  }
}

// 44 KB of shared memory at NG = 1 (2 CTAs per SM), 54 / 77 KB at NG = 2 / 4 (1
// CTA per SM).
template <ggml_type type, int NG>
void launch_packed(const char* vx, const block_q8_1* y, float* dst, int nrows,
                   int nblocks, int ncols, cudaStream_t stream) {
  constexpr size_t bytes = (size_t)WARPS * (8 * NG * Y_COL + NG * 128) * 4;
  static int ctas[16] = {0};
  const int ntiles = nrows / ROWS;
  iq3_mma_packed<type, NG>
      <<<std::min(ntiles, resident_ctas(ctas, iq3_mma_packed<type, NG>, bytes)),
         dim3(WARP_SIZE, WARPS), bytes, stream>>>(vx, y, dst, nrows, nblocks,
                                                  ncols);
}

// ---------------------------------------------------------------------------
// The packed layout at any number of activation rows (prefill chunks, decode at
// 9+ rows): a tiled int8 tensor-core product, op lcpp_mul_mat_iq3_packed.
//
// Inputs as the vendored MMQ's except W: X quantized by the shim's quantize_x
// into MMQ's block_q8_1_mmq layout (IQ3: D4, one fp32 scale per 32 values), W
// packed. A CTA is 8 warps; a warp owns 32 weight rows (two 16-row packed
// tiles) x TN = 16 / 32 / 48 / 64 activation columns (NT = TN / 8 mma column
// tiles); a CTA tile is 256 rows x TN columns.
//  - A: per weight block, each warp loads its two tile-blocks' records from
//  global memory
//    straight into registers (pload, one block ahead) and cuts the fragments
//    with the packed decode kernel's table and words (pdecode / pslice). No
//    decoded A tile in shared memory.
//  - B: per weight block (one pipeline step, one CTA barrier), the two
//  block_q8_1_mmq of every
//    column of the tile go to shared memory by cp.async, STAGES - 1 blocks
//    ahead: the quants at a 160-byte column stride (the lane's 8-byte B
//    fragment loads are then free of bank conflicts), the 4 scales per 128
//    values transposed to [slice][column] (a lane's two columns in one 8-byte
//    load). Columns >= ncols are zero-filled.
//  - scaling: per slice the vendored ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma term,
//    sum += float(C) * dA * dB (dA: x_df of ggml_cuda_mmq_load_tiles_iq3_s /
//    _iq3_xxs, exact), computed as fma(fma(M + C, dA, -M * dA), dB, sum) with M
//    = 1.5 * 2^23: the mma adds its int32 C to M's bits, so M + C is exact in
//    fp32 (|C| < 2^22), -M * dA is exact (dA is a half times the 5-bit integer
//    2 ls + 1: <= 16 significant bits), and the inner fma rounds float(C) * dA
//    once, the value of the vendored I2F + FMUL. The slices are added in K
//    order, as in MMQ; a tile one CTA computes whole is then bit-identical to
//    MMQ's fp32 sum if MMQ also computes it whole (MMQ's stream-k splits its
//    own 128 x J tiles).
//  - schedule: persistent CTAs, one per SM (G). Whole tiles go out in waves,
//  CTA c taking
//    tiles c, c + G, ... (row tile major, column tile fastest, so a wave's CTAs
//    share weight and activation blocks in L2); the (tile, weight block) units
//    of the last tiles % G tiles are split evenly over the G CTAs (stream-K),
//    unless whole-tile waves on fewer CTAs are estimated faster (plan below). A
//    tile a CTA covers whole is written in X's dtype; the pieces of a shared
//    tile go to fp32 scratch and a second kernel adds them in CTA (= K) order.

namespace tiled {

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * WARP_SIZE;
constexpr int TM = 2 * ROWS * WARPS;  // 256 weight rows per tile
constexpr int YB =
    sizeof(block_q8_1_mmq);  // 144 B: one column's 128 values (d4[4], then qs)
constexpr int YS = 160;      // staged column stride of the 128 quants
constexpr int STAGES = 3;
constexpr float MAGIC = 12582912.0f;  // 1.5 * 2^23
constexpr int MAGIC_BITS = 0x4B400000;
constexpr int FIXUP_MAX_CTAS = 256;
static_assert(YB == 144, "block_q8_1_mmq layout");

template <int NT_>
struct cfg {
  static constexpr int NT = NT_;
  static constexpr int TN = 8 * NT;  // activation columns per tile
  static constexpr int BLOCK_Q =
      TN * YS;  // one 128-value block's quants: [column][YS]
  static constexpr int BLOCK =
      BLOCK_Q + 4 * TN * 4;  // then its scales: [slice][column] fp32
  static constexpr int STAGE = 2 * BLOCK;  // a weight block's 256 values
  static constexpr size_t SMEM =
      PTABLE * 4 + (size_t)STAGES * STAGE + 16;  // + the mma addend
};

// The launch's work split (see above). Units of a tile: its nkb weight blocks.
struct sched {
  int nkb;    // weight blocks per row
  int nct;    // column tiles; tile = row tile * nct + column tile
  int tail0;  // tiles 0 .. tail0 - 1 go out whole (CTA c: c, c + G, ...); the
              // rest are split
  int tail_units;  // the split tiles' units, (tiles - tail0) * nkb
};

static __device__ __forceinline__ int tail_begin(const sched& s, int c,
                                                 int nctas) {
  return (int)((int64_t)c * s.tail_units / nctas);
}

static __device__ __forceinline__ void cp_async16(void* dst, const void* src,
                                                  int src_bytes) {
  const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d),
               "l"(src), "r"(src_bytes));
}
static __device__ __forceinline__ void cp_async4(void* dst, const void* src,
                                                 int src_bytes) {
  const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(d),
               "l"(src), "r"(src_bytes));
}
static __device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}
template <int N>
static __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// d = a x b + m as int32. m is M's bits x 4, loaded from shared memory by the
// kernel: as a known constant ptxas re-materializes the 4-register operand
// whenever it needs the registers
// (~2 moves per mma, 11 % of the loop at 8 column tiles).
static __device__ __forceinline__ void mma_m(int (&d)[4], const int (&a)[4],
                                             int b0, int b1,
                                             const int (&m)[4]) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, "
      "%5, %6, %7}, {%8, %9}, "
      "{%10, %11, %12, %13};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(m[0]),
        "r"(m[1]), "r"(m[2]), "r"(m[3]));
}

template <typename T>
static __device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ half from_float<half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

// One 32-value slice s of the warp's 32 rows x 8 NT columns: y points at the
// tile's first column's quants of the slice's 128-value block in the stage, ys
// at that block's scales
// ([slice % 4][TN columns]).
template <ggml_type type, int NT, int s>
static __device__ __forceinline__ void slice_product(
    const pfrag (&f)[2], const pwords<type> (&w)[2], const float (&dq)[2][2],
    const uint32_t* table, const char* __restrict__ y,
    const float* __restrict__ ys, int g, int t, const int (&mg)[4],
    float (&acc)[2][NT][4]) {
  int a[2][4];
  float dA[2][2], nm[2][2];
#pragma unroll
  for (int mt = 0; mt < 2; ++mt) {
    pslice<s>(w[mt], f[mt], table, a[mt][0], a[mt][1], a[mt][2], a[mt][3]);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      // (2 ls + 1) * dq, exact; float(ls) as the bits of 2^23 + ls (I2F is
      // quarter rate on sm_86)
      const uint32_t ls = ((r ? f[mt].sc.y : f[mt].sc.x) >> (4 * s)) & 0xF;
      dA[mt][r] = fmaf(__uint_as_float(0x4B000000u | ls) - 8388608.0f,
                       2.0f * dq[mt][r], dq[mt][r]);
      nm[mt][r] = dA[mt][r] * -MAGIC;
    }
  }
  constexpr int q = s % 4;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) {
    const int2 b = *(const int2*)(y + (nt * 8 + g) * YS + 32 * q + 8 * t);
    const float2 dB = *(const float2*)(ys + q * 8 * NT + nt * 8 +
                                       2 * t);  // columns 2t, 2t + 1
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      int c[4];
      mma_m(c, a[mt], b.x, b.y, mg);
      // ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma: sum += C * dA * dB
      acc[mt][nt][0] = fmaf(fmaf(__int_as_float(c[0]), dA[mt][0], nm[mt][0]),
                            dB.x, acc[mt][nt][0]);
      acc[mt][nt][1] = fmaf(fmaf(__int_as_float(c[1]), dA[mt][0], nm[mt][0]),
                            dB.y, acc[mt][nt][1]);
      acc[mt][nt][2] = fmaf(fmaf(__int_as_float(c[2]), dA[mt][1], nm[mt][1]),
                            dB.x, acc[mt][nt][2]);
      acc[mt][nt][3] = fmaf(fmaf(__int_as_float(c[3]), dA[mt][1], nm[mt][1]),
                            dB.y, acc[mt][nt][3]);
    }
  }
}

// Local unit j of a CTA = (tile, kb): its whole tiles (cta, cta + G, ...), then
// its run of the split part, from (tile0, kb0) on. Advanced one unit at a time
// (no division per step).
struct cursor {
  int j, tile, kb;
};
struct walk {
  int nw, G, nkb, tile0,
      kb0;  // units of whole tiles; the split run's first unit
  __device__ __forceinline__ cursor first(int cta) const {
    return nw > 0 ? cursor{0, cta, 0} : cursor{0, tile0, kb0};
  }
  __device__ __forceinline__ void next(cursor& c) const {
    ++c.j;
    if (c.j == nw) {
      c.tile = tile0;
      c.kb = kb0;
    } else if (++c.kb == nkb) {
      c.kb = 0;
      c.tile += c.j < nw ? G : 1;
    }
  }
};

// dst [ncols, nrows] in OutT; part: fp32 scratch for shared tiles, [2
// G][TN][TM].
template <ggml_type type, class C, typename OutT>
static __global__ void __launch_bounds__(THREADS, 1)
    iq3_packed_mmq(const char* __restrict__ vx, const char* __restrict__ vy,
                   OutT* __restrict__ dst, float* __restrict__ part,
                   const int nrows, const int ncols, const sched s) {
  constexpr int NT = C::NT, TN = C::TN;
  constexpr int64_t tbytes = ROWS * sizeof(typename traits<type>::block);
  extern __shared__ __align__(16) char smem_t[];
  uint32_t* const table = reinterpret_cast<uint32_t*>(smem_t);
  char* const stages = smem_t + PTABLE * 4;

  const int lane = threadIdx.x, warp = threadIdx.y,
            tid = warp * WARP_SIZE + lane;
  const int g = lane / 4, t = lane % 4;
  const int G = gridDim.x, cta = blockIdx.x;
  const int ntiles16 = nrows / ROWS;
  const int nw = (cta < s.tail0 ? (s.tail0 - cta + G - 1) / G : 0) *
                 s.nkb;  // units of the CTA's whole tiles
  const int tb = tail_begin(s, cta, G);
  const int nlocal = nw + tail_begin(s, cta + 1, G) - tb;
  const int first_tail_tile = s.tail0 + tb / s.nkb;

  build_ptable<type>(table, tid,
                     THREADS);  // read after the first step's barrier

  const walk wk{nw, G, s.nkb, first_tail_tile,
                tb - (first_tail_tile - s.tail0) * s.nkb};
  cursor cu = wk.first(cta), ci = cu,
         cw = cu;  // compute, copies (issue), weight loads (wload)

  // Unit ci's activations into stage i % STAGES: its 128-value blocks kq = 2 kb
  // + b (b = 0, 1) of the tile's columns, block_q8_1_mmq kq of each, are one
  // run of TN * 144 bytes in global memory each: every column's 128 quants (8
  // granules of 16 bytes after its 4 scales) go to [column][YS], its 4 scales
  // (4 bytes each) to [slice][column]. Columns >= ncols are zero-filled.
  constexpr int
      NQ = TN * 8,
      NS = TN * 4;  // copies per block: 16-byte quant granules, 4-byte scales
  auto issue = [&](int i) {
    const int ct = ci.tile % s.nct, nreal = ncols - ct * TN;
    char* st = stages + (i % STAGES) * C::STAGE;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const char* src = vy + ((int64_t)(2 * ci.kb + b) * ncols + ct * TN) * YB;
      char* sb = st + b * C::BLOCK;
#pragma unroll
      for (int c = 0; c < (NQ + THREADS - 1) / THREADS; ++c) {
        const int idx = tid + c * THREADS, col = idx / 8, gr = idx % 8;
        if (NQ % THREADS == 0 || idx < NQ) {
          const bool ok = col < nreal;
          cp_async16(sb + col * YS + 16 * gr,
                     src + (ok ? col * YB + 16 + 16 * gr : 0), ok ? 16 : 0);
        }
      }
#pragma unroll
      for (int c = 0; c < (NS + THREADS - 1) / THREADS; ++c) {
        const int idx = tid + c * THREADS, col = idx / 4, q = idx % 4;
        if (NS % THREADS == 0 || idx < NS) {
          const bool ok = col < nreal;
          cp_async4(sb + C::BLOCK_Q + (q * TN + col) * 4,
                    src + (ok ? col * YB + 4 * q : 0), ok ? 4 : 0);
        }
      }
    }
    wk.next(ci);
  };
  pfrag fn[2];
  auto wload = [&] {
    const int rt = cw.tile / s.nct;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int t16 = rt * (TM / ROWS) + warp * 2 + mt;
      if (t16 < ntiles16) {
        pload<type>(fn[mt], vx + ((int64_t)t16 * s.nkb + cw.kb) * tbytes, lane);
      } else {
        fn[mt] = pfrag{};
      }
    }
    wk.next(cw);
  };

  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < nlocal) issue(i);
    cp_async_commit();
  }
  if (nlocal > 0) wload();

  int4* const smagic = reinterpret_cast<int4*>(stages + STAGES * C::STAGE);
  if (tid == 0)
    *smagic = make_int4(MAGIC_BITS, MAGIC_BITS, MAGIC_BITS, MAGIC_BITS);
  __syncthreads();
  const int4 m4 = *smagic;
  const int mg[4] = {m4.x, m4.y, m4.z, m4.w};
  float acc[2][NT][4];
  auto zero = [&] {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int l = 0; l < 4; ++l) acc[mt][nt][l] = 0.0f;
  };
  zero();       // and after each store (not a select per unit)
  int kb0 = 0;  // the current tile's first block here
  for (; cu.j < nlocal; wk.next(cu)) {
    const int j = cu.j, tile = cu.tile, kb = cu.kb;
    if (j == nw) kb0 = kb;  // the split run's first unit
    const pfrag f[2] = {fn[0], fn[1]};
    if (j + 1 < nlocal) wload();
    pwords<type> w[2];
    float dq[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      pdecode(w[mt], f[mt]);
      const float2 d = __half22float2(*(const half2*)&f[mt].d);
      const float k = type == GGML_TYPE_IQ3_S ? 1.0f : 0.25f;
      dq[mt][0] = d.x * k;
      dq[mt][1] = d.y * k;
    }
    cp_async_wait<STAGES - 2>();
    __syncthreads();  // unit j's activations landed for every thread; unit j -
                      // 1's stage is free
    if (j + STAGES - 1 < nlocal) issue(j + STAGES - 1);
    cp_async_commit();
    const char* st = stages + (j % STAGES) * C::STAGE;
    // slices 4h .. 4h + 3 read the unit's 128-value block h
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const char* y = st + h * C::BLOCK;
      const float* ys = reinterpret_cast<const float*>(y + C::BLOCK_Q);
      if (h == 0) {
        slice_product<type, NT, 0>(f, w, dq, table, y, ys, g, t, mg, acc);
        slice_product<type, NT, 1>(f, w, dq, table, y, ys, g, t, mg, acc);
        slice_product<type, NT, 2>(f, w, dq, table, y, ys, g, t, mg, acc);
        slice_product<type, NT, 3>(f, w, dq, table, y, ys, g, t, mg, acc);
      } else {
        slice_product<type, NT, 4>(f, w, dq, table, y, ys, g, t, mg, acc);
        slice_product<type, NT, 5>(f, w, dq, table, y, ys, g, t, mg, acc);
        slice_product<type, NT, 6>(f, w, dq, table, y, ys, g, t, mg, acc);
        slice_product<type, NT, 7>(f, w, dq, table, y, ys, g, t, mg, acc);
      }
    }
    if (j + 1 == nlocal || j + 1 == nw ||
        kb + 1 == s.nkb) {  // the tile's last unit here: store
      const int rt = tile / s.nct, ct = tile - rt * s.nct;
      const int r0 = warp * 2 * ROWS + g,
                c0 = 2 * t;  // the lane's first row / column in the tile
      const int nvalid =
          min(2, ntiles16 - (rt * (TM / ROWS) +
                             warp * 2));  // the warp's 16-row tiles in W
      if (kb0 == 0 && kb + 1 == s.nkb) {  // whole: X's dtype into dst
        OutT* d = dst + (int64_t)(ct * TN + c0) * nrows + rt * TM + r0;
        const int cmax =
            ncols - ct * TN - c0;  // columns of this lane (c0 + ...) that exist
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          if (mt >= nvalid) break;
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
              const int cc = nt * 8 + l % 2;
              if (cc < cmax)
                d[(int64_t)cc * nrows + mt * ROWS + 8 * (l / 2)] =
                    from_float<OutT>(acc[mt][nt][l]);
            }
          }
        }
      } else {  // a piece: fp32 into the CTA's slot (columns >= ncols are
                // written, never read)
        float* pp =
            part +
            ((int64_t)(2 * cta + (tile == first_tail_tile ? 0 : 1)) * TN + c0) *
                TM +
            r0;
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int l = 0; l < 4; ++l)
              pp[(nt * 8 + l % 2) * TM + mt * ROWS + 8 * (l / 2)] =
                  acc[mt][nt][l];
          }
        }
      }
      zero();
      kb0 = 0;  // the next tile of the CTA's run starts at its first block
    }
  }
  cp_async_wait<0>();
}

// Per split tile (tail0 + blockIdx.x): the sum of its pieces in CTA order (= K
// order) into dst. The contributing CTAs' slots are found once per tile (thread
// 0), then each thread adds its elements' pieces.
template <class C, typename OutT>
static __global__ void __launch_bounds__(256)
    iq3_packed_fixup(const float* __restrict__ part, OutT* __restrict__ dst,
                     const int nrows, const int ncols, const sched s,
                     const int nctas) {
  constexpr int TN = C::TN;
  __shared__ int
      slots[FIXUP_MAX_CTAS];  // a tile's pieces come from at most every CTA
  __shared__ int npieces;
  const int tile = s.tail0 + blockIdx.x;
  if (threadIdx.x == 0) {
    // the CTAs whose tail units [tail_begin(c), tail_begin(c + 1)) meet the
    // tile's [u0, u0 + nkb)
    const int u0 = blockIdx.x * s.nkb, u1 = u0 + s.nkb;
    int n = 0;
    int c = (int)((int64_t)u0 * nctas /
                  s.tail_units);  // tail_begin(c) <= u0: floor(floor(u0 G / U)
                                  // U / G)
    for (int b = tail_begin(s, c, nctas); c < nctas && b < u1; ++c) {
      const int e = tail_begin(s, c + 1, nctas);
      if (e > b && e > u0) {  // a non-empty run that meets the tile
        slots[n++] = 2 * c + (tile == s.tail0 + b / s.nkb ? 0 : 1);
      }
      b = e;
    }
    npieces = n;
  }
  __syncthreads();
  const int n = npieces;
  if (n <= 1) return;  // one CTA covered the tile and wrote it
  const int rt = tile / s.nct, ct = tile - rt * s.nct;
  for (int i = threadIdx.x; i < TM * TN; i += blockDim.x) {
    const int cc = i / TM, r = i - cc * TM, row = rt * TM + r,
              col = ct * TN + cc;
    if (row >= nrows || col >= ncols) continue;
    float sum = part[(int64_t)slots[0] * TN * TM + i];
    for (int k = 1; k < n; ++k) sum += part[(int64_t)slots[k] * TN * TM + i];
    dst[(int64_t)col * nrows + row] = from_float<OutT>(sum);
  }
}

// Tile widths: 16 / 32 / 48 / 64 columns (2 / 4 / 6 / 8 mma column tiles per
// warp).
using C16 = cfg<2>;
using C32 = cfg<4>;
using C48 = cfg<6>;
using C64 = cfg<8>;

template <ggml_type type, class C>
static int sm_slots() {  // resident CTAs per device: 1 per SM (fewest over the
                         // output types)
  static int slots[GGML_CUDA_MAX_DEVICES] = {};
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  GGML_ASSERT(dev < GGML_CUDA_MAX_DEVICES);
  if (slots[dev] == 0) {
    const int bytes = (int)C::SMEM;
    int occ = INT_MAX, sms = 0;
    auto one = [&](auto kernel) {
      int o = 0;
      CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&o, kernel,
                                                               THREADS, bytes));
      occ = std::min(occ, o);
    };
    one(iq3_packed_mmq<type, C, float>);
    one(iq3_packed_mmq<type, C, half>);
    one(iq3_packed_mmq<type, C, nv_bfloat16>);
    CUDA_CHECK(
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    GGML_ASSERT(occ > 0);  // the tile's shared memory fits (C64: 100,368 of
                           // 101,376 B on sm_86)
    slots[dev] = occ * sms;
  }
  return slots[dev];
}

struct plan {
  int tn;  // tile width
  int nctas;
  sched s;
  bool split;  // some tile is shared: scratch + fixup
  size_t work_bytes;
  double cost;
};

// Estimated time of a launch in units of the unit cost: per SM, a unit (weight
// block x tile) costs ~1.7 + 0.55 NT us on the 3090 (measured 2.8 / 4.1 / 6.2
// us at NT = 2 / 4 / 8), and splitting tiles costs ~7 units more (scratch,
// fixup; whole-tile waves on 68 CTAs beat the split on 82 at 17408 rows, 32 and
// 128 columns). The plan's width was the fastest forced one at every n and
// shape measured. Calibrated on the 3090; other GPUs may prefer other constants
// (only speed depends on them, never results).
static double unit_cost(int nt) { return 2.9 + nt; }
constexpr double SPLIT_UNITS = 7.0;

template <ggml_type type, class C>
static plan make_plan(int nrows, int ncols, int nkb) {
  plan p{};
  p.tn = C::TN;
  const int nrt = (nrows + TM - 1) / TM, nct = (ncols + C::TN - 1) / C::TN,
            tiles = nrt * nct;
  const int gmax = std::max(1, std::min(sm_slots<type, C>(), tiles * nkb));
  const int waves = (tiles + gmax - 1) / gmax;
  const double whole = (double)waves * nkb,
               split = (double)tiles * nkb / gmax +
                       (tiles % gmax ? SPLIT_UNITS : 0.0);
  // whole-tile waves on as few CTAs as that many waves need, or the split tail
  // on gmax
  p.nctas = whole <= split ? (tiles + waves - 1) / waves : gmax;
  p.cost = std::min(whole, split) * unit_cost(C::NT);
  p.s.nkb = nkb;
  p.s.nct = nct;
  p.s.tail0 = whole <= split ? tiles : tiles / p.nctas * p.nctas;
  p.s.tail_units = (tiles - p.s.tail0) * nkb;
  p.split =
      p.s.tail_units > 0;  // fewer tail tiles than CTAs: some tile is shared
  p.work_bytes = p.split ? (size_t)2 * p.nctas * TM * C::TN * sizeof(float) : 0;
  return p;
}

// The cheapest of the tile widths (a narrower one on a tie).
template <ggml_type type>
static plan best_plan(int nrows, int ncols, int nkb) {
  plan best = make_plan<type, C16>(nrows, ncols, nkb);
  for (const plan& p : {make_plan<type, C32>(nrows, ncols, nkb),
                        make_plan<type, C48>(nrows, ncols, nkb),
                        make_plan<type, C64>(nrows, ncols, nkb)}) {
    if (p.cost < best.cost) best = p;
  }
  return best;
}

template <ggml_type type, class C, typename OutT>
static void launch_t(const char* vx, const void* vy, void* dst, float* work,
                     int nrows, int ncols, const plan& p, cudaStream_t stream) {
  GGML_ASSERT(!p.split || work != nullptr);
  GGML_ASSERT(p.nctas <= FIXUP_MAX_CTAS);  // the fixup's piece list
  iq3_packed_mmq<type, C, OutT>
      <<<p.nctas, dim3(WARP_SIZE, WARPS), C::SMEM, stream>>>(
          vx, (const char*)vy, (OutT*)dst, work, nrows, ncols, p.s);
  if (p.split) {
    iq3_packed_fixup<C, OutT><<<p.s.tail_units / p.s.nkb, 256, 0, stream>>>(
        work, (OutT*)dst, nrows, ncols, p.s, p.nctas);
  }
}

template <ggml_type type, typename OutT>
static void launch_out(const char* vx, const void* vy, void* dst, float* work,
                       int nrows, int ncols, int nkb, cudaStream_t stream) {
  const plan p = best_plan<type>(nrows, ncols, nkb);
  switch (p.tn) {
    case 16:
      launch_t<type, C16, OutT>(vx, vy, dst, work, nrows, ncols, p, stream);
      break;
    case 32:
      launch_t<type, C32, OutT>(vx, vy, dst, work, nrows, ncols, p, stream);
      break;
    case 48:
      launch_t<type, C48, OutT>(vx, vy, dst, work, nrows, ncols, p, stream);
      break;
    default:
      launch_t<type, C64, OutT>(vx, vy, dst, work, nrows, ncols, p, stream);
      break;
  }
}

template <ggml_type type>
static size_t work_bytes_t(int nrows, int ncols, int nkb) {
  return best_plan<type>(nrows, ncols, nkb).work_bytes;
}

template <ggml_type type>
static void launch_type(const char* vx, const void* vy, void* dst, int dst_kind,
                        float* work, int nrows, int ncols, int nkb,
                        cudaStream_t stream) {
  switch (dst_kind) {
    case 0:
      launch_out<type, float>(vx, vy, dst, work, nrows, ncols, nkb, stream);
      break;
    case 1:
      launch_out<type, half>(vx, vy, dst, work, nrows, ncols, nkb, stream);
      break;
    default:
      launch_out<type, nv_bfloat16>(vx, vy, dst, work, nrows, ncols, nkb,
                                    stream);
      break;
  }
}

}  // namespace tiled

}  // namespace

// W [nrows, row_bytes] IQ3_S/IQ3_XXS blocks, contiguous rows, 16-byte aligned;
// vy: ncols block_q8_1 rows of k values (16-byte aligned, k % 512 == 0); dst
// [ncols, nrows] fp32.
void iq3_mma_mul_mat_vec_cuda(ggml_type type, const char* vx, const void* vy,
                              float* dst, int nrows, int k, int64_t row_bytes,
                              int ncols, cudaStream_t stream) {
  const block_q8_1* y = (const block_q8_1*)vy;
  GGML_ASSERT(ncols >= 1 && ncols <= 8);
  (type == GGML_TYPE_IQ3_S ? launch<GGML_TYPE_IQ3_S>
                           : launch<GGML_TYPE_IQ3_XXS>)(vx, y, dst, nrows,
                                                        k / QK_K, row_bytes,
                                                        ncols, stream);
}

// The same on W packed by quantization/iq3_pack.py (nrows % 16 == 0), 1..32
// activation rows.
template <ggml_type type>
static void launch_packed_n(const char* vx, const block_q8_1* y, float* dst,
                            int nrows, int nblocks, int ncols,
                            cudaStream_t stream) {
  (ncols <= 8 ? launch_packed<type, 1>
   : ncols <= 16
       ? launch_packed<type, 2>
       : launch_packed<type, 4>)(vx, y, dst, nrows, nblocks, ncols, stream);
}

void iq3_mma_packed_mul_mat_vec_cuda(ggml_type type, const char* vx,
                                     const void* vy, float* dst, int nrows,
                                     int k, int ncols, cudaStream_t stream) {
  const block_q8_1* y = (const block_q8_1*)vy;
  GGML_ASSERT(ncols >= 1 && ncols <= 32 && nrows % ROWS == 0);
  (type == GGML_TYPE_IQ3_S
       ? launch_packed_n<GGML_TYPE_IQ3_S>
       : launch_packed_n<GGML_TYPE_IQ3_XXS>)(vx, y, dst, nrows, k / QK_K, ncols,
                                             stream);
}

// fp32 scratch iq3_packed_mmq_cuda needs for this product (0: none).
size_t iq3_packed_mmq_work_bytes(ggml_type type, int nrows, int k, int ncols) {
  return (type == GGML_TYPE_IQ3_S
              ? tiled::work_bytes_t<GGML_TYPE_IQ3_S>
              : tiled::work_bytes_t<GGML_TYPE_IQ3_XXS>)(nrows, ncols, k / QK_K);
}

// W packed by quantization/iq3_pack.py (nrows % 16 == 0); vy: quantize_x's
// block_q8_1_mmq (D4) for ncols >= 1 columns of k values (k % 256 == 0); dst
// [ncols, nrows] fp32 / fp16 / bf16 (dst_kind 0 / 1 / 2); work:
// iq3_packed_mmq_work_bytes of fp32 scratch or null.
void iq3_packed_mmq_cuda(ggml_type type, const char* vx, const void* vy,
                         void* dst, int dst_kind, float* work, int nrows, int k,
                         int ncols, cudaStream_t stream) {
  GGML_ASSERT(ncols >= 1 && nrows % ROWS == 0 && k % QK_K == 0);
  (type == GGML_TYPE_IQ3_S
       ? tiled::launch_type<GGML_TYPE_IQ3_S>
       : tiled::launch_type<GGML_TYPE_IQ3_XXS>)(vx, vy, dst, dst_kind, work,
                                                nrows, ncols, k / QK_K, stream);
}
