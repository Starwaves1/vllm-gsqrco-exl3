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
