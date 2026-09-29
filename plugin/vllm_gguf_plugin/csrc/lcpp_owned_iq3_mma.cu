// SPDX-License-Identifier: Apache-2.0
// IQ3_S / IQ3_XXS x q8_1 product for 1..8 activation rows on int8 tensor cores
// (mma.sync m16n8k32 s8 x s8 -> s32, sm_80+). Owned code, called from
// lcpp_shim.cu (op lcpp_mul_mat_vec_iq3_mma); cloud/results/phase3/k2 has the
// iteration log and numbers.
//
// One mma = 16 weight rows x one 32-value slice (one q8_1 block) x 8 activation
// columns (columns >= ncols are zero), so the per-column dp4a work of the dp4a
// kernel (iq3_mul_mat_vec in lcpp_shim.cu) is gone and the cost is flat in ncols.
//
// (Below the kernel on the GGUF bytes, the same kernel on the packed layout of
// quantization/iq3_pack.py, and the packed -> GGUF unpack for MMQ.)
//
// Layout: persistent CTAs of 4 warps; a CTA takes 16-row tiles, its warps split
// K by weight block (warp w: blocks w, w+4, ...). Per block a warp
//   - copies its 16 rows' block bytes into shared memory with coalesced 8-byte
//     loads and stages the block's 8 q8_1 slices of every activation column (both
//     fetched into registers one block ahead, while the previous block computes),
//   - decodes straight into mma fragments: lane (g, t) holds rows g and g+8 and,
//     of every slice, values 8t..8t+7 (words 2t, 2t+1; the mma's k order is
//     permuted the same way on A and B, which leaves every sum unchanged),
//   - looks each 4-value word up in a signed grid table built once per CTA:
//     T[grid index | sign nibble] = the grid word with the flagged bytes negated
//     (IQ3_S 512 x 16 words = 32 KB, IQ3_XXS 256 x 16 = 16 KB), one prmt + one
//     shared load per word,
//   - one mma per slice, then the sub-scale and d_q8 per output element; d_w is
//     applied once per weight block. The 4 warps' sums are added in warp order.
//
// Numerics: each slice's int32 sum is exact (the same 32 int8 products as the
// vendored dp4a chain), the sub-scale is applied in integers exactly as in
// vendored vec_dot_iq3_*_q8_1 (IQ3_XXS: (ls*s + s/2)/2 == trunc(s*(2ls+1)/4) for
// every int s), so each slice term d_q8 * float(scaled sum) is exact up to one
// fp32 rounding. The fp32 order differs from MMVQ: a block's 8 slice terms are
// summed first and multiplied by d_w once (MMVQ multiplies each term by
// d_w * d_q8), then blocks in K order per warp, then the 4 warps in order.

#include "common.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <cstddef>

namespace {

constexpr int WARPS = 4;                     // warps per CTA, splitting K
constexpr int THREADS = WARPS * WARP_SIZE;
constexpr int ROWS = 16;                     // weight rows per tile (mma M)
constexpr int SLICES = QK_K / QK8_1;         // q8_1 blocks per weight block
constexpr int Y_BLOCK = sizeof(block_q8_1) / 4;  // 9 words
constexpr int Y_COL = SLICES * Y_BLOCK;      // words per staged column
constexpr int W_ROW = 120;                   // staged bytes per weight row: the block's 8-byte window

template <ggml_type type>
struct traits;
template <>
struct traits<GGML_TYPE_IQ3_S> {
  using block = block_iq3_s;
  static constexpr int grid_size = 512;
  static constexpr int words = 15;  // 8-byte words covering 110 B at an even offset
  static __device__ const uint32_t* grid() { return iq3s_grid; }
};
template <>
struct traits<GGML_TYPE_IQ3_XXS> {
  using block = block_iq3_xxs;
  static constexpr int grid_size = 256;
  static constexpr int words = 13;  // 98 B
  static __device__ const uint32_t* grid() { return iq3xxs_grid; }
};

static __device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}

static __device__ __forceinline__ uint32_t lds16(const char* p) { return *(const uint16_t*)p; }

// g with the bytes flagged in the low nibble of signs negated. The IQ3 grids have no zero byte,
// so (g ^ 0xFF) + 1 per byte never carries.
static __device__ __forceinline__ uint32_t negate(uint32_t g, uint32_t signs) {
  const uint32_t ones = (signs * 0x00204081u) & 0x01010101u;
  return (g ^ (ones * 0xFFu)) + ones;
}

// The slice's int32 sum with its sub-scale applied, as the vendored vec_dot.
template <ggml_type type>
static __device__ __forceinline__ int scaled(int sumi, int ls2p1) {
  const int y = sumi * ls2p1;
  // y / 4 rounded toward zero: + 3 when negative (|y| < 2^24, so bits 31:30 are the sign)
  return type == GGML_TYPE_IQ3_S ? y : (y + (int)((uint32_t)y >> 30)) >> 2;
}

// Lane t's bytes of one row's block, read from the staged copy, and the per-block
// values derived from them. IQ3_S table index: qs | sign nibble << 8 | qh << 12.
struct raw_s {
  uint32_t d, sc0, sc1, qh[4], q[8], sb[8];
};
static __device__ __forceinline__ void load(raw_s& r, const char* b, int t) {
  r.d = lds16(b);
  r.sc0 = lds16(b + offsetof(block_iq3_s, scales));
  r.sc1 = lds16(b + offsetof(block_iq3_s, scales) + 2);
#pragma unroll
  for (int i = 0; i < 4; ++i) r.qh[i] = lds16(b + offsetof(block_iq3_s, qh) + 2 * i);
#pragma unroll
  for (int s = 0; s < SLICES; ++s) {
    r.q[s] = lds16(b + offsetof(block_iq3_s, qs) + 8 * s + 2 * t);
    r.sb[s] = *(const uint8_t*)(b + offsetof(block_iq3_s, signs) + 4 * s + t);
  }
}
struct row_s {
  float d;
  uint32_t sc, qa[2], qb[2];  // qh bit 2t / 2t+1 of slice s at bit 4 of byte s%4 of [s/4]
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
static __device__ __forceinline__ void slice(const row_s& p, const raw_s& r, int, const uint32_t* table,
                                             int& w0, int& w1, int& l) {
  constexpr uint32_t k = s % 4, z = 8 | k;  // z: sign of a 0x00/0x10 byte = 0x00
  const uint32_t h0 = prmt(p.qa[s / 4], 0, z | k << 4 | z << 8 | z << 12);  // qh bit at 12
  const uint32_t h1 = prmt(p.qb[s / 4], 0, z | k << 4 | z << 8 | z << 12);
  w0 = table[(prmt(r.q[s], r.sb[s], 0x3240) & 0x0FFFu) | h0];  // qs0 | sign nibble 0 << 8
  w1 = table[prmt(r.q[s], r.sb[s] >> 4, 0x3241) | h1];         // qs1 | sign nibble 1 << 8
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
static __device__ __forceinline__ void slice(const row_xxs&, const raw_xxs& r, int t, const uint32_t* table,
                                             int& w0, int& w1, int& l) {
  const uint32_t aux = r.a0[s] | r.a1[s] << 16;
  const uint32_t raw = (aux >> (7 * t)) & 0x7F;              // signs of words 2t, 2t+1
  const uint32_t full = raw | (__popc(raw) & 1) << 7;        // unpack_ksigns
  w0 = table[prmt(r.q[s], full, 0x3240) & 0x0FFFu];
  w1 = table[prmt(r.q[s], full >> 4, 0x3241)];
  l = 2 * (aux >> 28) + 1;
}

template <ggml_type type> struct types;
template <> struct types<GGML_TYPE_IQ3_S> { using raw = raw_s; using row = row_s; };
template <> struct types<GGML_TYPE_IQ3_XXS> { using raw = raw_xxs; using row = row_xxs; };

// 48.5 KB with the IQ3_S table: 2 CTAs per SM.
struct smem {
  __align__(16) int y[WARPS][8 * Y_COL];      // staged q8_1 slices, [column][slice] blocks
  __align__(16) char w[WARPS][ROWS * W_ROW];   // staged weight windows; the reduction reuses it
};

template <ggml_type type>
static __global__ void __launch_bounds__(THREADS)
iq3_mma(const char* __restrict__ vx, const block_q8_1* __restrict__ vy, float* __restrict__ dst,
        const int nrows, const int nblocks, const int64_t row_bytes, const int ncols) {
  using tr = traits<type>;
  using raw_t = typename types<type>::raw;
  using row_t = typename types<type>::row;
  constexpr int nstage = (8 * 18 + WARP_SIZE - 1) / WARP_SIZE;  // int4s per lane per block at ncols = 8
  constexpr int bsize = sizeof(typename tr::block);
  constexpr int nwin = ROWS * tr::words;                            // 8-byte words per warp per block
  constexpr int nw = (nwin + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ uint32_t table[tr::grid_size * 16];
  extern __shared__ __align__(16) char dyn[];
  smem& sm = *reinterpret_cast<smem*>(dyn);

  const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * WARP_SIZE + lane;
  for (int gi = tid; gi < tr::grid_size; gi += THREADS) {
    const uint32_t gv = tr::grid()[gi];
    const int base = type == GGML_TYPE_IQ3_S ? (gi & 0xFF) | (gi >> 8) << 12 : gi;
#pragma unroll
    for (int nib = 0; nib < 16; ++nib) table[base | nib << 8] = negate(gv, nib);
  }
  int* yt = sm.y[warp];
  for (int i = ncols * Y_COL + lane; i < 8 * Y_COL; i += WARP_SIZE) yt[i] = 0;  // columns >= ncols
  char* wt = sm.w[warp];
  __syncthreads();

  const int nby = nblocks * SLICES;  // q8_1 blocks per activation row (K % 512 == 0)
  const int g = lane / 4, t = lane % 4;
  const int ntiles = (nrows + ROWS - 1) / ROWS;
  // activation staging: copy lane + 32c is int4 e of column j's 288 B for the block
  int ysrc[nstage], ydst[nstage];
#pragma unroll
  for (int c = 0; c < nstage; ++c) {
    const int i = min(lane + WARP_SIZE * c, ncols * 18 - 1), j = i / 18, e = i - 18 * j;
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
  // Next block's bytes into registers: 8-byte words from the 8-byte boundary at or below the
  // block's first byte (tail rows read row nrows-1). The window can run up to 10 bytes past
  // the block; past the end of W, words are read as their low half or as 0. That drops no
  // byte of the last block only because K % 512 == 0 keeps W's size 0 or 4 mod 8 (last
  // block at offset 2 or 6); check_inputs enforces it.
  auto fetch = [&](int tl, int bl) {
#pragma unroll
    for (int c = 0; c < nw; ++c) {
      if (lane + WARP_SIZE * c < nwin) {
        const char* bp = vx + (int64_t)min(tl * ROWS + wrow[c], nrows - 1) * row_bytes + (int64_t)bl * bsize;
        const char* a = (const char*)((uintptr_t)bp & ~(uintptr_t)7) + 8 * wword[c];
        wn[c] = a + 8 <= wend ? *(const uint2*)a : make_uint2(a + 4 <= wend ? *(const uint32_t*)a : 0u, 0u);
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
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1)
    for (; b < nblocks; b += WARPS) {
#pragma unroll
      for (int c = 0; c < nstage; ++c) {
        if (lane + WARP_SIZE * c < ncols * 18) reinterpret_cast<int4*>(yt)[ydst[c]] = yn[c];
      }
#pragma unroll
      for (int c = 0; c < nw; ++c) {
        if (lane + WARP_SIZE * c < nwin) *(uint2*)(wt + wrow[c] * W_ROW + 8 * wword[c]) = wn[c];
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
      const int off0 = (int)(((int64_t)min(row0 + g, nrows - 1) * row_bytes + (int64_t)b * bsize) & 7);
      const int off1 = (int)(((int64_t)min(row0 + g + 8, nrows - 1) * row_bytes + (int64_t)b * bsize) & 7);
      raw_t r0, r1;
      load(r0, wt + g * W_ROW + off0, t);
      load(r1, wt + (g + 8) * W_ROW + off1, t);
      __syncwarp();
      row_t p0, p1;
      prep(p0, r0, t);
      prep(p1, r1, t);
      float bacc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#define IQ3_MMA_STEP(s)                                                                             \
      {                                                                                             \
        int a0, a1, a2, a3, l0, l1;                                                                 \
        slice<s>(p0, r0, t, table, a0, a2, l0);                                                     \
        slice<s>(p1, r1, t, table, a1, a3, l1);                                                     \
        const int b0 = yb[(s) * Y_BLOCK], b1 = yb[(s) * Y_BLOCK + 1];                               \
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;                                                         \
        asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)                                               \
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));                               \
        const float dq0 = __low2float(*(const half2*)(yd0 + (s) * Y_BLOCK));                       \
        const float dq1 = __low2float(*(const half2*)(yd1 + (s) * Y_BLOCK));                       \
        bacc[0] += dq0 * (float)scaled<type>(c0, l0);                                              \
        bacc[1] += dq1 * (float)scaled<type>(c1, l0);                                              \
        bacc[2] += dq0 * (float)scaled<type>(c2, l1);                                              \
        bacc[3] += dq1 * (float)scaled<type>(c3, l1);                                              \
      }
      IQ3_MMA_STEP(0) IQ3_MMA_STEP(1) IQ3_MMA_STEP(2) IQ3_MMA_STEP(3)
      IQ3_MMA_STEP(4) IQ3_MMA_STEP(5) IQ3_MMA_STEP(6) IQ3_MMA_STEP(7)
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
    for (int w = 1; w < WARPS; ++w) sum += ((const float*)sm.w[w])[e * WARP_SIZE + l];
    if (col < ncols && row < nrows) dst[(int64_t)col * nrows + row] = sum;
    __syncthreads();  // before the next tile's staging overwrites the buffers
  }
}

// Resident CTAs x SMs of kernel with `bytes` of dynamic shared memory, per device; set on the
// kernel's first (eager) call, with its dynamic shared memory limit. ctas: the kernel's own cache.
template <typename Kernel>
static int resident_ctas(int (&ctas)[16], Kernel kernel, size_t bytes) {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  GGML_ASSERT(dev < 16);
  if (ctas[dev] == 0) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
    int occ = 0, sms = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, THREADS, bytes));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    ctas[dev] = std::max(1, occ) * sms;
  }
  return ctas[dev];
}

template <ggml_type type>
void launch(const char* vx, const block_q8_1* y, float* dst, int nrows, int nblocks,
            int64_t row_bytes, int ncols, cudaStream_t stream) {
  constexpr size_t bytes = sizeof(smem);
  static int ctas[16] = {0};
  const int ntiles = (nrows + ROWS - 1) / ROWS;
  iq3_mma<type><<<std::min(ntiles, resident_ctas(ctas, iq3_mma<type>, bytes)), dim3(WARP_SIZE, WARPS), bytes,
                  stream>>>(
      vx, y, dst, nrows, nblocks, row_bytes, ncols);
}

// ---------------------------------------------------------------------------
// The same product on the packed layout (quantization/iq3_pack.py, applied at load by
// GGUFLinearMethod._pack_iq3), 1..32 activation rows. Per 16-row tile and weight block W holds
// one 16-byte aligned record with each lane's bytes in mma fragment order, so a lane fetches
// them with 6 (IQ3_S) / 7 (IQ3_XXS) coalesced loads straight into registers (no staging copy, no 2-byte shared
// loads), and a word's table index is one byte permute (IQ3_S: grid index byte + the 5 bits
// above it) or permute + mask (IQ3_XXS: the pack re-codes each pair's 7 sign bits so that the
// table rebuilds both 4th signs; no parity per word). Tiles, warps, mma, sub-scales and the
// fp32 order are the kernel above's: at 1..8 rows the two are bit-identical, and above 8 each
// output column is computed exactly as in an 8-row call. cloud/results/phase3/r1 has the data.

struct pfrag {
  int4 q0, q1;   // grid-index bytes of rows g, g+8: byte 2s+e (of 16) = word 2t+e of slice s
  uint4 h;       // IQ3_S: H0..H3; IQ3_XXS: B0, B1, B2 (bit 7 of each byte: B3's bytes 2, 3)
  uint32_t h4;   // IQ3_S: H4; IQ3_XXS: B3's bytes 0, 1
  uint2 sc;      // sub-scale nibbles of rows g, g+8
  uint32_t d;    // half2: d of rows g, g+8
};

template <ggml_type type>
static __device__ __forceinline__ void pload(pfrag& f, const char* __restrict__ tb, int lane) {
  constexpr int fb = type == GGML_TYPE_IQ3_S ? 1664 : 1472;  // lane bytes before the sub-scales and d
  f.q0 = *(const int4*)(tb + 16 * lane);
  f.q1 = *(const int4*)(tb + 512 + 16 * lane);
  if (type == GGML_TYPE_IQ3_S) {
    f.h = *(const uint4*)(tb + 1024 + 16 * lane);
    f.h4 = *(const uint32_t*)(tb + 1536 + 4 * lane);
  } else {
    const uint2 b01 = *(const uint2*)(tb + 1024 + 8 * lane);
    f.h = make_uint4(b01.x, b01.y, *(const uint32_t*)(tb + 1280 + 4 * lane), 0u);
    f.h4 = *(const uint16_t*)(tb + 1408 + 2 * lane);
  }
  f.sc = *(const uint2*)(tb + fb + 8 * (lane / 4));
  f.d = *(const uint32_t*)(tb + fb + 64 + 4 * (lane / 4));
}

template <int c>
static __device__ __forceinline__ uint32_t comp(const int4& v) {
  return (uint32_t)(c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w);
}

// Per-block words the table indices are cut from: IQ3_S X0..X7 (byte j of X[4R + c]: the 5
// bits above the grid index of word 2t + (j & 1), slice 2c + (j >> 1), row g + 8R, bits 5-7
// zero); IQ3_XXS B0..B3 (byte s % 4 of B[2R + s / 4]: slice s's re-coded sign byte, bit 7
// don't-care) and B >> 3 (the second word's nibble in bits 0-3).
template <ggml_type type>
struct pwords {
  uint32_t x[8];
};
static __device__ __forceinline__ void pdecode(pwords<GGML_TYPE_IQ3_S>& w, const pfrag& f) {
  w.x[0] = f.h.x & 0x1F1F1F1Fu;
  w.x[1] = f.h.y & 0x1F1F1F1Fu;
  w.x[2] = f.h.z & 0x1F1F1F1Fu;
  w.x[3] = f.h.w & 0x1F1F1F1Fu;
  w.x[4] = f.h4 & 0x1F1F1F1Fu;
  w.x[5] = ((f.h.x >> 5) & 0x07070707u) | ((f.h.y >> 2) & 0x18181818u);
  w.x[6] = ((f.h.z >> 5) & 0x07070707u) | ((f.h.w >> 2) & 0x18181818u);
  w.x[7] = ((f.h4 >> 5) & 0x07070707u) | ((f.h.y >> 4) & 0x08080808u) | ((f.h.w >> 3) & 0x10101010u);
}
static __device__ __forceinline__ uint32_t spare4(uint32_t v) {  // bit 7 of each byte, byte 0 first
  return (((v >> 7) & 0x01010101u) * 0x01020408u) >> 24;
}
static __device__ __forceinline__ void pdecode(pwords<GGML_TYPE_IQ3_XXS>& w, const pfrag& f) {
  const uint32_t sp = spare4(f.h.x) | spare4(f.h.y) << 4 | spare4(f.h.z) << 8 | ((f.h4 >> 7) & 1u) << 12 |
                      ((f.h4 >> 15) & 1u) << 13;
  w.x[0] = f.h.x;
  w.x[1] = f.h.y;
  w.x[2] = f.h.z;
  w.x[3] = f.h4 | (sp & 0x7Fu) << 16 | (sp >> 7) << 24;
#pragma unroll
  for (int i = 0; i < 4; ++i) w.x[4 + i] = w.x[i] >> 3;
}

// Slice s's A fragment: a0/a2 = row g words 2t/2t+1, a1/a3 = row g+8.
template <int s>
static __device__ __forceinline__ void pslice(const pwords<GGML_TYPE_IQ3_S>& w, const pfrag& f,
                                              const uint32_t* table, int& a0, int& a1, int& a2, int& a3) {
  constexpr int c = s / 2;
  constexpr uint32_t j0 = 2 * (s % 2), j1 = j0 + 1;
  // bytes: grid index, X byte j, then 0 (X's bit 7, replicated)
  constexpr uint32_t s0 = j0 | (4 + j0) << 4 | (0xC + j0) << 8 | (0xC + j0) << 12;
  constexpr uint32_t s1 = j1 | (4 + j1) << 4 | (0xC + j1) << 8 | (0xC + j1) << 12;
  const uint32_t qa = comp<c>(f.q0), qb = comp<c>(f.q1);
  a0 = table[prmt(qa, w.x[c], s0)];
  a2 = table[prmt(qa, w.x[c], s1)];
  a1 = table[prmt(qb, w.x[4 + c], s0)];
  a3 = table[prmt(qb, w.x[4 + c], s1)];
}
template <int s>
static __device__ __forceinline__ void pslice(const pwords<GGML_TYPE_IQ3_XXS>& w, const pfrag& f,
                                              const uint32_t* table, int& a0, int& a1, int& a2, int& a3) {
  constexpr int c = s / 2, k = s % 4, r = s / 4;
  constexpr uint32_t j0 = 2 * (s % 2), j1 = j0 + 1;
  constexpr uint32_t s0 = j0 | (4 + k) << 4, s1 = j1 | (4 + k) << 4;  // bytes 2, 3 masked off
  const uint32_t qa = comp<c>(f.q0), qb = comp<c>(f.q1);
  a0 = table[prmt(qa, w.x[r], s0) & 0x0FFFu];
  a2 = table[4096 + (prmt(qa, w.x[4 + r], s1) & 0x0FFFu)];
  a1 = table[prmt(qb, w.x[2 + r], s0) & 0x0FFFu];
  a3 = table[4096 + (prmt(qb, w.x[6 + r], s1) & 0x0FFFu)];
}

// NG groups of 8 activation columns (1..8 * NG columns): per slice the A fragment is cut once
// and used by NG mmas. Each output column is computed as at NG = 1, in the same order.
template <ggml_type type, int NG>
static __global__ void __launch_bounds__(THREADS)
iq3_mma_packed(const char* __restrict__ vx, const block_q8_1* __restrict__ vy, float* __restrict__ dst,
               const int nrows, const int nblocks, const int ncols) {
  using tr = traits<type>;
  constexpr int ycols = 8 * NG;
  constexpr int nstage = (ycols * 18 + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int64_t tbytes = ROWS * sizeof(typename tr::block);
  // IQ3_S: T[q | sign nibble << 8 | qh << 12]; IQ3_XXS: T[q | n << 8] for a pair's first word
  // (n = its 3 signs + the nibble's parity) and T[4096 + (q | n << 8)] for the second (n = the
  // first's parity + its 3 signs); the 4th sign is the parity of the other 3 and n's parity bit.
  __shared__ uint32_t table[8192];
  extern __shared__ __align__(16) int dyn_p[];  // [WARPS][ycols * Y_COL] staging, [WARPS][NG * 128] sums

  const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * WARP_SIZE + lane;
  for (int i = tid; i < 8192; i += THREADS) {
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
  int* yt = dyn_p + warp * ycols * Y_COL;
  float* red = (float*)(dyn_p + WARPS * ycols * Y_COL);
  for (int i = ncols * Y_COL + lane; i < ycols * Y_COL; i += WARP_SIZE) yt[i] = 0;  // columns >= ncols
  __syncthreads();

  const int nby = nblocks * SLICES;
  const int g = lane / 4, t = lane % 4;
  const int ntiles = nrows / ROWS;
  int ysrc[nstage], ydst[nstage];
#pragma unroll
  for (int c = 0; c < nstage; ++c) {
    const int i = min(lane + WARP_SIZE * c, ncols * 18 - 1), j = i / 18, e = i - 18 * j;
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
        if (lane + WARP_SIZE * c < ncols * 18) reinterpret_cast<int4*>(yt)[ydst[c]] = yn[c];
      }
      const pfrag f = fn;  // this block's weight words; fn takes the next block's
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
#define IQ3_MMA_PSTEP(s)                                                                            \
      {                                                                                             \
        int a0, a1, a2, a3;                                                                         \
        pslice<s>(w, f, table, a0, a1, a2, a3);                                                     \
        const int l0 = 1 + 2 * ((f.sc.x >> (4 * (s))) & 0xF), l1 = 1 + 2 * ((f.sc.y >> (4 * (s))) & 0xF); \
        _Pragma("unroll")                                                                           \
        for (int j = 0; j < NG; ++j) {                                                              \
          const int b0 = yb[8 * j * Y_COL + (s) * Y_BLOCK], b1 = yb[8 * j * Y_COL + (s) * Y_BLOCK + 1]; \
          int c0 = 0, c1 = 0, c2 = 0, c3 = 0;                                                       \
          asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" \
              : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)                                             \
              : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));                             \
          const float dq0 = __low2float(*(const half2*)(yd0 + 8 * j * Y_COL + (s) * Y_BLOCK));     \
          const float dq1 = __low2float(*(const half2*)(yd1 + 8 * j * Y_COL + (s) * Y_BLOCK));     \
          bacc[j][0] += dq0 * (float)scaled<type>(c0, l0);                                         \
          bacc[j][1] += dq1 * (float)scaled<type>(c1, l0);                                         \
          bacc[j][2] += dq0 * (float)scaled<type>(c2, l1);                                         \
          bacc[j][3] += dq1 * (float)scaled<type>(c3, l1);                                         \
        }                                                                                           \
      }
      IQ3_MMA_PSTEP(0) IQ3_MMA_PSTEP(1) IQ3_MMA_PSTEP(2) IQ3_MMA_PSTEP(3)
      IQ3_MMA_PSTEP(4) IQ3_MMA_PSTEP(5) IQ3_MMA_PSTEP(6) IQ3_MMA_PSTEP(7)
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
      for (int e = 0; e < 4; ++e) red[(warp * NG + j) * 128 + e * WARP_SIZE + lane] = acc[j][e];
    }
    __syncthreads();
    for (int i = tid; i < NG * 128; i += THREADS) {
      const int j = i / 128, e = (i / WARP_SIZE) % 4, l = i % WARP_SIZE;
      const int row = tile * ROWS + l / 4 + 8 * (e / 2), col = 8 * j + 2 * (l % 4) + e % 2;
      float sum = red[j * 128 + e * WARP_SIZE + l];
#pragma unroll
      for (int w = 1; w < WARPS; ++w) sum += red[(w * NG + j) * 128 + e * WARP_SIZE + l];
      if (col < ncols) dst[(int64_t)col * nrows + row] = sum;
    }
    __syncthreads();
  }
}

// 44 KB of shared memory at NG = 1 (2 CTAs per SM), 54 / 77 KB at NG = 2 / 4 (1 CTA per SM).
template <ggml_type type, int NG>
void launch_packed(const char* vx, const block_q8_1* y, float* dst, int nrows, int nblocks, int ncols,
                   cudaStream_t stream) {
  constexpr size_t bytes = (size_t)WARPS * (8 * NG * Y_COL + NG * 128) * 4;
  static int ctas[16] = {0};
  const int ntiles = nrows / ROWS;
  iq3_mma_packed<type, NG><<<std::min(ntiles, resident_ctas(ctas, iq3_mma_packed<type, NG>, bytes)),
                             dim3(WARP_SIZE, WARPS), bytes, stream>>>(
      vx, y, dst, nrows, nblocks, ncols);
}

// Packed -> GGUF bytes (the inverse of quantization/iq3_pack.py's pack), for the paths that
// read the GGUF layout (vendored MMQ). One warp per tile-block: it decodes its lane words as
// the product kernel does, assembles the 16 rows' blocks in shared memory and writes them out.
template <ggml_type type>
static __global__ void __launch_bounds__(THREADS)
iq3_unpack(const char* __restrict__ src, char* __restrict__ dst, const int ntb, const int nblocks) {
  using block = typename traits<type>::block;
  constexpr int bs = sizeof(block);
  constexpr int64_t tbytes = ROWS * bs;
  __shared__ __align__(4) uint8_t out[WARPS][ROWS * bs];
  const int lane = threadIdx.x, warp = threadIdx.y;
  const int g = lane / 4, t = lane % 4;
  uint8_t* o = out[warp];
  for (int tb = blockIdx.x * WARPS + warp; tb < ntb; tb += gridDim.x * WARPS) {
    pfrag f;
    pload<type>(f, src + tb * tbytes, lane);
    pwords<type> w;
    pdecode(w, f);
#pragma unroll
    for (int R = 0; R < 2; ++R) {
      uint8_t* ob = o + (g + 8 * R) * bs;
      const int4 q = R ? f.q1 : f.q0;
      const uint32_t qw[4] = {(uint32_t)q.x, (uint32_t)q.y, (uint32_t)q.z, (uint32_t)q.w};
      if (t == 0) {
        *(uint16_t*)ob = (uint16_t)((R ? f.d >> 16 : f.d) & 0xFFFF);
      }
#pragma unroll
      for (int s = 0; s < SLICES; ++s) {
        const int c = s / 2, j0 = 2 * (s % 2);
        ob[2 + 8 * s + 2 * t] = (uint8_t)(qw[c] >> (8 * j0));
        ob[2 + 8 * s + 2 * t + 1] = (uint8_t)(qw[c] >> (8 * (j0 + 1)));
        uint32_t v;  // this lane's bits of the slice's qh byte (IQ3_S) / sign word (IQ3_XXS)
        if (type == GGML_TYPE_IQ3_S) {
          const uint32_t h0 = (w.x[4 * R + c] >> (8 * j0)) & 0x1F, h1 = (w.x[4 * R + c] >> (8 * (j0 + 1))) & 0x1F;
          ob[offsetof(block_iq3_s, signs) + 4 * s + t] = (uint8_t)((h0 & 0xF) | (h1 & 0xF) << 4);
          v = (h0 >> 4 | (h1 >> 4) << 1) << (2 * t);
        } else {
          const uint32_t b = (w.x[2 * R + s / 4] >> (8 * (s % 4))) & 0x7F;  // re-coded 7 bits
          const uint32_t e3 = ((b >> 3) ^ __popc(b & 7)) & 1;
          v = ((b & 7) | e3 << 3 | (b & 0x70)) << (7 * t);
        }
        v |= __shfl_xor_sync(0xFFFFFFFFu, v, 1);
        v |= __shfl_xor_sync(0xFFFFFFFFu, v, 2);
        const uint32_t sc = ((R ? f.sc.y : f.sc.x) >> (4 * s)) & 0xF;
        if (t == 0) {
          if (type == GGML_TYPE_IQ3_S) {
            ob[offsetof(block_iq3_s, qh) + s] = (uint8_t)v;
          } else {
            const uint32_t aux = v | sc << 28;
#pragma unroll
            for (int i = 0; i < 4; ++i) ob[2 + QK_K / 4 + 4 * s + i] = (uint8_t)(aux >> (8 * i));
          }
        }
        if (type == GGML_TYPE_IQ3_S && t == 1 && s % 2 == 0) {
          const uint32_t sc2 = ((R ? f.sc.y : f.sc.x) >> (4 * s)) & 0xFF;
          ob[offsetof(block_iq3_s, scales) + s / 2] = (uint8_t)sc2;
        }
      }
    }
    __syncwarp();
    const int tile = tb / nblocks, b = tb - tile * nblocks;
    char* d0 = dst + (int64_t)tile * tbytes * nblocks + (int64_t)b * bs;  // row r at + r * nblocks * bs
    for (int i = lane; i < ROWS * bs / 2; i += WARP_SIZE) {
      const int r = i / (bs / 2), e = i - r * (bs / 2);
      *(uint16_t*)(d0 + (int64_t)r * nblocks * bs + 2 * e) = *(const uint16_t*)(o + r * bs + 2 * e);
    }
    __syncwarp();
  }
}

}  // namespace

// W [nrows, row_bytes] IQ3_S/IQ3_XXS blocks, contiguous rows, 16-byte aligned; vy: ncols
// block_q8_1 rows of k values (16-byte aligned, k % 512 == 0); dst [ncols, nrows] fp32.
void iq3_mma_mul_mat_vec_cuda(ggml_type type, const char* vx, const void* vy, float* dst,
                              int nrows, int k, int64_t row_bytes, int ncols, cudaStream_t stream) {
  const block_q8_1* y = (const block_q8_1*)vy;
  GGML_ASSERT(ncols >= 1 && ncols <= 8);
  (type == GGML_TYPE_IQ3_S ? launch<GGML_TYPE_IQ3_S> : launch<GGML_TYPE_IQ3_XXS>)(
      vx, y, dst, nrows, k / QK_K, row_bytes, ncols, stream);
}

// The same on W packed by quantization/iq3_pack.py (nrows % 16 == 0), 1..32 activation rows.
template <ggml_type type>
static void launch_packed_n(const char* vx, const block_q8_1* y, float* dst, int nrows, int nblocks, int ncols,
                            cudaStream_t stream) {
  (ncols <= 8 ? launch_packed<type, 1> : ncols <= 16 ? launch_packed<type, 2> : launch_packed<type, 4>)(
      vx, y, dst, nrows, nblocks, ncols, stream);
}

void iq3_mma_packed_mul_mat_vec_cuda(ggml_type type, const char* vx, const void* vy, float* dst,
                                     int nrows, int k, int ncols, cudaStream_t stream) {
  const block_q8_1* y = (const block_q8_1*)vy;
  GGML_ASSERT(ncols >= 1 && ncols <= 32 && nrows % ROWS == 0);
  (type == GGML_TYPE_IQ3_S ? launch_packed_n<GGML_TYPE_IQ3_S> : launch_packed_n<GGML_TYPE_IQ3_XXS>)(
      vx, y, dst, nrows, k / QK_K, ncols, stream);
}

// dst [nrows, nblocks * block bytes] GGUF blocks from src packed by quantization/iq3_pack.py.
void iq3_unpack_cuda(ggml_type type, const char* src, char* dst, int nrows, int nblocks, cudaStream_t stream) {
  GGML_ASSERT(nrows % ROWS == 0);
  const int ntb = nrows / ROWS * nblocks;
  const int grid = std::min((ntb + WARPS - 1) / WARPS, 65535);
  (type == GGML_TYPE_IQ3_S ? iq3_unpack<GGML_TYPE_IQ3_S> : iq3_unpack<GGML_TYPE_IQ3_XXS>)
      <<<grid, dim3(WARP_SIZE, WARPS), 0, stream>>>(src, dst, ntb, nblocks);
}
