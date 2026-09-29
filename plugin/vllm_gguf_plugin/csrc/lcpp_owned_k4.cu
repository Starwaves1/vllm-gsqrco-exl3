// SPDX-License-Identifier: Apache-2.0
// Owned Q4_K / IQ2_S product for 1..8 activation rows (MTP decode: 4 rows per
// sequence, 8 at c=2), op lcpp_mul_mat_vec_own in lcpp_shim.cu.
// cloud/results/opt/k1 has the data and the variants tried.
//
// Same CTA structure as the shim's IQ3 kernel (iq3_mul_mat_vec, phase 3 item 5):
// 4 warps x 4 weight rows per warp, lane l takes the 32-value slices l, l+32,
// ... of its rows, chunks of each activation row are staged in shared memory,
// and a lane decodes its slice of each of its rows once into 8 int8x4 words,
// then reads each activation slice once and dots it with all of them. The
// vendored MMVQ reuses each q8_1 load over only 1-2 weight rows per warp.
// Per type:
//  - Q4_K: the min term needs the sum of each activation slice's quants; it is
//    computed once per CTA at staging, not per warp.
//  - IQ2_S: grid in shared memory; at <= 4 rows a chunk is 2 slices per lane
//    (half the barriers), which measured faster there and slower at 8 rows.
// The CTA owns 16 rows, so it underfills the GPU below a few thousand rows:
// linear.py routes it only above 2048. Measured and dropped (see the results):
// an IQ4_XS version (no faster than MMVQ / MMQ in situ), an extra
// barrier-separated staging pass, weight loads hoisted above the staging
// barrier or prefetched a chunk ahead (more registers, slower), 8 warps x 2
// rows, 2 x 4, 4 x 2, __launch_bounds__ min-blocks 5 / 6.
//
// Numerics: IQ2_S computes each slice's integer sum and the term d_w * d_q8 *
// (sub-scaled sum) exactly as vendored vec_dot_iq2_s_q8_1 (vecdotq.cuh).
// Q4_K sums a slice's 32 products in one integer (the vendored vec_dot splits
// them over 4 threads), then folds the integer sub-scale into the fp32 block
// scale; its min term uses the integer sum of the q8_1 quants, as MMVQ does.
// Same model as MMVQ, different fp32 rounding and order.

#include "common.cuh"
#include "vecdotq.cuh"

constexpr int OWN_WARPS = 4;          // warps per CTA
constexpr int OWN_ROWS_PER_WARP = 4;  // weight rows per warp

// One weight row's decoded 32-value slice: 8 int8x4 words and its scales:
// Q4_K d = dm.x * sub-scale, b = dm.y * min; IQ2_S d = block scale, s / s2 =
// sub-scales of values 0..15 / 16..31.
struct own_slice {
  int w[8];
  int s, s2;
  float d, b;
};

// Negates the bytes of g whose bit is set in the low nibble of bits. Needs no
// zero byte in g (the IQ2_S grid holds 0x08 / 0x19 / 0x2b only): then the
// per-byte (g ^ 0xFF) + 1 never carries. Same int8 values as the vendored
// __vcmpne4 / __vsub4 sign step (the shim's iq3_negate).
static __device__ __forceinline__ int own_negate(uint32_t g, uint32_t bits) {
  const uint32_t ones = ((bits & 0xF) * 0x00204081u) & 0x01010101u;
  return (int)((g ^ (ones * 0xFFu)) + ones);
}

// Q4_K slice u of b = sub-block u: 6-bit scale and min (get_scale_min_k4).
static __device__ __forceinline__ void own_decode(const block_q4_K* b, int u, const uint2*, own_slice& o) {
  const int4 h = *(const int4*)b;  // dm, scales[0..11] (blocks are 16-byte aligned)
  const int4 qa = *(const int4*)(b->qs + 32 * (u / 2));
  const int4 qb = *(const int4*)(b->qs + 32 * (u / 2) + 16);
  const int q[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
  const int shift = 4 * (u & 1);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    o.w[i] = (q[i] >> shift) & 0x0F0F0F0F;
  }
  const int bs = 8 * (u & 3);
  const uint32_t s03 = (uint32_t)h.y >> bs, s47 = (uint32_t)h.z >> bs, s811 = (uint32_t)h.w >> bs;
  int sc, m;
  if (u < 4) {
    sc = s03 & 63;
    m = s47 & 63;
  } else {
    sc = (s811 & 0x0F) | ((s03 >> 2) & 0x30);
    m = ((s811 >> 4) & 0x0F) | ((s47 >> 2) & 0x30);
  }
  const float2 dm = __half22float2(__halves2half2(__ushort_as_half((unsigned short)(h.x & 0xFFFF)),
                                                  __ushort_as_half((unsigned short)((uint32_t)h.x >> 16))));
  o.d = dm.x * sc;
  o.b = dm.y * m;
}

// IQ2_S slice u of b: vec_dot_iq2_s_q8_1 with iqs = 2u, grid read from shared
// memory.
static __device__ __forceinline__ void own_decode(const block_iq2_s* b, int u, const uint2* grid, own_slice& o) {
  const uint32_t qs = get_int_b2(b->qs, u);
  const uint32_t signs = get_int_b2(b->qs, QK_K / 32 + u);
  const int qh = b->qh[u];
#pragma unroll
  for (int l = 0; l < 4; ++l) {
    const uint2 g = grid[((qs >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300)];
    o.w[2 * l + 0] = own_negate(g.x, signs >> (8 * l));
    o.w[2 * l + 1] = own_negate(g.y, signs >> (8 * l + 4));
  }
  o.s = b->scales[u] & 0x0F;
  o.s2 = b->scales[u] >> 4;
  o.d = __half2float(b->d);
}

// Adds one slice's term to acc given the integer dots of the slice's words
// 0..3 (sumi0) and 4..7 (sumi1) with the activation slice, its q8_1 scale d8
// and, Q4_K only, fs = d8 * (sum of its quants).
template <ggml_type type>
static __device__ __forceinline__ void own_term(float& acc, int sumi0, int sumi1, const own_slice& o,
                                                float d8, float fs) {
  if constexpr (type == GGML_TYPE_Q4_K) {
    acc += (o.d * d8) * (float)(sumi0 + sumi1) - fs * o.b;
  } else {  // IQ2_S
    acc += (o.d * d8) * ((sumi0 * o.s + sumi1 * o.s2 + (sumi0 + sumi1) / 2) / 4);
  }
}

template <ggml_type type> struct own_traits;
template <> struct own_traits<GGML_TYPE_Q4_K> { using block = block_q4_K; };
template <> struct own_traits<GGML_TYPE_IQ2_S> { using block = block_iq2_s; };

template <ggml_type type, int ncols>
static __global__ void __launch_bounds__(OWN_WARPS * WARP_SIZE)
own_mul_mat_vec(const char* __restrict__ vx, const block_q8_1* __restrict__ vy,
                float* __restrict__ dst, const int nrows, const int ncols_x,
                const int64_t row_bytes) {
  using block = typename own_traits<type>::block;
  constexpr int R = OWN_ROWS_PER_WARP, NT = OWN_WARPS * WARP_SIZE;
  constexpr int SPL = type == GGML_TYPE_IQ2_S && ncols <= 4 ? 2 : 1;  // slices per lane per chunk
  constexpr int CH = SPL * WARP_SIZE;   // q8_1 blocks staged per chunk
  constexpr int slices = QK_K / QK8_1;  // 32-value slices (q8_1 blocks) per weight block
  constexpr bool grid_type = type == GGML_TYPE_IQ2_S;
  __shared__ uint2 grid[grid_type ? 1024 : 1];
  __shared__ __align__(16) block_q8_1 ys[ncols][CH];  // int4 stores (IQ2_S staging)
  static_assert(sizeof(ys[0]) % sizeof(int4) == 0, "16-byte staging");
  __shared__ float ysum[type == GGML_TYPE_Q4_K ? ncols : 1][CH];  // Q4_K: d8 * sum of quants

  const int lane = threadIdx.x, tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  if constexpr (grid_type) {
    for (int i = tid; i < 1024; i += NT) {
      grid[i] = make_uint2((uint32_t)iq2s_grid[i], (uint32_t)(iq2s_grid[i] >> 32));
    }  // visible after the first __syncthreads below
  }
  const int row0 = (blockIdx.x * OWN_WARPS + threadIdx.y) * R;
  const int nby = ncols_x / QK8_1;  // q8_1 blocks per activation row
  const block* wr[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {  // tail rows: read row nrows-1, write nothing
    wr[r] = (const block*)(vx + (int64_t)min(row0 + r, nrows - 1) * row_bytes);
  }
  float acc[ncols][R] = {{0.0f}};

  for (int by0 = 0; by0 < nby; by0 += CH) {
    const int nb = min(CH, nby - by0);
    if constexpr (type == GGML_TYPE_IQ2_S) {  // plain copy; K % 512 == 0 (check_inputs): whole int4s
      constexpr int n16c = CH * (int)sizeof(block_q8_1) / (int)sizeof(int4);
      const int n16 = nb * (int)sizeof(block_q8_1) / (int)sizeof(int4);
      for (int i = tid; i < ncols * n16c; i += NT) {
        const int j = i / n16c, e = i % n16c;
        if (e < n16) {
          reinterpret_cast<int4*>(ys[j])[e] = reinterpret_cast<const int4*>(vy + (int64_t)j * nby + by0)[e];
        }
      }
    } else {  // Q4_K: one q8_1 block per thread, also storing d8 * (sum of its quants)
      for (int i = tid; i < ncols * CH; i += NT) {
        const int j = i / CH, b = i % CH;
        if (b >= nb) {
          continue;
        }
        const block_q8_1* src = vy + (int64_t)j * nby + by0 + b;
        int q[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          q[e] = get_int_b4(src->qs, e);
        }
        const half2 ds = src->ds;
        int sum = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          sum = ggml_cuda_dp4a(0x01010101, q[e], sum);
        }
        ysum[j][b] = __low2float(ds) * sum;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          ((int*)ys[j][b].qs)[e] = q[e];
        }
        ys[j][b].ds = ds;
      }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < SPL; ++h) {
      const int sl = h * WARP_SIZE + lane;  // staged slice
      if (sl < nb) {
        const int kbx = (by0 + sl) / slices, u = sl % slices;
        own_slice o[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
          own_decode(wr[r] + kbx, u, grid, o[r]);
        }
        // each activation slice is read from shared memory once, for all rows
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
          int y[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            y[i] = get_int_b4(ys[j][sl].qs, i);
          }
          const float d8 = __low2float(ys[j][sl].ds);
          float fs = 0.0f;
          if constexpr (type == GGML_TYPE_Q4_K) {
            fs = ysum[j][sl];
          }
#pragma unroll
          for (int r = 0; r < R; ++r) {
            int sumi0 = 0, sumi1 = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              sumi0 = ggml_cuda_dp4a(o[r].w[i], y[i], sumi0);
              sumi1 = ggml_cuda_dp4a(o[r].w[i + 4], y[i + 4], sumi1);
            }
            own_term<type>(acc[j][r], sumi0, sumi1, o[r], d8, fs);
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < ncols; ++j) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[j][r] = warp_reduce_sum<WARP_SIZE>(acc[j][r]);
      if (lane == 0 && row0 + r < nrows) {
        dst[(int64_t)j * nrows + row0 + r] = acc[j][r];
      }
    }
  }
}

template <ggml_type type>
static void own_launch(const char* vx, const block_q8_1* y, float* dst, int nrows, int k,
                       int64_t row_bytes, int ncols, cudaStream_t stream) {
  const dim3 grid((nrows + OWN_WARPS * OWN_ROWS_PER_WARP - 1) / (OWN_WARPS * OWN_ROWS_PER_WARP));
  const dim3 block(WARP_SIZE, OWN_WARPS);
  switch (ncols) {
    case 1: own_mul_mat_vec<type, 1><<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes); break;
    case 2: own_mul_mat_vec<type, 2><<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes); break;
    case 3: own_mul_mat_vec<type, 3><<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes); break;
    case 4: own_mul_mat_vec<type, 4><<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes); break;
    case 5: own_mul_mat_vec<type, 5><<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes); break;
    case 6: own_mul_mat_vec<type, 6><<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes); break;
    case 7: own_mul_mat_vec<type, 7><<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes); break;
    case 8: own_mul_mat_vec<type, 8><<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes); break;
    default: GGML_ABORT("own_mul_mat_vec: %d activation rows", ncols);
  }
}

bool own_mul_mat_vec_supported(int type) {
  return type == GGML_TYPE_Q4_K || type == GGML_TYPE_IQ2_S;
}

// W [nrows, row_bytes] blocks of type (own_mul_mat_vec_supported), 16-byte
// aligned with contiguous rows; y: ncols block_q8_1 rows of k values (k % 512
// == 0, check_inputs); dst [ncols, nrows] fp32.
void own_mul_mat_vec_cuda(int type, const char* vx, const void* vy, float* dst, int nrows, int k,
                          int64_t row_bytes, int ncols, cudaStream_t stream) {
  const block_q8_1* y = (const block_q8_1*)vy;
  switch (type) {
    case GGML_TYPE_Q4_K: own_launch<GGML_TYPE_Q4_K>(vx, y, dst, nrows, k, row_bytes, ncols, stream); break;
    case GGML_TYPE_IQ2_S: own_launch<GGML_TYPE_IQ2_S>(vx, y, dst, nrows, k, row_bytes, ncols, stream); break;
    default: GGML_ABORT("own_mul_mat_vec: type %d", type);
  }
}
