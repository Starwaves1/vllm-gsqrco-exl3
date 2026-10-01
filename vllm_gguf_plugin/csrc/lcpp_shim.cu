// SPDX-License-Identifier: Apache-2.0
// lcpp shim: run llama.cpp b11211 ggml-cuda MMVQ / MMQ (vendored unmodified
// under csrc/lcpp, see csrc/lcpp/VENDORED.md) on torch tensors.
//
// This file provides the few pieces of ggml-base / ggml-cuda.cu the vendored
// kernels link against (device info, context, pool, type sizes, error hooks),
// an owned q8_1 quantizer and IQ3 kernel, and these torch ops, all
// (W, X, type, row) -> [n, row] in X's dtype:
//   lcpp_mul_mat_vec_q               MMVQ, 1..8 activation rows
//   lcpp_mul_mat_q                   MMQ (int8 tensor cores), any rows
//   lcpp_mul_mat_vec_iq3             owned IQ3_S / IQ3_XXS dp4a kernel,
//                                    1..8 rows
//   lcpp_mul_mat_vec_iq3_mma         the same on int8 tensor cores
//                                    (lcpp_owned_iq3_mma.cu)
// W: uint8 [>= row, row_bytes] GGUF blocks, contiguous rows. X: [n, K] fp32 /
// fp16 / bf16, unit inner stride.
// All checks run before any launch. Nothing here touches CUDA until an op is
// called on CUDA tensors.

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include "common.cuh"
#include "mmq.cuh"
#include "mmvq.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

// ---------------------------------------------------------------------------
// ggml-base pieces used by the vendored files. Errors throw instead of abort()
// so a bad call surfaces as a Python exception, not a dead server.

void ggml_abort(const char* file, int line, const char* fmt, ...) {
  char msg[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, sizeof(msg), fmt, args);
  va_end(args);
  throw std::runtime_error(std::string("lcpp: ") + file + ":" +
                           std::to_string(line) + ": " + msg);
}

void ggml_log_internal(enum ggml_log_level level, const char* fmt, ...) {
  if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_INFO) {
    return;
  }
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
}

void ggml_cuda_error(const char* stmt, const char* func, const char* file,
                     int line, const char* msg) {
  throw std::runtime_error(std::string("lcpp CUDA error: ") + msg + " in " +
                           func + " at " + file + ":" + std::to_string(line) +
                           ": " + stmt);
}

// Block size / bytes per block for the types this shim serves (plus the float
// types used for src1/dst). Anything else aborts.
static void type_traits(ggml_type t, int64_t* blck, size_t* size) {
  switch (t) {
    case GGML_TYPE_F32:
      *blck = 1;
      *size = sizeof(float);
      return;
    case GGML_TYPE_F16:
      *blck = 1;
      *size = sizeof(ggml_fp16_t);
      return;
    case GGML_TYPE_BF16:
      *blck = 1;
      *size = sizeof(ggml_bf16_t);
      return;
    case GGML_TYPE_Q2_K:
      *blck = QK_K;
      *size = sizeof(block_q2_K);
      return;
    case GGML_TYPE_Q4_K:
      *blck = QK_K;
      *size = sizeof(block_q4_K);
      return;
    case GGML_TYPE_Q6_K:
      *blck = QK_K;
      *size = sizeof(block_q6_K);
      return;
    case GGML_TYPE_IQ2_XXS:
      *blck = QK_K;
      *size = sizeof(block_iq2_xxs);
      return;
    case GGML_TYPE_IQ2_XS:
      *blck = QK_K;
      *size = sizeof(block_iq2_xs);
      return;
    case GGML_TYPE_IQ2_S:
      *blck = QK_K;
      *size = sizeof(block_iq2_s);
      return;
    case GGML_TYPE_IQ3_XXS:
      *blck = QK_K;
      *size = sizeof(block_iq3_xxs);
      return;
    case GGML_TYPE_IQ3_S:
      *blck = QK_K;
      *size = sizeof(block_iq3_s);
      return;
    case GGML_TYPE_IQ4_XS:
      *blck = QK_K;
      *size = sizeof(block_iq4_xs);
      return;
    case GGML_TYPE_IQ1_M:
      *blck = QK_K;
      *size = sizeof(block_iq1_m);
      return;
    default:
      GGML_ABORT("lcpp shim: unsupported ggml type %d", (int)t);
  }
}

int64_t ggml_blck_size(enum ggml_type type) {
  int64_t b;
  size_t s;
  type_traits(type, &b, &s);
  return b;
}

size_t ggml_type_size(enum ggml_type type) {
  int64_t b;
  size_t s;
  type_traits(type, &b, &s);
  return s;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
  return ggml_type_size(type) * ne / ggml_blck_size(type);
}

bool ggml_is_quantized(enum ggml_type type) {
  return type != GGML_TYPE_F32 && type != GGML_TYPE_F16 &&
         type != GGML_TYPE_BF16;
}

const char* ggml_type_name(enum ggml_type type) {
  static thread_local char buf[16];
  snprintf(buf, sizeof(buf), "type%d", (int)type);
  return buf;
}

int64_t ggml_nelements(const struct ggml_tensor* t) {
  return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}

size_t ggml_element_size(const struct ggml_tensor* t) {
  return ggml_type_size(t->type);
}

size_t ggml_nbytes(const struct ggml_tensor* t) {
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (t->ne[i] <= 0) return 0;
  }
  const size_t blck = ggml_blck_size(t->type);
  size_t n = blck == 1 ? ggml_type_size(t->type) : t->ne[0] * t->nb[0] / blck;
  for (int i = blck == 1 ? 0 : 1; i < GGML_MAX_DIMS; ++i) {
    n += (t->ne[i] - 1) * t->nb[i];
  }
  return n;
}

bool ggml_is_contiguous(const struct ggml_tensor* t) {
  size_t next = ggml_type_size(t->type);
  if (t->ne[0] != ggml_blck_size(t->type) && t->nb[0] != next) return false;
  next *= t->ne[0] / ggml_blck_size(t->type);
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    if (t->ne[i] != 1) {
      if (t->nb[i] != next) return false;
      next *= t->ne[i];
    }
  }
  return true;
}

bool ggml_is_contiguously_allocated(const struct ggml_tensor* t) {
  return ggml_nbytes(t) ==
         ggml_nelements(t) * ggml_type_size(t->type) / ggml_blck_size(t->type);
}

bool ggml_are_same_stride(const struct ggml_tensor* a,
                          const struct ggml_tensor* b) {
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (a->nb[i] != b->nb[i]) return false;
  }
  return true;
}

// Weights live in torch storage, never in a ggml compute buffer, so the
// vendored "clear padding of compute buffers" branches never run.
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(
    ggml_backend_buffer_t /*buffer*/) {
  return GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t /*buffer*/,
                                          const struct ggml_tensor* t) {
  return ggml_nbytes(t);
}

// ---------------------------------------------------------------------------
// ggml-cuda.cu pieces: device info (filled on first call, i.e. first op
// launch, never at import), device get/set, context.

const ggml_cuda_device_info& ggml_cuda_info() {
  static const ggml_cuda_device_info info = [] {
    ggml_cuda_device_info i{};
    CUDA_CHECK(cudaGetDeviceCount(&i.physical_device_count));
    i.device_count = std::min(i.physical_device_count, GGML_CUDA_MAX_DEVICES);
    for (int id = 0; id < i.device_count; ++id) {
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
      auto& d = i.devices[id];
      d.cc = 100 * prop.major + 10 * prop.minor;
      d.nsm = prop.multiProcessorCount;
      d.smpb = prop.sharedMemPerBlock;
      d.smpbo = prop.sharedMemPerBlockOptin;
      d.integrated = false;  // as upstream
      d.vmm = false;         // no ggml VMM pool here
      d.total_vram = prop.totalGlobalMem;
      d.warp_size = prop.warpSize;
      d.supports_cooperative_launch = prop.cooperativeLaunch != 0;
      d.physical_device = id;
      d.physical_share_count = 1;
      d.virtual_index = 0;
    }
    return i;
  }();
  return info;
}

void ggml_cuda_set_device(int device) {
  int current;
  CUDA_CHECK(cudaGetDevice(&current));
  if (device != current) {
    CUDA_CHECK(cudaSetDevice(device));
  }
}

int ggml_cuda_get_device() {
  int id;
  CUDA_CHECK(cudaGetDevice(&id));
  return id;
}

// The context borrows torch's stream and pool; it owns neither CUDA stream
// nor cuBLAS handle, so there is nothing to destroy.
ggml_backend_cuda_context::~ggml_backend_cuda_context() = default;

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(
    int /*device*/, int /*stream_no*/) {
  GGML_ABORT("lcpp shim: pool must be installed by the caller");
}

// Every allocation is an uninitialised torch tensor (caching allocator, so
// scratch is reused across calls; stream ordered; CUDA-graph safe). Nothing is
// zero-filled: the quantizers write every byte of their q8 buffer (zeros past
// ne00), and stream-k tmp_fixup is written before it is read. Upstream's ggml
// pool doesn't zero either. The one exception, the MMQ read tail, is zeroed by
// a memset (mul_mat_q below).
struct TorchPool final : public ggml_cuda_pool {
  explicit TorchPool(const Tensor& like) : like_(like) {}
  void* alloc(size_t size, size_t* actual_size) override {
    const int64_t n = (int64_t)((std::max<size_t>(size, 1) + 3) / 4);
    owners_.push_back(torch::stable::new_empty(like_, {n}, ScalarType::Int));
    *actual_size = (size_t)n * 4;
    return owners_.back().data_ptr();
  }
  void free(void* /*ptr*/, size_t /*size*/) override {}  // freed with the pool

 private:
  const Tensor& like_;
  std::vector<Tensor> owners_;
};

// ---------------------------------------------------------------------------
// q8_1 activation quantizers for fp32 / fp16 / bf16 X. The same arithmetic and
// output layout as the vendored quantize_q8_1 (MMVQ) and
// quantize_mmq_q8_1 (MMQ; 2-D, no ids) in quantize.cu, which take fp32 only;
// reading 16-bit X directly saves a cast kernel per call. Converting fp16/bf16
// to float is exact, so the q8 blocks are bit-identical to the vendored ones on
// X.float() (tests/test_lcpp_kernels.py::test_lcpp_quantize_vs_vendored).
// Scalar loads: X needs no alignment beyond its element size.

template <typename T>
static __global__ void quantize_q8_1_x(const T* __restrict__ x,
                                       block_q8_1* __restrict__ y,
                                       const int64_t ne00, const int64_t s01,
                                       const int64_t ne0) {
  const int64_t i0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
  if (i0 >= ne0) return;
  const int64_t i_cont = blockIdx.y * ne0 + i0;
  const int64_t ib = i_cont / QK8_1;
  const int64_t iqs = i_cont % QK8_1;

  const float xi = i0 < ne00 ? float(x[blockIdx.y * s01 + i0]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;
  amax = warp_reduce_max<QK8_1>(amax);
  sum = warp_reduce_sum<QK8_1>(sum);

  const float d = amax / 127.0f;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
  y[ib].qs[iqs] = q;
  if (iqs > 0) return;
  y[ib].ds = make_half2(d, sum);
}

template <typename T, mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1_x(const T* __restrict__ x,
                                           block_q8_1_mmq* __restrict__ y,
                                           const int64_t ne00,
                                           const int64_t s01, const int64_t ne0,
                                           const int ne1) {
  constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
  constexpr int vals_per_sum = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;
  if (i0 >= ne0) return;

  const T* xr = x + blockIdx.x * s01 + i0;
  const float4 xi = i0 < ne00 ? make_float4(float(xr[0]), float(xr[1]),
                                            float(xr[2]), float(xr[3]))
                              : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float amax = fabsf(xi.x);
  amax = fmaxf(amax, fabsf(xi.y));
  amax = fmaxf(amax, fabsf(xi.z));
  amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
  for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
  }
  float sum;
  if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
    sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
    }
  }

  const float d_inv = 127.0f / amax;
  char4 q;
  q.x = roundf(xi.x * d_inv);
  q.y = roundf(xi.y * d_inv);
  q.z = roundf(xi.z * d_inv);
  q.w = roundf(xi.w * d_inv);
  const float d = 1.0f / d_inv;

  const int64_t ib = (i0 / QK8_1_MMQ) * ne1 + blockIdx.x;
  const int64_t iqs = i0 % QK8_1_MMQ;
  ((char4*)y[ib].qs)[iqs / 4] = q;
  if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
    if (iqs % 16 == 0 && iqs < 96) {
      y[ib].d2s6[2 + iqs / 16] = sum;
      if (iqs % 64 == 0) y[ib].d2s6[iqs / 64] = d;
    }
  } else if (iqs % 32 == 0) {
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
      y[ib].ds4[iqs / 32] = make_half2(d, sum);
    } else {
      y[ib].d4[iqs / 32] = d;
    }
  }
}

// X [n, k] (row stride s01 elements) -> q8 at vy: block_q8_1 (mmq false) or
// block_q8_1_mmq in type's ds layout, k_padded values per row. Launch geometry
// as upstream's quantize_row_q8_1_cuda / quantize_mmq_q8_1_cuda.
template <typename T>
static void quantize_x_t(const T* x, void* vy, ggml_type type, bool mmq,
                         int64_t n, int64_t k, int64_t s01, int64_t k_padded,
                         cudaStream_t stream) {
  if (!mmq) {
    const dim3 grid(
        (k_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE, n,
        1);
    quantize_q8_1_x<T><<<grid, CUDA_QUANTIZE_BLOCK_SIZE, 0, stream>>>(
        x, (block_q8_1*)vy, k, s01, k_padded);
    return;
  }
  const dim3 grid(n,
                  (k_padded + 4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) /
                      (4 * CUDA_QUANTIZE_BLOCK_SIZE_MMQ),
                  1);
  block_q8_1_mmq* y = (block_q8_1_mmq*)vy;
  switch (mmq_get_q8_1_ds_layout(type)) {
    case MMQ_Q8_1_DS_LAYOUT_D4:
      quantize_mmq_q8_1_x<T, MMQ_Q8_1_DS_LAYOUT_D4>
          <<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(x, y, k, s01,
                                                              k_padded, n);
      break;
    case MMQ_Q8_1_DS_LAYOUT_DS4:
      quantize_mmq_q8_1_x<T, MMQ_Q8_1_DS_LAYOUT_DS4>
          <<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(x, y, k, s01,
                                                              k_padded, n);
      break;
    case MMQ_Q8_1_DS_LAYOUT_D2S6:
      quantize_mmq_q8_1_x<T, MMQ_Q8_1_DS_LAYOUT_D2S6>
          <<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(x, y, k, s01,
                                                              k_padded, n);
      break;
  }
}

static void quantize_x(const Tensor& X, void* vy, ggml_type type, bool mmq,
                       int64_t k_padded, cudaStream_t stream) {
  const int64_t n = X.size(0), k = X.size(1), s01 = n == 1 ? k : X.stride(0);
  switch (X.scalar_type()) {
    case ScalarType::Float:
      quantize_x_t((const float*)X.data_ptr(), vy, type, mmq, n, k, s01,
                   k_padded, stream);
      break;
    case ScalarType::Half:
      quantize_x_t((const half*)X.data_ptr(), vy, type, mmq, n, k, s01,
                   k_padded, stream);
      break;
    default:  // BFloat16 (check_inputs admits nothing else)
      quantize_x_t((const nv_bfloat16*)X.data_ptr(), vy, type, mmq, n, k, s01,
                   k_padded, stream);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// IQ3_S / IQ3_XXS dp4a product for 1..8 activation rows (MTP decode: 4 rows per
// sequence, 8 at c=2); linear.py routes it at 1..5 rows and 6..8 to the mma
// kernel (lcpp_owned_iq3_mma.cu). Owned code.
//
// The vendored MMVQ does not re-decode weights per activation row: nvcc merges
// the per-row decodes (its sm_86 loop has 27 / 70 / 106 global loads at 1 / 4 /
// 8 rows, +9 per row = one q8_1 block). Per activation row it pays those q8_1
// loads (a lane reads its 36-byte block as 8 separate words), shared by only
// 2 weight rows per warp, on top of a heavy sign step.
// Here a CTA of 4 warps owns 16 weight rows, 4 per warp, and lane l takes the
// 32-value slices l, l+32, ... of each. The grid sits in shared memory, and
// each 32-block chunk of every activation row is staged there with 16-byte
// loads (read back at a 36-byte lane stride = 9 words: no bank conflicts). A
// lane decodes its slice of each of its 4 rows once into 8 int8x4 words, then
// reads each activation slice once and dots it with all 4.
// From ninfer-all's ggml_bridge_vec.cuh (Apache-2.0; read, not copied): the
// slice-per-lane layout, the shared-memory grid and the sign step (iq3_negate).
// The staging and the 4-row reuse are ours. 4 warps x 4 rows was the fastest of
// 2..16 warps x 2..8 rows measured.
//
// Numerics: each slice's term d_w * d_q8 * sumi (integer sub-scales included)
// is vendored vec_dot_iq3_*_q8_1's (vecdotq.cuh) bit for bit; only the fp32
// order in which a row's slice terms are summed differs from MMVQ (lane-
// strided + one warp reduction here, 4 warps + shared memory there).

constexpr int IQ3_WARPS = 4;          // warps per CTA
constexpr int IQ3_ROWS_PER_WARP = 4;  // weight rows per warp
constexpr int IQ3_CHUNK =
    WARP_SIZE;  // q8_1 blocks staged per chunk: one slice per lane

template <ggml_type type>
struct iq3_traits;
template <>
struct iq3_traits<GGML_TYPE_IQ3_S> {
  using block = block_iq3_s;
  static constexpr int grid_size = 512;
  static __device__ const uint32_t* grid() { return iq3s_grid; }
};
template <>
struct iq3_traits<GGML_TYPE_IQ3_XXS> {
  using block = block_iq3_xxs;
  static constexpr int grid_size = 256;
  static __device__ const uint32_t* grid() { return iq3xxs_grid; }
};

// Negates the bytes of g whose bit is set in the low nibble of bits. The IQ3
// grids have no zero byte, so the per-byte (g ^ 0xFF) + 1 never carries: the
// same int8 values as the vendored __vcmpne4 / __vsub4 sign step, in 4
// instructions instead of ~10 (ninfer-all's negate_bytes, ggml_bridge_vec.cuh).
static __device__ __forceinline__ int iq3_negate(uint32_t g, uint32_t bits) {
  const uint32_t ones = ((bits & 0xF) * 0x00204081u) & 0x01010101u;
  return (int)((g ^ (ones * 0xFFu)) + ones);
}

// Slice u (values 32u..32u+31) of block b: 8 signed int8x4 words in value
// order, the block scale d and the slice's integer sub-scale ls, with the
// grid read from shared memory. Same values as vec_dot_iq3_s_q8_1 /
// vec_dot_iq3_xxs_q8_1 (vecdotq.cuh) with iqs = 2u.
static __device__ __forceinline__ void iq3_decode(const block_iq3_s* b, int u,
                                                  const uint32_t* grid,
                                                  int (&w)[8], int& ls,
                                                  float& d) {
  const int2 qs_packed =
      make_int2(get_int_b2(b->qs, 2 * u), get_int_b2(b->qs, 2 * u + 1));
  const uint8_t* qs = (const uint8_t*)&qs_packed;
  const int qh = b->qh[u];
  const uint32_t signs = get_int_b2(b->signs, u);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    w[e] =
        iq3_negate(grid[qs[e] | ((qh << (8 - e)) & 0x100)], signs >> (4 * e));
  }
  ls = (b->scales[u / 2] >> (4 * (u & 1))) & 0x0F;
  d = __half2float(b->d);
}

static __device__ __forceinline__ void iq3_decode(const block_iq3_xxs* b, int u,
                                                  const uint32_t* grid,
                                                  int (&w)[8], int& ls,
                                                  float& d) {
  const int2 q3_packed =
      make_int2(get_int_b2(b->qs, 2 * u), get_int_b2(b->qs, 2 * u + 1));
  const uint8_t* q3 = (const uint8_t*)&q3_packed;
  const uint32_t aux32 = get_int_b2(b->qs, QK_K / 16 + u);
#pragma unroll
  for (int l0 = 0; l0 < 8; l0 += 2) {
    const uint32_t signs =
        unpack_ksigns(aux32 >> (7 * l0 / 2));  // 8 sign bits, 1 per value
    w[l0 + 0] = iq3_negate(grid[q3[l0 + 0]], signs);
    w[l0 + 1] = iq3_negate(grid[q3[l0 + 1]], signs >> 4);
  }
  ls = aux32 >> 28;
  d = __half2float(b->d);
}

// The slice's integer sum with its sub-scale applied, as the vendored vec_dot.
template <ggml_type type>
static __device__ __forceinline__ int iq3_scale(int sumi, int ls) {
  return type == GGML_TYPE_IQ3_S ? sumi * (1 + 2 * ls)
                                 : (ls * sumi + sumi / 2) / 2;
}

// Output in X's dtype, rounded to nearest even as torch's cast from fp32.
static __device__ __forceinline__ void iq3_store(float* p, float v) { *p = v; }
static __device__ __forceinline__ void iq3_store(half* p, float v) {
  *p = __float2half_rn(v);
}
static __device__ __forceinline__ void iq3_store(nv_bfloat16* p, float v) {
  *p = __float2bfloat16_rn(v);
}

template <ggml_type type, int ncols, typename dst_t>
static __global__ void __launch_bounds__(IQ3_WARPS* WARP_SIZE)
    iq3_mul_mat_vec(const char* __restrict__ vx,
                    const block_q8_1* __restrict__ vy, dst_t* __restrict__ dst,
                    const int nrows, const int ncols_x,
                    const int64_t row_bytes) {
  using traits = iq3_traits<type>;
  constexpr int slices =
      QK_K / QK8_1;  // 32-value slices (q8_1 blocks) per weight block
  __shared__ uint32_t grid[traits::grid_size];
  __shared__ __align__(16)
      block_q8_1 ys[ncols][IQ3_CHUNK];  // int4 stores below
  static_assert(sizeof(ys[0]) % sizeof(int4) == 0, "16-byte staging");

  const int lane = threadIdx.x, tid = threadIdx.y * WARP_SIZE + threadIdx.x;
  for (int i = tid; i < traits::grid_size; i += IQ3_WARPS * WARP_SIZE) {
    grid[i] = traits::grid()[i];  // visible after the first __syncthreads below
  }
  const int row0 = (blockIdx.x * IQ3_WARPS + threadIdx.y) * IQ3_ROWS_PER_WARP;
  const int nby = ncols_x / QK8_1;  // q8_1 blocks per activation row
  const typename traits::block* wr[IQ3_ROWS_PER_WARP];
#pragma unroll
  for (int r = 0; r < IQ3_ROWS_PER_WARP;
       ++r) {  // tail rows: read row nrows-1, write nothing
    wr[r] = (const typename traits::block*)(vx + min(row0 + r, nrows - 1) *
                                                     row_bytes);
  }
  float acc[ncols][IQ3_ROWS_PER_WARP] = {{0.0f}};

  for (int by0 = 0; by0 < nby; by0 += IQ3_CHUNK) {
    // K % 512 == 0 (check_inputs), so a chunk is 32 or 16 blocks: whole int4s.
    const int nb = min(IQ3_CHUNK, nby - by0);
    const int n16 = nb * (int)sizeof(block_q8_1) / (int)sizeof(int4);
    for (int i = tid; i < ncols * n16; i += IQ3_WARPS * WARP_SIZE) {
      const int j = i / n16, e = i % n16;
      reinterpret_cast<int4*>(ys[j])[e] =
          reinterpret_cast<const int4*>(vy + (int64_t)j * nby + by0)[e];
    }
    __syncthreads();
    if (lane < nb) {
      const int kbx = (by0 + lane) / slices, u = lane % slices;
      int w[IQ3_ROWS_PER_WARP][8], ls[IQ3_ROWS_PER_WARP];
      float d[IQ3_ROWS_PER_WARP];
#pragma unroll
      for (int r = 0; r < IQ3_ROWS_PER_WARP; ++r) {
        iq3_decode(wr[r] + kbx, u, grid, w[r], ls[r], d[r]);
      }
      // each activation slice is read from shared memory once, for all rows
#pragma unroll
      for (int j = 0; j < ncols; ++j) {
        int y[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          y[i] = get_int_b4(ys[j][lane].qs, i);
        }
        const float d8 = __low2float(ys[j][lane].ds);
#pragma unroll
        for (int r = 0; r < IQ3_ROWS_PER_WARP; ++r) {
          int sumi = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            sumi = ggml_cuda_dp4a(w[r][i], y[i], sumi);
          }
          acc[j][r] += (d[r] * d8) * iq3_scale<type>(sumi, ls[r]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < ncols; ++j) {
#pragma unroll
    for (int r = 0; r < IQ3_ROWS_PER_WARP; ++r) {
      acc[j][r] = warp_reduce_sum<WARP_SIZE>(acc[j][r]);
      if (lane == 0 && row0 + r < nrows) {
        iq3_store(dst + (int64_t)j * nrows + row0 + r, acc[j][r]);
      }
    }
  }
}

// W [nrows, row_bytes] IQ3_S/IQ3_XXS blocks, y: ncols block_q8_1 rows of k
// values (16-byte aligned; k % 512 == 0), dst [ncols, nrows] fp32, fp16 or bf16
// (written directly: no separate output cast kernel).
template <ggml_type type, typename dst_t>
static void iq3_mul_mat_vec_cuda(const char* vx, const void* vy, dst_t* dst,
                                 int nrows, int k, int64_t row_bytes, int ncols,
                                 cudaStream_t stream) {
  const dim3 grid((nrows + IQ3_WARPS * IQ3_ROWS_PER_WARP - 1) /
                  (IQ3_WARPS * IQ3_ROWS_PER_WARP));
  const dim3 block(WARP_SIZE, IQ3_WARPS);
  const block_q8_1* y = (const block_q8_1*)vy;
  switch (ncols) {
    case 1:
      iq3_mul_mat_vec<type, 1, dst_t>
          <<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes);
      break;
    case 2:
      iq3_mul_mat_vec<type, 2, dst_t>
          <<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes);
      break;
    case 3:
      iq3_mul_mat_vec<type, 3, dst_t>
          <<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes);
      break;
    case 4:
      iq3_mul_mat_vec<type, 4, dst_t>
          <<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes);
      break;
    case 5:
      iq3_mul_mat_vec<type, 5, dst_t>
          <<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes);
      break;
    case 6:
      iq3_mul_mat_vec<type, 6, dst_t>
          <<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes);
      break;
    case 7:
      iq3_mul_mat_vec<type, 7, dst_t>
          <<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes);
      break;
    case 8:
      iq3_mul_mat_vec<type, 8, dst_t>
          <<<grid, block, 0, stream>>>(vx, y, dst, nrows, k, row_bytes);
      break;
    default:
      GGML_ABORT("iq3_mul_mat_vec: %d activation rows", ncols);
  }
}

template <ggml_type type>
static void iq3_mul_mat_vec_y(const char* vx, const void* vy, const Tensor& y,
                              int nrows, int k, int64_t row_bytes, int ncols,
                              cudaStream_t stream) {
  switch (y.scalar_type()) {
    case ScalarType::Float:
      iq3_mul_mat_vec_cuda<type>(vx, vy, (float*)y.data_ptr(), nrows, k,
                                 row_bytes, ncols, stream);
      break;
    case ScalarType::Half:
      iq3_mul_mat_vec_cuda<type>(vx, vy, (half*)y.data_ptr(), nrows, k,
                                 row_bytes, ncols, stream);
      break;
    default:  // BFloat16
      iq3_mul_mat_vec_cuda<type>(vx, vy, (nv_bfloat16*)y.data_ptr(), nrows, k,
                                 row_bytes, ncols, stream);
      break;
  }
}

// ---------------------------------------------------------------------------
// MMQ host side. Mirrors the non-MoE, q8_1 branch of ggml_cuda_mul_mat_q in
// llama.cpp b11211 ggml-cuda/mmq.cu (not vendored: its type switch needs all
// 22 MMQ instances and mmid.cu). 2-D only: one channel, one sample.

static void mul_mat_q(ggml_backend_cuda_context& ctx, const ggml_tensor* src0,
                      const Tensor& X, ggml_tensor* dst, cudaStream_t stream) {
  const int64_t ne00 = src0->ne[0], ne01 = src0->ne[1];
  const int64_t ne10 = X.size(1), ne11 = X.size(0);
  const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
  const bool fallback = ne01 % 128 != 0;
  const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

  // MMQ reads full J-column tiles of q8 past the last quantized column.
  // Upstream sizes that tail as J_max blocks, but J_max is 0 below 8 columns,
  // and garbage there gave IMAs / NaNs (Maxwell-Lyu/vllm-gguf-plugin
  // f1d38ffdd0). So add 128 block_q8_1_mmq (18 KiB; 128 is the largest J
  // mul_mat_q_switch_J selects) and zero everything past the quantized data:
  // one small memset, not the whole buffer.
  const size_t nbytes_quant =
      ne11 * ne10_padded * sizeof(block_q8_1_mmq) / QK8_1_MMQ;
  const size_t nbytes_tail =
      (ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) + 128) *
      sizeof(block_q8_1_mmq);
  ggml_cuda_pool_alloc<char> q8(ctx.pool(), nbytes_quant + nbytes_tail);
  CUDA_CHECK(cudaMemsetAsync(q8.get() + nbytes_quant, 0, nbytes_tail, stream));
  quantize_x(X, q8.get(), src0->type, true, ne10_padded, stream);

  const int64_t s01 = src0->nb[1] / ggml_type_size(src0->type);
  const int64_t s02 = s01 * ne01;
  const int64_t s1 = dst->nb[1] / sizeof(float);
  const int64_t s2 = s1 * ne11;
  const int64_t s12 =
      ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
  // clang-format off
  const mmq_args args = {
      (const char*)src0->data, src0->type, (const int*)q8.get(), nullptr, nullptr,
      (float*)dst->data, nullptr,
      ne00, ne01, ne11, s01, ne11, s1,
      1, 1, s02, s12, s2,
      1, 1, s02, s12, s2,
      ne11, ne11};
  // clang-format on

  switch (src0->type) {
    case GGML_TYPE_Q2_K:
      mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
      break;
    case GGML_TYPE_Q4_K:
      mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
      break;
    case GGML_TYPE_Q6_K:
      mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
      break;
    case GGML_TYPE_IQ2_XXS:
      mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
      break;
    case GGML_TYPE_IQ2_XS:
      mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
      break;
    case GGML_TYPE_IQ2_S:
      mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
      break;
    case GGML_TYPE_IQ3_XXS:
      mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
      break;
    case GGML_TYPE_IQ3_S:
      mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
      break;
    case GGML_TYPE_IQ4_XS:
      mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
      break;
    default:
      GGML_ABORT("lcpp shim: no MMQ instance for type %d", (int)src0->type);
  }
}

// ---------------------------------------------------------------------------
// Ops

static bool lcpp_type_supported(int64_t type) {
  switch (type) {
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_IQ2_XXS:
    case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ3_XXS:
    case GGML_TYPE_IQ3_S:
    case GGML_TYPE_IQ4_XS:
      return true;
    default:
      return false;
  }
}

// Shape / stride / alignment guards; max_rows: activation row limit (0: none).
// Returns K (logical columns).
static int64_t check_inputs(const Tensor& W, const Tensor& X, int64_t type,
                            int64_t row, int64_t max_rows, const char* op) {
  STD_TORCH_CHECK(lcpp_type_supported(type), op, ": unsupported ggml type ",
                  type);
  STD_TORCH_CHECK(W.dim() == 2 && X.dim() == 2, op, ": W and X must be 2-D");
  STD_TORCH_CHECK(W.scalar_type() == ScalarType::Byte, op, ": W must be uint8");
  STD_TORCH_CHECK(X.scalar_type() == ScalarType::Float ||
                      X.scalar_type() == ScalarType::Half ||
                      X.scalar_type() == ScalarType::BFloat16,
                  op, ": X must be fp32, fp16 or bf16");

  const int64_t blck = ggml_blck_size((ggml_type)type);
  const int64_t ts = (int64_t)ggml_type_size((ggml_type)type);
  const int64_t row_bytes = W.size(1);
  STD_TORCH_CHECK(row_bytes > 0 && row_bytes % ts == 0, op, ": W row bytes ",
                  row_bytes, " not a multiple of the block size ", ts);
  const int64_t k = row_bytes / ts * blck;
  STD_TORCH_CHECK(k % MATRIX_ROW_PADDING == 0, op, ": K=", k,
                  " must be a multiple of ", MATRIX_ROW_PADDING,
                  " (no weight tail padding)");
  STD_TORCH_CHECK(k <= INT_MAX && W.size(0) <= INT_MAX, op, ": W too large");
  STD_TORCH_CHECK(row > 0 && row <= W.size(0), op, ": row=", row,
                  " out of range (W has ", W.size(0), " rows)");
  STD_TORCH_CHECK(W.stride(1) == 1, op, ": W inner stride must be 1");
  STD_TORCH_CHECK(W.size(0) == 1 || W.stride(0) == row_bytes, op,
                  ": W rows must be contiguous (row stride ", W.stride(0),
                  ", row bytes ", row_bytes, ")");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0, op,
                  ": W data must be 16-byte aligned");

  STD_TORCH_CHECK(X.size(1) == k, op, ": X has ", X.size(1),
                  " columns, W rows hold K=", k);
  STD_TORCH_CHECK(X.stride(1) == 1 && (X.size(0) <= 1 || X.stride(0) >= k), op,
                  ": X inner stride must be 1");
  if (max_rows > 0) {
    STD_TORCH_CHECK(X.size(0) <= max_rows, op, ": at most ", max_rows,
                    " rows, got ", X.size(0));
  }
  STD_TORCH_CHECK(X.size(0) <= INT_MAX, op, ": X too large");
  // Device checks last, so the CPU registration below exercises every guard
  // above without a GPU and then rejects.
  STD_TORCH_CHECK(W.is_cuda() && X.is_cuda(), op,
                  ": W and X must be CUDA tensors");
  STD_TORCH_CHECK(W.get_device_index() == X.get_device_index(), op,
                  ": W and X must be on the same device");
  return k;
}

static cudaStream_t torch_stream(int32_t device) {
  void* s = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &s));
  // torch's default stream is the null handle; ggml treats null as "create a
  // private stream", so hand it the legacy default stream explicitly.
  return s != nullptr ? static_cast<cudaStream_t>(s) : cudaStreamLegacy;
}

// lcpp_owned_iq3_mma.cu
void iq3_mma_mul_mat_vec_cuda(ggml_type type, const char* vx, const void* vy,
                              float* dst, int nrows, int k, int64_t row_bytes,
                              int ncols, cudaStream_t stream);

enum class Kernel { mmvq, mmq, iq3, iq3_mma };

static Tensor run(Tensor W, Tensor X, int64_t type, int64_t row,
                  Kernel kernel) {
  // mmvq: the kernel reads MMVQ-layout q8_1 (quantize_x) and writes
  // fp32 dst: MMVQ and the 1..8-row owned kernels
  const bool mmvq = kernel != Kernel::mmq;
  const bool iq3 = kernel == Kernel::iq3 || kernel == Kernel::iq3_mma;
  const char* op = kernel == Kernel::mmvq  ? "lcpp_mul_mat_vec_q"
                   : kernel == Kernel::mmq ? "lcpp_mul_mat_q"
                   : kernel == Kernel::iq3 ? "lcpp_mul_mat_vec_iq3"
                                           : "lcpp_mul_mat_vec_iq3_mma";
  STD_TORCH_CHECK(!iq3 || type == GGML_TYPE_IQ3_S || type == GGML_TYPE_IQ3_XXS,
                  op, ": IQ3_S or IQ3_XXS only, got type ", type);
  const int64_t max_rows = kernel == Kernel::mmq ? 0 : MMVQ_MAX_BATCH_SIZE;
  const int64_t k = check_inputs(W, X, type, row, max_rows, op);
  const int64_t n = X.size(0);
  const ScalarType out_dtype = X.scalar_type();
  const int64_t k_padded = GGML_PAD(k, MATRIX_ROW_PADDING);
  const int64_t q8_bytes = n * k_padded * (int64_t)sizeof(block_q8_1) / QK8_1;
  if (n == 0) {
    return torch::stable::new_empty(X, {0, row}, out_dtype);
  }

  const int32_t device = X.get_device_index();
  const DeviceGuard guard(device);
  const cudaStream_t stream = torch_stream(device);

  // X is quantized to q8_1 by the owned quantize_x (any float dtype, no cast).
  // The dp4a IQ3 kernel writes X's dtype; MMVQ / MMQ, the IQ3 mma kernels write
  // fp32 dst only, so for them 16-bit X costs one output cast. Launches per
  // call with 16-bit X:
  //   IQ3 dp4a: quantize, iq3_mul_mat_vec = 2
  //   MMVQ, iq3_mma: quantize, the kernel, cast Y = 3
  //   MMQ: tail memset, quantize, mul_mat_q, [stream-k fixup], cast Y = 4 or 5
  const ScalarType y_dtype =
      kernel == Kernel::iq3 ? out_dtype : ScalarType::Float;
  Tensor y = torch::stable::new_empty(X, {n, row}, y_dtype);

  ggml_tensor src0{};
  src0.type = (ggml_type)type;
  src0.ne[0] = k;
  src0.ne[1] = row;
  src0.ne[2] = 1;
  src0.ne[3] = 1;
  src0.nb[0] = ggml_type_size(src0.type);
  src0.nb[1] = W.size(1);
  src0.nb[2] = src0.nb[1] * row;
  src0.nb[3] = src0.nb[2];
  src0.data = W.data_ptr();

  ggml_tensor src1{};  // only ne[0] is read (ggml_cuda_op_mul_mat_vec_q)
  src1.type = GGML_TYPE_F32;
  src1.ne[0] = k;
  src1.ne[1] = n;
  src1.ne[2] = 1;
  src1.ne[3] = 1;

  ggml_tensor dst{};  // fp32 dst of all but the dp4a IQ3 kernel (that one
                      // writes y in X's dtype)
  dst.type = GGML_TYPE_F32;
  dst.ne[0] = row;
  dst.ne[1] = n;
  dst.ne[2] = 1;
  dst.ne[3] = 1;
  dst.nb[0] = sizeof(float);
  dst.nb[1] = row * sizeof(float);
  dst.nb[2] = dst.nb[1] * n;
  dst.nb[3] = dst.nb[2];
  dst.data = y.data_ptr();

  {
    ggml_backend_cuda_context ctx(device);
    ctx.streams[device][0] = stream;
    ctx.pools[device][0] = std::make_unique<TorchPool>(X);
    if (mmvq) {
      // upstream's ggml_cuda_mul_mat_vec_q minus its fp32 quantize: the q8_1
      // entry point of the legacy op path, one channel, rows row_low..row_high
      // = 0..row.
      ggml_cuda_pool_alloc<char> q8(ctx.pool());
      quantize_x(X, q8.alloc(q8_bytes), src0.type, false, k_padded, stream);
      const char* vy = q8.get();
      if (kernel == Kernel::iq3) {
        (type == GGML_TYPE_IQ3_S
             ? iq3_mul_mat_vec_y<GGML_TYPE_IQ3_S>
             : iq3_mul_mat_vec_y<GGML_TYPE_IQ3_XXS>)((const char*)src0.data, vy,
                                                     y, (int)row, (int)k,
                                                     src0.nb[1], (int)n,
                                                     stream);
      } else if (kernel == Kernel::iq3_mma) {
        iq3_mma_mul_mat_vec_cuda(src0.type, (const char*)src0.data, vy,
                                 (float*)dst.data, (int)row, (int)k, src0.nb[1],
                                 (int)n, stream);
      } else {
        ggml_cuda_op_mul_mat_vec_q(
            ctx, &src0, &src1, &dst, (const char*)src0.data, nullptr, vy,
            (float*)dst.data, 0, row, n, k_padded, stream);
      }
    } else {
      mul_mat_q(ctx, &src0, X, &dst, stream);
    }
    CUDA_CHECK(cudaGetLastError());
  }  // pool tensors released here, stream-ordered by the caching allocator

  return y_dtype == out_dtype
             ? y
             : torch::stable::to(y, std::optional<ScalarType>(out_dtype));
}

Tensor lcpp_mul_mat_vec_q(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run(W, X, type, row, Kernel::mmvq);
}

Tensor lcpp_mul_mat_q(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run(W, X, type, row, Kernel::mmq);
}

Tensor lcpp_mul_mat_vec_iq3(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run(W, X, type, row, Kernel::iq3);
}

Tensor lcpp_mul_mat_vec_iq3_mma(Tensor W, Tensor X, int64_t type, int64_t row) {
  return run(W, X, type, row, Kernel::iq3_mma);
}

// The q8_1 bytes quantize_x makes for MMVQ (mmq false) or MMQ (mmq true, in
// type's ds layout), or those of the vendored fp32 quantizers (vendored true;
// X must then be fp32 with a row stride that is a multiple of 4). For tests.
// Every byte is written: no zero fill.
Tensor lcpp_quantize_q8_1(Tensor X, int64_t type, bool mmq, bool vendored) {
  STD_TORCH_CHECK(lcpp_type_supported(type) && X.is_cuda() && X.dim() == 2 &&
                      X.stride(1) == 1 && X.size(1) % MATRIX_ROW_PADDING == 0 &&
                      (X.scalar_type() == ScalarType::Float ||
                       X.scalar_type() == ScalarType::Half ||
                       X.scalar_type() == ScalarType::BFloat16),
                  "lcpp_quantize_q8_1: bad arguments");
  const int64_t n = X.size(0), k = X.size(1), s01 = n == 1 ? k : X.stride(0);
  const int64_t bytes =
      mmq ? n * k / QK8_1_MMQ * (int64_t)sizeof(block_q8_1_mmq)
          : n * k / QK8_1 * (int64_t)sizeof(block_q8_1);
  Tensor q = torch::stable::new_empty(X, {bytes}, ScalarType::Byte);
  const DeviceGuard guard(X.get_device_index());
  const cudaStream_t stream = torch_stream(X.get_device_index());
  if (vendored) {
    STD_TORCH_CHECK(X.scalar_type() == ScalarType::Float && s01 % 4 == 0,
                    "vendored: fp32 X");
    (mmq ? quantize_mmq_q8_1_cuda : quantize_row_q8_1_cuda)(
        (const float*)X.data_ptr(), nullptr, q.data_ptr(), (ggml_type)type, k,
        s01, s01 * n, s01 * n, k, n, 1, 1, stream);
    CUDA_CHECK(cudaGetLastError());
  } else {
    quantize_x(X, q.data_ptr(), (ggml_type)type, mmq, k, stream);
  }
  return q;
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C_gguf, ops) {
  ops.def(
      "lcpp_mul_mat_vec_q(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def("lcpp_mul_mat_q(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "lcpp_mul_mat_vec_iq3(Tensor W, Tensor X, int type, SymInt row) -> "
      "Tensor");
  ops.def(
      "lcpp_mul_mat_vec_iq3_mma(Tensor W, Tensor X, int type, SymInt row) -> "
      "Tensor");
  ops.def(
      "lcpp_quantize_q8_1(Tensor X, int type, bool mmq, bool vendored) -> "
      "Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CUDA, ops) {
  ops.impl("lcpp_mul_mat_vec_q", TORCH_BOX(&lcpp_mul_mat_vec_q));
  ops.impl("lcpp_mul_mat_q", TORCH_BOX(&lcpp_mul_mat_q));
  ops.impl("lcpp_mul_mat_vec_iq3", TORCH_BOX(&lcpp_mul_mat_vec_iq3));
  ops.impl("lcpp_mul_mat_vec_iq3_mma", TORCH_BOX(&lcpp_mul_mat_vec_iq3_mma));
  ops.impl("lcpp_quantize_q8_1", TORCH_BOX(&lcpp_quantize_q8_1));
}

// CPU: the checks only (a call that passes them all ends in "must be CUDA
// tensors"), so they can be tested without a GPU.
STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CPU, ops) {
  ops.impl("lcpp_mul_mat_vec_q", TORCH_BOX(&lcpp_mul_mat_vec_q));
  ops.impl("lcpp_mul_mat_q", TORCH_BOX(&lcpp_mul_mat_q));
  ops.impl("lcpp_mul_mat_vec_iq3", TORCH_BOX(&lcpp_mul_mat_vec_iq3));
  ops.impl("lcpp_mul_mat_vec_iq3_mma", TORCH_BOX(&lcpp_mul_mat_vec_iq3_mma));
}
