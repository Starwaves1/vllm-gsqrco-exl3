// SPDX-License-Identifier: Apache-2.0
// lcpp shim: run llama.cpp b11211 ggml-cuda MMVQ / MMQ (vendored unmodified
// under csrc/lcpp, see csrc/lcpp/VENDORED.md) on torch tensors.
//
// This file provides the few pieces of ggml-base / ggml-cuda.cu the vendored
// kernels link against (device info, context, pool, type sizes, error hooks),
// and these torch ops, all
// (W, X, type, row) -> [n, row] in X's dtype:
//   lcpp_mul_mat_vec_q               MMVQ, 1..8 activation rows
//   lcpp_mul_mat_q                   MMQ (int8 tensor cores), any rows
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
// q8_1 activation quantization with the vendored quantizers (quantize.cu).
// They take fp32 X only, so 16-bit X is cast first (one more kernel), and they
// load fp32 X as float4: 16-byte aligned, row stride a multiple of 4.
static void quantize_x(const Tensor& X, void* vy, ggml_type type, bool mmq,
                       int64_t k_padded, cudaStream_t stream) {
  const Tensor xf =
      X.scalar_type() == ScalarType::Float
          ? X
          : torch::stable::to(X, std::optional<ScalarType>(ScalarType::Float));
  const int64_t n = xf.size(0), k = xf.size(1);
  const int64_t s01 = n == 1 ? k : xf.stride(0);
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(xf.data_ptr()) % 16 == 0 && s01 % 4 == 0,
      "lcpp: fp32 X must be 16-byte aligned with a row stride that is a "
      "multiple of 4");
  (mmq ? quantize_mmq_q8_1_cuda : quantize_row_q8_1_cuda)(
      (const float*)xf.data_ptr(), nullptr, vy, type, k, s01, s01 * n, s01 * n,
      k_padded, n, 1, 1, stream);
  CUDA_CHECK(cudaGetLastError());
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

enum class Kernel { mmvq, mmq };

static Tensor run(Tensor W, Tensor X, int64_t type, int64_t row,
                  Kernel kernel) {
  const bool mmvq = kernel == Kernel::mmvq;
  const char* op =
      kernel == Kernel::mmvq ? "lcpp_mul_mat_vec_q" : "lcpp_mul_mat_q";
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

  // X is quantized to q8_1 by quantize_x (the vendored fp32 quantizers, after a
  // cast for 16-bit X), and MMVQ / MMQ write fp32 dst only, so 16-bit X also
  // costs an output cast. Launches per call with 16-bit X:
  //   MMVQ: cast X, quantize, mul_mat_vec_q, cast Y                    = 4
  //   MMQ:  cast X, tail memset, quantize, mul_mat_q, [stream-k fixup],
  //         cast Y                                                     = 5 or 6
  const ScalarType y_dtype = ScalarType::Float;
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

  ggml_tensor dst{};  // fp32
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
      ggml_cuda_op_mul_mat_vec_q(ctx, &src0, &src1, &dst,
                                 (const char*)src0.data, nullptr, vy,
                                 (float*)dst.data, 0, row, n, k_padded, stream);
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

STABLE_TORCH_LIBRARY_FRAGMENT(_C_gguf, ops) {
  ops.def(
      "lcpp_mul_mat_vec_q(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def("lcpp_mul_mat_q(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CUDA, ops) {
  ops.impl("lcpp_mul_mat_vec_q", TORCH_BOX(&lcpp_mul_mat_vec_q));
  ops.impl("lcpp_mul_mat_q", TORCH_BOX(&lcpp_mul_mat_q));
}

// CPU: the checks only (a call that passes them all ends in "must be CUDA
// tensors"), so they can be tested without a GPU.
STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CPU, ops) {
  ops.impl("lcpp_mul_mat_vec_q", TORCH_BOX(&lcpp_mul_mat_vec_q));
  ops.impl("lcpp_mul_mat_q", TORCH_BOX(&lcpp_mul_mat_q));
}
