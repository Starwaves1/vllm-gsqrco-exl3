#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"

using torch::headeronly::ScalarType;
using torch::stable::Tensor;
using torch::stable::accelerator::DeviceGuard;

static inline cudaStream_t get_current_cuda_stream(int32_t device_index) {
  void* raw_stream = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(device_index, &raw_stream));
  return static_cast<cudaStream_t>(raw_stream);
}

// Q8 gemv
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

// Input checks for the dense ops below. The kernels only take data_ptr(), so
// without them a strided or misaligned W, a non-contiguous X, a row count
// past W or an X of the wrong width gave silently wrong results, NaNs or a
// device fault, and an unsupported type returned zeros (or, in
// ggml_dequantize, called a null function). Device checks come last, so the
// CPU registration (torch_bindings.cpp) runs every other check without a GPU.

// Values and bytes per block of the types these ops take; false for others.
static bool ggml_block_size(int64_t type, int64_t* values, int64_t* bytes) {
  switch (type) {
#define GGML_BLOCK(t, qk, block) \
  case t:                        \
    *values = qk;                \
    *bytes = sizeof(block);      \
    return true;
    GGML_BLOCK(2, QK4_0, block_q4_0)
    GGML_BLOCK(3, QK4_1, block_q4_1)
    GGML_BLOCK(6, QK5_0, block_q5_0)
    GGML_BLOCK(7, QK5_1, block_q5_1)
    GGML_BLOCK(8, QK8_0, block_q8_0)
    GGML_BLOCK(10, QK_K, block_q2_K)
    GGML_BLOCK(11, QK_K, block_q3_K)
    GGML_BLOCK(12, QK_K, block_q4_K)
    GGML_BLOCK(13, QK_K, block_q5_K)
    GGML_BLOCK(14, QK_K, block_q6_K)
    GGML_BLOCK(16, QK_K, block_iq2_xxs)
    GGML_BLOCK(17, QK_K, block_iq2_xs)
    GGML_BLOCK(18, QK_K, block_iq3_xxs)
    GGML_BLOCK(19, QK_K, block_iq1_s)
    GGML_BLOCK(20, QK4_NL, block_iq4_nl)
    GGML_BLOCK(21, QK_K, block_iq3_s)
    GGML_BLOCK(22, QK_K, block_iq2_s)
    GGML_BLOCK(23, QK_K, block_iq4_xs)
    GGML_BLOCK(29, QK_K, block_iq1_m)
#undef GGML_BLOCK
    default:
      return false;
  }
}

// The types ggml_mul_mat_a8 has a kernel for.
static bool ggml_mmq_type(int64_t type) {
  return type == 2 || type == 3 || type == 6 || type == 7 || type == 8 ||
         (type >= 10 && type <= 14);
}

static bool is_float_dtype(ScalarType t) {
  return t == ScalarType::Float || t == ScalarType::Half ||
         t == ScalarType::BFloat16;
}

// W [>= row, row_bytes] uint8 blocks with contiguous rows, X [n, K] with
// contiguous rows. Returns K.
static int64_t check_mul_mat_inputs(const Tensor& W, const Tensor& X,
                                    int64_t type, int64_t row, bool mmq,
                                    const char* op) {
  int64_t values = 0, bytes = 0;
  STD_TORCH_CHECK(
      ggml_block_size(type, &values, &bytes) && (!mmq || ggml_mmq_type(type)),
      op, ": unsupported ggml type ", type);
  STD_TORCH_CHECK(W.dim() == 2 && X.dim() == 2, op, ": W and X must be 2-D");
  STD_TORCH_CHECK(W.scalar_type() == ScalarType::Byte, op, ": W must be uint8");
  STD_TORCH_CHECK(is_float_dtype(X.scalar_type()), op,
                  ": X must be fp32, fp16 or bf16");
  const int64_t row_bytes = W.size(1);
  STD_TORCH_CHECK(row_bytes > 0 && row_bytes % bytes == 0, op, ": W row bytes ",
                  row_bytes, " not a multiple of the block size ", bytes);
  const int64_t k = row_bytes / bytes * values;
  STD_TORCH_CHECK(X.size(1) == k, op, ": X has ", X.size(1),
                  " columns, W rows hold K=", k);
  STD_TORCH_CHECK(row > 0 && row <= W.size(0), op, ": row=", row,
                  " out of range (W has ", W.size(0), " rows)");
  STD_TORCH_CHECK(k <= INT_MAX && W.size(0) <= INT_MAX && X.size(0) <= INT_MAX,
                  op, ": W or X too large");
  STD_TORCH_CHECK(
      W.stride(1) == 1 && (W.size(0) <= 1 || W.stride(0) == row_bytes), op,
      ": W rows must be contiguous (row stride ", W.stride(0), ", row bytes ",
      row_bytes, ")");
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0, op,
                  ": W data must be 16-byte aligned");
  STD_TORCH_CHECK(X.stride(1) == 1 && (X.size(0) <= 1 || X.stride(0) == k), op,
                  ": X rows must be contiguous");
  STD_TORCH_CHECK(mmq || X.size(0) <= 65535, op, ": at most 65535 rows");
  STD_TORCH_CHECK(W.is_cuda() && X.is_cuda(), op,
                  ": W and X must be CUDA tensors");
  STD_TORCH_CHECK(W.get_device_index() == X.get_device_index(), op,
                  ": W and X must be on the same device");
  return k;
}

Tensor ggml_dequantize(Tensor W,  // quant weight
                       int64_t type, int64_t m, int64_t n,
                       std::optional<ScalarType> dtype) {
  int64_t values = 0, bytes = 0;
  STD_TORCH_CHECK(ggml_block_size(type, &values, &bytes),
                  "ggml_dequantize: unsupported ggml type ", type);
  STD_TORCH_CHECK(W.scalar_type() == ScalarType::Byte,
                  "ggml_dequantize: W must be uint8");
  // The kernels see m * n values as one flat run of blocks (the embedding
  // path passes m = hidden size, n = tokens).
  STD_TORCH_CHECK(m >= 0 && n >= 0 && (m * n) % values == 0,
                  "ggml_dequantize: m * n = ", m * n,
                  " must be a multiple of the block's ", values, " values");
  int64_t numel = 1, expected_stride = 1;
  for (int64_t d = W.dim() - 1; d >= 0; --d) {
    STD_TORCH_CHECK(W.size(d) <= 1 || W.stride(d) == expected_stride,
                    "ggml_dequantize: W must be contiguous");
    expected_stride *= W.size(d);
    numel *= W.size(d);
  }
  STD_TORCH_CHECK(numel >= m * n / values * bytes, "ggml_dequantize: W has ",
                  numel, " bytes, ", m, " x ", n, " values need ",
                  m * n / values * bytes);
  STD_TORCH_CHECK(reinterpret_cast<uintptr_t>(W.data_ptr()) % 16 == 0,
                  "ggml_dequantize: W data must be 16-byte aligned");
  STD_TORCH_CHECK(!dtype || is_float_dtype(*dtype),
                  "ggml_dequantize: dtype must be fp32, fp16 or bf16");
  STD_TORCH_CHECK(W.is_cuda(), "ggml_dequantize: W must be a CUDA tensor");
  const int32_t device_idx = W.get_device_index();
  const DeviceGuard device_guard(device_idx);
  const auto dtype_ = dtype.value_or(ScalarType::Half);
  Tensor DW = torch::stable::new_zeros(W, {m, n}, dtype_);
  cudaStream_t stream = get_current_cuda_stream(device_idx);

  VLLM_DISPATCH_FLOATING_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

Tensor ggml_mul_mat_vec_a8(Tensor W,  // quant weight
                           Tensor X,  // input
                           int64_t type, int64_t row) {
  check_mul_mat_inputs(W, X, type, row, false, "ggml_mul_mat_vec_a8");
  int64_t col = X.sizes()[1];
  int64_t vecs = X.sizes()[0];
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y = torch::stable::new_zeros(W, {vecs, row}, X.scalar_type());
  cudaStream_t stream = get_current_cuda_stream(device_idx);
  Tensor quant_X =
      torch::stable::new_empty(W, {vecs, padded / 32 * 9}, ScalarType::Int);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>(
        (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), col, vecs, stream);
    switch (type) {
      case 2:
        mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 3:
        mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 6:
        mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 7:
        mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 8:
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 10:
        mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 11:
        mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 12:
        mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 13:
        mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 14:
        mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 16:
        mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 17:
        mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 18:
        mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 19:
        mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 20:
        mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 21:
        mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 22:
        mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 23:
        mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
      case 29:
        mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
        break;
    }
  });
  return Y;
}

Tensor ggml_mul_mat_a8(Tensor W,  // quant weight
                       Tensor X,  // input
                       int64_t type, int64_t row) {
  check_mul_mat_inputs(W, X, type, row, true, "ggml_mul_mat_a8");
  int64_t col = X.sizes()[1];
  int64_t padded = (col + 512 - 1) / 512 * 512;
  int64_t batch = X.sizes()[0];
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y = torch::stable::new_zeros(W, {batch, row}, X.scalar_type());
  cudaStream_t stream = get_current_cuda_stream(device_idx);
  Tensor quant_X =
      torch::stable::new_empty(W, {batch, padded / 32 * 9}, ScalarType::Int);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
    }
  });
  return Y;
}

Tensor ggml_moe_a8(Tensor X,  // input
                   Tensor W,  // expert weights
                   Tensor sorted_token_ids, Tensor expert_ids,
                   Tensor num_tokens_post_padded, int64_t type, int64_t row,
                   int64_t top_k, int64_t tokens) {
  int64_t col = X.sizes()[1];
  int64_t padded = (col + 512 - 1) / 512 * 512;
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y =
      torch::stable::new_zeros(W, {tokens * top_k, row}, X.scalar_type());
  cudaStream_t stream = get_current_cuda_stream(device_idx);
  Tensor quant_X =
      torch::stable::new_empty(W, {tokens, padded / 32 * 9}, ScalarType::Int);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
    }
  });
  return Y;
}

Tensor ggml_moe_a8_vec(Tensor X,  // input
                       Tensor W,  // expert weights
                       Tensor topk_ids, int64_t top_k, int64_t type,
                       int64_t row, int64_t tokens) {
  int64_t col = X.sizes()[1];
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  const int32_t device_idx = X.get_device_index();
  const DeviceGuard device_guard(device_idx);
  Tensor Y =
      torch::stable::new_zeros(W, {tokens * top_k, row}, X.scalar_type());
  cudaStream_t stream = get_current_cuda_stream(device_idx);
  Tensor quant_X =
      torch::stable::new_empty(W, {tokens, padded / 32 * 9}, ScalarType::Int);
  VLLM_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                     (void*)quant_X.data_ptr(), col, tokens,
                                     stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
    }
  });
  return Y;
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
    case 16:
    case 17:
    case 18:
    case 19:
    case 20:
    case 21:
    case 22:
    case 23:
    case 29:
      return MOE_X_Q2_K;
  }
  return 0;
}
