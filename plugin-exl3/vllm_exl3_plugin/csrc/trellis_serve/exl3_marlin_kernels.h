// Kernel declaration, parameter list and the instance table shared by the host code (exl3_marlin.cu) and the
// generated instantiation units (setup.py writes one build/gen/inst_mb<M>_cb<C>.cu per row-block family and
// codebook so they compile in parallel; same scheme as vLLM's Marlin generate_kernels.py).
#pragma once

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME trellis_exl3_marlin
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

#define TRELLIS_KERNEL_PARAMS                                                   \
  const int4 *__restrict__ A, const int4 *__restrict__ B,                      \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                          \
      const int4 *__restrict__ b_bias_ptr,                                     \
      const float *__restrict__ a_scales_ptr,                                  \
      const int4 *__restrict__ scales_ptr,                                     \
      const float *__restrict__ global_scale_ptr,                              \
      const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx,          \
      int num_groups, int prob_m, int prob_n, int prob_k, int lda, int *locks, \
      bool has_bias, bool use_atomic_add, bool use_fp32_reduce,                \
      int max_shared_mem, int a_shard_stride, int shard_end0, int shard_end1,  \
      int shard_end2, int out_flags, const int4 *__restrict__ x_raw,           \
      const int4 *__restrict__ suh_ptr, int in_flags, int in_shards

namespace MARLIN_NAMESPACE_NAME {

#ifndef TRELLIS_KERNEL_DEFINED
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id, const int threads,
          const int thread_m_blocks, const int thread_n_blocks, const int thread_k_blocks,
          const bool m_block_size_8, const int stages, const int group_blocks, const bool is_zp_float,
          const int exl3_cb>
__global__ void Marlin(TRELLIS_KERNEL_PARAMS);
#endif

using FuncPtr = void (*)(TRELLIS_KERNEL_PARAMS);

static constexpr int kStages = 4;

// K3: EXL3 K = 3: a 3-bit "weight type" that only selects the byte-exact 24-word tile staging in the template
// (vLLM's ScalarType has no 3-bit constant; the id is what the kernel template switches on). Same id as the MoE front.
static constexpr vllm::ScalarTypeId kExl3K3Id = vllm::ScalarType::uint(3, 0).id();
// K5: EXL3 K = 5: same scheme with 40-word tiles (byte-exact staging)
static constexpr vllm::ScalarTypeId kExl3K5Id = vllm::ScalarType::uint(5, 0).id();

// Kernel type for (threads, thread_n, thread_k, mb, cb, kb); mb = thread_m_blocks, 0 = rows <= 8 (transposed MMA);
// kb = EXL3 bits per weight: 3 / 5 (byte-exact tile staging, K3 / K5), 4 (Marlin 4-bit staging) or 6 (Marlin 8-bit staging,
// two vectors per lane)
#define TRELLIS_KERNEL(threads, tn, tk, mb, cb, kb)                                                          \
  Marlin<vllm::kFloat16.id(), (kb == 4 ? vllm::kU4B8.id() : kb == 3 ? kExl3K3Id : kb == 5 ? kExl3K5Id /* K5 */ : vllm::kU8B128.id()), \
         vllm::kFloat16.id(),                                                                               \
         vllm::kFloat16.id(), threads, (mb == 0 ? 1 : mb), tn / 16, tk / 16, (mb == 0), kStages, -1, false, cb>

// X(threads, thread_n, thread_k): Marlin's thread configurations
#define TRELLIS_THREAD_CFGS(X, mb, cb, kb) \
  X(256, 128, 128, mb, cb, kb)            \
  X(256, 256, 64, mb, cb, kb)             \
  X(128, 128, 64, mb, cb, kb)             \
  X(128, 64, 128, mb, cb, kb)

}  // namespace MARLIN_NAMESPACE_NAME
