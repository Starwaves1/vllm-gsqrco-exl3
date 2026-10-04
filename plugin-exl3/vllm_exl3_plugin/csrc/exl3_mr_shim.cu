// SPDX-License-Identifier: Apache-2.0
// EXL3 multi-row shim: run trellis-serve's Marlin-template EXL3 kernels (vendored unmodified
// under csrc/trellis_serve, see csrc/trellis_serve/VENDORED.md) as torch ops in the _C_exl3
// namespace. Built into its own extension, vllm_exl3_plugin._C_exl3_mr, so _C_exl3 (the
// phase-1 library) is unchanged; the extension's Python module init is the vendored
// PYBIND11_MODULE.
//
// Ops (x: fp16 activations [m, k]; suh fp16 [k]; svh fp16 [n]; mcg/mul1 the checkpoint's
// codebook flags):
//   exl3_gemm_mr(x, trellis, suh, svh, mcg, mul1, out_fp32) -> [m, n] fp32 or fp16; bf16 x -> bf16
//       x @ W through the vendored exl3_linear_marlin_out: one input-Hadamard launch, one GEMM
//       launch with the output Hadamard in its epilogue. exl3_gemm's signature and output
//       dtypes. The kernel writes fp16 or bf16 (its MMA accumulates in fp32); with out_fp32
//       it writes bf16 and the result is widened, so a bf16 model sees one rounding of the
//       fp32 epilogue value, as with exl3_gemm's fp32 output. bf16 x (a bf16 model, the glue)
//       goes in as is (the input Hadamard converts it to fp16, the value conversion .to(fp16)
//       does) and the bf16 result comes out as is: the same bits as fp16 x + out_fp32 + .to(bf16),
//       without the two cast launches. Up to 64 rows per weight pass
//       (thread_m_blocks 4), so 17..64 rows cost one pass, 65..128 two, 129..144 three.
//       trellis: exllamav3's int16 [k/16, n/16, 48 | 80] for K3 / K5 (read in place as int32
//       [k/16, n/64, 4, 24 | 40], no copy), or exl3_mr_repack's int32 [k/16, n/64, 32, 4]
//       for K4. mul1 only (the one codebook compiled).
//   exl3_mr_repack(trellis) -> int32 [k/16, n/64, 32, 4]: K4 int16 trellis to the kernel's
//       layout (vendored repack_trellis: 32-bit word l of tile (i, 4g + j) moves to
//       [i, g, l, j]; a permutation, lossless, same size). CPU or CUDA.
//   exl3_mr_unpack(b) -> int16 [k/16, n/16, 64]: the inverse (vendored unpack_trellis), for
//       the dequant routes above 144 rows. CPU or CUDA.
//   exl3_mr_warmup(trellis, suh, svh, mcg, mul1, rows, out_fp32): the vendored per-device
//       state (locks zeroed, fp32 reduce scratch: at::zeros / at::empty on first use) and one
//       call per row count, which loads every kernel instance those rows select. Must run
//       outside CUDA graph capture; throws if the stream is capturing.
//
//   exl3_embed_host_register(table) -> id: page-locks a bf16 CPU table [rows, cols] in place
//       (cudaHostRegister, mapped: exactly its size, where torch's pinned allocator rounds 2.37 GiB
//       up to 4 GiB) and keeps it for the life of the process. Ids count from 0 in registration
//       order, so the same model loads to the same id in every process: a compiled graph (vLLM's
//       AOT cache reloads them without guards) holds the id, never an address.
//   exl3_embed_host(ids, table_id, cols) -> bf16 [n, cols] on ids' device: rows of a registered
//       table, gathered by a kernel reading it over PCIe (UVA), so the token embedding needs no
//       VRAM. Out-of-range ids give zero rows. No host work: graph-capturable.
//
// Capture safety: exl3_gemm_mr refuses to run while the current stream is capturing unless
// exl3_mr_warmup ran that (device, k, n, K, rows) first. Keyed per row count, not per bucket:
// the vendored launcher picks the thread config from m (narrow config, row family, 64-row
// splits), so every routed m is warmed (the plugin passes 17..144, or 1..144 for a repack).
//
// Guards (shapes, dtypes, strides, 16-byte alignment, bit width, codebook) run before any
// CUDA call. exl3_gemm_mr and exl3_mr_warmup are registered for CPU too, where the guards run
// and the call then fails with "must be CUDA tensors"; repack and unpack run on CPU for real.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <climits>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

// Defined in the vendored trellis_serve/exl3_marlin.cu (no header declares them).
void exl3_linear_marlin_out(const at::Tensor& x, const at::Tensor& b, const at::Tensor& suh, const at::Tensor& svh,
                            int64_t cb, at::Tensor& xh, at::Tensor& y);
void exl3_linear_marlin_multi_out(const at::Tensor& x, const at::Tensor& b, const at::Tensor& suh_cat,
                                  const at::Tensor& svh_cat, std::vector<int64_t> shard_ends, int64_t cb,
                                  at::Tensor& xh, at::Tensor& y);
at::Tensor repack_trellis(const at::Tensor& trellis);
at::Tensor unpack_trellis(const at::Tensor& b);
void init_device(int64_t device);
void set_force_cfg(int64_t tk, int64_t tn);

namespace {

constexpr int64_t kCbMul1 = 2;  // trellis-serve codebook id: 0 = 3INST, 1 = MCG, 2 = MUL1

// ---------------------------------------------------------------------------
// Guards (same rules and messages as exl3_shim.cu's)

bool aligned16(const at::Tensor& t) { return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0; }

void check_scale(const at::Tensor& s, int64_t size, const char* name, const char* op) {
  TORCH_CHECK(s.dim() == 1 && s.size(0) == size, op, ": ", name, " must be 1-D of size ", size,
              ", got ", s.sizes());
  TORCH_CHECK(s.scalar_type() == at::kHalf, op, ": ", name, " must be fp16");
  TORCH_CHECK(s.is_contiguous(), op, ": ", name, " must be contiguous");
  TORCH_CHECK(aligned16(s), op, ": ", name, " must be 16-byte aligned");
}

void check_cuda(std::initializer_list<const at::Tensor*> ts, const char* op) {
  const at::Tensor* first = *ts.begin();
  for (const at::Tensor* t : ts) {
    TORCH_CHECK(t->is_cuda(), op, ": inputs must be CUDA tensors");
    TORCH_CHECK(t->get_device() == first->get_device(), op, ": inputs must be on one device");
  }
}

int64_t check_x(const at::Tensor& x, int64_t k, const char* op) {
  TORCH_CHECK(x.dim() == 2, op, ": x must be 2-D");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, op, ": x must be fp16 or bf16");
  TORCH_CHECK(x.size(1) == k, op, ": x has ", x.size(1), " columns, the weight has k=", k);
  TORCH_CHECK(x.is_contiguous(), op, ": x must be contiguous");
  TORCH_CHECK(aligned16(x), op, ": x must be 16-byte aligned");
  TORCH_CHECK(x.size(0) <= INT_MAX, op, ": x too large");
  return x.size(0);
}

void check_k_n(int64_t k, int64_t n, const char* op) {
  TORCH_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0, op,
              ": k and n must be multiples of 128 (128-wide Hadamard), got k=", k, " n=", n);
  TORCH_CHECK(k <= INT_MAX && n <= INT_MAX, op, ": weight too large");
}

// The kernel's view of a trellis: int32, plus K, k and n.
struct Weight {
  at::Tensor b;
  int64_t bits, k, n;
};

Weight check_weight(const at::Tensor& trellis, bool mcg, bool mul1, const char* op) {
  TORCH_CHECK(!(mcg && mul1), op, ": mcg and mul1 are exclusive");
  TORCH_CHECK(mul1, op, ": only the mul1 codebook is built");
  TORCH_CHECK(trellis.is_contiguous(), op, ": trellis must be contiguous");
  TORCH_CHECK(aligned16(trellis), op, ": trellis must be 16-byte aligned");
  if (trellis.scalar_type() == at::kShort) {
    TORCH_CHECK(trellis.dim() == 3 && (trellis.size(2) == 48 || trellis.size(2) == 80), op,
                ": an int16 trellis must be K3 or K5 [k/16, n/16, 48 | 80]; K4 needs exl3_mr_repack, got ",
                trellis.sizes());
    const int64_t kt = trellis.size(0), nt = trellis.size(1), words = trellis.size(2) / 2;
    check_k_n(kt * 16, nt * 16, op);
    return {trellis.view(at::kInt).view({kt, nt / 4, 4, words}), trellis.size(2) / 16, kt * 16, nt * 16};
  }
  TORCH_CHECK(trellis.scalar_type() == at::kInt && trellis.dim() == 4 && trellis.size(2) == 32 &&
                  trellis.size(3) == 4,
              op, ": trellis must be int16 K3/K5 or exl3_mr_repack's int32 K4 [k/16, n/64, 32, 4], got ",
              trellis.scalar_type(), " ", trellis.sizes());
  check_k_n(trellis.size(0) * 16, trellis.size(1) * 64, op);
  return {trellis, 4, trellis.size(0) * 16, trellis.size(1) * 64};
}

// ---------------------------------------------------------------------------
// Host-resident embedding gather

__global__ void embed_host_kernel(const int64_t* __restrict__ ids, const uint4* __restrict__ table,
                                  uint4* __restrict__ out, int64_t v, int64_t d16) {
  const int64_t row = blockIdx.x, id = ids[row];
  uint4* dst = out + row * d16;
  if (id < 0 || id >= v) {
    for (int64_t j = threadIdx.x; j < d16; j += blockDim.x) dst[j] = make_uint4(0, 0, 0, 0);
    return;
  }
  const uint4* src = table + id * d16;
  for (int64_t j = threadIdx.x; j < d16; j += blockDim.x) dst[j] = src[j];
}

std::mutex g_tables_mutex;
std::vector<at::Tensor> g_tables;  // registered (page-locked) host tables, never released

int64_t exl3_embed_host_register_op(const at::Tensor& table) {
  const char* op = "exl3_embed_host_register";
  TORCH_CHECK(table.device().is_cpu() && table.dim() == 2 && table.scalar_type() == at::kBFloat16 &&
                  table.is_contiguous(),
              op, ": table must be a contiguous 2-D bf16 CPU tensor");
  TORCH_CHECK(table.size(1) % 8 == 0 && aligned16(table) && table.numel() > 0, op,
              ": table rows must be a nonzero multiple of 16 bytes, 16-byte aligned");
  C10_CUDA_CHECK(cudaHostRegister(table.data_ptr(), table.nbytes(), cudaHostRegisterMapped | cudaHostRegisterPortable));
  std::lock_guard<std::mutex> lock(g_tables_mutex);
  g_tables.push_back(table);
  return (int64_t)g_tables.size() - 1;
}

at::Tensor exl3_embed_host_op(const at::Tensor& ids, int64_t table_id, int64_t cols) {
  const char* op = "exl3_embed_host";
  TORCH_CHECK(ids.scalar_type() == at::kLong && ids.dim() == 1 && ids.is_contiguous(), op,
              ": ids must be contiguous 1-D int64");
  TORCH_CHECK(ids.is_cuda(), op, ": ids must be a CUDA tensor");
  at::Tensor table;
  {
    std::lock_guard<std::mutex> lock(g_tables_mutex);
    TORCH_CHECK(table_id >= 0 && table_id < (int64_t)g_tables.size(), op, ": table ", table_id, " is not registered");
    table = g_tables[table_id];
  }
  TORCH_CHECK(table.size(1) == cols, op, ": table ", table_id, " has ", table.size(1), " columns, not ", cols);
  const c10::cuda::CUDAGuard guard(ids.device());
  at::Tensor out = at::empty({ids.size(0), cols}, ids.options().dtype(at::kBFloat16));
  if (ids.size(0) == 0) return out;
  void* dev_table = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&dev_table, table.data_ptr(), 0));
  embed_host_kernel<<<(unsigned)ids.size(0), 128, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      ids.data_ptr<int64_t>(), (const uint4*)dev_table, (uint4*)out.data_ptr(), table.size(0), cols / 8);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// ---------------------------------------------------------------------------
// Warmup registry and the capture guard

using Key = std::tuple<int, int64_t, int64_t, int64_t, int64_t>;  // device, k, n, K, rows
std::mutex g_mutex;
std::set<Key> g_warmed;

bool capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  return status != cudaStreamCaptureStatusNone;
}

// ---------------------------------------------------------------------------
// Ops

// shard_ends empty: one tensor. Else a fused group of equal-K tensors concatenated along n (the vendored
// exl3_linear_marlin_multi_out): suh = the shards' suh stacked [shards * k], svh [n], shard_ends = column
// ends of all shards but the last, multiples of 128; one input-Hadamard launch for all shards, one GEMM.
at::Tensor gemm_mr_impl(const at::Tensor& x, const at::Tensor& trellis, const at::Tensor& suh, const at::Tensor& svh,
                        at::IntArrayRef shard_ends, bool mcg, bool mul1, bool out_fp32, bool warming) {
  const char* op = shard_ends.empty() ? "exl3_gemm_mr" : "exl3_gemm_mr_multi";
  const Weight w = check_weight(trellis, mcg, mul1, op);
  const int64_t m = check_x(x, w.k, op);
  const int64_t shards = (int64_t)shard_ends.size() + 1;
  TORCH_CHECK(shards <= 4, op, ": at most 4 shards");
  for (size_t i = 0; i < shard_ends.size(); i++)
    TORCH_CHECK(shard_ends[i] > (i ? shard_ends[i - 1] : 0) && shard_ends[i] < w.n && shard_ends[i] % 128 == 0, op,
                ": shard ends must be ascending multiples of 128 inside n=", w.n, ", got ", shard_ends);
  check_scale(suh, shards * w.k, "suh", op);
  check_scale(svh, w.n, "svh", op);
  check_cuda({&x, &trellis, &suh, &svh}, op);

  const bool bf16_io = x.scalar_type() == at::kBFloat16;
  const at::ScalarType out_dtype = bf16_io ? at::kBFloat16 : out_fp32 ? at::kFloat : at::kHalf;
  if (m == 0) return at::empty({0, w.n}, x.options().dtype(out_dtype));
  const c10::cuda::CUDAGuard guard(x.device());
  if (!warming && capturing(at::cuda::getCurrentCUDAStream().stream())) {
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(g_warmed.count({x.get_device(), w.k, w.n, w.bits, m}), op, ": shape k=", w.k, " n=", w.n,
                " K=", w.bits, " rows=", m, " was not warmed up (exl3_mr_warmup) before CUDA graph capture");
  }
  // job 15 (3090, the model's shapes): at 17..64 rows (thread_m_blocks 2..4, one launch) thread_k
  // 128 x thread_n 128 beats the launcher's default pick by 4-6 % (target pass at 24 rows 33.2 ->
  // 31.7 ms, 48 rows 46.4 -> 44.2); the vendored launcher keeps its default where it is invalid
  // (not for narrow outputs: k_proj, n = 1024, keeps the launcher's narrow pick, job 15 +30 %).
  // A process-wide knob, set on every call: one model thread per process, as vLLM runs it
  const bool wide = m > 16 && m <= 64 && w.n >= 2048;
  set_force_cfg(wide ? 128 : 0, wide ? 128 : 0);
  at::Tensor xh = at::empty({shards * m, w.k}, x.options().dtype(at::kHalf));
  at::Tensor y = at::empty({m, w.n}, x.options().dtype(bf16_io || out_fp32 ? at::kBFloat16 : at::kHalf));
  if (shards == 1) {
    exl3_linear_marlin_out(x, w.b, suh, svh, kCbMul1, xh, y);
  } else {
    exl3_linear_marlin_multi_out(x, w.b, suh, svh, shard_ends.vec(), kCbMul1, xh, y);
  }
  return out_fp32 && !bf16_io ? y.to(at::kFloat) : y;
}

at::Tensor exl3_gemm_mr_op(const at::Tensor& x, const at::Tensor& trellis, const at::Tensor& suh,
                           const at::Tensor& svh, bool mcg, bool mul1, bool out_fp32) {
  return gemm_mr_impl(x, trellis, suh, svh, {}, mcg, mul1, out_fp32, false);
}

at::Tensor exl3_gemm_mr_multi_op(const at::Tensor& x, const at::Tensor& trellis, const at::Tensor& suh,
                                 const at::Tensor& svh, at::IntArrayRef shard_ends, bool mcg, bool mul1, bool out_fp32) {
  TORCH_CHECK(!shard_ends.empty(), "exl3_gemm_mr_multi: needs at least one shard end (else exl3_gemm_mr)");
  return gemm_mr_impl(x, trellis, suh, svh, shard_ends, mcg, mul1, out_fp32, false);
}

at::Tensor exl3_mr_repack_op(const at::Tensor& trellis) {
  const char* op = "exl3_mr_repack";
  TORCH_CHECK(trellis.dim() == 3 && trellis.scalar_type() == at::kShort && trellis.size(2) == 64, op,
              ": only a K4 int16 trellis [k/16, n/16, 64] is repacked (K3/K5 are read as stored), got ",
              trellis.scalar_type(), " ", trellis.sizes());
  TORCH_CHECK(trellis.is_contiguous(), op, ": trellis must be contiguous");
  check_k_n(trellis.size(0) * 16, trellis.size(1) * 16, op);
  const c10::OptionalDeviceGuard guard(trellis.device());
  return repack_trellis(trellis);
}

at::Tensor exl3_mr_unpack_op(const at::Tensor& b) {
  const char* op = "exl3_mr_unpack";
  TORCH_CHECK(b.dim() == 4 && b.scalar_type() == at::kInt && b.size(2) == 32 && b.size(3) == 4, op,
              ": b must be exl3_mr_repack's int32 [k/16, n/64, 32, 4], got ", b.scalar_type(), " ", b.sizes());
  check_k_n(b.size(0) * 16, b.size(1) * 64, op);
  const c10::OptionalDeviceGuard guard(b.device());
  return unpack_trellis(b);
}

void mr_warmup_impl(const at::Tensor& trellis, const at::Tensor& suh, const at::Tensor& svh, at::IntArrayRef shard_ends,
                    bool mcg, bool mul1, at::IntArrayRef rows, bool out_fp32) {
  const char* op = "exl3_mr_warmup";
  const Weight w = check_weight(trellis, mcg, mul1, op);
  check_scale(suh, ((int64_t)shard_ends.size() + 1) * w.k, "suh", op);
  check_scale(svh, w.n, "svh", op);
  for (int64_t m : rows) TORCH_CHECK(m >= 1 && m <= INT_MAX, op, ": row counts must be >= 1, got ", m);
  check_cuda({&trellis, &suh, &svh}, op);

  const int device = trellis.get_device();
  const c10::cuda::CUDAGuard guard(trellis.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(!capturing(stream), op, ": must run outside CUDA graph capture");
  init_device(device);  // vendored dev_state: SM count, smem limit, locks (zeroed), fp32 reduce scratch
  int64_t max_m = 0;
  for (int64_t m : rows) max_m = std::max(max_m, m);
  if (max_m > 0) {
    for (at::ScalarType dt : {at::kHalf, at::kBFloat16}) {  // the input Hadamard has an instance per dtype
      at::Tensor x = at::zeros({max_m, w.k}, trellis.options().dtype(dt));
      for (int64_t m : rows) gemm_mr_impl(x.narrow(0, 0, m), trellis, suh, svh, shard_ends, mcg, mul1, out_fp32, true);
    }
  }
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::lock_guard<std::mutex> lock(g_mutex);
  for (int64_t m : rows) g_warmed.insert({device, w.k, w.n, w.bits, m});
}

void exl3_mr_warmup_op(const at::Tensor& trellis, const at::Tensor& suh, const at::Tensor& svh, bool mcg, bool mul1,
                       at::IntArrayRef rows, bool out_fp32) {
  mr_warmup_impl(trellis, suh, svh, {}, mcg, mul1, rows, out_fp32);
}

void exl3_mr_warmup_multi_op(const at::Tensor& trellis, const at::Tensor& suh, const at::Tensor& svh,
                             at::IntArrayRef shard_ends, bool mcg, bool mul1, at::IntArrayRef rows, bool out_fp32) {
  mr_warmup_impl(trellis, suh, svh, shard_ends, mcg, mul1, rows, out_fp32);
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C_exl3, m) {
  m.def("exl3_gemm_mr(Tensor x, Tensor trellis, Tensor suh, Tensor svh, bool mcg, bool mul1, bool out_fp32) -> Tensor");
  m.def("exl3_gemm_mr_multi(Tensor x, Tensor trellis, Tensor suh, Tensor svh, int[] shard_ends, bool mcg, bool mul1, "
        "bool out_fp32) -> Tensor");
  m.def("exl3_mr_warmup_multi(Tensor trellis, Tensor suh, Tensor svh, int[] shard_ends, bool mcg, bool mul1, int[] rows, "
        "bool out_fp32) -> ()");
  m.def("exl3_mr_repack(Tensor trellis) -> Tensor");
  m.def("exl3_mr_unpack(Tensor b) -> Tensor");
  m.def("exl3_mr_warmup(Tensor trellis, Tensor suh, Tensor svh, bool mcg, bool mul1, int[] rows, bool out_fp32) -> ()");
  m.def("exl3_embed_host_register(Tensor table) -> int");
  m.def("exl3_embed_host(Tensor ids, int table_id, int cols) -> Tensor");
}

TORCH_LIBRARY_IMPL(_C_exl3, CUDA, m) {
  m.impl("exl3_gemm_mr_multi", &exl3_gemm_mr_multi_op);
  m.impl("exl3_mr_warmup_multi", &exl3_mr_warmup_multi_op);
  m.impl("exl3_embed_host", &exl3_embed_host_op);
  m.impl("exl3_gemm_mr", &exl3_gemm_mr_op);
  m.impl("exl3_mr_repack", &exl3_mr_repack_op);
  m.impl("exl3_mr_unpack", &exl3_mr_unpack_op);
  m.impl("exl3_mr_warmup", &exl3_mr_warmup_op);
}

// CPU: exl3_gemm_mr and exl3_mr_warmup run their guards and stop at "must be CUDA tensors";
// repack and unpack are plain tensor ops and run.
TORCH_LIBRARY_IMPL(_C_exl3, CPU, m) {
  m.impl("exl3_gemm_mr_multi", &exl3_gemm_mr_multi_op);
  m.impl("exl3_mr_warmup_multi", &exl3_mr_warmup_multi_op);
  m.impl("exl3_embed_host_register", &exl3_embed_host_register_op);
  m.impl("exl3_embed_host", &exl3_embed_host_op);
  m.impl("exl3_gemm_mr", &exl3_gemm_mr_op);
  m.impl("exl3_mr_repack", &exl3_mr_repack_op);
  m.impl("exl3_mr_unpack", &exl3_mr_unpack_op);
  m.impl("exl3_mr_warmup", &exl3_mr_warmup_op);
}
