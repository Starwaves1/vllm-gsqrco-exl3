// SPDX-License-Identifier: Apache-2.0
// EXL3 shim: run exllamav3 d3739fd's dense-linear kernels (vendored unmodified under
// csrc/exl3, see csrc/exl3/VENDORED.md) as torch ops _C_exl3::*.
//
// Ops (x: activations [m, k]; trellis: int16 [k/16, n/16, 16*K] (+8 for half-integer K);
// suh: fp16 [k]; svh: fp16 [n]; mcg/mul1: the codebook flags from the checkpoint):
//   exl3_gemm(x, trellis, suh, svh, mcg, mul1, out_fp32) -> [m, n] fp32 or fp16
//       vendored exl3_gemm (QTIP GEMV where its heuristic picks it, else the cooperative
//       kernel, autotuned per shape) on fp16 x (the kernels' only activation dtype; the caller
//       casts once per layer). Any m; the kernel re-streams the weight per 16 rows (EXL3.md).
//   exl3_dequant(trellis, suh, svh, mcg, mul1, n_start, n_count, had) -> fp16 [k, n_count]
//       columns n_start..n_start+n_count of the weight, W[in, out] (y = x @ W). had=true: the
//       original basis (reconstruct_had_slice: both Hadamards, suh and svh folded in);
//       had=false: the rotated basis (reconstruct_slice), for had_r_128 on x and y.
//   exl3_had_r_128(x, pre_scale?, post_scale?, scale) -> fp16: 128-wide Hadamard of the rows
//       of fp16 x, times pre_scale before or post_scale after (at most one), times scale.
//   exl3_hgemm(a, b) -> fp16 [m, n] = a @ b, a fp16 [m, k], b fp16 [k, n]: vendored
//       hgemm_recon (fp16-accumulate MMA where the rate probe enables it, else cuBLAS).
//   exl3_warmup(trellis, suh, svh, mcg, mul1, rows, out_fp32): the one-time host work (DevCtx
//       cudaMalloc of locks and workspace, the f16acc rate probe, autotune sessions, kernel
//       attributes) for this weight shape: exl3_gemm once per row count in rows, and one
//       f16acc-sized hgemm. Must run outside CUDA graph capture (it synchronizes); it throws
//       if the current stream is capturing.
//
// Capture safety: exl3_gemm and exl3_hgemm refuse to run while the current stream is
// capturing unless exl3_warmup already ran that shape and row bucket on that device. The
// buckets follow the vendored choices: 1 row (GEMV m == 1 mode), then 2, 4, 8, 16 by
// rounding up (GEMV m <= 8, autotune key min(pow2(max(m, 2)), 16)); 16 covers every m >= 9.
// Without that, the vendored code could cudaMalloc, synchronize or autotune inside capture.
//
// exllamav3's int8-activation GEMV (EXL3_INT8_GEMV, on by default in d3739fd for mul1 K <= 5
// at m <= 2 on Ampere) is switched off when this library loads: it changes the numerics
// (~0.9 % of output RMS), its first call cudaMallocs a workspace and the vendored comment
// calls it not graph-capturable. EXL3_GEMM_H_ACC (fp16 MMA accumulation on sm_86) is a
// compile-time vendored choice and stays on.
//
// Guards (shapes, dtypes, strides, 16-byte alignment, bit width, codebook) run before any
// CUDA call. The ops are registered for CPU too: there the guards run and the call then
// fails with "must be CUDA tensors", so the guard tests need no GPU.

#include <Python.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <climits>
#include <cstdlib>
#include <stdexcept>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

#include "cuda_drv.h"
#include "hgemm.cuh"
#include "quant/exl3_devctx.cuh"
#include "quant/exl3_gemm.cuh"

// Defined in the vendored quant/reconstruct.cu and quant/hadamard.cu; their headers are not
// part of the vendored closure (nothing vendored includes them), so declared here.
void reconstruct_slice(at::Tensor unpacked, at::Tensor packed, float K, bool mcg, bool mul1, int64_t n_offset);
void reconstruct_had_slice(at::Tensor unpacked, at::Tensor packed, at::Tensor suh, at::Tensor svh, float K,
                           bool mcg, bool mul1, int64_t n_offset);
void had_r_128(const at::Tensor& input, const at::Tensor& output, const c10::optional<at::Tensor>& pre_scale,
               const c10::optional<at::Tensor>& post_scale, const float scale);

// ---------------------------------------------------------------------------
// Linked-in pieces the vendored files expect from the rest of exllamav3_ext.

// graph.cu (exllamav3's own CUDA graph recorder) resolves driver entry points through
// cuda_drv.cpp, which is not vendored. The shim never passes a Graph, so the recorder never
// runs; this stub keeps libcuda out of the link and fails loudly if it ever is reached.
const CudaDrv& CudaDrv::instance() {
  throw std::runtime_error("exl3 shim: exllamav3's Graph recorder is not supported");
}

namespace {

struct Int8GemvOff {
  Int8GemvOff() { setenv("EXL3_INT8_GEMV", "0", 1); }
} int8_gemv_off;

// ---------------------------------------------------------------------------
// Guards

// Bit width K from the trellis tile width, as the vendored code derives it (16 * K uint16
// per tile, 16 * K + 8 for half-integer K, mul1 only). Returns K as the float the vendored
// entry points take.
float check_trellis(const at::Tensor& trellis, bool mcg, bool mul1, const char* op) {
  TORCH_CHECK(!(mcg && mul1), op, ": mcg and mul1 are exclusive");
  TORCH_CHECK(trellis.dim() == 3, op, ": trellis must be 3-D [k/16, n/16, 16*K]");
  TORCH_CHECK(trellis.scalar_type() == at::kShort, op, ": trellis must be int16");
  TORCH_CHECK(trellis.is_contiguous(), op, ": trellis must be contiguous");
  const int64_t w = trellis.size(2);
  const bool half_k = w % 16 == 8;
  TORCH_CHECK(w % 16 == 0 || half_k, op, ": trellis tile width ", w, " is not 16*K or 16*K+8");
  const int64_t bits = w / 16;
  TORCH_CHECK(bits >= 1 && bits <= 8 && (!half_k || bits <= 3), op, ": unsupported bit width (tile width ", w, ")");
  TORCH_CHECK(!half_k || mul1, op, ": half-integer bit widths need the mul1 codebook");
  TORCH_CHECK(trellis.size(0) > 0 && trellis.size(1) > 0, op, ": empty trellis");
  TORCH_CHECK(trellis.size(0) * 16 % 128 == 0 && trellis.size(1) * 16 % 128 == 0, op,
              ": k and n must be multiples of 128 (128-wide Hadamard), got k=", trellis.size(0) * 16,
              " n=", trellis.size(1) * 16);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(trellis.data_ptr()) % 16 == 0, op, ": trellis must be 16-byte aligned");
  return (float)bits + (half_k ? 0.5f : 0.0f);
}

void check_scale(const at::Tensor& s, int64_t size, const char* name, const char* op) {
  TORCH_CHECK(s.dim() == 1 && s.size(0) == size, op, ": ", name, " must be 1-D of size ", size,
              ", got ", s.sizes());
  TORCH_CHECK(s.scalar_type() == at::kHalf, op, ": ", name, " must be fp16");
  TORCH_CHECK(s.is_contiguous(), op, ": ", name, " must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(s.data_ptr()) % 16 == 0, op, ": ", name, " must be 16-byte aligned");
}

// Device checks last, so the CPU registration runs every guard above them without a GPU.
void check_cuda(std::initializer_list<const at::Tensor*> ts, const char* op) {
  const at::Tensor* first = *ts.begin();
  for (const at::Tensor* t : ts) {
    TORCH_CHECK(t->is_cuda(), op, ": inputs must be CUDA tensors");
    TORCH_CHECK(t->get_device() == first->get_device(), op, ": inputs must be on one device");
  }
}

// x [m, k] fp16, rows contiguous; returns m
int64_t check_x(const at::Tensor& x, int64_t k, const char* op) {
  TORCH_CHECK(x.dim() == 2, op, ": x must be 2-D");
  TORCH_CHECK(x.scalar_type() == at::kHalf, op, ": x must be fp16");
  TORCH_CHECK(x.size(1) == k, op, ": x has ", x.size(1), " columns, the weight has k=", k);
  TORCH_CHECK(x.is_contiguous(), op, ": x must be contiguous");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, op, ": x must be 16-byte aligned");
  TORCH_CHECK(x.size(0) <= INT_MAX, op, ": x too large");
  return x.size(0);
}

// ---------------------------------------------------------------------------
// Warmup registry and the capture guard

using Key = std::tuple<int, int64_t, int64_t, int64_t, int, bool, int64_t>;  // device, k, n, tile, cb, fp32, rows
std::mutex g_mutex;
std::set<Key> g_warmed;
std::set<int> g_hgemm_warmed;

int64_t row_bucket(int64_t m) {
  int64_t b = 1;
  while (b < m && b < 16) b *= 2;
  return b;
}

Key gemm_key(int device, const at::Tensor& trellis, bool mcg, bool mul1, bool fp32, int64_t m) {
  return {device, trellis.size(0) * 16, trellis.size(1) * 16, trellis.size(2), mul1 ? 2 : mcg ? 1 : 0, fp32,
          row_bucket(m)};
}

bool capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
  return status != cudaStreamCaptureStatusNone;
}

// ---------------------------------------------------------------------------
// Ops

at::Tensor gemm_impl(const at::Tensor& x, const at::Tensor& trellis, const at::Tensor& suh, const at::Tensor& svh,
                     bool mcg, bool mul1, bool out_fp32, bool warming) {
  const char* op = "exl3_gemm";
  check_trellis(trellis, mcg, mul1, op);
  const int64_t k = trellis.size(0) * 16, n = trellis.size(1) * 16;
  const int64_t m = check_x(x, k, op);
  check_scale(suh, k, "suh", op);
  check_scale(svh, n, "svh", op);
  check_cuda({&x, &trellis, &suh, &svh}, op);

  const at::ScalarType out_dtype = out_fp32 ? at::kFloat : at::kHalf;
  if (m == 0) return at::empty({0, n}, x.options().dtype(out_dtype));
  const c10::cuda::CUDAGuard guard(x.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (!warming && capturing(stream)) {
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(g_warmed.count(gemm_key(x.get_device(), trellis, mcg, mul1, out_fp32, m)), op,
                ": shape k=", k, " n=", n, " rows=", m, " was not warmed up (exl3_warmup) before CUDA graph capture");
  }
  at::Tensor c = at::empty({m, n}, x.options().dtype(out_dtype));
  at::Tensor a_had = at::empty_like(x);
  exl3_gemm(x, trellis, c, suh, a_had, svh, -1, mcg, mul1, 0);
  return c;
}

at::Tensor exl3_gemm_op(const at::Tensor& x, const at::Tensor& trellis, const at::Tensor& suh,
                        const at::Tensor& svh, bool mcg, bool mul1, bool out_fp32) {
  return gemm_impl(x, trellis, suh, svh, mcg, mul1, out_fp32, false);
}

at::Tensor exl3_dequant_op(const at::Tensor& trellis, const at::Tensor& suh, const at::Tensor& svh, bool mcg,
                           bool mul1, int64_t n_start, int64_t n_count, bool had) {
  const char* op = "exl3_dequant";
  const float K = check_trellis(trellis, mcg, mul1, op);
  const int64_t k = trellis.size(0) * 16, n = trellis.size(1) * 16;
  check_scale(suh, k, "suh", op);
  check_scale(svh, n, "svh", op);
  TORCH_CHECK(n_start >= 0 && n_count > 0 && n_start % 128 == 0 && n_count % 128 == 0 && n_start + n_count <= n, op,
              ": columns ", n_start, "..", n_start + n_count, " must be a 128-aligned range inside n=", n);
  check_cuda({&trellis, &suh, &svh}, op);

  const c10::cuda::CUDAGuard guard(trellis.device());
  at::Tensor w = at::empty({k, n_count}, trellis.options().dtype(at::kHalf));
  if (had) {
    // as LinearEXL3.reconstruct_hgemm: svh offset by the caller, n_offset for the trellis
    reconstruct_had_slice(w, trellis, suh, svh.narrow(0, n_start, n_count), K, mcg, mul1, n_start);
  } else {
    reconstruct_slice(w, trellis, K, mcg, mul1, n_start);
  }
  return w;
}

at::Tensor exl3_had_r_128_op(const at::Tensor& x, const std::optional<at::Tensor>& pre_scale,
                             const std::optional<at::Tensor>& post_scale, double scale) {
  const char* op = "exl3_had_r_128";
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == at::kHalf && x.is_contiguous(), op,
              ": x must be contiguous 2-D fp16");
  TORCH_CHECK(x.size(1) % 128 == 0, op, ": columns must be a multiple of 128, got ", x.size(1));
  TORCH_CHECK(!(pre_scale && post_scale), op, ": at most one of pre_scale and post_scale");
  if (pre_scale) check_scale(*pre_scale, x.size(1), "pre_scale", op);
  if (post_scale) check_scale(*post_scale, x.size(1), "post_scale", op);
  check_cuda({&x}, op);
  if (pre_scale) check_cuda({&x, &*pre_scale}, op);
  if (post_scale) check_cuda({&x, &*post_scale}, op);

  const c10::cuda::CUDAGuard guard(x.device());
  at::Tensor y = at::empty_like(x);
  if (x.numel()) had_r_128(x, y, pre_scale, post_scale, (float)scale);
  return y;
}

at::Tensor hgemm_impl(const at::Tensor& a, const at::Tensor& b, bool warming) {
  const char* op = "exl3_hgemm";
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, op, ": a and b must be 2-D");
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf, op, ": a and b must be fp16");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), op, ": a and b must be contiguous");
  TORCH_CHECK(a.size(1) == b.size(0), op, ": a is ", a.sizes(), ", b is ", b.sizes());
  TORCH_CHECK(a.size(0) <= INT_MAX && a.size(1) <= INT_MAX && b.size(1) <= INT_MAX, op, ": too large");
  check_cuda({&a, &b}, op);

  const c10::cuda::CUDAGuard guard(a.device());
  if (!warming && capturing(at::cuda::getCurrentCUDAStream().stream())) {
    std::lock_guard<std::mutex> lock(g_mutex);
    TORCH_CHECK(g_hgemm_warmed.count(a.get_device()), op, ": not warmed up (exl3_warmup) before CUDA graph capture");
  }
  at::Tensor c = at::empty({a.size(0), b.size(1)}, a.options());
  if (c.numel()) hgemm_recon(a, b, c);
  return c;
}

at::Tensor exl3_hgemm_op(const at::Tensor& a, const at::Tensor& b) { return hgemm_impl(a, b, false); }

void exl3_warmup_op(const at::Tensor& trellis, const at::Tensor& suh, const at::Tensor& svh, bool mcg, bool mul1,
                    at::IntArrayRef rows, bool out_fp32) {
  const char* op = "exl3_warmup";
  check_trellis(trellis, mcg, mul1, op);
  const int64_t k = trellis.size(0) * 16, n = trellis.size(1) * 16;
  check_scale(suh, k, "suh", op);
  check_scale(svh, n, "svh", op);
  for (int64_t m : rows) TORCH_CHECK(m >= 1 && m <= INT_MAX, op, ": row counts must be >= 1, got ", m);
  check_cuda({&trellis, &suh, &svh}, op);

  const int device = trellis.get_device();
  const c10::cuda::CUDAGuard guard(trellis.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  TORCH_CHECK(!capturing(stream), op, ": must run outside CUDA graph capture");

  // DevCtx: SM count, compute capability, smem limit, the lock buffer (cudaMalloc + memset)
  // and the cuBLAS workspace (cudaMalloc) hgemm uses; then the f16acc rate probe (syncs).
  prepare_ctx(device);
  DevCtx::instance().get_ws(device);
  hgemm_f16acc_status(device);

  // One call per row count: autotune sessions (syncs, disk cache), kernel attributes,
  // GEMV occupancy lookups. x is zeros; outputs are discarded.
  for (int64_t m : rows) {
    at::Tensor x = at::zeros({m, k}, trellis.options().dtype(at::kHalf));
    gemm_impl(x, trellis, suh, svh, mcg, mul1, out_fp32, true);
  }
  // hgemm_recon at a size its f16acc kernel takes (hgemm_f16acc.cu worthwhile(): >= 384 rows,
  // >= one block per SM): the kernel's one-time attribute set (the >144-row route)
  {
    at::Tensor a = at::zeros({384, k}, trellis.options().dtype(at::kHalf));
    at::Tensor b = at::zeros({k, 4096}, trellis.options().dtype(at::kHalf));
    hgemm_impl(a, b, true);
  }
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::lock_guard<std::mutex> lock(g_mutex);
  for (int64_t m : rows) g_warmed.insert(gemm_key(device, trellis, mcg, mul1, out_fp32, m));
  g_hgemm_warmed.insert(device);
}

}  // namespace

TORCH_LIBRARY(_C_exl3, m) {
  m.def("exl3_gemm(Tensor x, Tensor trellis, Tensor suh, Tensor svh, bool mcg, bool mul1, bool out_fp32) -> Tensor");
  m.def("exl3_dequant(Tensor trellis, Tensor suh, Tensor svh, bool mcg, bool mul1, int n_start, int n_count, "
        "bool had) -> Tensor");
  m.def("exl3_had_r_128(Tensor x, Tensor? pre_scale, Tensor? post_scale, float scale) -> Tensor");
  m.def("exl3_hgemm(Tensor a, Tensor b) -> Tensor");
  m.def("exl3_warmup(Tensor trellis, Tensor suh, Tensor svh, bool mcg, bool mul1, int[] rows, bool out_fp32) -> ()");
}

TORCH_LIBRARY_IMPL(_C_exl3, CUDA, m) {
  m.impl("exl3_gemm", &exl3_gemm_op);
  m.impl("exl3_dequant", &exl3_dequant_op);
  m.impl("exl3_had_r_128", &exl3_had_r_128_op);
  m.impl("exl3_hgemm", &exl3_hgemm_op);
  m.impl("exl3_warmup", &exl3_warmup_op);
}

// CPU: guards only (a valid call ends in "must be CUDA tensors"); lets the guard tests run
// on a machine without a GPU.
TORCH_LIBRARY_IMPL(_C_exl3, CPU, m) {
  m.impl("exl3_gemm", &exl3_gemm_op);
  m.impl("exl3_dequant", &exl3_dequant_op);
  m.impl("exl3_had_r_128", &exl3_had_r_128_op);
  m.impl("exl3_hgemm", &exl3_hgemm_op);
  m.impl("exl3_warmup", &exl3_warmup_op);
}

static struct PyModuleDef _module_def = {
    PyModuleDef_HEAD_INIT, "_C_exl3", nullptr, -1, nullptr,
};

extern "C" PyObject* PyInit__C_exl3(void) { return PyModule_Create(&_module_def); }
