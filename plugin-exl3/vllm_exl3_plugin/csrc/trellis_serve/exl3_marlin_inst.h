// One instantiation unit: TRELLIS_INST_MB=<0..4>, TRELLIS_INST_CB=<0..2>, TRELLIS_INST_KB=<3|4|5|6> (see setup.py; 3 = K3, 5 = K5).
#define MARLIN_NAMESPACE_NAME trellis_exl3_marlin
#define TRELLIS_KERNEL_DEFINED
#include "exl3_marlin_template.h"
#include "exl3_marlin_kernels.h"

namespace MARLIN_NAMESPACE_NAME {
#define TRELLIS_INST(threads, tn, tk, mb, cb, kb) \
  template __global__ void TRELLIS_KERNEL(threads, tn, tk, mb, cb, kb)(TRELLIS_KERNEL_PARAMS);
TRELLIS_THREAD_CFGS(TRELLIS_INST, TRELLIS_INST_MB, TRELLIS_INST_CB, TRELLIS_INST_KB)
}  // namespace MARLIN_NAMESPACE_NAME
