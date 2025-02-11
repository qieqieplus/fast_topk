#ifndef INTERNAL_GPU_COMMON_HPP_
#define INTERNAL_GPU_COMMON_HPP_

#include "internal/common.hpp"
#include <cuda_runtime.h>

namespace internal::gpu {

// Constants
constexpr size_t kVectorizedReadSize = 16; // Size for vectorized reads in bytes
constexpr int kWarpSize = 32;             // Standard warp size for CUDA GPUs

// Macros for Host/Device Attributes
#ifdef __CUDACC__
#define FAST_TOPK_HOST __host__
#define FAST_TOPK_DEVICE __device__
#else
#define FAST_TOPK_HOST
#define FAST_TOPK_DEVICE
#endif

#define FAST_TOPK_HOST_DEVICE FAST_TOPK_HOST FAST_TOPK_DEVICE
#define FAST_TOPK_DEVICE_INLINE FAST_TOPK_DEVICE FAST_TOPK_INLINE

#define FAST_TOPK_HIDDEN __attribute__((visibility("hidden")))

// Kernel Declaration Macro
#if defined(__CUDACC_RDC__)
#define FAST_TOPK_KERNEL FAST_TOPK_HIDDEN __global__ void
#elif defined(__CUDACC__)
#define FAST_TOPK_KERNEL static __global__ void
#else
#define FAST_TOPK_KERNEL static void
#endif

/**
 * @brief Checks CUDA API call status and throws error if the call failed.
 *
 * This macro checks the return status of a CUDA runtime API call. If the call
 * returns anything other than `cudaSuccess`, it prints an error message to
 * stderr including file name, line number, and the CUDA error string,
 * then aborts execution.
 *
 * Usage:
 * ```cpp
 * INTERNAL_CUDA_CHECK(cudaMalloc(&device_ptr, size));
 * ```
 *
 * @param call CUDA runtime API call to be checked.
 */
#define INTERNAL_CUDA_CHECK(call)                                                \
  do {                                                                           \
    cudaError_t error = (call);                                                  \
    if (error != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error (%s:%d): %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(error));                                        \
      std::abort();                                                               \
    }                                                                            \
  } while (0)

} // namespace internal::gpu

#endif // INTERNAL_GPU_COMMON_HPP_