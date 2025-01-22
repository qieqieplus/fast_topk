#ifndef INTERNAL_GPU_MACROS_H
#define INTERNAL_GPU_MACROS_H

#include "internal/common.hpp"
#include <cuda_runtime.h>


constexpr size_t VECTORIZED_READ_SIZE = 16;
constexpr int WarpSize = 32;


#ifdef __CUDACC__

#define FAST_TOPK_HOST __host__
#define FAST_TOPK_DEVICE __device__
#else

#define FAST_TOPK_HOST
#define FAST_TOPK_DEVICE

#endif

#define FAST_TOPK_HOST_DEVICE FAST_TOPK_HOST FAST_TOPK_DEVICE
#define FATS_TOPK_DEVICE_INLINE FAST_TOPK_DEVICE FAST_TOPK_INLINE

#define FAST_TOPK_HIDDEN __attribute__((visibility("hidden")))

#if defined(__CUDACC_RDC__)
#define FAST_TOPK_KERNEL FAST_TOPK_HIDDEN __global__ void
#elif defined(__CUDACC__)
#define FAST_TOPK_KERNEL static __global__ void
#else
#define FAST_TOPK_KERNEL static void
#endif


#define INTERNAL_CUDA_CHECK(call)                    \
  do {                                      \
    cudaError_t err__ = (call);             \
    if (err__ != cudaSuccess) {             \
      fprintf(stderr,                       \
              "CUDA Error (%s:%d): %s\n",   \
              __FILE__,                     \
              __LINE__,                     \
              cudaGetErrorString(err__));   \
      std::abort();                         \
    }                                       \
  } while (0)


#endif // INTERNAL_GPU_MACROS_H

