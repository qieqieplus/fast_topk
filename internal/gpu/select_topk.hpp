#ifndef INTERNAL_GPU_SELECT_TOPK_HPP_
#define INTERNAL_GPU_SELECT_TOPK_HPP_

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <algorithm>

#include "internal/gpu/common.hpp"
#include "internal/gpu/bits.cuh"
#include "internal/gpu/util.cuh"
#include "internal/gpu/radix_kernels.cuh" // Include kernels here as select_k will launch them

namespace internal::gpu {

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void select_k(const T* in,
              const IdxT* in_idx,
              int batch_size,
              IdxT len,
              IdxT k,
              T* out,
              IdxT* out_idx,
              bool select_min,
              bool fused_last_filter,
              cudaStream_t stream); // Removed mr parameter


template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void select_k(const T* in,
              const IdxT* in_idx,
              int batch_size,
              IdxT len,
              IdxT k,
              T* out,
              IdxT* out_idx,
              bool select_min,
              bool fused_last_filter,
              cudaStream_t stream) // Removed mr parameter
{
  if (k == len) {
    INTERNAL_CUDA_CHECK(
      cudaMemcpyAsync(out, in, sizeof(T) * batch_size * len, cudaMemcpyDeviceToDevice, stream));
    if (in_idx) {
      INTERNAL_CUDA_CHECK(cudaMemcpyAsync(
        out_idx, in_idx, sizeof(IdxT) * batch_size * len, cudaMemcpyDeviceToDevice, stream));
    } else {
      thrust::device_vector<IdxT> out_idx_thrust(out_idx, out_idx + static_cast<size_t>(len) * batch_size);
      thrust::transform(thrust::make_counting_iterator<IdxT>(0), thrust::make_counting_iterator<IdxT>(static_cast<size_t>(len) * batch_size), thrust::raw_pointer_cast(out_idx_thrust.data()),
                         [len] FAST_TOPK_DEVICE (IdxT i){ return i % len; });
    }
    return;
  }

  int sm_cnt;
  {
    int dev;
    INTERNAL_CUDA_CHECK(cudaGetDevice(&dev));
    INTERNAL_CUDA_CHECK(cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev));
  }

  constexpr int items_per_thread = 32;

  if (len <= BlockSize * items_per_thread) {
    impl::radix_topk_one_block<T, IdxT, BitsPerPass, BlockSize>(
      in, in_idx, batch_size, len, k, out, out_idx, select_min, sm_cnt, stream); // Removed mr
  } else {
    unsigned grid_dim =
      impl::calc_grid_dim<T, IdxT, BitsPerPass, BlockSize>(batch_size, len, sm_cnt);
    if (grid_dim == 1) {
      impl::radix_topk_one_block<T, IdxT, BitsPerPass, BlockSize>(
        in, in_idx, batch_size, len, k, out, out_idx, select_min, sm_cnt, stream); // Removed mr
    } else {
      impl::radix_topk<T, IdxT, BitsPerPass, BlockSize>(in,
                                                        in_idx,
                                                        batch_size,
                                                        len,
                                                        k,
                                                        out,
                                                        out_idx,
                                                        select_min,
                                                        fused_last_filter,
                                                        grid_dim,
                                                        sm_cnt,
                                                        stream); // Removed mr
    }
  }
}

} // namespace internal::gpu

#endif // INTERNAL_GPU_SELECT_TOPK_HPP_