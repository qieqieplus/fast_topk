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

/**
 * @brief Selects the top-k smallest or largest elements from a batch of rows using radix sort.
 *
 * This is the main interface function for the radix-based top-k selection. It dispatches
 * to optimized kernel implementations based on problem size and configuration. It efficiently
 * finds the top-k elements for each row in a batch of data, leveraging GPU parallelism.
 *
 * @tparam T Data type of the elements to select from.
 * @tparam IdxT Index type for element indexing.
 * @tparam BitsPerPass Number of bits to process in each radix sort pass.
 * @tparam BlockSize CUDA block size for kernel launch.
 *
 * @param in Device memory pointer to the input array of values (row-major batch).
 * @param in_idx Device memory pointer to the input index array (optional, row-major batch).
 * @param batch_size Number of rows in the input batch.
 * @param len Length of each row (number of columns in the batch).
 * @param k Number of top elements to select from each row.
 * @param out Device memory pointer to the output array for top-k values.
 * @param out_idx Device memory pointer to the output array for top-k indices.
 * @param select_min Boolean flag to select smallest (true) or largest (false) top-k elements.
 * @param fused_last_filter Boolean flag to enable fused last filter optimization.
 * @param stream CUDA stream for asynchronous kernel execution.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void SelectK(const T* in, const IdxT* in_idx, int batch_size, IdxT len, IdxT k, T* out,
             IdxT* out_idx, bool select_min, bool fused_last_filter, cudaStream_t stream); // Removed mr parameter (not used)


template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void SelectK(const T* in, const IdxT* in_idx, int batch_size, IdxT len, IdxT k, T* out,
             IdxT* out_idx, bool select_min, bool fused_last_filter, cudaStream_t stream) // Removed mr parameter (not used)
{
  if (k == len) {
    // Optimization for k == len: simply copy all input to output
    INTERNAL_CUDA_CHECK(cudaMemcpyAsync(out, in, sizeof(T) * batch_size * len, cudaMemcpyDeviceToDevice, stream));
    if (in_idx) {
      INTERNAL_CUDA_CHECK(cudaMemcpyAsync(
          out_idx, in_idx, sizeof(IdxT) * batch_size * len, cudaMemcpyDeviceToDevice, stream));
    } else {
      // Generate identity indices if input indices are not provided
      thrust::device_vector<IdxT> out_idx_thrust(out_idx, out_idx + static_cast<size_t>(len) * batch_size);
      thrust::transform(thrust::make_counting_iterator<IdxT>(0), thrust::make_counting_iterator<IdxT>(static_cast<size_t>(len) * batch_size), thrust::raw_pointer_cast(out_idx_thrust.data()),
                         [len] FAST_TOPK_DEVICE (IdxT i){ return i % len; });
    }
    return; // Early return for k == len case
  }

  int sm_count;
  {
    // Fetch device attribute: number of multiprocessors
    int device_id;
    INTERNAL_CUDA_CHECK(cudaGetDevice(&device_id));
    INTERNAL_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  }

  constexpr int items_per_thread = 32; // Define items per thread for kernel dispatch

  // Dispatch to appropriate kernel based on input length
  if (len <= BlockSize * items_per_thread) {
    // Use one-block-per-row kernel for smaller inputs
    impl::RadixTopKOneBlock<T, IdxT, BitsPerPass, BlockSize>(
        in, in_idx, batch_size, len, k, out, out_idx, select_min, sm_count, stream); // Removed mr
  } else {
    // Use multi-block kernel for larger inputs
    unsigned grid_dim =
        impl::CalculateGridDim<T, IdxT, BitsPerPass, BlockSize>(batch_size, len, sm_count);
    if (grid_dim == 1) {
      // Fallback to one-block kernel if grid dimension is 1
      impl::RadixTopKOneBlock<T, IdxT, BitsPerPass, BlockSize>(
          in, in_idx, batch_size, len, k, out, out_idx, select_min, sm_count, stream); // Removed mr
    } else {
      // Use standard multi-block radix top-k kernel
      impl::RadixTopK<T, IdxT, BitsPerPass, BlockSize>(in,
                                                        in_idx,
                                                        batch_size,
                                                        len,
                                                        k,
                                                        out,
                                                        out_idx,
                                                        select_min,
                                                        fused_last_filter,
                                                        grid_dim,
                                                        sm_count,
                                                        stream); // Removed mr
    }
  }
}

} // namespace internal::gpu

#endif // INTERNAL_GPU_SELECT_TOPK_HPP_