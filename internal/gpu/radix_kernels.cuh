#ifndef INTERNAL_GPU_RADIX_KERNELS_CUH_
#define INTERNAL_GPU_RADIX_KERNELS_CUH_

#include "internal/gpu/common.hpp"
#include "internal/gpu/bits.cuh"
#include "internal/gpu/util.cuh"
#include "internal/gpu/vectorize.cuh"
#include <cub/cub.cuh>
#include <thrust/device_vector.h>

namespace internal::gpu::impl {

// Moved Counter struct here - keep it in impl namespace as it's kernel internal
/**
 * @brief Counter structure for radix top-k algorithm.
 *
 * Manages counters and state for each batch row during the radix top-k process.
 * Used in shared memory to coordinate and track progress within kernels.
 *
 * @tparam T Data type of the elements being sorted.
 * @tparam IdxT Index type for array indexing.
 */
template <typename T, typename IdxT>
struct alignas(128) Counter {
  // Public members
  IdxT k;                     // Current k-value for top-k selection
  IdxT length;                // Length of input in current pass
  IdxT previous_length;      // Length of input in previous pass

  // Bits of the k-th value known so far
  typename cub::Traits<T>::UnsignedBits kth_value_bits;

  IdxT filter_count;         // Count of elements passed in filtering
  unsigned int finished_block_count; // Count of blocks finished processing
  IdxT output_count;          // Count of elements in the output (top-k results)
  IdxT output_back_count;     // Count of elements equal to k-th value, from back

  // Default constructor to initialize counters
  FAST_TOPK_DEVICE Counter()
      : k(0),
        length(0),
        previous_length(0),
        kth_value_bits(0),
        filter_count(0),
        finished_block_count(0),
        output_count(0),
        output_back_count(0) {}
};

/**
 * @brief Kernel function to filter and compute histogram in radix sort.
 *
 * Fuses filtering of the current pass and histogram building for the next pass
 * to optimize memory access. Handles early stopping to avoid unnecessary work.
 */
template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_DEVICE void FilterAndHistogram(const T* in_buf, const IdxT* in_idx_buf, T* out_buf,
                                            IdxT* out_idx_buf, T* out, IdxT* out_idx,
                                            IdxT previous_len, Counter<T, IdxT>* counter,
                                            IdxT* histogram, bool select_min, int pass,
                                            bool early_stop) {
  constexpr int num_buckets = CalculateNumBuckets<BitsPerPass>();
  __shared__ IdxT histogram_shared_memory[num_buckets]; // Shared memory for histogram

  // Initialize shared histogram to zero at the beginning of each kernel invocation
  for (IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram_shared_memory[i] = 0;
  }
  __syncthreads();

  const int start_bit = CalculateStartBit<T, BitsPerPass>(pass);
  const unsigned mask = CalculateMask<T, BitsPerPass>(pass);

  if (pass == 0) {
    // Pass 0: Compute histogram for the first pass
    auto histogram_func = [select_min, start_bit, mask](T value, IdxT) {
      int bucket = CalculateBucket<T, BitsPerPass>(value, start_bit, mask, select_min);
      atomicAdd(histogram_shared_memory + bucket, static_cast<IdxT>(1)); // Atomic increment histogram bin
    };
    VectorizedProcess(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                        static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len,
                        histogram_func);
  } else {
    // Subsequent passes: Filter elements and compute histogram for candidates
    IdxT* filter_count_ptr = &counter->filter_count;
    IdxT* output_count_ptr = &counter->output_count;
    const auto kth_value_bits = counter->kth_value_bits;
    const int previous_start_bit = CalculateStartBit<T, BitsPerPass>(pass - 1);

    auto filter_histogram_func =
        [&](T value, IdxT i) {
          const auto previous_bits =
              (TwiddleIn(value, select_min) >> previous_start_bit) << previous_start_bit;
          if (previous_bits == kth_value_bits) {
            // Elements in the k-th bucket: Candidates for the next pass
            if (early_stop) {
              // Early stop condition: Directly write to output arrays
              IdxT pos = atomicAdd(output_count_ptr, static_cast<IdxT>(1)); // Atomic increment output count
              out[pos] = value;
              out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
            } else {
              if (out_buf) {
                // Write candidate elements to buffer for next pass
                IdxT pos = atomicAdd(filter_count_ptr, static_cast<IdxT>(1)); // Atomic increment filter count
                out_buf[pos] = value;
                out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
              }
              // Compute histogram for elements in the k-th bucket
              int bucket = CalculateBucket<T, BitsPerPass>(value, start_bit, mask, select_min);
              atomicAdd(histogram_shared_memory + bucket, static_cast<IdxT>(1)); // Atomic increment histogram bin
            }
          } else if ((out_buf || early_stop) && previous_bits < kth_value_bits) {
            // Elements smaller than k-th bucket: Part of top-k results
            IdxT pos = atomicAdd(output_count_ptr, static_cast<IdxT>(1)); // Atomic increment output count
            out[pos] = value;
            out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
          }
        }; // End of lambda

    VectorizedProcess(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                        static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len,
                        filter_histogram_func);
  }

  if (early_stop) {
    return; // Early exit if early stopping is triggered
  }
  __syncthreads();

  // Merge histograms from all blocks in the grid by atomic addition
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    if (histogram_shared_memory[i] != 0) {
      atomicAdd(histogram + i, histogram_shared_memory[i]); // Atomic add to global histogram
    }
  }
}

/**
 * @brief Kernel function to calculate inclusive prefix sum of a histogram.
 *
 * Uses CUB's BlockScan for efficient parallel prefix sum calculation within each block.
 * Handles both large and small histograms using block-level or warp-level scans.
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
FAST_TOPK_DEVICE void ScanHistogram(volatile IdxT* histogram) {
  constexpr int num_buckets = CalculateNumBuckets<BitsPerPass>();
  if constexpr (num_buckets >= BlockSize) {
    // Block-level scan for histograms larger than block size
    static_assert(num_buckets % BlockSize == 0, "Number of buckets must be divisible by block size");
    constexpr int items_per_thread = num_buckets / BlockSize;
    using BlockLoadType = cub::BlockLoad<IdxT, BlockSize, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE>;
    using BlockStoreType = cub::BlockStore<IdxT, BlockSize, items_per_thread, cub::BLOCK_STORE_TRANSPOSE>;
    using BlockScanType = cub::BlockScan<IdxT, BlockSize>;

    __shared__ union {
      typename BlockLoadType::TempStorage load;
      typename BlockScanType::TempStorage scan;
      typename BlockStoreType::TempStorage store;
    } temp_storage; // Shared memory for BlockScan operations
    IdxT thread_data[items_per_thread];

    // Load histogram data into shared memory
    BlockLoadType(temp_storage.load).Load(histogram, thread_data);
    __syncthreads();

    // Calculate inclusive prefix sum using BlockScan
    BlockScanType(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    // Store results back to histogram array
    BlockStoreType(temp_storage.store).Store(histogram, thread_data);
  } else {
    // Warp-level scan for smaller histograms (less than block size)
    using BlockScanType = cub::BlockScan<IdxT, BlockSize>;
    __shared__ typename BlockScanType::TempStorage temp_storage; // Shared memory for BlockScan

    IdxT thread_data = 0;
    if (threadIdx.x < num_buckets) {
      thread_data = histogram[threadIdx.x]; // Load histogram value into thread data
    }

    // Calculate inclusive prefix sum at warp level
    BlockScanType(temp_storage).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    if (threadIdx.x < num_buckets) {
      histogram[threadIdx.x] = thread_data; // Write back scanned value to histogram
    }
  }
}

/**
 * @brief Kernel function to choose the bucket for the k-th element.
 *
 * Determines the bucket index where the k-th smallest/largest element resides
 * by checking the prefix sum histogram. Updates the counter with relevant info.
 */
template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_DEVICE void ChooseBucket(Counter<T, IdxT>* counter, const IdxT* histogram, const IdxT k,
                                     const int pass) {
  constexpr int num_buckets = CalculateNumBuckets<BitsPerPass>();
  // Iterate through each bucket to find the bucket containing the k-th element
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    IdxT prev_count = (i == 0) ? 0 : histogram[i - 1]; // Count of elements in previous buckets
    IdxT current_count = histogram[i];                 // Count of elements up to current bucket

    // one and only one thread will satisfy this condition, so counter is written by only one thread
    if (prev_count < k && current_count >= k) {
      counter->k = k - prev_count;             // Update remaining k for next pass
      counter->length = current_count - prev_count; // Set length of candidates for next pass
      typename cub::Traits<T>::UnsignedBits bucket_bits = i; // Bucket index
      int start_bit = CalculateStartBit<T, BitsPerPass>(pass);
      counter->kth_value_bits |= bucket_bits << start_bit; // Store bits of k-th value
    }
  }
}

/**
 * @brief Kernel function for last filter pass to collect top-k elements.
 *
 * Performs final filtering to gather top-k elements based on the k-th value's bits
 * determined in previous passes. Writes results to the output arrays.
 */
template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_DEVICE void LastFilter(const T* in_buf, const IdxT* in_idx_buf, T* out, IdxT* out_idx,
                                   IdxT current_len, IdxT k, Counter<T, IdxT>* counter,
                                   const bool select_min, const int pass) {
  const auto kth_value_bits = counter->kth_value_bits; // Bits of the k-th value
  const int start_bit = CalculateStartBit<T, BitsPerPass>(pass); // Start bit for current pass

  const IdxT num_kth_needed = counter->k;           // Number of k-th elements needed
  IdxT* output_count_ptr = &counter->output_count;     // Pointer to output count
  IdxT* output_back_count_ptr = &counter->output_back_count; // Pointer to output back count

  // Iterate through candidate elements to perform final filtering
  for (IdxT i = threadIdx.x; i < current_len; i += blockDim.x) {
    const T value = in_buf[i];
    const auto bits = (TwiddleIn(value, select_min) >> start_bit) << start_bit; // Extract relevant bits

    if (bits < kth_value_bits) {
      // Elements smaller than k-th value: part of the final top-k results
      IdxT pos = atomicAdd(output_count_ptr, static_cast<IdxT>(1)); // Atomic increment output count
      out[pos] = value;
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      // Elements equal to k-th value: also part of the final top-k results, up to k
      IdxT back_pos = atomicAdd(output_back_count_ptr, static_cast<IdxT>(1)); // Atomic increment back count
      if (back_pos < num_kth_needed) {
        IdxT pos = k - 1 - back_pos; // Position from the back to fill k-th elements
        out[pos] = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

/**
 * @brief Standalone kernel for the last filtering stage of radix top-k.
 *
 * Launched as a separate kernel to handle the final filtering pass, especially
 * when not fused with the main radix kernel. Optimizes for cases where
 * last-pass filtering is computationally intensive.
 */
template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_KERNEL LastFilterKernel(const T* in, const IdxT* in_idx, const T* in_buf,
                                     const IdxT* in_idx_buf, T* out, IdxT* out_idx, const IdxT len,
                                     const IdxT k, Counter<T, IdxT>* counters,
                                     const bool select_min) {
  const size_t batch_id = blockIdx.y; // Batch ID from blockIdx.y

  Counter<T, IdxT>* counter = counters + batch_id;
  IdxT previous_len = counter->previous_length;
  if (previous_len == 0) {
    return; // Early return if no elements to filter
  }
  const IdxT buf_len = CalculateBufferLength<T>(len);

  // Determine input buffer based on length and previous pass buffer usage
  if (previous_len > buf_len || in_buf == in) {
    in_buf = in + batch_id * len; // Use original input if buffer was skipped or too large
    in_idx_buf = in_idx ? (in_idx + batch_id * len) : nullptr;
    previous_len = len;
  } else {
    in_buf += batch_id * buf_len; // Use temporary buffer from previous pass
    in_idx_buf += batch_id * buf_len;
  }
  out += batch_id * k;
  out_idx += batch_id * k;

  constexpr int pass = CalculateNumPasses<T, BitsPerPass>() - 1; // Last pass index
  constexpr int start_bit = CalculateStartBit<T, BitsPerPass>(pass); // Start bit for last pass

  const auto kth_value_bits = counter->kth_value_bits; // K-th value bits from counter
  const IdxT num_kth_needed = counter->k;
  IdxT* output_count_ptr = &counter->output_count;
  IdxT* output_back_count_ptr = &counter->output_back_count;

  auto last_pass_filter_func = [&](T value, IdxT i) {
    const auto bits = (TwiddleIn(value, select_min) >> start_bit) << start_bit; // Extract last pass bits
    if (bits < kth_value_bits) {
      // Elements less than k-th value: add to top-k results
      IdxT pos = atomicAdd(output_count_ptr, static_cast<IdxT>(1)); // Atomic increment output count
      out[pos] = value;
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      // Elements equal to k-th value: conditionally add to top-k results
      IdxT back_pos = atomicAdd(output_back_count_ptr, static_cast<IdxT>(1)); // Atomic increment back count
      if (back_pos < num_kth_needed) {
        IdxT pos = k - 1 - back_pos;
        out[pos] = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  };

  VectorizedProcess(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                     static_cast<size_t>(blockDim.x) * gridDim.x, in_buf, previous_len,
                     last_pass_filter_func);
}

/**
 * @brief Main radix top-k kernel function (multi-block per batch row).
 *
 * Orchestrates multiple passes of radix sort to find top-k elements. Fuses
 * histogramming and filtering for efficiency. Manages buffer switching and early stopping.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize, bool FusedLastFilter>
FAST_TOPK_KERNEL RadixKernel(const T* in, const IdxT* in_idx, const T* in_buf,
                               const IdxT* in_idx_buf, T* out_buf, IdxT* out_idx_buf, T* out,
                               IdxT* out_idx, Counter<T, IdxT>* counters, IdxT* histograms,
                               const IdxT len, const IdxT k, const bool select_min,
                               const int pass) {
  const size_t batch_id = blockIdx.y; // Batch ID from block index
  auto counter          = counters + batch_id; // Counter for this batch row
  IdxT current_k;
  IdxT previous_len;
  IdxT current_len;

  // Initialize pass-specific variables from counter or input parameters
  if (pass == 0) {
    current_k = k;
    previous_len = len;
    current_len = len;
  } else {
    current_k = counter->k;
    current_len = counter->length;
    previous_len = counter->previous_length;
  }
  if (current_len == 0) {
    return; // Early return if no candidates left
  }

  const bool early_stop = (current_len == current_k); // Check for early stopping condition
  const IdxT buf_len = CalculateBufferLength<T>(len);   // Calculate buffer length for current row

  // Determine input buffer pointers based on pass and buffer length
  if (pass == 0 || pass == 1 || previous_len > buf_len) {
    in_buf = in + batch_id * len; // Use original input for first passes or if skipping buffer
    in_idx_buf = in_idx ? (in_idx + batch_id * len) : nullptr;
    previous_len = len;
  } else {
    in_buf += batch_id * buf_len; // Use temporary buffer from previous pass
    in_idx_buf += batch_id * buf_len;
  }
  // Determine output buffer pointers based on pass and candidate length
  if (pass == 0 || current_len > buf_len) {
    out_buf = nullptr; // Skip writing to buffer for first pass or if too many candidates
    out_idx_buf = nullptr;
  } else {
    out_buf += batch_id * buf_len; // Use temporary buffer for output candidates
    out_idx_buf += batch_id * buf_len;
  }
  out += batch_id * k;
  out_idx += batch_id * k;

  IdxT* histogram = histograms + batch_id * CalculateNumBuckets<BitsPerPass>(); // Histogram in device memory

  // Execute fused filter and histogram kernel
  FilterAndHistogram<T, IdxT, BitsPerPass>(in_buf, in_idx_buf, out_buf, out_idx_buf, out,
                                             out_idx, previous_len, counter, histogram,
                                             select_min, pass, early_stop);
  __threadfence(); // Ensure histogram is fully computed before scanning

  bool isLastBlock = false;
  if (threadIdx.x == 0) {
    unsigned int finished = atomicInc(&counter->finished_block_count, gridDim.x - 1); // Corrected to atomicInc
    isLastBlock = (finished == (gridDim.x - 1)); // Check if this is the last block to finish
  }

  // Only last block in grid performs histogram scan and bucket choice
  if (__syncthreads_or(isLastBlock)) {
    if (early_stop) {
      // Early stopping: reset counter lengths and return
      if (threadIdx.x == 0) {
        counter->previous_length = 0;
        counter->length = 0;
      }
      return;
    }

    ScanHistogram<IdxT, BitsPerPass, BlockSize>(histogram); // Calculate prefix sum of histogram
    __syncthreads();
    ChooseBucket<T, IdxT, BitsPerPass>(counter, histogram, current_k, pass); // Choose bucket for k-th element
    __syncthreads();

    constexpr int num_passes = CalculateNumPasses<T, BitsPerPass>();
    constexpr int num_buckets = CalculateNumBuckets<BitsPerPass>();
    // Reset histogram for next pass, except for the last one
    if (pass != num_passes - 1) {
      for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        histogram[i] = 0; // Reset histogram bins to zero
      }
    }
    if (threadIdx.x == 0) {
      counter->previous_length = current_len; // Update previous length for next iteration
      counter->filter_count = 0;          // Reset filter count
    }

    // Execute last filter if fused and if it's the last pass
    if constexpr (FusedLastFilter) {
      if (pass == num_passes - 1) {
        LastFilter<T, IdxT, BitsPerPass>(
            out_buf ? out_buf : in_buf, out_idx_buf ? out_idx_buf : in_idx_buf, out, out_idx,
            out_buf ? current_len : len, k, counter, select_min, pass);
      }
    }
  }
}

/**
 * @brief Calculates chunk size for kernel execution.
 */
template <typename T, typename IdxT, int BlockSize, typename Kernel>
int CalculateChunkSize(int batch_size, IdxT len, int sm_cnt, Kernel kernel, bool one_block) {
  int active_blocks;
  INTERNAL_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, reinterpret_cast<void*>(kernel), BlockSize, 0));

  constexpr int num_waves = 10; // Number of waves to saturate GPU
  int chunk_size;
  if (one_block) {
    chunk_size = num_waves * sm_cnt * active_blocks; // Chunk size for one-block kernels
  } else {
    constexpr int items_per_thread = 32; // Items per thread for multi-block kernels
    chunk_size =
        std::max<int>(1, num_waves * sm_cnt * active_blocks * BlockSize * items_per_thread / len);
  }
  return std::min(chunk_size, batch_size); // Ensure chunk size does not exceed batch size
}

/**
 * @brief Calculates grid dimension for kernel launch.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
unsigned CalculateGridDim(int batch_size, IdxT len, int sm_cnt) {
  static_assert(kVectorizedReadSize / sizeof(T) >= 1,
                "Vectorized read size must not be smaller than element size.");

  int active_blocks;
  INTERNAL_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, reinterpret_cast<void*>(RadixKernel<T, IdxT, BitsPerPass, BlockSize, false>), BlockSize, 0));
  active_blocks *= sm_cnt; // Total active blocks across all SMs

  IdxT best_num_blocks = 0;
  float best_tail_wave_penalty = 1.0f;
  const IdxT max_num_blocks = CeilingDivide<IdxT>(len, kVectorizedReadSize / sizeof(T) * BlockSize);

  // Iterate through possible number of blocks to minimize tail wave penalty
  for (int num_waves = 1;; ++num_waves) {
    IdxT num_blocks = std::min(
        max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
    IdxT items_per_thread = CeilingDivide<IdxT>(len, num_blocks * BlockSize);
    items_per_thread = AlignUp<IdxT>(items_per_thread, kVectorizedReadSize / sizeof(T));
    num_blocks = CeilingDivide<IdxT>(len, items_per_thread * BlockSize);
    float actual_num_waves = static_cast<float>(num_blocks) * batch_size / active_blocks;
    float tail_wave_penalty =
        (ceilf(actual_num_waves) - actual_num_waves) / ceilf(actual_num_waves);

    if (tail_wave_penalty < 0.15f) {
      best_num_blocks = num_blocks; // Accept if tail wave penalty is low enough
      break;
    } else if (tail_wave_penalty < best_tail_wave_penalty) {
      best_num_blocks = num_blocks;        // Update best blocks if penalty is better
      best_tail_wave_penalty = tail_wave_penalty;
    }

    if (num_blocks == max_num_blocks) {
      break; // Stop if max blocks are reached
    }
  }
  return best_num_blocks; // Return the number of blocks with the best tail wave penalty
}


/**
 * @brief Sets buffer pointers for different radix passes (Host version).
 */
template <typename T, typename IdxT>
FAST_TOPK_HOST void SetBufferPointers(const T* in, const IdxT* in_idx, T* buf1, IdxT* idx_buf1, T* buf2,
                                       IdxT* idx_buf2, int pass, const T*& in_buf,
                                       const IdxT*& in_idx_buf, T*& out_buf,
                                       IdxT*& out_idx_buf) {
  if (pass == 0) {
    in_buf      = in;
    in_idx_buf  = nullptr;
    out_buf     = nullptr;
    out_idx_buf = nullptr;
  } else if (pass == 1) {
    in_buf      = in;
    in_idx_buf  = in_idx;
    out_buf     = buf1;
    out_idx_buf = idx_buf1;
  } else if (pass % 2 == 0) {
    in_buf      = buf1;
    in_idx_buf  = idx_buf1;
    out_buf     = buf2;
    out_idx_buf = idx_buf2;
  } else {
    in_buf      = buf2;
    in_idx_buf  = idx_buf2;
    out_buf     = buf1;
    out_idx_buf = idx_buf1;
  }
}

/**
 * @brief Sets buffer pointers for different radix passes (Device version).
 */
template <typename T, typename IdxT>
FAST_TOPK_DEVICE void SetBufferPointers(const T* in, const IdxT* in_idx, char* bufs, IdxT buf_len,
                                         int pass, const T*& in_buf, const IdxT*& in_idx_buf,
                                         T*& out_buf, IdxT*& out_idx_buf) {
  // bufs layout: [buf1][buf2][idx_buf1][idx_buf2]
  if (pass == 0) {
    in_buf      = in;
    in_idx_buf  = nullptr;
    out_buf     = nullptr;
    out_idx_buf = nullptr;
  } else if (pass == 1) {
    in_buf      = in;
    in_idx_buf  = in_idx;
    out_buf     = reinterpret_cast<T*>(bufs);
    out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
  } else if (pass % 2 == 0) {
    in_buf      = reinterpret_cast<T*>(bufs);
    in_idx_buf  = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
    out_buf     = const_cast<T*>(in_buf + buf_len);
    out_idx_buf = const_cast<IdxT*>(in_idx_buf + buf_len);
  } else {
    out_buf     = reinterpret_cast<T*>(bufs);
    out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
    in_buf      = out_buf + buf_len;
    in_idx_buf  = out_idx_buf + buf_len;
  }
}

/**
 * @brief High-level function to execute multi-block radix top-k sort.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void RadixTopK(const T* in, const IdxT* in_idx, int batch_size, IdxT len, IdxT k, T* out,
                IdxT* out_idx, bool select_min, bool fused_last_filter, unsigned grid_dim,
                int sm_cnt, cudaStream_t stream) {
  static_assert(CalculateNumPasses<T, BitsPerPass>() > 1);
  constexpr int num_buckets = CalculateNumBuckets<BitsPerPass>();

  auto kernel = RadixKernel<T, IdxT, BitsPerPass, BlockSize, false>;
  const size_t max_chunk_size =
    CalculateChunkSize<T, IdxT, BlockSize>(batch_size, len, sm_cnt, kernel, false);
  if (max_chunk_size != static_cast<size_t>(batch_size)) {
    grid_dim = CalculateGridDim<T, IdxT, BitsPerPass, BlockSize>(max_chunk_size, len, sm_cnt);
  }
  const IdxT buf_len = CalculateBufferLength<T>(len);

  size_t req_buf = max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  thrust::device_vector<Counter<T, IdxT>> counters(max_chunk_size); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<IdxT> histograms(max_chunk_size * num_buckets); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<T> buf1(max_chunk_size * buf_len); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<IdxT> idx_buf1(max_chunk_size * buf_len); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<T> buf2(max_chunk_size * buf_len); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<IdxT> idx_buf2(max_chunk_size * buf_len); //TODO: Replace rmm uvector with thrust vector

  cudaMemsetAsync(thrust::raw_pointer_cast(counters.data()), 0, GetSizeBytes(counters), stream);
  cudaMemsetAsync(thrust::raw_pointer_cast(histograms.data()), 0, GetSizeBytes(histograms), stream);


  for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - static_cast<int>(offset));
    auto kernel_instance = RadixKernel<T, IdxT, BitsPerPass, BlockSize, false>;

    const T* chunk_in        = in + offset * len;
    const IdxT* chunk_in_idx = in_idx ? (in_idx + offset * len) : nullptr;
    T* chunk_out             = out + offset * k;
    IdxT* chunk_out_idx      = out_idx + offset * k;

    const T* in_buf        = nullptr;
    const IdxT* in_idx_buf = nullptr;
    T* out_buf             = nullptr;
    IdxT* out_idx_buf      = nullptr;

    dim3 blocks_config(grid_dim, chunk_size);
    constexpr int num_passes = CalculateNumPasses<T, BitsPerPass>();

    for (int pass = 0; pass < num_passes; ++pass) {
      SetBufferPointers(chunk_in, chunk_in_idx, thrust::raw_pointer_cast(buf1.data()), thrust::raw_pointer_cast(idx_buf1.data()),
                          thrust::raw_pointer_cast(buf2.data()), thrust::raw_pointer_cast(idx_buf2.data()), pass, input_buffer_ptr,
                          input_index_buffer_ptr, output_buffer_ptr, output_index_buffer_ptr);

      if (fused_last_filter && pass == num_passes - 1) {
        kernel_instance = RadixKernel<T, IdxT, BitsPerPass, BlockSize, true>;
      }

      kernel_instance<<<blocks_config, BlockSize, 0, stream>>>(
          chunk_in, chunk_in_idx, input_buffer_ptr, input_index_buffer_ptr,
          output_buffer_ptr, output_index_buffer_ptr, chunk_out, chunk_out_idx,
          thrust::raw_pointer_cast(counters.data()), thrust::raw_pointer_cast(histograms.data()), len, k, select_min, pass);
      INTERNAL_CUDA_CHECK(cudaPeekAtLastError());
    }

    if (!fused_last_filter) {
      LastFilterKernel<T, IdxT, BitsPerPass><<<blocks_config, BlockSize, 0, stream>>>(
          chunk_in, chunk_in_idx, output_buffer_ptr, output_index_buffer_ptr, chunk_out,
          chunk_out_idx, len, k, thrust::raw_pointer_cast(counters.data()), select_min);
      INTERNAL_CUDA_CHECK(cudaPeekAtLastError());
    }
     cudaStreamSynchronize(stream); //TODO: Remove sync if not needed
  }
}

// ========================================================================================
// One-block-per-row kernel and related functions
// ========================================================================================

/**
 * @brief Kernel function to filter and compute histogram (one block per row version).
 */
template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_DEVICE void FilterAndHistogramOneBlock(const T* in_buf, const IdxT* in_idx_buf,
                                                     T* out_buf, IdxT* out_idx_buf, T* out,
                                                     IdxT* out_idx, const IdxT previous_len,
                                                     Counter<T, IdxT>* counter, IdxT* histogram,
                                                     bool select_min, int pass) {
  constexpr int num_buckets = CalculateNumBuckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram[i] = 0;
  }
  IdxT* filter_count_ptr = &counter->filter_count;
  if (threadIdx.x == 0) {
    *filter_count_ptr = 0;
  }
  __syncthreads();

  const int start_bit = CalculateStartBit<T, BitsPerPass>(pass);
  const unsigned mask = CalculateMask<T, BitsPerPass>(pass);

  if (pass == 0) {
    auto histogram_func = [&](T value, IdxT) {
      int bucket = CalculateBucket<T, BitsPerPass>(value, start_bit, mask, select_min);
      atomicAdd(histogram + bucket, static_cast<IdxT>(1)); // Corrected to atomicAdd
    };
    VectorizedProcess(threadIdx.x, blockDim.x, in_buf, previous_len, histogram_func);
  } else if (!out_buf) {
    const auto kth_value_bits = counter->kth_value_bits;
    const int previous_start_bit = CalculateStartBit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value = in_buf[i];
      const auto previous_bits =
          (TwiddleIn(value, select_min) >> previous_start_bit) << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        int bucket = CalculateBucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1)); // Corrected to atomicAdd
      }
    }
  } else {
    IdxT* output_count_ptr = &counter->output_count;
    const auto kth_value_bits = counter->kth_value_bits;
    const int previous_start_bit = CalculateStartBit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value = in_buf[i];
      const auto previous_bits =
          (TwiddleIn(value, select_min) >> previous_start_bit) << previous_start_bit;
      if (previous_bits == kth_value_bits) {
#if CUDART_VERSION < 12000
        volatile // Workaround for potential compiler bug in CUDA 11
#endif
          IdxT pos = atomicAdd(filter_count_ptr, static_cast<IdxT>(1)); // Corrected to atomicAdd
        out_buf[pos] = value;
        out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;

        int bucket = CalculateBucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1)); // Corrected to atomicAdd
      } else if (previous_bits < kth_value_bits) {
        IdxT pos = atomicAdd(output_count_ptr, static_cast<IdxT>(1)); // Corrected to atomicAdd
        out[pos] = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

/**
 * @brief Kernel for one-block-per-row radix top-k sort.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
FAST_TOPK_KERNEL RadixTopKOneBlockKernel(const T* in, const IdxT* in_idx, const IdxT len,
                                            const IdxT k, T* out, IdxT* out_idx,
                                            const bool select_min, char* bufs) {
  constexpr int num_buckets = CalculateNumBuckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  if (threadIdx.x == 0) {
    counter.k = k;
    counter.length = len;
    counter.previous_length = len;
    counter.kth_value_bits = 0;
    counter.output_count = 0;
    counter.output_back_count = 0;
  }
  __syncthreads();

  const size_t batch_id = blockIdx.x;
  in += batch_id * len;
  if (in_idx) { in_idx += batch_id * len; }
  out += batch_id * k;
  out_idx += batch_id * k;
  const IdxT buf_len = CalculateBufferLength<T, IdxT, unsigned>(len);
  bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  constexpr int num_passes = CalculateNumPasses<T, BitsPerPass>();
  for (int pass = 0; pass < num_passes; ++pass) {
    const T* input_buffer_ptr;
    const IdxT* input_index_buffer_ptr;
    T* output_buffer_ptr;
    IdxT* output_index_buffer_ptr;
    SetBufferPointers(in, in_idx, bufs, buf_len, pass, input_buffer_ptr, input_index_buffer_ptr, output_buffer_ptr, output_index_buffer_ptr);

    const IdxT current_len = counter.length;
    const IdxT current_k = counter.k;
    IdxT previous_len = counter.previous_length;
    if (previous_len > buf_len) {
      input_buffer_ptr = in;
      input_index_buffer_ptr = in_idx;
      previous_len = len;
    }
    if (current_len > buf_len) {
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }

    FilterAndHistogramOneBlock<T, IdxT, BitsPerPass>(
        input_buffer_ptr, input_index_buffer_ptr, output_buffer_ptr, output_index_buffer_ptr, out,
        out_idx, previous_len, &counter, histogram, select_min, pass);
    __syncthreads();

    ScanHistogram<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();

    ChooseBucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
    if (threadIdx.x == 0) { counter.previous_length = current_len; }
    __syncthreads();

    if (counter.length == counter.k || pass == num_passes - 1) {
      LastFilter<T, IdxT, BitsPerPass>(
          output_buffer_ptr ? output_buffer_ptr : input_buffer_ptr,
          output_buffer_ptr ? output_index_buffer_ptr : input_index_buffer_ptr, out, out_idx,
          output_buffer_ptr ? current_len : len, k, &counter, select_min, pass);
      break;
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void RadixTopKOneBlock(const T* in, const IdxT* in_idx, int batch_size, IdxT len, IdxT k, T* out,
                          IdxT* out_idx, bool select_min, int sm_cnt, cudaStream_t stream) {
  static_assert(CalculateNumPasses<T, BitsPerPass>() > 1);

  auto kernel = RadixTopKOneBlockKernel<T, IdxT, BitsPerPass, BlockSize>;
  const IdxT buf_len = CalculateBufferLength<T, IdxT, unsigned>(len);
  const size_t max_chunk_size =
    CalculateChunkSize<T, IdxT, BlockSize>(batch_size, len, sm_cnt, reinterpret_cast<void*>(kernel), true);

  thrust::device_vector<char> bufs(
      max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT))); //TODO: replace rmm uvector

    cudaMemsetAsync(thrust::raw_pointer_cast(bufs.data()), 0, GetSizeBytes(bufs), stream);


  for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - static_cast<int>(offset));
    RadixTopKOneBlockKernel<T, IdxT, BitsPerPass, BlockSize><<<chunk_size, BlockSize, 0, stream>>>(
        in + offset * len, in_idx ? (in_idx + offset * len) : nullptr, len, k, out + offset * k,
        out_idx + offset * k, select_min, thrust::raw_pointer_cast(bufs.data()));
     INTERNAL_CUDA_CHECK(cudaPeekAtLastError());
     cudaStreamSynchronize(stream); //TODO: Remove sync if not needed
  }
}

} // namespace internal::gpu::impl

#endif // INTERNAL_GPU_RADIX_KERNELS_CUH_