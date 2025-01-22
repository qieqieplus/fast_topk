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
template <typename T, typename IdxT>
struct alignas(128) Counter {
  // We are processing the values in multiple passes, from most significant to least significant. In
  // each pass, we keep the length of input (`len`) and the `k` of current pass, and update them at
  // the end of the pass.
  IdxT k;
  IdxT len;

  //  `previous_len` is the length of input in previous pass. Note that `previous_len` rather
  //  than `len` is used for the filtering step because filtering is indeed for previous pass (see
  //  comments before `radix_kernel`).
  IdxT previous_len;

  // We determine the bits of the k_th value inside the mask processed by the pass. The
  // already known bits are stored in `kth_value_bits`. It's used to discriminate a element is a
  // result (written to `out`), a candidate for next pass (written to `out_buf`), or not useful
  // (discarded). The bits that are not yet processed do not matter for this purpose.
  typename cub::Traits<T>::UnsignedBits kth_value_bits;

  // Record how many elements have passed filtering. It's used to determine the position in the
  // `out_buf` where an element should be written.
  alignas(128) IdxT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This counter is used to
  // determine if the current block is the last running block. If so, this block will execute scan()
  // and choose_bucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements less (if
  // select_min==true) than the k-th value are written from front to back.
  alignas(128) IdxT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements equal to the k-th
  // value are written from back to front. We need to keep count of them separately because the
  // number of elements that <= the k-th value might exceed k.
  alignas(128) IdxT out_back_cnt;
};

template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_DEVICE void filter_and_histogram(const T* in_buf,
                                       const IdxT* in_idx_buf,
                                       T* out_buf,
                                       IdxT* out_idx_buf,
                                       T* out,
                                       IdxT* out_idx,
                                       IdxT previous_len,
                                       Counter<T, IdxT>* counter,
                                       IdxT* histogram,
                                       bool select_min,
                                       int pass,
                                       bool early_stop)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ IdxT histogram_smem[num_buckets];
  for (IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram_smem[i] = 0;
  }
  __syncthreads();

  const int start_bit = calc_start_bit<T, BitsPerPass>(pass);
  const unsigned mask = calc_mask<T, BitsPerPass>(pass);

  if (pass == 0) {
    // Passed to vectorized_process, this function executes in all blocks in parallel,
    // i.e. the work is split along the input (both, in batches and chunks of a single row).
    // Later, the histograms are merged using atomicAdd.
    auto f = [select_min, start_bit, mask](T value, IdxT) {
      int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
      atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
    };
    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                       static_cast<size_t>(blockDim.x) * gridDim.x,
                       in_buf,
                       previous_len,
                       f);
  } else {
    IdxT* p_filter_cnt           = &counter->filter_cnt;
    IdxT* p_out_cnt              = &counter->out_cnt;
    const auto kth_value_bits    = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    // See the remark above on the distributed execution of `f` using vectorized_process.
    auto f = [in_idx_buf,
              out_buf,
              out_idx_buf,
              out,
              out_idx,
              select_min,
              start_bit,
              mask,
              previous_start_bit,
              kth_value_bits,
              p_filter_cnt,
              p_out_cnt,
              early_stop](T value, IdxT i) {
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        if (early_stop) {
          IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
          out[pos]     = value;
          out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
        } else {
          if (out_buf) {
            IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
            out_buf[pos]     = value;
            out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
          }

          int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
          atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
        }
      }
      // the condition `(out_buf || early_stop)` is a little tricky:
      // If we skip writing to `out_buf` (when `out_buf` is nullptr), we should skip writing to
      // `out` too. So we won't write the same value to `out` multiple times in different passes.
      // And if we keep skipping the writing, values will be written in `last_filter_kernel()` at
      // last. But when `early_stop` is true, we need to write to `out` since it's the last chance.
      else if ((out_buf || early_stop) && previous_bits < kth_value_bits) {
        IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    };
    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                       static_cast<size_t>(blockDim.x) * gridDim.x,
                       in_buf,
                       previous_len,
                       f);
  }
  if (early_stop) { return; }
  __syncthreads();

  // merge histograms produced by individual blocks
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    if (histogram_smem[i] != 0) { atomicAdd(histogram + i, histogram_smem[i]); }
  }
}

/**
 * Replace histogram with its own prefix sum
 * (step 2 in `radix_kernel` description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
FAST_TOPK_DEVICE void scan(volatile IdxT* histogram)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  if constexpr (num_buckets >= BlockSize) {
    static_assert(num_buckets % BlockSize == 0);
    constexpr int items_per_thread = num_buckets / BlockSize;
    typedef cub::BlockLoad<IdxT, BlockSize, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
    typedef cub::BlockStore<IdxT, BlockSize, items_per_thread, cub::BLOCK_STORE_TRANSPOSE>
      BlockStore;
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;

    __shared__ union {
      typename BlockLoad::TempStorage load;
      typename BlockScan::TempStorage scan;
      typename BlockStore::TempStorage store;
    } temp_storage;
    IdxT thread_data[items_per_thread];

    BlockLoad(temp_storage.load).Load(histogram, thread_data);
    __syncthreads();

    BlockScan(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    BlockStore(temp_storage.store).Store(histogram, thread_data);
  } else {
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    IdxT thread_data = 0;
    if (threadIdx.x < num_buckets) { thread_data = histogram[threadIdx.x]; }

    BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    if (threadIdx.x < num_buckets) { histogram[threadIdx.x] = thread_data; }
  }
}

/**
 * Calculate in which bucket the k-th value will fall
 *  (steps 3 in `radix_kernel` description)
 */
template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_DEVICE void choose_bucket(Counter<T, IdxT>* counter,
                                const IdxT* histogram,
                                const IdxT k,
                                const int pass)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    IdxT prev = (i == 0) ? 0 : histogram[i - 1];
    IdxT cur  = histogram[i];

    // one and only one thread will satisfy this condition, so counter is written by only one thread
    if (prev < k && cur >= k) {
      counter->k   = k - prev;    // how many values still are there to find
      counter->len = cur - prev;  // number of values in next pass
      typename cub::Traits<T>::UnsignedBits bucket = i;
      int start_bit                                = calc_start_bit<T, BitsPerPass>(pass);
      counter->kth_value_bits |= bucket << start_bit;
    }
  }
}

// For one-block version, last_filter() could be called when pass < num_passes - 1.
// So `pass` could not be constexpr
template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_DEVICE void last_filter(const T* in_buf,
                              const IdxT* in_idx_buf,
                              T* out,
                              IdxT* out_idx,
                              IdxT current_len,
                              IdxT k,
                              Counter<T, IdxT>* counter,
                              const bool select_min,
                              const int pass)
{
  const auto kth_value_bits = counter->kth_value_bits;
  const int start_bit       = calc_start_bit<T, BitsPerPass>(pass);

  // changed in choose_bucket(); need to reload
  const IdxT num_of_kth_needed = counter->k;
  IdxT* p_out_cnt              = &counter->out_cnt;
  IdxT* p_out_back_cnt         = &counter->out_back_cnt;
  for (IdxT i = threadIdx.x; i < current_len; i += blockDim.x) {
    const T value   = in_buf[i];
    const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
    if (bits < kth_value_bits) {
      IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
      out[pos] = value;
      // For one-block version, `in_idx_buf` could be nullptr at pass 0.
      // For non one-block version, if writing has been skipped, `in_idx_buf` could be nullptr if
      // `in_buf` is `in`
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
      if (back_pos < num_of_kth_needed) {
        IdxT pos     = k - 1 - back_pos;
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_KERNEL last_filter_kernel(const T* in,
                               const IdxT* in_idx,
                               const T* in_buf,
                               const IdxT* in_idx_buf,
                               T* out,
                               IdxT* out_idx,
                               const IdxT len,
                               const IdxT k,
                               Counter<T, IdxT>* counters,
                               const bool select_min)
{
  const size_t batch_id = blockIdx.y;  // size_t to avoid multiplication overflow

  Counter<T, IdxT>* counter = counters + batch_id;
  IdxT previous_len         = counter->previous_len;
  if (previous_len == 0) { return; }
  const IdxT buf_len = calc_buf_len<T>(len);
  if (previous_len > buf_len || in_buf == in) {
    in_buf       = in + batch_id * len;
    in_idx_buf   = in_idx ? (in_idx + batch_id * len) : nullptr;
    previous_len = len;
  } else {
    in_buf += batch_id * buf_len;
    in_idx_buf += batch_id * buf_len;
  }
  out += batch_id * k;
  out_idx += batch_id * k;

  constexpr int pass      = calc_num_passes<T, BitsPerPass>() - 1;
  constexpr int start_bit = calc_start_bit<T, BitsPerPass>(pass);

  const auto kth_value_bits    = counter->kth_value_bits;
  const IdxT num_of_kth_needed = counter->k;
  IdxT* p_out_cnt              = &counter->out_cnt;
  IdxT* p_out_back_cnt         = &counter->out_back_cnt;

  auto f = [k,
            select_min,
            kth_value_bits,
            num_of_kth_needed,
            p_out_cnt,
            p_out_back_cnt,
            in_idx_buf,
            out,
            out_idx](T value, IdxT i) {
    const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
    if (bits < kth_value_bits) {
      IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
      out[pos]     = value;
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
      if (back_pos < num_of_kth_needed) {
        IdxT pos     = k - 1 - back_pos;
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  };

  vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                     static_cast<size_t>(blockDim.x) * gridDim.x,
                     in_buf,
                     previous_len,
                     f);
}

/**
 *
 * It is expected to call this kernel multiple times (passes), in each pass we process a radix,
 * going from the most significant towards the least significant bits (MSD).
 *
 * Conceptually, each pass consists of 4 steps:
 *
 * 1. Calculate histogram
 *      First, transform bits into a digit, the value of which is in the range
 *      [0, 2^{BITS_PER_PASS}-1]. Then count the frequency of each digit value and the result is a
 *      histogram. That is, histogram[i] contains the count of inputs having value i.
 *
 * 2. Scan the histogram
 *      Inclusive prefix sum is computed for the histogram. After this step, histogram[i] contains
 *      the count of inputs having value <= i.
 *
 * 3. Find the bucket j of the histogram that the k-th value falls into
 *
 * 4. Filtering
 *      Input elements whose digit value <j are the top-k elements. We put them into the result
 *      array out. The number of such elements is histogram[j-1]. Since the k-th value must be in
 *      the bucket j, we write all elements in bucket j into a intermediate buffer out_buf. For the
 *      next pass, these elements are used as input, and we would like to find the
 *      (k - histogram[j-1])-th value among them. That is, the k in the next pass is set to
 *      (k - histogram[j-1]).
 *
 * In the implementation, the filtering step is delayed to the next pass so the filtering and
 * histogram computation are fused. In this way, inputs are read once rather than twice.
 *
 * During the filtering step, we won't write candidates (elements in bucket j) to `out_buf` if the
 * number of candidates is larger than the length of `out_buf` (this could happen when the leading
 * bits of input values are almost the same). And then in the next pass, inputs are read from `in`
 * rather than from `out_buf`. The benefit is that we can save the cost of writing candidates and
 * their indices.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize, bool fused_last_filter>
FAST_TOPK_KERNEL radix_kernel(const T* in,
                         const IdxT* in_idx,
                         const T* in_buf,
                         const IdxT* in_idx_buf,
                         T* out_buf,
                         IdxT* out_idx_buf,
                         T* out,
                         IdxT* out_idx,
                         Counter<T, IdxT>* counters,
                         IdxT* histograms,
                         const IdxT len,
                         const IdxT k,
                         const bool select_min,
                         const int pass)
{
  const size_t batch_id = blockIdx.y;
  auto counter          = counters + batch_id;
  IdxT current_k;
  IdxT previous_len;
  IdxT current_len;
  if (pass == 0) {
    current_k    = k;
    previous_len = len;
    // Need to do this so setting counter->previous_len for the next pass is correct.
    // This value is meaningless for pass 0, but it's fine because pass 0 won't be the
    // last pass in this implementation so pass 0 won't hit the "if (pass ==
    // num_passes - 1)" branch.
    // Maybe it's better to reload counter->previous_len and use it rather than
    // current_len in last_filter()
    current_len = len;
  } else {
    current_k    = counter->k;
    current_len  = counter->len;
    previous_len = counter->previous_len;
  }
  if (current_len == 0) { return; }

  // When k=len, early_stop will be true at pass 0. It means filter_and_histogram() should handle
  // correctly the case that pass=0 and early_stop=true. However, this special case of k=len is
  // handled in other way in select_k() so such case is not possible here.
  const bool early_stop = (current_len == current_k);
  const IdxT buf_len    = calc_buf_len<T>(len);

  // "previous_len > buf_len" means previous pass skips writing buffer
  if (pass == 0 || pass == 1 || previous_len > buf_len) {
    in_buf       = in + batch_id * len;
    in_idx_buf   = in_idx ? (in_idx + batch_id * len) : nullptr;
    previous_len = len;
  } else {
    in_buf += batch_id * buf_len;
    in_idx_buf += batch_id * buf_len;
  }
  // "current_len > buf_len" means current pass will skip writing buffer
  if (pass == 0 || current_len > buf_len) {
    out_buf     = nullptr;
    out_idx_buf = nullptr;
  } else {
    out_buf += batch_id * buf_len;
    out_idx_buf += batch_id * buf_len;
  }
  out += batch_id * k;
  out_idx += batch_id * k;

  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  auto histogram            = histograms + batch_id * num_buckets;

  filter_and_histogram<T, IdxT, BitsPerPass>(in_buf,
                                             in_idx_buf,
                                             out_buf,
                                             out_idx_buf,
                                             out,
                                             out_idx,
                                             previous_len,
                                             counter,
                                             histogram,
                                             select_min,
                                             pass,
                                             early_stop);
  __threadfence();

  bool isLastBlock = false;
  if (threadIdx.x == 0) {
    unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
    isLastBlock           = (finished == (gridDim.x - 1));
  }

  if (__syncthreads_or(isLastBlock)) {
    if (early_stop) {
      if (threadIdx.x == 0) {
        // `last_filter_kernel()` requires setting previous_len
        counter->previous_len = 0;
        counter->len          = 0;
      }
      return;
    }

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();
    choose_bucket<T, IdxT, BitsPerPass>(counter, histogram, current_k, pass);
    __syncthreads();

    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
    // reset for next pass
    if (pass != num_passes - 1) {
      for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        histogram[i] = 0;
      }
    }
    if (threadIdx.x == 0) {
      // `last_filter_kernel()` requires setting previous_len even in the last pass
      counter->previous_len = current_len;
      // not necessary for the last pass, but put it here anyway
      counter->filter_cnt = 0;
    }

    if constexpr (fused_last_filter) {
      if (pass == num_passes - 1) {
        last_filter<T, IdxT, BitsPerPass>(out_buf ? out_buf : in_buf,
                                          out_idx_buf ? out_idx_buf : in_idx_buf,
                                          out,
                                          out_idx,
                                          out_buf ? current_len : len,
                                          k,
                                          counter,
                                          select_min,
                                          pass);
      }
    }
  }
}

template <typename T, typename IdxT, int BlockSize, typename Kernel>
int calc_chunk_size(int batch_size, IdxT len, int sm_cnt, Kernel kernel, bool one_block)
{
  int active_blocks;
  INTERNAL_CUDA_CHECK(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, BlockSize, 0));

  // The chunk size is chosen so that there is enough workload to fully utilize GPU.
  // One full wave contains (sm_cnt * active_blocks) blocks, and 10 waves is an empirically safe
  // estimation of enough workload. It also counteracts imbalance if some blocks run slower
  // than others.
  constexpr int num_waves = 10;
  int chunk_size;
  if (one_block) {
    // For one-block version, one block processes one instance in the chunk. Just ensure that there
    // are enough blocks.
    chunk_size = num_waves * sm_cnt * active_blocks;
  } else {
    // One instance in the chunk contains len items and is processed by multiple blocks.
    // The total number of items in a chunk (chunk_size * len) should be large enough that every
    // thread has enough items to processes. So set it to num_waves * "max num of active threads"
    // (sm_cnt * active_blocks * BlockSize) * items_per_thread.
    //
    // Also, the upper bound of the total number of items in a chunk is:
    // 10 (num_waves) * ~100 (sm_cnt) * 2048 (active_blocks*BlockSize) * 32 (items_per_thread) =64M.
    // So temporary buffer size required for one chunk won't be too large.
    constexpr int items_per_thread = 32;
    chunk_size =
      std::max<int>(1, num_waves * sm_cnt * active_blocks * BlockSize * items_per_thread / len);
  }
  return std::min(chunk_size, batch_size);
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
unsigned calc_grid_dim(int batch_size, IdxT len, int sm_cnt)
{
  static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);

  int active_blocks;
  INTERNAL_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks, radix_kernel<T, IdxT, BitsPerPass, BlockSize, false>, BlockSize, 0));
  active_blocks *= sm_cnt;

  IdxT best_num_blocks         = 0;
  float best_tail_wave_penalty = 1.0f;
  const IdxT max_num_blocks    = ceildiv<IdxT>(len, VECTORIZED_READ_SIZE / sizeof(T) * BlockSize);
  for (int num_waves = 1;; ++num_waves) {
    IdxT num_blocks = std::min(
      max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
    IdxT items_per_thread  = ceildiv<IdxT>(len, num_blocks * BlockSize);
    items_per_thread       = alignTo<IdxT>(items_per_thread, VECTORIZED_READ_SIZE / sizeof(T));
    num_blocks             = ceildiv<IdxT>(len, items_per_thread * BlockSize);
    float actual_num_waves = static_cast<float>(num_blocks) * batch_size / active_blocks;
    float tail_wave_penalty =
      (ceilf(actual_num_waves) - actual_num_waves) / ceilf(actual_num_waves);

    // 0.15 is determined experimentally. It also ensures breaking the loop early,
    // e.g. when num_waves > 7, tail_wave_penalty will always <0.15
    if (tail_wave_penalty < 0.15) {
      best_num_blocks = num_blocks;
      break;
    } else if (tail_wave_penalty < best_tail_wave_penalty) {
      best_num_blocks        = num_blocks;
      best_tail_wave_penalty = tail_wave_penalty;
    }

    if (num_blocks == max_num_blocks) { break; }
  }
  return best_num_blocks;
}


template <typename T, typename IdxT>
FAST_TOPK_HOST void set_buf_pointers(const T* in,
                                 const IdxT* in_idx,
                                 T* buf1,
                                 IdxT* idx_buf1,
                                 T* buf2,
                                 IdxT* idx_buf2,
                                 int pass,
                                 const T*& in_buf,
                                 const IdxT*& in_idx_buf,
                                 T*& out_buf,
                                 IdxT*& out_idx_buf)
{
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

template <typename T, typename IdxT>
FAST_TOPK_DEVICE void set_buf_pointers(const T* in,
                                   const IdxT* in_idx,
                                   char* bufs,
                                   IdxT buf_len,
                                   int pass,
                                   const T*& in_buf,
                                   const IdxT*& in_idx_buf,
                                   T*& out_buf,
                                   IdxT*& out_idx_buf)
{
  // bufs consists of 4 pieces in order: buf1, buf2, idx_buf1, idx_buf2
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

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void radix_topk(const T* in,
                const IdxT* in_idx,
                int batch_size,
                IdxT len,
                IdxT k,
                T* out,
                IdxT* out_idx,
                bool select_min,
                bool fused_last_filter,
                unsigned grid_dim,
                int sm_cnt,
                cudaStream_t stream)
{
  // TODO: is it possible to relax this restriction?
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

  auto kernel = radix_kernel<T, IdxT, BitsPerPass, BlockSize, false>;
  const size_t max_chunk_size =
    calc_chunk_size<T, IdxT, BlockSize>(batch_size, len, sm_cnt, kernel, false);
  if (max_chunk_size != static_cast<size_t>(batch_size)) {
    grid_dim = calc_grid_dim<T, IdxT, BitsPerPass, BlockSize>(max_chunk_size, len, sm_cnt);
  }
  const IdxT buf_len = calc_buf_len<T>(len);

  size_t req_buf = max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  thrust::device_vector<Counter<T, IdxT>> counters(max_chunk_size); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<IdxT> histograms(max_chunk_size * num_buckets); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<T> buf1(max_chunk_size * buf_len); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<IdxT> idx_buf1(max_chunk_size * buf_len); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<T> buf2(max_chunk_size * buf_len); //TODO: Replace rmm uvector with thrust vector
  thrust::device_vector<IdxT> idx_buf2(max_chunk_size * buf_len); //TODO: Replace rmm uvector with thrust vector

  cudaMemsetAsync(counters.data().get(), 0, size_bytes(counters), stream);
  cudaMemsetAsync(histograms.data().get(), 0, size_bytes(histograms), stream);


  for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - offset);

    auto kernel = radix_kernel<T, IdxT, BitsPerPass, BlockSize, false>;

    const T* chunk_in        = in + offset * len;
    const IdxT* chunk_in_idx = in_idx ? (in_idx + offset * len) : nullptr;
    T* chunk_out             = out + offset * k;
    IdxT* chunk_out_idx      = out_idx + offset * k;

    const T* in_buf        = nullptr;
    const IdxT* in_idx_buf = nullptr;
    T* out_buf             = nullptr;
    IdxT* out_idx_buf      = nullptr;

    dim3 blocks(grid_dim, chunk_size);
    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();

    for (int pass = 0; pass < num_passes; ++pass) {
      impl::set_buf_pointers(chunk_in,
                       chunk_in_idx,
                       buf1.data().get(),
                       idx_buf1.data().get(),
                       buf2.data().get(),
                       idx_buf2.data().get(),
                       pass,
                       in_buf,
                       in_idx_buf,
                       out_buf,
                       out_idx_buf);

      if (fused_last_filter && pass == num_passes - 1) {
        kernel = radix_kernel<T, IdxT, BitsPerPass, BlockSize, true>;
      }

      kernel<<<blocks, BlockSize, 0, stream>>>(chunk_in,
                                               chunk_in_idx,
                                               in_buf,
                                               in_idx_buf,
                                               out_buf,
                                               out_idx_buf,
                                               chunk_out,
                                               chunk_out_idx,
                                               thrust::raw_pointer_cast(counters.data()),
                                               thrust::raw_pointer_cast(histograms.data()),
                                               len,
                                               k,
                                               select_min,
                                               pass);
      INTERNAL_CUDA_CHECK(cudaPeekAtLastError());
    }

    if (!fused_last_filter) {
      last_filter_kernel<T, IdxT, BitsPerPass><<<blocks, BlockSize, 0, stream>>>(chunk_in,
                                                                                 chunk_in_idx,
                                                                                 out_buf,
                                                                                 out_idx_buf,
                                                                                 chunk_out,
                                                                                 chunk_out_idx,
                                                                                 len,
                                                                                 k,
                                                                                 thrust::raw_pointer_cast(counters.data()),
                                                                                 select_min);
      INTERNAL_CUDA_CHECK(cudaPeekAtLastError());
    }
     cudaStreamSynchronize(stream); //TODO: Remove sync if not needed
  }
}

// The following a few functions are for the one-block version, which uses single thread block for
// each row of a batch.
template <typename T, typename IdxT, int BitsPerPass>
FAST_TOPK_DEVICE void filter_and_histogram_for_one_block(const T* in_buf,
                                                     const IdxT* in_idx_buf,
                                                     T* out_buf,
                                                     IdxT* out_idx_buf,
                                                     T* out,
                                                     IdxT* out_idx,
                                                     const IdxT previous_len,
                                                     Counter<T, IdxT>* counter,
                                                     IdxT* histogram,
                                                     bool select_min,
                                                     int pass)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram[i] = 0;
  }
  IdxT* p_filter_cnt = &counter->filter_cnt;
  if (threadIdx.x == 0) { *p_filter_cnt = 0; }
  __syncthreads();

  const int start_bit = calc_start_bit<T, BitsPerPass>(pass);
  const unsigned mask = calc_mask<T, BitsPerPass>(pass);

  if (pass == 0) {
    auto f = [histogram, select_min, start_bit, mask](T value, IdxT) {
      int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
      atomicAdd(histogram + bucket, static_cast<IdxT>(1));
    };
    vectorized_process(threadIdx.x, blockDim.x, in_buf, previous_len, f);
  } else if (!out_buf) {
    // not use vectorized_process here because it increases #registers a lot
    const auto kth_value_bits    = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value            = in_buf[i];
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1));
      }
    }
  } else {
    // not use vectorized_process here because it increases #registers a lot
    IdxT* p_out_cnt              = &counter->out_cnt;
    const auto kth_value_bits    = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value            = in_buf[i];
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
#if CUDART_VERSION < 12000
        // Avoiding potential compiler bug in CUDA 11
        volatile
#endif
          IdxT pos       = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
        out_buf[pos]     = value;
        out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;

        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1));
      } else if (previous_bits < kth_value_bits) {
        IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
FAST_TOPK_KERNEL radix_topk_one_block_kernel(const T* in,
                                        const IdxT* in_idx,
                                        const IdxT len,
                                        const IdxT k,
                                        T* out,
                                        IdxT* out_idx,
                                        const bool select_min,
                                        char* bufs)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  if (threadIdx.x == 0) {
    counter.k              = k;
    counter.len            = len;
    counter.previous_len   = len;
    counter.kth_value_bits = 0;
    counter.out_cnt        = 0;
    counter.out_back_cnt   = 0;
  }
  __syncthreads();

  const size_t batch_id = blockIdx.x;  // size_t to avoid multiplication overflow
  in += batch_id * len;
  if (in_idx) { in_idx += batch_id * len; }
  out += batch_id * k;
  out_idx += batch_id * k;
  const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);
  bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
  for (int pass = 0; pass < num_passes; ++pass) {
    const T* in_buf;
    const IdxT* in_idx_buf;
    T* out_buf;
    IdxT* out_idx_buf;
    impl::set_buf_pointers(in, in_idx, bufs, buf_len, pass, in_buf, in_idx_buf, out_buf, out_idx_buf);

    const IdxT current_len = counter.len;
    const IdxT current_k   = counter.k;
    IdxT previous_len      = counter.previous_len;
    if (previous_len > buf_len) {
      in_buf       = in;
      in_idx_buf   = in_idx;
      previous_len = len;
    }
    if (current_len > buf_len) {
      // so "out_buf==nullptr" denotes skipping writing buffer in current pass
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }

    filter_and_histogram_for_one_block<T, IdxT, BitsPerPass>(in_buf,
                                                             in_idx_buf,
                                                             out_buf,
                                                             out_idx_buf,
                                                             out,
                                                             out_idx,
                                                             previous_len,
                                                             &counter,
                                                             histogram,
                                                             select_min,
                                                             pass);
    __syncthreads();

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();

    choose_bucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
    if (threadIdx.x == 0) { counter.previous_len = current_len; }
    __syncthreads();

    if (counter.len == counter.k || pass == num_passes - 1) {
      last_filter<T, IdxT, BitsPerPass>(out_buf ? out_buf : in,
                                        out_buf ? out_idx_buf : in_idx,
                                        out,
                                        out_idx,
                                        out_buf ? current_len : len,
                                        k,
                                        &counter,
                                        select_min,
                                        pass);
      break;
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void radix_topk_one_block(const T* in,
                          const IdxT* in_idx,
                          int batch_size,
                          IdxT len,
                          IdxT k,
                          T* out,
                          IdxT* out_idx,
                          bool select_min,
                          int sm_cnt,
                          cudaStream_t stream)
{
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);

  auto kernel        = radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize>;
  const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);
  const size_t max_chunk_size =
    calc_chunk_size<T, IdxT, BlockSize>(batch_size, len, sm_cnt, reinterpret_cast<void*>(kernel), true);

  thrust::device_vector<char> bufs(
    max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT))); //TODO: replace rmm uvector

    cudaMemsetAsync(bufs.data().get(), 0, size_bytes(bufs), stream);


  for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - offset);
    kernel<<<chunk_size, BlockSize, 0, stream>>>(in + offset * len,
                                                 in_idx ? (in_idx + offset * len) : nullptr,
                                                 len,
                                                 k,
                                                 out + offset * k,
                                                 out_idx + offset * k,
                                                 select_min,
                                                 thrust::raw_pointer_cast(bufs.data()));
     INTERNAL_CUDA_CHECK(cudaPeekAtLastError());
     cudaStreamSynchronize(stream); //TODO: Remove sync if not needed
  }
}

} // namespace internal::gpu::impl

#endif // INTERNAL_GPU_RADIX_KERNELS_CUH_