#ifndef INTERNAL_GPU_UTIL_H_
#define INTERNAL_GPU_UTIL_H_

#include <cub/util_type.cuh>
#include <thrust/device_vector.h>

#include "internal/gpu/common.hpp"
#include "internal/gpu/vectorize.cuh"

namespace internal::gpu {

/**
 * @brief Returns the size in bytes of a thrust::device_vector.
 *
 * @tparam T Element type of the vector.
 * @param vec The thrust::device_vector to get the size of.
 * @return Size in bytes.
 */
template <typename T>
FAST_TOPK_HOST_DEVICE inline size_t GetSizeBytes(const thrust::device_vector<T>& vec) {
  return vec.size() * sizeof(T);
}

/**
 * @brief Calculates the ceiling division of two numbers.
 *
 * Computes `(a + b - 1) / b` for integer types, which is equivalent to
 * ceiling division.
 *
 * @tparam T Integer type.
 * @param a Numerator.
 * @param b Denominator.
 * @return Ceiling of a divided by b.
 */
template <typename T>
constexpr FAST_TOPK_HOST_DEVICE inline T CeilingDivide(T a, T b) {
  return (a + b - 1) / b;
}

/**
 * @brief Aligns a value up to the nearest multiple of m.
 *
 * If v is not a multiple of m, this function rounds v up to the nearest
 * multiple of m.
 *
 * @tparam T Integer type.
 * @param v Value to align.
 * @param m Alignment value, must be a power of 2 for bitwise efficiency.
 * @return Aligned value.
 */
template <typename T>
constexpr FAST_TOPK_HOST_DEVICE inline T AlignUp(T v, T m) {
  return CeilingDivide(v, m) * m;
}

/**
 * @brief Checks if a value is a power of two.
 *
 * @tparam I Integer type.
 * @param val Value to check.
 * @return True if val is a power of two, false otherwise.
 */
template <typename I>
constexpr FAST_TOPK_HOST_DEVICE inline auto IsPowerOfTwo(I val) noexcept
    -> std::enable_if_t<std::is_integral<I>::value, bool> {
  return (val != 0) && (((val - 1) & val) == 0);
}

/**
 * @brief Utility struct for power of 2 operations.
 *
 * Provides compile-time utilities for rounding and modulo operations based on
 * a power of 2 value N.
 *
 * @tparam N Power of two value.
 */
template <size_t N>
struct Pow2 {
  static constexpr size_t kMask = N - 1; // Mask for bitwise operations

  /**
   * @brief Rounds a size up to the nearest power of 2.
   *
   * @param p Size to round up.
   * @return Rounded size.
   */
  static constexpr FAST_TOPK_HOST_DEVICE inline size_t RoundUp(size_t p) {
    return ((p + kMask) & ~kMask);
  }

  /**
   * @brief Rounds a size down to the nearest power of 2.
   *
   * @param p Size to round down.
   * @return Rounded size.
   */
  static constexpr FAST_TOPK_HOST_DEVICE inline size_t RoundDown(size_t p) {
    return (p & ~kMask);
  }

  /**
   * @brief Calculates modulo N using bitwise AND.
   *
   * @param x Value to calculate modulo.
   * @return x modulo N.
   */
  static constexpr FAST_TOPK_HOST_DEVICE inline size_t Mod(size_t x) {
    return (x & kMask);
  }
};

/**
 * @brief Helper to calculate the size of the intermediate buffer.
 *
 * Determines the buffer length based on data type sizes and an empirical ratio
 * to optimize buffer usage in radix sort operations.
 *
 * @tparam T Data type.
 * @tparam IdxT Index type.
 * @tparam RATIO_T Ratio type, defaults to float.
 * @param len Length of the data.
 * @return Calculated buffer length.
 */
template <typename T, typename IdxT, typename RATIO_T = float>
FAST_TOPK_HOST_DEVICE inline IdxT CalculateBufferLength(IdxT len) {
  // Ratio based on read and write operations in different buffer scenarios
  constexpr RATIO_T ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
  IdxT buffer_length = len / (ratio * 8); // Empirical adjustment of buffer length

  // Align buffer length down to the nearest multiple for memory alignment
  static_assert(IsPowerOfTwo(sizeof(T)));
  static_assert(IsPowerOfTwo(sizeof(IdxT)));
  constexpr IdxT aligned_size = 256 / std::min(sizeof(T), sizeof(IdxT));
  buffer_length = Pow2<aligned_size>::RoundDown(buffer_length);
  return buffer_length;
}

/**
 * @brief Processes elements in a vectorized manner using a provided function.
 *
 * Applies a function `f` to each element of an input array `in`. Vectorizes
 * memory access when beneficial based on element size and alignment.
 *
 * @tparam T Data type of input array.
 * @tparam IdxT Index type.
 * @tparam Func Function object or lambda to apply to each element (T value, IdxT index).
 * @param thread_rank Rank of the current thread.
 * @param num_threads Total number of threads.
 * @param in Input array pointer.
 * @param len Length of the input array.
 * @param f Function to apply.
 */
template <typename T, typename IdxT, typename Func>
FAST_TOPK_DEVICE void VectorizedProcess(size_t thread_rank, size_t num_threads, const T* in,
                                        IdxT len, Func f) {
  if constexpr (sizeof(T) >= kVectorizedReadSize || kVectorizedReadSize % sizeof(T) != 0) {
    // Scalar processing for large types or non-divisible sizes
    for (IdxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    // Vectorized processing for smaller, divisible types
    using VecBlockType = VecBlock<T, kVectorizedReadSize / sizeof(T)>;
    using AlignBytes = Pow2<kVectorizedReadSize>;
    using AlignElems = Pow2<VecBlockType::kNumElems>;
    VecBlockType wide;

    size_t in_address = reinterpret_cast<size_t>(in);
    const IdxT skip_count_left =
        std::min((IdxT)(AlignBytes::RoundUp(in_address) - in_address), len);

    // Main vectorized loop
    for (IdxT i = thread_rank * VecBlockType::kNumElems + skip_count_left;
         i + VecBlockType::kNumElems <= len; i += num_threads * VecBlockType::kNumElems) {
      wide.Load(in, i);
#pragma unroll
      for (int j = 0; j < VecBlockType::kNumElems; ++j) {
        f(wide.elems.data[j], i + j);
      }
    }

    static_assert(kWarpSize >= VecBlockType::kNumElems,
                  "Warp size is smaller than vectorized block size.");

    // Handle skipped elements before vector alignment
    if (thread_rank < skip_count_left) {
      f(in[thread_rank], thread_rank);
    }

    // Handle remaining elements after vector alignment
    const IdxT skip_count_right = AlignElems::Mod(len - skip_count_left);
    const IdxT remain_index = len - skip_count_right + thread_rank;
    if (remain_index < len) {
      f(in[remain_index], remain_index);
    }
  }
}

} // namespace internal::gpu

#endif // INTERNAL_GPU_UTIL_H_