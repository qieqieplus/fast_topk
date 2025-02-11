#ifndef INTERNAL_GPU_BITS_CUH_
#define INTERNAL_GPU_BITS_CUH_

#include "internal/gpu/common.hpp"
#include <cub/cub.cuh>

namespace internal::gpu {

/**
 * @brief Calculates the number of passes needed for radix sort-based algorithms.
 *
 * For a given data type `T` and bits per pass `BitsPerPass`, this function
 * computes the number of passes required to process all bits of `T`.
 *
 * @tparam T           Data type being processed.
 * @tparam BitsPerPass Number of bits processed in each pass.
 * @return             The number of passes needed (ceil(#bits_in_T / BitsPerPass)).
 */
template <typename T, int BitsPerPass>
FAST_TOPK_HOST_DEVICE constexpr int CalculateNumPasses() {
  constexpr int bits_in_T = static_cast<int>(sizeof(T) * 8);
  return (bits_in_T + BitsPerPass - 1) / BitsPerPass;
}

/**
 * @brief Calculates the number of buckets based on bits per pass.
 *
 * For a given number of bits per pass `BitsPerPass`, this function
 * calculates the total number of buckets, which is 2 raised to the power of `BitsPerPass`.
 *
 * @tparam BitsPerPass Number of bits used to determine buckets in each pass.
 * @return             The number of buckets = 2^(BitsPerPass).
 */
template <int BitsPerPass>
FAST_TOPK_HOST_DEVICE constexpr int CalculateNumBuckets() {
  return 1 << BitsPerPass;
}

/**
 * @brief Calculates the starting bit index for a given pass.
 *
 * In radix sort, processing typically goes from most significant bit (MSB) to
 * least significant bit (LSB). This function determines the starting bit index
 * (0-based from LSB) for a specific pass.
 *
 * @tparam T           Data type being processed.
 * @tparam BitsPerPass Number of bits processed in each pass.
 * @param  pass        The current pass number (0-indexed).
 * @return             The starting bit index for the given pass.
 *                     If the calculated index is negative, it's clamped to 0.
 */
template <typename T, int BitsPerPass>
constexpr FAST_TOPK_DEVICE FAST_TOPK_INLINE int CalculateStartBit(int pass) {
  int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
  return (start_bit < 0) ? 0 : start_bit;
}

/**
 * @brief Creates a mask for extracting bits in a given pass.
 *
 * For a given pass, this function generates a mask to isolate the relevant
 * bits for processing in that pass.
 *
 * @tparam T           Data type being processed.
 * @tparam BitsPerPass Number of bits processed in each pass.
 * @param  pass        The current pass number (0-indexed).
 * @return             A mask with bits set for the range of bits in the pass.
 */
template <typename T, int BitsPerPass>
constexpr FAST_TOPK_DEVICE FAST_TOPK_INLINE unsigned CalculateMask(int pass) {
  int num_bits = CalculateStartBit<T, BitsPerPass>(pass - 1) - CalculateStartBit<T, BitsPerPass>(pass);
  return (1U << num_bits) - 1U;
}

template <typename T>
using UnsignedBits = typename cub::Traits<T>::UnsignedBits;

/**
 * @brief Twiddles bits for correct float comparison (input).
 *
 * Prepares the bits of a floating-point value for integer-based comparison,
 * as required for radix sort. Handles both min and max selection by optionally
 * inverting the bits for max selection.
 *
 * @tparam T         Data type being processed.
 * @param  key       The value to twiddle.
 * @param  select_min True if selecting minimum values, false for maximum.
 * @return           Twiddled bits as an unsigned integer type.
 */
template <typename T>
FAST_TOPK_DEVICE FAST_TOPK_INLINE UnsignedBits<T> TwiddleIn(T key, bool select_min) {
  UnsignedBits<T>& bits = reinterpret_cast<UnsignedBits<T>&>(key);
  bits        = cub::Traits<T>::TwiddleIn(bits);
  if (!select_min) bits = ~bits;
  return bits;
}

/**
 * @brief Untwiddles bits back to the original float representation (output).
 *
 * Reverses the bit twiddling process to restore the original floating-point
 * value after integer-based operations. Handles both min and max selection
 * by optionally inverting the bits if max selection was performed earlier.
 *
 * @tparam T         Data type being processed.
 * @param  bits      The twiddled bits.
 * @param  select_min True if selecting minimum values, false for maximum.
 * @return           The original value of type T.
 */
template <typename T>
FAST_TOPK_DEVICE FAST_TOPK_INLINE T TwiddleOut(UnsignedBits<T> bits, bool select_min) {
  if (!select_min) bits = ~bits;
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T&>(bits);
}

/**
 * @brief Calculates the bucket index for a given value in a radix pass.
 *
 * Extracts the relevant bits for the current radix pass from the input value
 * and converts them into a bucket index.
 *
 * @tparam T           Data type being processed.
 * @tparam BitsPerPass Number of bits processed in each pass.
 * @param  x           The input value.
 * @param  start_bit   The starting bit index for the current pass.
 * @param  mask        The bit mask for the current pass.
 * @param  select_min True if selecting minimum values, false for maximum.
 * @return             The calculated bucket index.
 */
template <typename T, int BitsPerPass>
FAST_TOPK_DEVICE FAST_TOPK_INLINE int CalculateBucket(T x, int start_bit, unsigned mask, bool select_min) {
  UnsignedBits<T> b      = TwiddleIn(x, select_min);
  return static_cast<int>((b >> start_bit) & mask);
}

} // namespace internal::gpu

#endif // INTERNAL_GPU_BITS_CUH_