#ifndef INTERNAL_GPU_BITS_H_
#define INTERNAL_GPU_BITS_H_

#include "internal/gpu/common.hpp"
#include <cub/cub.cuh>

namespace internal::gpu {

/** Return the number of passes needed (ceil(#bits_in_T / BitsPerPass)). */
template <typename T, int BitsPerPass>
FAST_TOPK_HOST_DEVICE constexpr int calc_num_passes()
{
  constexpr int bits_in_T = static_cast<int>(sizeof(T) * 8);
  return (bits_in_T + BitsPerPass - 1) / BitsPerPass;
}

/** Return the number of buckets = 2^(BitsPerPass). */
template <int BitsPerPass>
FAST_TOPK_HOST_DEVICE constexpr int calc_num_buckets()
{
  return 1 << BitsPerPass;
}

/**
 * Return the starting bit index (0-based from LSB) for a given pass (processing from MSB→LSB).
 * If it becomes negative, clamp it to 0.
 */
template <typename T, int BitsPerPass>
constexpr FAST_TOPK_DEVICE FAST_TOPK_INLINE int calc_start_bit(int pass)
{
  int start_bit = (int)(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
  return (start_bit < 0) ? 0 : start_bit;
}

/**
 * For a given pass, mask for extracting just the portion of bits in that pass.
 */
template <typename T, int BitsPerPass>
constexpr FAST_TOPK_DEVICE FAST_TOPK_INLINE unsigned calc_mask(int pass)
{
  int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
  return (1U << num_bits) - 1U;
}

template <typename T>
using UBITS = typename cub::Traits<T>::UnsignedBits;

/**
 * Twiddle bits in/out for float comparison as in cub::Traits<T>.
 */
template <typename T>
FAST_TOPK_DEVICE FAST_TOPK_INLINE UBITS<T> twiddle_in(T key, bool select_min)
{
  UBITS<T>& bits = reinterpret_cast<UBITS<T>&>(key);
  bits        = cub::Traits<T>::TwiddleIn(bits);
  if (!select_min) bits = ~bits;
  return bits;
}

template <typename T>
FAST_TOPK_DEVICE FAST_TOPK_INLINE T twiddle_out(UBITS<T> bits, bool select_min)
{
  if (!select_min) bits = ~bits;
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T&>(bits);
}

template <typename T, int BitsPerPass>
FAST_TOPK_DEVICE FAST_TOPK_INLINE int calc_bucket(T x, int start_bit, unsigned mask, bool select_min)
{
  UBITS<T> b      = twiddle_in(x, select_min);
  return static_cast<int>((b >> start_bit) & mask);
}

} // namespace

#endif // INTERNAL_GPU_BITS_H_