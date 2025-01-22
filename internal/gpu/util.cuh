#ifndef INTERNAL_GPU_UTIL_H
#define INTERNAL_GPU_UTIL_H

#include <cub/util_type.cuh> 

#include "internal/gpu/common.hpp"
#include "internal/gpu/vectorize.cuh"

namespace internal::gpu {

template <typename T>
FAST_TOPK_INLINE size_t size_bytes(const thrust::device_vector<T>& vec) {
    return vec.size() * sizeof(T);
}

template <typename T>
constexpr FAST_TOPK_HOST_DEVICE FAST_TOPK_INLINE T ceildiv(T a, T b)
{
  return (a + b - 1) / b;
}

template <typename T>
constexpr FAST_TOPK_HOST_DEVICE FAST_TOPK_INLINE T alignTo(T v, T m)
{
  return ceildiv(v, m) * m;
}

template <typename I>
constexpr FAST_TOPK_HOST_DEVICE FAST_TOPK_INLINE auto is_a_power_of_two(I val) noexcept
  -> std::enable_if_t<std::is_integral<I>::value, bool>
{
  return (val != 0) && (((val - 1) & val) == 0);
}

template <size_t N>
struct Pow2 {
  static constexpr size_t Mask = N - 1;

  static constexpr FAST_TOPK_HOST_DEVICE FAST_TOPK_INLINE size_t roundUp(size_t p)
  {
    return ((p + Mask) & ~Mask);
  }

  static constexpr FAST_TOPK_HOST_DEVICE FAST_TOPK_INLINE size_t roundDown(size_t p)
  {
    return (p & ~Mask);
  }

  static constexpr FAST_TOPK_HOST_DEVICE FAST_TOPK_INLINE size_t mod(size_t x)
  {
    return (x & Mask);
  }
};


/** Helper to figure out the size of the intermediate buffer. */
template <typename T, typename IdxT, typename RATIO_T = float>
FAST_TOPK_HOST_DEVICE inline IdxT calc_buf_len(IdxT len)
{
  // When writing is skipped, only read `in`(type T).
  // When writing is not skipped, read `in_buf`(T) and `in_idx_buf`(IdxT), and write `out_buf`(T)
  // and `out_idx_buf`(IdxT).
  // The ratio between these cases determines whether to skip writing and hence the buffer size.
  constexpr RATIO_T ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
  // Even such estimation is too conservative, so further decrease buf_len by 1/8
  IdxT buf_len = len / (ratio * 8);

  // one-block kernel splits one large buffer into smaller ones, so round buf size to 256 bytes to
  // avoid alignment issues
  static_assert(is_a_power_of_two(sizeof(T)));
  static_assert(is_a_power_of_two(sizeof(IdxT)));
  constexpr IdxT aligned = 256 / std::min(sizeof(T), sizeof(IdxT));
  buf_len                = Pow2<aligned>::roundDown(buf_len);
  return buf_len;
}

/**
 * Launch a function f(T val, IdxT idx) over an array of length len, 
 * distributing elements among multiple threads in a vectorized manner if possible.
 */
template <typename T, typename IdxT, typename Func>
FAST_TOPK_DEVICE void vectorized_process(
  size_t thread_rank, size_t num_threads, const T* in, IdxT len, Func f)
{
  if constexpr (sizeof(T) >= VECTORIZED_READ_SIZE || VECTORIZED_READ_SIZE % sizeof(T) != 0) {
    for (IdxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    using wide_t      = VecBlock<T, VECTORIZED_READ_SIZE / sizeof(T)>;
    using align_bytes = Pow2<(size_t)VECTORIZED_READ_SIZE>;
    using align_elems = Pow2<wide_t::NumElems>;
    wide_t wide;

    // how many elements to skip in order to do aligned vectorized load
    size_t in_idx = reinterpret_cast<size_t>(in);
    const IdxT skip_cnt_left = std::min((IdxT)(align_bytes::roundUp(in_idx) - in_idx), len);

    // The main loop: process all aligned data
    for (IdxT i = thread_rank * wide_t::NumElems + skip_cnt_left; i + wide_t::NumElems <= len;
         i += num_threads * wide_t::NumElems) {
      wide.load(in, i);
#pragma unroll
      for (int j = 0; j < wide_t::NumElems; ++j) {
        f(wide.elems.data[j], i + j);
      }
    }

    static_assert(WarpSize >= wide_t::NumElems);
    // Processes the skipped elements on the left
    if (thread_rank < skip_cnt_left) { f(in[thread_rank], thread_rank); }
    // Processes the skipped elements on the right
    const IdxT skip_cnt_right = align_elems::mod(len - skip_cnt_left);
    const IdxT remain_i       = len - skip_cnt_right + thread_rank;
    if (remain_i < len) { f(in[remain_i], remain_i); }
  }
}

} // namespace

#endif // INTERNAL_GPU_UTIL_H