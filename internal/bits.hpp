// This fast top-k library adopts code from ScaNN
// (https://github.com/google-research/google-research/tree/master/scann)
//
// Copyright 2024 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef INTERNAL_BITS_H_
#define INTERNAL_BITS_H_

#include <bit>

#include "common.hpp"

namespace internal {

inline size_t NextPowerOfTwo(size_t x) {
  return (x & (x - 1)) ? static_cast<size_t>(1)
                             << (sizeof(size_t) * 8 - std::countl_zero(x))
                       : x;
}

template <typename T>
inline bool IsPowerOfTwo(const T x) {
  return x && ((x & (x - 1)) == 0);
}

template <typename T>
inline int Log2Floor(T n) {
  return std::bit_width(n) - 1;
}

template <typename T>
inline int Log2Ceiling(T n) {
  int floor = Log2Floor(n);
  return (n & (n - 1)) ? (floor + 1) : floor;
}

FAST_TOPK_INLINE uint32_t GetFinalMask32(const size_t num_dp) {
  const size_t remainder_bits = num_dp % 32;
  return remainder_bits ? (1u << remainder_bits) - 1 : 0xFFFFFFFF;
}

}  // namespace internal

#endif  // INTERNAL_BITS_H_
