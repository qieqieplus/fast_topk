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

#ifndef INTERNAL_UTIL_H
#define INTERNAL_UTIL_H

#include <type_traits>
#include <utility>

#include "common.hpp"

template <typename T, typename U>
constexpr bool IsSame() {
  return std::is_same_v<T, U>;
}

template <typename T, typename... UU>
constexpr bool IsSameAny() {
  return (std::is_same_v<T, UU> || ...);
}

template <size_t... kInts>
inline constexpr size_t index_sequence_sum_v = (kInts + ...);

template <size_t... kInts>
inline constexpr size_t index_sequence_last_v = (kInts, ...);

template <typename Seq>
struct index_sequence_all_but_last_impl;

template <size_t First, size_t... Rest>
struct index_sequence_all_but_last_impl<std::index_sequence<First, Rest...>> {
  using type = std::conditional_t<(sizeof...(Rest) > 1),
                                  typename index_sequence_all_but_last_impl<
                                      std::index_sequence<Rest...>>::type,
                                  std::index_sequence<First>>;
};

template <size_t Last>
struct index_sequence_all_but_last_impl<std::index_sequence<Last>> {
  using type = std::index_sequence<>;
};

// Alias for cleaner usage
template <size_t... Ints>
using index_sequence_all_but_last_t = typename index_sequence_all_but_last_impl<
    std::index_sequence<Ints...>>::type;

template <typename T>
FAST_TOPK_INLINE constexpr bool IsFloatingType() {
  return std::is_floating_point<std::decay_t<T>>::value;
}

template <typename T>
constexpr T MaxOrInfinity() {
  // This can be customized for integral or floating-point
  return std::numeric_limits<T>::has_infinity
             ? std::numeric_limits<T>::infinity()
             : std::numeric_limits<T>::max();
}

template <typename Int, typename DenomInt>
constexpr Int DivRoundUp(Int num, DenomInt denom) {
  return (num + static_cast<Int>(denom) - static_cast<Int>(1)) /
         static_cast<Int>(denom);
}

template <typename Int, typename DenomInt>
constexpr Int NextMultipleOf(Int num, DenomInt denom) {
  return DivRoundUp(num, denom) * denom;
}

class DistanceComparator {
 public:
  template <typename DistanceT, typename DatapointIndexT = DatapointIndex>
  bool operator()(const pair<DatapointIndexT, DistanceT>& a,
                  const pair<DatapointIndexT, DistanceT>& b) const {
    /*
    if (IsFloatingType<DistanceT>()) {
      DCHECK(!std::isnan(a.second));
      DCHECK(!std::isnan(b.second));
    }
    */
    if (a.second < b.second) return true;
    if (a.second > b.second) return false;
    return a.first < b.first;
  }

  template <typename DistanceT, typename DatapointIndexT = DatapointIndex>
  static bool CompareBranchOptimized(
      const pair<DatapointIndexT, DistanceT>& a,
      const pair<DatapointIndexT, DistanceT>& b) {
    const bool is_eq_or_nan =
        (a.second == b.second ||
         (IsFloatingType<DistanceT>() && std::isunordered(a.second, b.second)));

    if (FAST_TOPK_PREDICT_FALSE(is_eq_or_nan)) {
      return a.first < b.first;
    }
    return a.second < b.second;
  }
};

class DistanceComparatorBranchOptimized {
 public:
  template <typename DistanceT, typename DatapointIndexT>
  bool operator()(const pair<DatapointIndexT, DistanceT>& a,
                  const pair<DatapointIndexT, DistanceT>& b) const {
    return DistanceComparator::CompareBranchOptimized(a, b);
  }
};

#endif  // INTERNAL_UTIL_H
