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

#ifndef INTERNAL_TOPK_AVX2_H_
#define INTERNAL_TOPK_AVX2_H_

#include "internal/avx2.hpp"
#include "internal/bits.hpp"
#include "internal/common.hpp"
#include "internal/span.hpp"
#include "internal/util.hpp"
#include "internal/zip_sort.hpp"

namespace internal {

namespace {

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_INLINE bool CompIV(DatapointIndexT idx_a, DatapointIndexT idx_b,
                             DistT value_a, DistT value_b) {
  const bool is_eq_or_nan =
      value_a == value_b || std::isunordered(value_a, value_b);
  if (FAST_TOPK_PREDICT_FALSE(is_eq_or_nan)) {
    return idx_a < idx_b;
  }
  return value_a < value_b;
}

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_INLINE void ZipSwap(size_t a, size_t b, DatapointIndexT *indices,
                              DistT *values) {
  std::swap(indices[a], indices[b]);
  std::swap(values[a], values[b]);
}

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_INLINE void CompOrSwap(size_t a, size_t b, DatapointIndexT *indices,
                                 DistT *values) {
  if (!CompIV(indices[a], indices[b], values[a], values[b])) {
    ZipSwap(a, b, indices, values);
  }
}

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_INLINE void SelectionSort(DatapointIndexT *indices, DistT *values,
                                    size_t sz) {
  // DCHECK_LE(sz, 3);
  switch (sz) {
    case 3:
      CompOrSwap(0, 1, indices, values);
      CompOrSwap(1, 2, indices, values);
      FAST_TOPK_FALLTHROUGH_INTENDED;
    case 2:
      CompOrSwap(0, 1, indices, values);
      FAST_TOPK_FALLTHROUGH_INTENDED;
    case 1:
      break;
  }
}

template <typename DistT>
FAST_TOPK_INLINE DistT FastMedianOf3(DistT v0, DistT v1, DistT v2) {
  DistT big = std::max(v0, v1);
  DistT sml = std::min(v0, v1);
  return std::max(sml, std::min(big, v2));
}

}  // namespace

namespace avx2 {

template <bool kIsEquality, typename DistT>
FAST_TOPK_AVX2_INLINE size_t CalculateSwapMasks(const DistT *values,
                                                uint32_t *masks, size_t n_masks,
                                                uint32_t final_mask,
                                                DistT threshold) {
  const auto simd_threshold = Avx2<DistT>::Broadcast(threshold);

  size_t n_kept = 0;
  for (size_t j = 0; j < n_masks; j++) {
    auto vals = Avx2For<DistT, 32>::Load(values + 32 * j);
    const uint32_t mask = kIsEquality
                              ? GetComparisonMask(vals == simd_threshold)
                              : GetComparisonMask(vals < simd_threshold);
    n_kept += std::popcount(mask);
    masks[j] = mask;
  }

  uint32_t &last_mask = masks[n_masks - 1];
  n_kept -= std::popcount(last_mask);
  last_mask &= final_mask;
  n_kept += std::popcount(last_mask);

  return n_kept;
}

template <typename DistT>
FAST_TOPK_AVX2_INLINE size_t CalculateLtSwapMasks(const DistT *values,
                                                  uint32_t *masks,
                                                  size_t n_masks,
                                                  uint32_t final_mask,
                                                  DistT threshold) {
  return CalculateSwapMasks<false>(values, masks, n_masks, final_mask,
                                   threshold);
}

template <typename DistT>
FAST_TOPK_AVX2_INLINE size_t CalculateEqSwapMasks(const DistT *values,
                                                  uint32_t *masks,
                                                  size_t n_masks,
                                                  uint32_t final_mask,
                                                  DistT threshold) {
  return CalculateSwapMasks<true>(values, masks, n_masks, final_mask,
                                  threshold);
}

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_AVX2_OUTLINE size_t UseMasksToPartition(DatapointIndexT *indices,
                                                  DistT *values,
                                                  uint32_t *masks,
                                                  size_t n_masks) {
  size_t mask_idx1 = 0;
  size_t mask_idx2 = n_masks - 1;

  uint32_t mask1 = ~masks[mask_idx1];

  uint32_t mask2 = masks[mask_idx2];

  if (n_masks > 1) {
    for (;;) {
      while (mask1 && mask2) {
        const int offset1 = std::countr_zero(mask1);
        const int offset2 = std::countr_zero(mask2);
        mask1 &= mask1 - 1;
        mask2 &= mask2 - 1;
        const size_t idx1 = mask_idx1 * 32 + offset1;
        const size_t idx2 = mask_idx2 * 32 + offset2;
        ZipSwap(idx1, idx2, indices, values);
      }
      if (!mask1) {
        ++mask_idx1;
        if (mask_idx1 == mask_idx2) {
          break;
        }
        mask1 = ~masks[mask_idx1];
      }
      if (!mask2) {
        --mask_idx2;
        if (mask_idx1 == mask_idx2) {
          mask2 = ~mask1;
          break;
        }
        mask2 = masks[mask_idx2];
      }
    }
  }

  size_t write_offset = mask_idx2 * 32;
  while (mask2) {
    const int offset = std::countr_zero(mask2);
    const size_t idx = mask_idx2 * 32 + offset;
    mask2 &= mask2 - 1;
    ZipSwap(write_offset++, idx, indices, values);
  }
  return write_offset;
}

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_AVX2_INLINE size_t UseMaskToCompact(DatapointIndexT *indices,
                                              DistT *values, uint32_t mask) {
  size_t write_idx = 0;
  while (mask) {
    const int offset = std::countr_zero(mask);
    mask &= (mask - 1);
    indices[write_idx] = indices[offset];
    values[write_idx] = values[offset];
    ++write_idx;
  }
  return write_idx;
}

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_AVX2_OUTLINE size_t UseMasksToCompactDoublePorted(
    DatapointIndexT *indices, DistT *values, uint32_t *masks, size_t n_masks) {
  // DCHECK_GE(n_masks, 2);

  std::copy(values, values + 64, values + n_masks * 32);
  std::copy(indices, indices + 64, indices + n_masks * 32);
  std::copy(masks, masks + 2, masks + n_masks);
  n_masks += 2;

  uint32_t mask1 = masks[2];
  DatapointIndexT *indices1 = indices + 2 * 32;
  DistT *values1 = values + 2 * 32;

  uint32_t mask2 = masks[3];
  DatapointIndexT *indices2 = indices + 3 * 32;
  DistT *values2 = values + 3 * 32;

  uint32_t *masks_ptr = masks + 3;
  uint32_t *masks_end = masks + n_masks;

  DatapointIndexT *indices_write_ptr = indices;
  DistT *values_write_ptr = values;

  for (;;) {
    if (FAST_TOPK_PREDICT_FALSE(!mask1 || !mask2)) {
      bool proceed_to_cooldown = false;

      do {
        if (!mask1) {
          mask1 = mask2;
          indices1 = indices2;
          values1 = values2;
        }

        if (++masks_ptr >= masks_end) {
          proceed_to_cooldown = true;
          break;
        }

        mask2 = *masks_ptr;
        indices2 += 32;
        values2 += 32;

      } while (FAST_TOPK_PREDICT_FALSE(!mask1 || !mask2));

      if (proceed_to_cooldown) break;
    }
    // DCHECK(mask1);
    // DCHECK(mask2);
    // DCHECK_LT(indices_write_ptr, indices1);
    // DCHECK_LT(indices_write_ptr, indices2);
    // DCHECK_LT(values_write_ptr, values1);
    // DCHECK_LT(values_write_ptr, values2);

    const int offset2 = std::countr_zero(mask2);
    const int offset1 = std::countr_zero(mask1);

    *indices_write_ptr++ = indices2[offset2];
    *values_write_ptr++ = values2[offset2];

    *indices_write_ptr++ = indices1[offset1];
    *values_write_ptr++ = values1[offset1];

    mask2 &= (mask2 - 1);
    mask1 &= (mask1 - 1);
  }

  while (mask1) {
    const int offset1 = std::countr_zero(mask1);
    mask1 &= (mask1 - 1);
    *indices_write_ptr++ = indices1[offset1];
    *values_write_ptr++ = values1[offset1];
  }

  // DCHECK_EQ(indices_write_ptr - indices, values_write_ptr - values);
  return indices_write_ptr - indices;
}

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_AVX2_INLINE size_t UseMasksToCompact(DatapointIndexT *indices,
                                               DistT *values, uint32_t *masks,
                                               size_t n_masks) {
  if (n_masks == 1) {
    return UseMaskToCompact(indices, values, masks[0]);
  }
  return UseMasksToCompactDoublePorted(indices, values, masks, n_masks);
}

template <typename DatapointIndexT>
FAST_TOPK_AVX2_OUTLINE size_t UseMasksToSelect(DatapointIndexT *to,
                                               DatapointIndexT *from,
                                               uint32_t *masks,
                                               size_t n_masks) {
  size_t write_idx = 0;
  for (size_t j = 0; j < n_masks; j++) {
    uint32_t mask = masks[j];
    while (mask) {
      const int offset = std::countr_zero(mask);
      mask &= (mask - 1);
      to[write_idx++] = from[32 * j + offset];
    }
  }
  return write_idx;
}

template <typename DistT>
FAST_TOPK_AVX2_INLINE DistT UseMasksToFindNewMedian(DistT *values,
                                                    uint32_t *lt_masks,
                                                    uint32_t *eq_masks,
                                                    size_t n_masks) {
  size_t n_idx = 0;
  DistT vals[3];
  for (size_t j = 0; j < n_masks; j++) {
    uint32_t mask = ~(lt_masks[j] + eq_masks[j]);
    while (mask) {
      const int offset = std::countr_zero(mask);
      mask &= (mask - 1);
      const size_t idx = j * 32 + offset;
      vals[n_idx++] = values[idx];
      if (n_idx == 3) {
        return FastMedianOf3(vals[0], vals[1], vals[2]);
      }
    }
  }

  return vals[0];
}

template <typename DistT>
FAST_TOPK_AVX2_INLINE DistT DecrementThreshold(DistT threshold) {
  if (IsFloatingType<DistT>()) {
    constexpr DistT kNegativeInfinity = -numeric_limits<DistT>::infinity();
    return std::nextafter(threshold, kNegativeInfinity);
  } else {
    return threshold - 1;
  }
}

template <typename DistT, typename DatapointIndexT>
FAST_TOPK_AVX2_OUTLINE size_t ApproxNthElementImpl(size_t keep_min,
                                                   size_t keep_max, size_t sz,
                                                   DatapointIndexT *indices,
                                                   DistT *values,
                                                   uint32_t *masks) {
  // DCHECK_LE(keep_min, keep_max);
  // DCHECK_LT(keep_max, sz);

  DistT threshold_value;
  size_t n_already_kept = 0;
  bool skip_threshold_selection = false;

  for (;;) {
    if (!skip_threshold_selection) {
      // DCHECK_GT(sz, keep_min);
      // DCHECK_GT(sz, keep_max);
      // DCHECK_LE(keep_min, keep_max);
      if (FAST_TOPK_PREDICT_FALSE(sz <= 3)) {
        const size_t final_size = n_already_kept + keep_min;
        /*
        SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
        "SelectionSort for range:[%d,%d) = %d; will return idx %d",
        n_already_kept, final_size, sz, keep_min);
        */
        SelectionSort(indices, values, sz);

        values[keep_min] = values[keep_min - 1];
        indices[keep_min] = indices[keep_min - 1];
        return final_size;
      }

      threshold_value =
          FastMedianOf3(values[0], values[sz / 2], values[sz - 1]);
      // Pushing NaN values to Top-K is expressly forbidden
      if (FAST_TOPK_PREDICT_FALSE(std::isnan(threshold_value))) {
        //
      }
    }
    skip_threshold_selection = false;

    const uint32_t final_mask = GetFinalMask32(sz);
    const size_t n_masks = DivRoundUp(sz, 32);
    size_t n_kept = CalculateLtSwapMasks(values, masks, n_masks, final_mask,
                                         threshold_value);
    /*
    SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
    "CalculateLtSwapMasks w/ %s returns %d / %d bits (goal=%d). "
    "final_mask=%08x",
    absl::StrCat(threshold_value), n_kept, sz, keep_min, final_mask);
    SCANN_LOG_NOOP(INFO, kShouldLog)
    << DebugLogArrayContents(indices, values, masks, sz);
    */

    auto handle_overly_picky_pivot = [&]() FAST_TOPK_AVX2_INLINE_LAMBDA {
      if (n_kept < sz * 3 / 4) {
        /*
        SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
        "UseMasksToFindNewMedian; range:[%d,%d) = %d; masks=%d << "
        "goal=%d",
        n_already_kept, n_already_kept + sz, sz, n_kept, keep_min);
        */
        uint32_t *eq_masks = masks + n_masks;
        threshold_value =
            UseMasksToFindNewMedian(values, masks, eq_masks, n_masks);
        skip_threshold_selection = true;
      } else {
        /*
        SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
        "UseMasksToPartition; range:[%d,%d) = %d; masks=%d < goal=%d;",
        n_already_kept, n_already_kept + sz, sz, n_kept, keep_min);
        */
        const size_t pivot_idx =
            UseMasksToPartition(indices, values, masks, n_masks);
        // DCHECK_EQ(pivot_idx, n_kept);

        n_already_kept += n_kept;
        keep_min -= n_kept;
        keep_max -= n_kept;
        sz -= n_kept;
        indices += n_kept;
        values += n_kept;
      }
    };

    const bool compute_eq_masks = n_kept < keep_min;
    if (compute_eq_masks) {
      const size_t n_needed = keep_min - n_kept;
      size_t n_found;
      uint32_t *eq_masks = masks + n_masks;
      n_found = CalculateEqSwapMasks(values, eq_masks, n_masks, final_mask,
                                     threshold_value);
      /*
      SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
      "CalculateEqSwapMasks w/ %s returns %d / %d bits (goal=%d). "
      "final_mask=%08x",
      absl::StrCat(threshold_value), n_found, sz, n_needed, final_mask);
      */

      if (n_found < n_needed) {
        handle_overly_picky_pivot();
        continue;
      }

      DatapointIndexT *scratch = indices + 32 * n_masks + 64;
      const size_t sz = UseMasksToSelect(scratch, indices, eq_masks, n_masks);
      // DCHECK_EQ(sz, n_found);

      if (FAST_TOPK_PREDICT_TRUE(n_found > n_needed)) {
        zip_sort::ZipNthElementBranchOptimized(std::less<DatapointIndexT>(),
                                               n_needed - 1, scratch,
                                               scratch + n_found);
      }
    }

    /*
    SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
    "UseMasksToCompact; range:[%d,%d) = %d; masks=%d >= goal=%d;",
    n_already_kept, n_already_kept + sz, sz, n_kept, keep_min);
    */
    sz = UseMasksToCompact(indices, values, masks, n_masks);
    // DCHECK_EQ(sz, n_kept);

    if (n_kept > keep_max) {
      continue;
    }

    DatapointIndexT tiebreaker_idx = numeric_limits<DatapointIndexT>::max();
    if (compute_eq_masks) {
      const size_t n_needed = keep_min - n_kept;
      /*
      SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat(
      "RecoverPivotElement(s); %f x %d", threshold_value, n_needed);
      */
      DatapointIndexT *scratch = indices + 32 * n_masks + 64;
      std::copy(scratch, scratch + n_needed, indices + n_kept);
      std::fill(values + n_kept, values + n_kept + n_needed, threshold_value);
      n_kept = keep_min;
      tiebreaker_idx = scratch[n_needed - 1];
    } else {
      threshold_value = DecrementThreshold(threshold_value);
    }

    // DCHECK_GE(n_kept, keep_min);
    // DCHECK_LE(n_kept, keep_max);

    values[n_kept] = threshold_value;
    indices[n_kept] = tiebreaker_idx;

    const size_t final_size = n_already_kept + n_kept;
    /*
    SCANN_LOG_NOOP(INFO, kShouldLog)
    << StrFormat("Returning %d elements w/ pivot = %f", final_size,
    static_cast<double>(values[n_kept]));
    */
    return final_size;
  }
}

template <typename DistT, typename DocidFn, typename Mutator>
FAST_TOPK_AVX2_INLINE static void PushBlockToTopNeighbors(
    ConstSpan<DistT> distances, DocidFn docid_fn, Mutator &mutator,
    DatapointIndex &dist_idx) {
  Avx2<DistT> sse_epsilon = mutator.epsilon();
  constexpr size_t kNumFloatsPerSimdRegister =
      Avx2<DistT>::kElementsPerRegister;
  const size_t num_avx2_registers =
      distances.size() / kNumFloatsPerSimdRegister;
  for (uint32_t simd_idx = 0; simd_idx < num_avx2_registers; simd_idx++) {
    const uint32_t i0 = simd_idx * kNumFloatsPerSimdRegister;
    Avx2<float> simd_dists = Avx2<DistT>::Load(&distances[i0]);
    uint32_t push_mask = GetComparisonMask(simd_dists <= sse_epsilon);
    while (FAST_TOPK_PREDICT_FALSE(push_mask)) {
      const int offset = std::countr_zero(push_mask);
      push_mask &= (push_mask - 1);
      if (FAST_TOPK_PREDICT_FALSE(
              mutator.Push(docid_fn(i0 + offset), (*simd_dists)[offset]))) {
        mutator.GarbageCollect();
        sse_epsilon = mutator.epsilon();

        push_mask &= GetComparisonMask(simd_dists < sse_epsilon);
      }
    }
  }
  dist_idx = num_avx2_registers * kNumFloatsPerSimdRegister;
}

}  // namespace avx2
}  // namespace internal

#endif  // INTERNAL_TOPK_AVX2_H_