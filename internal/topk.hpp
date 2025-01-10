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

#ifndef INTERNAL_TOPK_H_
#define INTERNAL_TOPK_H_

#include <atomic>

#include "common.hpp"
#include "span.hpp"
#include "util.hpp"
#include "zip_sort.hpp"

#ifdef __AVX2__
#include "topk_avx2.hpp"
#endif

namespace internal {

namespace {

template <typename Comparator, typename T, typename... U>
void ZipSort(Comparator comp, T begin, T end, U... rest) {
  zip_sort::ZipSortBranchOptimized<Comparator, T, U...>(
      comp, 0, end - begin, zip_sort::ComputeDepthLimit(end - begin), begin,
      rest...);
}

template <typename T, typename... U>
void ZipSort(T begin, T end, U... rest) {
  ZipSort(zip_sort::DefaultComparator(), begin, end, rest...);
}

}  // namespace

template <typename DistT, typename DatapointIndexT = DatapointIndex>
class TopNeighbors {
 public:
  TopNeighbors() = default;

  explicit TopNeighbors(size_t max_results,
                        DistT epsilon = MaxOrInfinity<DistT>()) {
    Init(max_results, epsilon);
  }

  TopNeighbors(TopNeighbors &&rhs) noexcept { *this = std::move(rhs); }

  TopNeighbors &operator=(TopNeighbors &&rhs) noexcept {
    indices_ = std::move(rhs.indices_);
    distances_ = std::move(rhs.distances_);
    masks_ = std::move(rhs.masks_);
    sz_ = rhs.sz_;
    max_results_ = rhs.max_results_;
    capacity_ = rhs.capacity_;
    max_capacity_ = rhs.max_capacity_;
    epsilon_ = rhs.epsilon_.load(std::memory_order_relaxed);
    tiebreaker_idx_ = rhs.tiebreaker_idx_;
    mutator_held_ = rhs.mutator_held_;
    return *this;
  }

  void Init(size_t max_results, DistT epsilon = MaxOrInfinity<DistT>()) {
    // CHECK(!mutator_held_);
    sz_ = 0;
    epsilon_.store(epsilon, std::memory_order_relaxed);
    if (max_results_ == max_results && indices_) {
      return;
    }

    max_results_ = max_results;

    const size_t max_no_realloc_results =
        (epsilon < MaxOrInfinity<DistT>()) ? 128 : 16384;
    if (max_results == 0) {
      capacity_ = 32;
    } else if (max_results <= max_no_realloc_results) {
      capacity_ = max_capacity_ = NextMultipleOf(2 * max_results, 32);
    } else {
      capacity_ = 2 * max_no_realloc_results;

      constexpr size_t kMaxPossibleResults =
          (numeric_limits<size_t>::max() ^ size_t(31)) / 2;
      max_capacity_ =
          NextMultipleOf(2 * std::min(kMaxPossibleResults, max_results), 32);
    }

    AllocateArrays(capacity_);
    FillDistancesForASan();
  }

  void InitWithCapacity(size_t capacity) {
    epsilon_.store(MaxOrInfinity<DistT>(), std::memory_order_relaxed);
    capacity_ = max_capacity_ = capacity;
    AllocateArrays(capacity_);
    FillDistancesForASan();
  }

  FAST_TOPK_INLINE DistT epsilon() const {
    return epsilon_.load(std::memory_order_relaxed);
  }

  size_t max_results() const { return max_results_; }

  size_t capacity() const { return capacity_; }

  void PushBlock(ConstSpan<DistT> distances, DatapointIndexT base_dp_idx) {
    PushBlockToTopNeighbors(
        distances,
        [base_dp_idx](DatapointIndex offset) { return base_dp_idx + offset; },
        this);
  }

  template <typename LocalDistT>
  void PushBlock(ConstSpan<DistT> distances,
                 ConstSpan<LocalDistT> local_dp_indices,
                 DatapointIndexT base_dp_idx) {
    PushBlockToTopNeighbors(
        distances,
        [&](DatapointIndex offset) {
          return base_dp_idx + local_dp_indices[offset];
        },
        this);
  }

  FAST_TOPK_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  FinishUnsorted(size_t max_results) {
    // CHECK(!mutator_held_);
    GarbageCollect(max_results, max_results);
    auto indices = MutableSpan<DatapointIndexT>(indices_.get(), sz_);
    auto dists = MutableSpan<DistT>(distances_.get(), sz_);
    return std::make_pair(indices, dists);
  }

  FAST_TOPK_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  FinishUnsorted() {
    return FinishUnsorted(max_results_);
  }

  FAST_TOPK_INLINE pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>>
  GetRawStorage(size_t set_size_to) {
    // CHECK(set_size_to <= capacity_);
    sz_ = set_size_to;
    auto indices = MutableSpan<DatapointIndexT>(indices_.get(), sz_);
    auto dists = MutableSpan<DistT>(distances_.get(), sz_);
    return std::make_pair(indices, dists);
  }

  pair<MutableSpan<DatapointIndexT>, MutableSpan<DistT>> FinishSorted() {
    MutableSpan<DatapointIndexT> ii;
    MutableSpan<DistT> vv;
    std::tie(ii, vv) = FinishUnsorted();

    ZipSort(vv.begin(), vv.end(), ii.begin(), ii.end());

    return {ii, vv};
  }

  void FinishUnsorted(std::vector<pair<DatapointIndexT, DistT>> *results) {
    ConstSpan<DatapointIndexT> idxs;
    ConstSpan<DistT> dists;
    std::tie(idxs, dists) = FinishUnsorted();
    // DCHECK_EQ(idxs.size(), dists.size());

    results->resize(idxs.size());
    auto *rr = results->data();
    for (size_t j = 0; j < idxs.size(); j++) {
      rr[j] = std::make_pair(idxs[j], dists[j]);
    }
  }

  void FinishSorted(std::vector<pair<DatapointIndexT, DistT>> *results) {
    FinishUnsorted(results);
    ZipSort(DistanceComparatorBranchOptimized(), results->begin(),
            results->end());
  }

  class Mutator;
  friend class Mutator;

 private:
  template <typename DocidFn, typename TopN>
  static void PushBlockToTopNeighbors(ConstSpan<DistT> distances,
                                      DocidFn docid_fn, TopN *top_n) {
    typename TopN::Mutator mutator;
    top_n->AcquireMutator(&mutator);
    DatapointIndex dist_idx = 0;

#ifdef __AVX2__
    avx2::PushBlockToTopNeighbors(distances, docid_fn, mutator, dist_idx);
#endif

    DistT eps = mutator.epsilon();
    for (; dist_idx < distances.size(); ++dist_idx) {
      const DistT dist = distances[dist_idx];
      if (dist < eps) {
        if (FAST_TOPK_PREDICT_FALSE(mutator.Push(docid_fn(dist_idx), dist))) {
          mutator.GarbageCollect();
          eps = mutator.epsilon();
        }
      }
    }
  }

  FAST_TOPK_INLINE void AcquireMutator(Mutator *mutator) {
    // DCHECK(!mutator_held_);
    mutator_held_ = true;
    return mutator->Init(this);
  }

  void GarbageCollect(size_t keep_min, size_t keep_max) {
    // DCHECK_LE(keep_min, keep_max);
    if (keep_min == 0) {
      sz_ = 0;
      return;
    }
    if (sz_ <= keep_max) return;
    sz_ = ApproxNthElement(keep_min, keep_max, sz_, indices_.get(),
                           distances_.get(), masks_.get());
    const DistT old_epsilon = epsilon_;
    epsilon_ = distances_[sz_];
    /*
    SCANN_LOG_NOOP(INFO, kShouldLog)
        << DebugLogArrayContents(indices_.get(), distances_.get(), nullptr,
    sz_); SCANN_LOG_NOOP(INFO, kShouldLog) << StrFormat( "Threshold change: %f
    => %f (sz = %d)", static_cast<double>(old_epsilon),
        static_cast<double>(epsilon_), sz_);
    */
  }

  FAST_TOPK_INLINE void GarbageCollectApproximate() {
    if (capacity_ < max_capacity_) {
      return ReallocateForPureEnn();
    }
    const size_t keep_max = (max_results_ + capacity_) / 2 - 1;
    GarbageCollect(max_results_, keep_max);
  }

  void AllocateArrays(size_t capacity) {
    constexpr size_t kPadding = 96;

    capacity_ = capacity;
    indices_.reset(new DatapointIndexT[2 * capacity_ + kPadding]);
    distances_.reset(new DistT[capacity_ + kPadding]);
    masks_.reset(new uint32_t[2 * capacity_ / 32 + 2]);
  }
  void FillDistancesForASan() {
    constexpr size_t kPadding = 96;
    std::fill(distances_.get() + sz_, distances_.get() + capacity_ + kPadding,
              epsilon_);
  }
  void ReallocateForPureEnn() {
    if (sz_ < capacity_) return;

    unique_ptr<DatapointIndexT[]> old_indices = std::move(indices_);
    unique_ptr<DistT[]> old_distances = std::move(distances_);

    AllocateArrays(std::min(capacity_ * 2, max_capacity_));

    std::copy(old_indices.get(), old_indices.get() + sz_, indices_.get());
    std::copy(old_distances.get(), old_distances.get() + sz_, distances_.get());
    FillDistancesForASan();
  }

  FAST_TOPK_INLINE void ReleaseMutator(ssize_t pushes_remaining_negated) {
    mutator_held_ = false;
    sz_ = pushes_remaining_negated + capacity_;
  }

  static size_t ApproxNthElement(size_t keep_min, size_t keep_max, size_t sz,
                                 DatapointIndexT *ii, DistT *dd, uint32_t *mm) {
#ifdef __AVX2__
    return avx2::ApproxNthElementImpl(keep_min, keep_max, sz, ii, dd, mm);
#endif

    zip_sort::ZipNthElementBranchOptimized(std::less<DistT>(), keep_min - 1, dd,
                                           dd + sz, ii, ii + sz);

    dd[keep_min] = dd[keep_min - 1];
    ii[keep_min] = ii[keep_min - 1];
    return keep_min;
  }

  unique_ptr<DatapointIndexT[]> indices_;

  unique_ptr<DistT[]> distances_;

  unique_ptr<uint32_t[]> masks_;

  size_t sz_ = 0;

  size_t max_results_ = 0;

  size_t capacity_ = 0;

  size_t max_capacity_ = 0;

  std::atomic<DistT> epsilon_ = MaxOrInfinity<DistT>();
  DatapointIndexT tiebreaker_idx_ = kInvalidDatapointIndex;

  bool mutator_held_ = false;
};

template <typename DistT, typename DatapointIndexT>
class TopNeighbors<DistT, DatapointIndexT>::Mutator {
 public:
  Mutator() = default;
  Mutator(Mutator &&) = default;
  Mutator &operator=(Mutator &&) = default;
  Mutator(const Mutator &) = delete;
  Mutator &operator=(const Mutator &) = delete;

  void Release() {
    if (parent_) {
      parent_->ReleaseMutator(pushes_remaining_negated_);
      parent_ = nullptr;
    }
  }

  ~Mutator() { Release(); }

  FAST_TOPK_INLINE bool Push(DatapointIndexT dp_idx, DistT distance) {
    /*
    // DCHECK_LE(distance, epsilon());
    // DCHECK(!std::isnan(distance));
    SCANN_LOG_NOOP(1) << StrFormat("Pushing {%d, %f}", dp_idx,
                                   static_cast<double>(distance));
    // DCHECK_LT(pushes_remaining_negated_, 0);
    */
    indices_end_[pushes_remaining_negated_] = dp_idx;
    distances_end_[pushes_remaining_negated_] = distance;
    ++pushes_remaining_negated_;
    return pushes_remaining_negated_ == 0;
  }

  FAST_TOPK_INLINE DistT epsilon() const {
    return parent_->epsilon_.load(std::memory_order_relaxed);
  }

  FAST_TOPK_INLINE void GarbageCollect() {
    parent_->sz_ = parent_->capacity_ + pushes_remaining_negated_;

    parent_->GarbageCollectApproximate();

    InitImpl();
  }

 private:
  FAST_TOPK_INLINE void Init(TopNeighbors *parent) {
    // DCHECK(!parent_);
    parent_ = parent;
    InitImpl();
  }

  FAST_TOPK_INLINE void InitImpl() {
    // DCHECK(parent_);
    indices_end_ = parent_->indices_.get() + parent_->capacity_;
    distances_end_ = parent_->distances_.get() + parent_->capacity_;
    pushes_remaining_negated_ = parent_->sz_ - parent_->capacity_;
  }

  TopNeighbors *parent_ = nullptr;

  DatapointIndexT *indices_end_;

  DistT *distances_end_;

  ssize_t pushes_remaining_negated_;

  friend class TopNeighbors;
};

}  // namespace internal

#endif  // INTERNAL_TOPK_H_
