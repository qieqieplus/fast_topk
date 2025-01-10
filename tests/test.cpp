#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <random>

#include "fast_topk.hpp"

using namespace fast_topk;
using kv_pair = std::pair<size_t, float>;

template <typename T, size_t array_size>
std::array<T, array_size> generateRandomValues() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::array<T, array_size> values;
  for (size_t i = 0; i < array_size; ++i) {
    values[i] = dist(gen);
  }
  return values;
}

TEST(FastTopK, Sorted) {
  constexpr size_t array_size = 10000;
  constexpr size_t block_offset = 256;
  constexpr size_t top_k_limit = 10;

  auto rand_values = generateRandomValues<float, array_size>();

  std::array<kv_pair, array_size> data;
  for (size_t i = 0; i < array_size; ++i) {
    data[i] = {i + block_offset, rand_values[i]};
  }

  // Ground truth
  std::partial_sort(
      data.begin(), data.begin() + top_k_limit, data.end(),
      [](const kv_pair& a, const kv_pair& b) { return a.second < b.second; });

  std::array<size_t, top_k_limit> truth_indices;
  std::array<float, top_k_limit> truth_values;
  for (size_t i = 0; i < top_k_limit; ++i) {
    std::tie(truth_indices[i], truth_values[i]) = data[i];
  }

  // Initialize
  TopNeighbors<float> top_k(top_k_limit, std::numeric_limits<float>::max());
  top_k.PushBlock(MakeConstSpan(rand_values), block_offset);
  // Retrieve sorted results
  MutableSpan<DatapointIndex> indices;
  MutableSpan<float> values;
  std::tie(indices, values) = top_k.FinishSorted();

  EXPECT_EQ(indices.size(), top_k_limit);
  EXPECT_EQ(indices.size(), values.size());

  for (size_t i = 0; i < top_k_limit; ++i) {
    EXPECT_EQ(truth_indices[i], indices[i]);
    EXPECT_EQ(truth_values[i], values[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
