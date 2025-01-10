#include <algorithm>
#include <queue>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"
#include "fast_topk.hpp"

using namespace fast_topk;
using kv_pair = std::pair<size_t, float>;
const std::vector<int64_t> k_values = {20, 50, 100, 200, 500, 1000, 2000, 5000};
const std::vector<int64_t> n_values = {10 * 1000 * 1000};
const std::vector<std::vector<int64_t>> arg_lists = {k_values, n_values};

namespace {

// Helper function to generate random floats
std::vector<float> GenerateRandomFloats(size_t count) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
  std::vector<float> data(count);
  std::generate(data.begin(), data.end(), [&]() { return distrib(gen); });
  return data;
}

}  // namespace

// Benchmarking FastTopNeighbors::PushBlock and FinishSorted
static void FastTopNeighbors(benchmark::State& state) {
  int k = state.range(0);
  int n = state.range(1);
  std::vector<float> data = GenerateRandomFloats(n);

  for (auto _ : state) {
    TopNeighbors<float> top_n(k);
    top_n.PushBlock(MakeConstSpan(data), 0);
    std::vector<kv_pair> results(k);
    top_n.FinishSorted(&results);
    // benchmark::DoNotOptimize(results);
  }
  state.SetComplexityN(state.range(1));
}
BENCHMARK(FastTopNeighbors)
    ->ArgsProduct(arg_lists)
    ->Complexity(benchmark::oNLogN);

// Benchmarking HeapBasedTopK
static void HeapBasedTopK(benchmark::State& state) {
  int k = state.range(0);
  int n = state.range(1);
  std::vector<float> data = GenerateRandomFloats(n);

  for (auto _ : state) {
    std::priority_queue<kv_pair, std::vector<kv_pair>> max_heap;
    for (size_t i = 0; i < data.size(); ++i) {
      if (max_heap.size() < k) {
        max_heap.push({i, data[i]});
      } else if (data[i] < max_heap.top().second) {
        max_heap.pop();
        max_heap.push({i, data[i]});
      }
    }
    std::vector<kv_pair> result;
    result.reserve(k);
    while (!max_heap.empty()) {
      result.emplace_back(max_heap.top());
      max_heap.pop();
    }
  }
  state.SetComplexityN(state.range(1));
}
BENCHMARK(HeapBasedTopK)
    ->ArgsProduct(arg_lists)
    ->Complexity(benchmark::oNLogN);

/*
static void PartialSort(benchmark::State& state) {
  int k = state.range(0);
  int n = state.range(1);
  std::vector<float> data = GenerateRandomFloats(n);
  std::vector<kv_pair> kv_data(n);
  for (size_t i = 0; i < n; ++i) {
    kv_data[i] = {i, data[i]};
  }

  for (auto _ : state) {
    state.PauseTiming();
    std::vector<kv_pair> kv_copy = kv_data;
    state.ResumeTiming();

    std::partial_sort(
        kv_copy.begin(), kv_copy.begin() + k, kv_copy.end(),
        [](const kv_pair& a, const kv_pair& b) {
          return a.second < b.second;
        });
  }
  state.SetComplexityN(state.range(1));
}
BENCHMARK(PartialSort)
    ->ArgsProduct(arg_lists)
    ->Complexity(benchmark::oNLogN);
*/

// Boilerplate to register benchmarks
BENCHMARK_MAIN();
