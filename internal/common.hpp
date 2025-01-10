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

#ifndef INTERNAL_COMMON_H
#define INTERNAL_COMMON_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

using ::std::array;
using ::std::index_sequence;
using ::std::numeric_limits;
using ::std::pair;
using ::std::unique_ptr;

#ifdef __has_attribute
#define FAST_TOPK_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define FAST_TOPK_HAVE_ATTRIBUTE(x) 0
#endif

#if FAST_TOPK_HAVE_ATTRIBUTE(format) || \
    (defined(__GNUC__) && !defined(__clang__))
#define FAST_TOPK_PRINTF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__printf__, string_index, first_to_check)))
#define FAST_TOPK_SCANF_ATTRIBUTE(string_index, first_to_check) \
  __attribute__((__format__(__scanf__, string_index, first_to_check)))
#else
#define FAST_TOPK_PRINTF_ATTRIBUTE(string_index, first_to_check)
#define FAST_TOPK_SCANF_ATTRIBUTE(string_index, first_to_check)
#endif

// FAST_TOPK_ATTRIBUTE_ALWAYS_INLINE
// FAST_TOPK_ATTRIBUTE_NOINLINE
//
// Forces functions to either inline or not inline. Introduced in gcc 3.1.
#if FAST_TOPK_HAVE_ATTRIBUTE(always_inline) || \
    (defined(__GNUC__) && !defined(__clang__))
#define FAST_TOPK_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#define FAST_TOPK_HAVE_ATTRIBUTE_ALWAYS_INLINE 1
#else
#define FAST_TOPK_ATTRIBUTE_ALWAYS_INLINE
#endif

#if FAST_TOPK_HAVE_ATTRIBUTE(noinline) || \
    (defined(__GNUC__) && !defined(__clang__))
#define FAST_TOPK_ATTRIBUTE_NOINLINE __attribute__((noinline))
#define FAST_TOPK_HAVE_ATTRIBUTE_NOINLINE 1
#else
#define FAST_TOPK_ATTRIBUTE_NOINLINE
#endif

#define FAST_TOPK_INLINE inline FAST_TOPK_ATTRIBUTE_ALWAYS_INLINE

#define FAST_TOPK_INLINE_LAMBDA FAST_TOPK_ATTRIBUTE_ALWAYS_INLINE

#define FAST_TOPK_OUTLINE FAST_TOPK_ATTRIBUTE_NOINLINE

#ifdef __x86_64__
#define FAST_TOPK_AVX2 __attribute((target("avx,avx2,fma")))
#else
#define FAST_TOPK_AVX2
#endif

#define FAST_TOPK_AVX2_INLINE FAST_TOPK_AVX2 FAST_TOPK_INLINE
#define FAST_TOPK_AVX2_INLINE_LAMBDA FAST_TOPK_AVX2 FAST_TOPK_INLINE_LAMBDA
#define FAST_TOPK_AVX2_OUTLINE FAST_TOPK_AVX2 FAST_TOPK_OUTLINE

#ifdef __has_builtin
#define FAST_TOPK_HAVE_BUILTIN(x) __has_builtin(x)
#else
#define FAST_TOPK_HAVE_BUILTIN(x) 0
#endif

#ifdef __has_feature
#define FAST_TOPK_HAVE_FEATURE(f) __has_feature(f)
#else
#define FAST_TOPK_HAVE_FEATURE(f) 0
#endif

#if defined(__cplusplus) && defined(__has_cpp_attribute)
// NOTE: requiring __cplusplus above should not be necessary, but
// works around https://bugs.llvm.org/show_bug.cgi?id=23435.
#define FAST_TOPK_HAVE_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define FAST_TOPK_HAVE_CPP_ATTRIBUTE(x) 0
#endif

#if FAST_TOPK_HAVE_BUILTIN(__builtin_expect)
#define FAST_TOPK_PREDICT_FALSE(x) (__builtin_expect(false || (x), false))
#define FAST_TOPK_PREDICT_TRUE(x) (__builtin_expect(false || (x), true))
#else
#define FAST_TOPK_PREDICT_FALSE(x) (x)
#define FAST_TOPK_PREDICT_TRUE(x) (x)
#endif

#ifdef FAST_TOPK_FALLTHROUGH_INTENDED
#error "FAST_TOPK_FALLTHROUGH_INTENDED should not be defined."
#elif FAST_TOPK_HAVE_CPP_ATTRIBUTE(fallthrough)
#define FAST_TOPK_FALLTHROUGH_INTENDED [[fallthrough]]
#elif FAST_TOPK_HAVE_CPP_ATTRIBUTE(clang::fallthrough)
#define FAST_TOPK_FALLTHROUGH_INTENDED [[clang::fallthrough]]
#elif FAST_TOPK_HAVE_CPP_ATTRIBUTE(gnu::fallthrough)
#define FAST_TOPK_FALLTHROUGH_INTENDED [[gnu::fallthrough]]
#else
#define FAST_TOPK_FALLTHROUGH_INTENDED \
  do {                                 \
  } while (0)
#endif

using DatapointIndex = size_t;
enum : DatapointIndex {
  kInvalidDatapointIndex = numeric_limits<DatapointIndex>::max(),
};

using DimensionIndex = size_t;
enum : DimensionIndex {
  kInvalidDimension = numeric_limits<DimensionIndex>::max(),
};

#endif  // INTERNAL_COMMON_H
