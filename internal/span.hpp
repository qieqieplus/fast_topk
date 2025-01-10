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

#ifndef INTERNAL_SPAN_H_
#define INTERNAL_SPAN_H_

#include <array>
#include <iterator>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace internal {

template <typename T>
using ConstSpan = std::span<const T>;

template <typename T>
using MutableSpan = std::span<T>;

// ===================================
// Type Traits and Concepts Definitions
// ===================================

// Type alias for the element type of container (removes const/volatile
// qualifiers)
template <typename Container>
using element_t = std::remove_cv_t<std::ranges::range_value_t<Container>>;

// Concept to check if a type has data() and size() members
template <typename T>
concept ContiguousContainer = requires(T a) {
  { std::data(a) }
  ->std::convertible_to<element_t<T>*>;
  { std::size(a) }
  ->std::convertible_to<std::size_t>;
};

// ===================================
// MakeSpan: Mutable Span Factory
// ===================================

// MakeSpan for non-const containers using concepts
template <ContiguousContainer Container>
constexpr auto MakeSpan(Container& c) noexcept
    -> std::span<element_t<Container>> {
  return std::span<element_t<Container>>(std::data(c), std::size(c));
}

// MakeSpan for C-style arrays (deduced extent)
template <typename T, std::size_t N>
constexpr std::span<T, N> MakeSpan(T (&arr)[N]) noexcept {
  return std::span<T, N>(arr);
}

// MakeSpan for pointer and size (mutable)
template <typename T>
constexpr std::span<T> MakeSpan(T* ptr, std::size_t size) noexcept {
  return std::span<T>(ptr, size);
}

// MakeSpan for pointer ranges [begin, end) (mutable)
template <typename T>
constexpr std::span<T> MakeSpan(T* begin, T* end) noexcept {
  return std::span<T>(begin, static_cast<std::size_t>(end - begin));
}

// ===================================
// MakeConstSpan: Const Span Factory
// ===================================

// MakeConstSpan for const containers using concepts
template <ContiguousContainer Container>
constexpr auto MakeConstSpan(const Container& c) noexcept
    -> std::span<const element_t<Container>> {
  return std::span<const element_t<Container>>(std::data(c), std::size(c));
}

// MakeConstSpan for C-style arrays (deduced extent)
template <typename T, std::size_t N>
constexpr std::span<const T, N> MakeConstSpan(const T (&arr)[N]) noexcept {
  return std::span<const T, N>(arr);
}

// MakeConstSpan for pointer and size (const)
template <typename T>
constexpr std::span<const T> MakeConstSpan(const T* ptr,
                                           std::size_t size) noexcept {
  return std::span<const T>(ptr, size);
}

// MakeConstSpan for pointer ranges [begin, end) (const)
template <typename T>
constexpr std::span<const T> MakeConstSpan(const T* begin,
                                           const T* end) noexcept {
  return std::span<const T>(begin, static_cast<std::size_t>(end - begin));
}

}  // namespace internal

#endif  // INTERNAL_SPAN_H_
