#pragma once

#include <cuda_fp16.h>
#include "internal/gpu/common.hpp"

namespace internal::gpu {

/**
 * @brief Storage type traits struct.
 *
 * Defines underlying storage types for vectorized operations based on
 * data type and vector length.
 *
 * @tparam DataType Element data type.
 * @tparam N Vector length.
 */
template <typename DataType, int N>
struct StorageType {};

// Explicit specializations for StorageType
template <>
struct StorageType<bool, 1> {
  static_assert(sizeof(bool) == sizeof(int8_t), "StorageType bool size assumption failed");
  using UnderlyingType = int8_t;
};

template <>
struct StorageType<bool, 2> {
  using UnderlyingType = int16_t;
};

template <>
struct StorageType<bool, 4> {
  using UnderlyingType = int32_t;
};

template <>
struct StorageType<bool, 8> {
  using UnderlyingType = int2;
};

template <>
struct StorageType<bool, 16> {
  using UnderlyingType = int4;
};

template <>
struct StorageType<int8_t, 1> {
  using UnderlyingType = int8_t;
};

template <>
struct StorageType<int8_t, 2> {
  using UnderlyingType = int16_t;
};

template <>
struct StorageType<int8_t, 4> {
  using UnderlyingType = int32_t;
};

template <>
struct StorageType<int8_t, 8> {
  using UnderlyingType = int2;
};

template <>
struct StorageType<int8_t, 16> {
  using UnderlyingType = int4;
};

template <>
struct StorageType<uint8_t, 1> {
  using UnderlyingType = uint8_t;
};

template <>
struct StorageType<uint8_t, 2> {
  using UnderlyingType = uint16_t;
};

template <>
struct StorageType<uint8_t, 4> {
  using UnderlyingType = uint32_t;
};

template <>
struct StorageType<uint8_t, 8> {
  using UnderlyingType = uint2;
};

template <>
struct StorageType<uint8_t, 16> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<int16_t, 1> {
  using UnderlyingType = int16_t;
};

template <>
struct StorageType<int16_t, 2> {
  using UnderlyingType = int32_t;
};

template <>
struct StorageType<int16_t, 4> {
  using UnderlyingType = int2;
};

template <>
struct StorageType<int16_t, 8> {
  using UnderlyingType = int4;
};

template <>
struct StorageType<uint16_t, 1> {
  using UnderlyingType = uint16_t;
};

template <>
struct StorageType<uint16_t, 2> {
  using UnderlyingType = uint32_t;
};

template <>
struct StorageType<uint16_t, 4> {
  using UnderlyingType = uint2;
};

template <>
struct StorageType<uint16_t, 8> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<__half, 1> {
  using UnderlyingType = __half;
};

template <>
struct StorageType<__half, 2> {
  using UnderlyingType = __half2;
};

template <>
struct StorageType<__half, 4> {
  using UnderlyingType = uint2;
};

template <>
struct StorageType<__half, 8> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<__half2, 1> {
  using UnderlyingType = __half2;
};

template <>
struct StorageType<__half2, 2> {
  using UnderlyingType = uint2;
};

template <>
struct StorageType<__half2, 4> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<int32_t, 1> {
  using UnderlyingType = int32_t;
};

template <>
struct StorageType<int32_t, 2> {
  using UnderlyingType = uint2;
};

template <>
struct StorageType<int32_t, 4> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<uint32_t, 1> {
  using UnderlyingType = uint32_t;
};

template <>
struct StorageType<uint32_t, 2> {
  using UnderlyingType = uint2;
};

template <>
struct StorageType<uint32_t, 4> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<float, 1> {
  using UnderlyingType = float;
};

template <>
struct StorageType<float, 2> {
  using UnderlyingType = float2;
};

template <>
struct StorageType<float, 4> {
  using UnderlyingType = float4;
};

template <>
struct StorageType<int64_t, 1> {
  using UnderlyingType = int64_t;
};

template <>
struct StorageType<int64_t, 2> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<uint64_t, 1> {
  using UnderlyingType = uint64_t;
};

template <>
struct StorageType<uint64_t, 2> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<unsigned long long, 1> {
  using UnderlyingType = unsigned long long;
};

template <>
struct StorageType<unsigned long long, 2> {
  using UnderlyingType = uint4;
};

template <>
struct StorageType<double, 1> {
  using UnderlyingType = double;
};

template <>
struct StorageType<double, 2> {
  using UnderlyingType = double2;
};

/**
 * @brief Structure for vectorized memory operations.
 *
 * Provides vectorized load and store operations for POD types in CUDA.
 * Optimizes memory bandwidth by using wider data types for memory access.
 *
 * @tparam DataType Element data type for computation.
 * @tparam N Number of DataType elements per vector block (vector length).
 */
template <typename DataType, int N>
struct VecBlock {
  // Type definitions
  using ElementType = DataType;                     // Element type
  using VectorType = typename StorageType<DataType, N>::UnderlyingType; // Vector storage type

  // Static constant
  static constexpr int kNumElems = N; // Number of elements in VecBlock

  // Data storage
  struct alignas(VectorType) { 
    ElementType data[N]; 
  } elems; // Aligned storage for vector elements

  /**
   * @brief Get a pointer to the underlying storage.
   *
   * @return Pointer to the underlying vector storage type.
   */
  FAST_TOPK_DEVICE_INLINE VectorType* GetStoragePtr() { 
    return reinterpret_cast<VectorType*>(elems.data); 
  }

  /**
   * @brief Fill the vector block with a constant value.
   *
   * @param val Constant value to fill the block with.
   */
  FAST_TOPK_DEVICE_INLINE void Fill(ElementType val) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      elems.data[i] = val;
    }
  }

  /**
   * @brief Vectorized load operations.
   * @{
   */

  /**
   * @brief Load data from a given pointer with index.
   *
   * @tparam IdxType Index type.
   * @param ptr Base pointer to load from.
   * @param idx Index offset from the base pointer, must be aligned to N.
   */
  template <typename IdxType = int>
  FAST_TOPK_DEVICE_INLINE void Load(const ElementType* ptr, IdxType idx) {
    const VectorType* basePtr = reinterpret_cast<const VectorType*>(&ptr[idx]);
    *GetStoragePtr() = __ldg(basePtr);
  }

  /**
   * @brief Load data from a non-const pointer with index.
   *
   * @tparam IdxType Index type.
   * @param ptr Base pointer to load from.
   * @param idx Index offset from the base pointer, must be aligned to N.
   */
  template <typename IdxType = int>
  FAST_TOPK_DEVICE_INLINE void Load(ElementType* ptr, IdxType idx) {
    VectorType* basePtr = reinterpret_cast<VectorType*>(&ptr[idx]);
    *GetStoragePtr() = *basePtr;
  }

  /**
   * @brief Vectorized store operation.
   *
   * @tparam IdxType Index type.
   * @param ptr Base pointer to store to.
   * @param idx Index offset from the base pointer, must be aligned to N.
   */
  template <typename IdxType = int>
  FAST_TOPK_DEVICE_INLINE void Store(ElementType* ptr, IdxType idx) {
    VectorType* basePtr = reinterpret_cast<VectorType*>(&ptr[idx]);
    *basePtr = *GetStoragePtr();
  }
  /** @} */
};

/**
 * @brief Specialization for scalar VecBlock (non-vectorized fallback).
 *
 * Acts as a scalar type with a single element for non-vectorized operations.
 * When N=0, VecBlock defaults to scalar operations.
 *
 * @tparam DataType Element data type.
 */
template <typename DataType>
struct VecBlock<DataType, 0> {
  using ElementType = DataType;
  static constexpr int kNumElems = 1;

  struct { 
    ElementType data[1]; 
  } elems;

  FAST_TOPK_DEVICE_INLINE void Fill(ElementType val) {}

  template <typename IdxType = int>
  FAST_TOPK_DEVICE_INLINE void Load(const ElementType* ptr, IdxType idx) {}

  template <typename IdxType = int>
  FAST_TOPK_DEVICE_INLINE void Load(ElementType* ptr, IdxType idx) {}

  template <typename IdxType = int>
  FAST_TOPK_DEVICE_INLINE void Store(ElementType* ptr, IdxType idx) {}
};

}  // namespace internal::gpu