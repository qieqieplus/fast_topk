#pragma once

#include <cuda_fp16.h>
#include "internal/gpu/common.hpp"

namespace internal::gpu {

template <typename DataType, int N>
struct StorageType {};

template <>
struct StorageType<bool, 1> {
  static_assert(sizeof(bool) == sizeof(int8_t), "StorageType bool size assumption failed");
  typedef int8_t UnderlyingType;
};
template <>
struct StorageType<bool, 2> {
  typedef int16_t UnderlyingType;
};
template <>
struct StorageType<bool, 4> {
  typedef int32_t UnderlyingType;
};
template <>
struct StorageType<bool, 8> {
  typedef int2 UnderlyingType;
};
template <>
struct StorageType<bool, 16> {
  typedef int4 UnderlyingType;
};
template <>
struct StorageType<int8_t, 1> {
  typedef int8_t UnderlyingType;
};
template <>
struct StorageType<int8_t, 2> {
  typedef int16_t UnderlyingType;
};
template <>
struct StorageType<int8_t, 4> {
  typedef int32_t UnderlyingType;
};
template <>
struct StorageType<int8_t, 8> {
  typedef int2 UnderlyingType;
};
template <>
struct StorageType<int8_t, 16> {
  typedef int4 UnderlyingType;
};
template <>
struct StorageType<uint8_t, 1> {
  typedef uint8_t UnderlyingType;
};
template <>
struct StorageType<uint8_t, 2> {
  typedef uint16_t UnderlyingType;
};
template <>
struct StorageType<uint8_t, 4> {
  typedef uint32_t UnderlyingType;
};
template <>
struct StorageType<uint8_t, 8> {
  typedef uint2 UnderlyingType;
};
template <>
struct StorageType<uint8_t, 16> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<int16_t, 1> {
  typedef int16_t UnderlyingType;
};
template <>
struct StorageType<int16_t, 2> {
  typedef int32_t UnderlyingType;
};
template <>
struct StorageType<int16_t, 4> {
  typedef int2 UnderlyingType;
};
template <>
struct StorageType<int16_t, 8> {
  typedef int4 UnderlyingType;
};
template <>
struct StorageType<uint16_t, 1> {
  typedef uint16_t UnderlyingType;
};
template <>
struct StorageType<uint16_t, 2> {
  typedef uint32_t UnderlyingType;
};
template <>
struct StorageType<uint16_t, 4> {
  typedef uint2 UnderlyingType;
};
template <>
struct StorageType<uint16_t, 8> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<__half, 1> {
  typedef __half UnderlyingType;
};
template <>
struct StorageType<__half, 2> {
  typedef __half2 UnderlyingType;
};
template <>
struct StorageType<__half, 4> {
  typedef uint2 UnderlyingType;
};
template <>
struct StorageType<__half, 8> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<__half2, 1> {
  typedef __half2 UnderlyingType;
};
template <>
struct StorageType<__half2, 2> {
  typedef uint2 UnderlyingType;
};
template <>
struct StorageType<__half2, 4> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<int32_t, 1> {
  typedef int32_t UnderlyingType;
};
template <>
struct StorageType<int32_t, 2> {
  typedef uint2 UnderlyingType;
};
template <>
struct StorageType<int32_t, 4> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<uint32_t, 1> {
  typedef uint32_t UnderlyingType;
};
template <>
struct StorageType<uint32_t, 2> {
  typedef uint2 UnderlyingType;
};
template <>
struct StorageType<uint32_t, 4> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<float, 1> {
  typedef float UnderlyingType;
};
template <>
struct StorageType<float, 2> {
  typedef float2 UnderlyingType;
};
template <>
struct StorageType<float, 4> {
  typedef float4 UnderlyingType;
};
template <>
struct StorageType<int64_t, 1> {
  typedef int64_t UnderlyingType;
};
template <>
struct StorageType<int64_t, 2> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<uint64_t, 1> {
  typedef uint64_t UnderlyingType;
};
template <>
struct StorageType<uint64_t, 2> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<unsigned long long, 1> {
  typedef unsigned long long UnderlyingType;
};
template <>
struct StorageType<unsigned long long, 2> {
  typedef uint4 UnderlyingType;
};
template <>
struct StorageType<double, 1> {
  typedef double UnderlyingType;
};
template <>
struct StorageType<double, 2> {
  typedef double2 UnderlyingType;
};

/**
 * @struct VecBlock
 *
 * @brief Internal data structure for vectorized loads/stores of POD types.
 * Provides CUDA programmers an easy way to issue vectorized load/store
 * instructions to memory (global or shared). Vectorized memory accesses
 * improve bandwidth utilization compared to scalar accesses.  If vectorization
 * isn't possible, scalar POD types can be used directly.
 *
 * Vectorization concept: Threads process multiple elements for speedup.
 * Elements are loaded in a single read via type promotion and reinterpreted
 * as a vector for kernel work.
 *
 * Caution: Vectorized accesses require memory alignment of input addresses
 * to the promoted type, not the input type.
 *
 * Pointer alignment is caller's responsibility!
 *
 * @tparam DataType Data type for computation/math.
 * @tparam N Number of 'DataType' elements per load/store.
 */
template <typename DataType, int N>
struct VecBlock {
  /** Underlying math data type */
  typedef DataType ElemType;
  /** Internal storage data type */
  typedef typename StorageType<DataType, N>::UnderlyingType UnderlyingType;

  /** Number of 'DataType' elements in this block */
  static const int NumElems = N; // Alternative shorter name, choose one

  struct alignas(UnderlyingType) {
    /** Vectorized data for operations */
    ElemType data[N];
  } elems; // Changed 'values' to 'elems' - shorter and common for elements

  FATS_TOPK_DEVICE_INLINE  UnderlyingType* storagePtr() { return reinterpret_cast<UnderlyingType*>(elems.data); } // Shorter name

  /**
   * @brief Fill block with a constant value
   * @param val Constant value to fill with.
   */
  FATS_TOPK_DEVICE_INLINE  void fill(ElemType val) // Shorter name
  {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      elems.data[i] = val;
    }
  }

  /**
   * @defgroup LoadsStores Vectorized Loads/Stores
   *
   * @brief Vectorized load/store operations for this block.
   * @tparam IdxType Index data type
   * @param ptr Base pointer for load/store. Must be aligned to 'sizeof(UnderlyingType)'!
   * @param idx Offset from base pointer. Must be aligned to 'N'!
   *
   * @note Loads populate elems.data; stores write elems.data to memory.
   * @{
   */
  template <typename IdxType = int> // Shorter name
  FATS_TOPK_DEVICE_INLINE  void load(const ElemType* ptr, IdxType idx) // Shorter name
  {
    const UnderlyingType* basePtr   = reinterpret_cast<const UnderlyingType*>(&ptr[idx]); // shorter ptr name
    *storagePtr() = __ldg(basePtr);
  }

  template <typename IdxType = int> // Shorter name
  FATS_TOPK_DEVICE_INLINE  void load(ElemType* ptr, IdxType idx) // Shorter name
  {
    UnderlyingType* basePtr         = reinterpret_cast<UnderlyingType*>(&ptr[idx]); // shorter ptr name
    *storagePtr() = *basePtr;
  }

  template <typename IdxType = int> // Shorter name
  FATS_TOPK_DEVICE_INLINE  void store(ElemType* ptr, IdxType idx) // Shorter name
  {
    UnderlyingType* basePtr = reinterpret_cast<UnderlyingType*>(&ptr[idx]); // shorter ptr name
    *basePtr      = *storagePtr();
  }
  /** @} */
};

/**
 * @brief Specialization for VecBlock with N = 0 (scalar fallback).
 *        Acts as a scalar type with a single element for non-vectorized cases.
 */
template <typename DataType>
struct VecBlock<DataType, 0> {
  typedef DataType ElemType;
  static const int NumElems = 1;

  struct {
    ElemType data[1];
  } elems;

  FATS_TOPK_DEVICE_INLINE  void fill(ElemType val) {} // Shorter name
  template <typename IdxType = int> // Shorter name
  FATS_TOPK_DEVICE_INLINE  void load(const ElemType* ptr, IdxType idx) // Shorter name
  {
  }
  template <typename IdxType = int> // Shorter name
  FATS_TOPK_DEVICE_INLINE  void load(ElemType* ptr, IdxType idx) // Shorter name
  {
  }
  template <typename IdxType = int> // Shorter name
  FATS_TOPK_DEVICE_INLINE  void store(ElemType* ptr, IdxType idx) // Shorter name
  {
  }
};

}  // namespace my_namespace (or remove if not needed)