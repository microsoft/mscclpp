// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_REDUCE_KERNEL_HPP_
#define MSCCLPP_REDUCE_KERNEL_HPP_

#include <mscclpp/algorithm.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <type_traits>

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE)

// Generic element-wise calculation helper
template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE T cal_elements(const T& a, const T& b) {
  if constexpr (OpType == SUM) {
    return a + b;
  } else if constexpr (OpType == MIN) {
    return mscclpp::min(a, b);
  }
  static_assert(OpType == SUM || OpType == MIN, "Unsupported ReduceOp");
}

// Generic vector reduction helpers
template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE int4 cal_vector_helper(const int4& a, const int4& b) {
  int4 ret;
  ret.w = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE uint2 cal_vector_helper(const uint2& a, const uint2& b) {
  uint2 ret;
  ret.x = bit_cast<uint32_t, T>(cal_elements<T, OpType>(bit_cast<T, uint32_t>(a.x), bit_cast<T, uint32_t>(b.x)));
  ret.y = bit_cast<uint32_t, T>(cal_elements<T, OpType>(bit_cast<T, uint32_t>(a.y), bit_cast<T, uint32_t>(b.y)));
  return ret;
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE int cal_vector_helper(const int& a, const int& b) {
  return bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE uint32_t cal_vector_helper(const uint32_t& a, const uint32_t& b) {
  return bit_cast<uint32_t, T>(cal_elements<T, OpType>(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

// cal_vector wrapper - converts scalar types to vector types and calls cal_vector_helper
template <typename T, ReduceOp OpType, typename DataType>
MSCCLPP_DEVICE_INLINE DataType cal_vector(const DataType& a, const DataType& b) {
  // Define the vectorized computation type based on the element type
  static_assert(sizeof(DataType) % sizeof(T) == 0, "DataType size must be multiple of T size");
  static_assert(sizeof(DataType) >= 4, "DataType size must be at least 4 bytes");
  using CompType = typename std::conditional_t<
      std::is_same_v<T, __half>, f16x2,
      std::conditional_t<
          std::is_same_v<T, __bfloat16>, bf16x2,
          std::conditional_t<
              std::is_same_v<T, uint8_t>, u8x4,
              std::conditional_t<std::is_same_v<T, __fp8_e4m3b15>, f8_e4m3b15x4,
#if defined(__FP8_TYPES_EXIST__)
                                 std::conditional_t<std::is_same_v<T, __fp8_e4m3>, f8_e4m3x4,
                                                    std::conditional_t<std::is_same_v<T, __fp8_e5m2>, f8_e5m2x4, T>>
#else
                                 T
#endif
                                 >>>>;
  return cal_vector_helper<CompType, OpType>(a, b);
}

/// Upcast a packed DataType (containing T elements) to a packed AccDataType (containing AccumT elements).
/// Uses the optimized to<>() specializations when available (e.g. FP8 -> float hardware intrinsics).
/// When AccumT == T, this is a no-op identity.
template <typename T, typename AccumT, typename AccDataType, typename DataType>
MSCCLPP_DEVICE_INLINE AccDataType upcast_vector(const DataType& val) {
  if constexpr (std::is_same_v<T, AccumT>) {
    return val;
  } else {
    constexpr int nElems = sizeof(DataType) / sizeof(T);
    using FromVec = VectorType<T, nElems>;
    using ToVec = VectorType<AccumT, nElems>;
    ToVec result = mscclpp::to<ToVec>(reinterpret_cast<const FromVec&>(val));
    return reinterpret_cast<const AccDataType&>(result);
  }
}

/// Downcast a packed AccDataType (containing AccumT elements) back to DataType (containing T elements).
/// Uses the optimized to<>() specializations when available.
/// When AccumT == T, this is a no-op identity.
template <typename T, typename AccumT, typename DataType, typename AccDataType>
MSCCLPP_DEVICE_INLINE DataType downcast_vector(const AccDataType& val) {
  if constexpr (std::is_same_v<T, AccumT>) {
    return val;
  } else {
    constexpr int nElems = sizeof(DataType) / sizeof(T);
    using FromVec = VectorType<T, nElems>;
    using ToVec = VectorType<AccumT, nElems>;
    FromVec result = mscclpp::to<FromVec>(reinterpret_cast<const ToVec&>(val));
    return reinterpret_cast<const DataType&>(result);
  }
}

/// Accumulate `val` (packed T elements in DataType) into `acc` (packed AccumT elements in AccDataType).
/// When AccumT == T, falls back to the standard cal_vector.
/// Otherwise, upcasts val to AccumT, reduces element-wise, and returns the AccumT accumulator.
template <typename T, typename AccumT, ReduceOp OpType, typename AccDataType, typename DataType>
MSCCLPP_DEVICE_INLINE AccDataType cal_vector_accum(const AccDataType& acc, const DataType& val) {
  if constexpr (std::is_same_v<T, AccumT>) {
    return cal_vector<T, OpType>(acc, val);
  } else {
    constexpr int nElems = sizeof(DataType) / sizeof(T);
    using FromVec = VectorType<T, nElems>;
    using ToVec = VectorType<AccumT, nElems>;

    ToVec fv = mscclpp::to<ToVec>(reinterpret_cast<const FromVec&>(val));
    const ToVec& fa = reinterpret_cast<const ToVec&>(acc);
    ToVec fr;
#pragma unroll
    for (int i = 0; i < nElems; ++i) {
      fr.data[i] = cal_elements<AccumT, OpType>(fa.data[i], fv.data[i]);
    }
    return reinterpret_cast<const AccDataType&>(fr);
  }
}

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

}  // namespace mscclpp

#endif  // MSCCLPP_REDUCE_KERNEL_HPP_
