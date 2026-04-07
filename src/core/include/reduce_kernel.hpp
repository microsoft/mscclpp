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
MSCCLPP_DEVICE_INLINE T calElements(const T& a, const T& b) {
  if constexpr (OpType == SUM) {
    return a + b;
  } else if constexpr (OpType == MIN) {
    return mscclpp::min(a, b);
  }
  static_assert(OpType == SUM || OpType == MIN, "Unsupported ReduceOp");
}

// Generic vector reduction helpers

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE uint2 calVectorHelper(const uint2& a, const uint2& b) {
  uint2 ret;
  ret.x = bit_cast<uint32_t, T>(calElements<T, OpType>(bit_cast<T, uint32_t>(a.x), bit_cast<T, uint32_t>(b.x)));
  ret.y = bit_cast<uint32_t, T>(calElements<T, OpType>(bit_cast<T, uint32_t>(a.y), bit_cast<T, uint32_t>(b.y)));
  return ret;
}

/// f32x2 specialization for uint2: uses packed f32x2 operator+ (Blackwell __fadd2_rn when available).
template <>
MSCCLPP_DEVICE_INLINE uint2 calVectorHelper<f32x2, SUM>(const uint2& a, const uint2& b) {
  f32x2 fa = bit_cast<f32x2, uint2>(a);
  f32x2 fb = bit_cast<f32x2, uint2>(b);
  f32x2 fr = fa + fb;
  return bit_cast<uint2, f32x2>(fr);
}

template <>
MSCCLPP_DEVICE_INLINE uint2 calVectorHelper<f32x2, MIN>(const uint2& a, const uint2& b) {
  f32x2 fa = bit_cast<f32x2, uint2>(a);
  f32x2 fb = bit_cast<f32x2, uint2>(b);
  f32x2 fr = mscclpp::min(fa, fb);
  return bit_cast<uint2, f32x2>(fr);
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE int4 calVectorHelper(const int4& a, const int4& b) {
  int4 ret;
  ret.w = bit_cast<int, T>(calElements<T, OpType>(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(calElements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(calElements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(calElements<T, OpType>(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

/// f32x2 specialization for int4: process as two uint2 pairs using packed f32x2 arithmetic.
template <>
MSCCLPP_DEVICE_INLINE int4 calVectorHelper<f32x2, SUM>(const int4& a, const int4& b) {
  uint2 lo_a = {(uint32_t)a.x, (uint32_t)a.y};
  uint2 hi_a = {(uint32_t)a.z, (uint32_t)a.w};
  uint2 lo_b = {(uint32_t)b.x, (uint32_t)b.y};
  uint2 hi_b = {(uint32_t)b.z, (uint32_t)b.w};
  uint2 lo_r = calVectorHelper<f32x2, SUM>(lo_a, lo_b);
  uint2 hi_r = calVectorHelper<f32x2, SUM>(hi_a, hi_b);
  return {(int)lo_r.x, (int)lo_r.y, (int)hi_r.x, (int)hi_r.y};
}

template <>
MSCCLPP_DEVICE_INLINE int4 calVectorHelper<f32x2, MIN>(const int4& a, const int4& b) {
  uint2 lo_a = {(uint32_t)a.x, (uint32_t)a.y};
  uint2 hi_a = {(uint32_t)a.z, (uint32_t)a.w};
  uint2 lo_b = {(uint32_t)b.x, (uint32_t)b.y};
  uint2 hi_b = {(uint32_t)b.z, (uint32_t)b.w};
  uint2 lo_r = calVectorHelper<f32x2, MIN>(lo_a, lo_b);
  uint2 hi_r = calVectorHelper<f32x2, MIN>(hi_a, hi_b);
  return {(int)lo_r.x, (int)lo_r.y, (int)hi_r.x, (int)hi_r.y};
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE int calVectorHelper(const int& a, const int& b) {
  return bit_cast<int, T>(calElements<T, OpType>(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE uint32_t calVectorHelper(const uint32_t& a, const uint32_t& b) {
  return bit_cast<uint32_t, T>(calElements<T, OpType>(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

/// f32x2 specialization for uint32_t: a single float packed in 32 bits (scalar fallback).
template <>
MSCCLPP_DEVICE_INLINE uint32_t calVectorHelper<f32x2, SUM>(const uint32_t& a, const uint32_t& b) {
  float fa = bit_cast<float, uint32_t>(a);
  float fb = bit_cast<float, uint32_t>(b);
  return bit_cast<uint32_t, float>(fa + fb);
}

template <>
MSCCLPP_DEVICE_INLINE uint32_t calVectorHelper<f32x2, MIN>(const uint32_t& a, const uint32_t& b) {
  float fa = bit_cast<float, uint32_t>(a);
  float fb = bit_cast<float, uint32_t>(b);
  return bit_cast<uint32_t, float>(fminf(fa, fb));
}

// calVector wrapper – converts scalar types to vector types and calls calVectorHelper
template <typename T, ReduceOp OpType, typename DataType>
MSCCLPP_DEVICE_INLINE DataType calVector(const DataType& a, const DataType& b) {
  // Define the vectorized computation type based on the element type
  static_assert(sizeof(DataType) % sizeof(T) == 0, "DataType size must be multiple of T size");
  static_assert(sizeof(DataType) >= 4, "DataType size must be at least 4 bytes");
  using CompType = typename std::conditional_t<
      std::is_same_v<T, float>, f32x2,
      std::conditional_t<
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
                                     >>>>>;
  return calVectorHelper<CompType, OpType>(a, b);
}

/// Upcast a packed DataType (containing T elements) to a packed AccDataType (containing AccumT elements).
/// Uses the optimized to<>() specializations when available (e.g. FP8 -> float hardware intrinsics).
/// When AccumT == T, this is a no-op identity.
template <typename T, typename AccumT, typename AccDataType, typename DataType>
MSCCLPP_DEVICE_INLINE AccDataType upcastVector(const DataType& val) {
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
MSCCLPP_DEVICE_INLINE DataType downcastVector(const AccDataType& val) {
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
/// When AccumT == T, falls back to the standard calVector.
/// Otherwise, upcasts val to AccumT, reduces element-wise, and returns the AccumT accumulator.
template <typename T, typename AccumT, ReduceOp OpType, typename AccDataType, typename DataType>
MSCCLPP_DEVICE_INLINE AccDataType calVectorAccum(const AccDataType& acc, const DataType& val) {
  if constexpr (std::is_same_v<T, AccumT>) {
    return calVector<T, OpType>(acc, val);
  } else {
    constexpr int nElems = sizeof(DataType) / sizeof(T);
    using FromVec = VectorType<T, nElems>;
    using ToVec = VectorType<AccumT, nElems>;

    ToVec fv = mscclpp::to<ToVec>(reinterpret_cast<const FromVec&>(val));
    const ToVec& fa = reinterpret_cast<const ToVec&>(acc);
    ToVec fr;
#pragma unroll
    for (int i = 0; i < nElems; ++i) {
      fr.data[i] = calElements<AccumT, OpType>(fa.data[i], fv.data[i]);
    }
    return reinterpret_cast<const AccDataType&>(fr);
  }
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
