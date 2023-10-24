// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel_device.hpp>
#include <cuda_fp16.h>

template<typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
    static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

    union {
        From f;
        To   t;
    } u;
    u.f = src;
    return u.t;
}

template<typename T>
__forceinline__ __device__ T add_elements(T a, T b){
  return a + b;
}

template<>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b){
  return __hadd2(a, b);
}

template<typename T>
__forceinline__ __device__ uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template<typename T>
__forceinline__ __device__ uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a,b);
}

template<>
__forceinline__ __device__ uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

