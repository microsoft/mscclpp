// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdint.h>

template <typename T, int N>
class Plist {
 public:
#ifdef __CUDACC__
  __forceinline__ __device__ T& operator[](int i) { return data[i]; }
  __forceinline__ __device__ const T& operator[](int i) const { return data[i]; }
#endif

 private:
  T data[N];
};
