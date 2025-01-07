// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <assert.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

// Numerical Recipes ranqd1, Chapter 7.1, §An Even Quicker Generator, Eq. 7.1.6
// parameters from Knuth and H. W. Lewis
static __device__ unsigned int ranqd1(unsigned int seed) {
  const unsigned int a = 1664525;
  const unsigned int c = 1013904223;
  return a * seed + c;
}

// fill/test kernel pairs must have the same thread block size to
// match their random number series.

#define FILL_DATA(FuncNameType, DataType)                                                                \
  extern "C" __global__ void __launch_bounds__(1024, 1)                                                  \
      fill_data_##FuncNameType(DataType* input_buf, size_t num_elems, int rank, int seq) {               \
    unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + rank + seq);              \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
      seed = ranqd1(seed);                                                                               \
      input_buf[i] = DataType(seed % blockDim.x) / DataType(blockDim.x);                                 \
    }                                                                                                    \
  }

FILL_DATA(float16, __half)
FILL_DATA(float32, float)
FILL_DATA(int32, int)

#define TEST_DATA_ALL_GATHER(FuncNameType, DataType)                                                       \
  extern "C" __global__ void __launch_bounds__(1024, 1) test_data_all_gather_##FuncNameType(               \
      DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, int my_rank, int seq) {   \
    for (int rank = 0; rank < num_ranks; rank++) {                                                         \
      size_t rank_offset = rank * num_elems;                                                               \
      unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + rank + seq);              \
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        seed = ranqd1(seed);                                                                               \
        test_buf[rank_offset + i] = DataType(seed % blockDim.x) / DataType(blockDim.x);                    \
        assert(result_buf[rank_offset + i] == test_buf[rank_offset + i]);                                  \
      }                                                                                                    \
    }                                                                                                      \
  }

TEST_DATA_ALL_GATHER(float16, __half)
TEST_DATA_ALL_GATHER(float32, float)
TEST_DATA_ALL_GATHER(int32, int)

#define TEST_DATA_ALL_REDUCE(FuncNameType, DataType)                                                       \
  extern "C" __global__ void __launch_bounds__(1024, 1) test_data_all_reduce_##FuncNameType(               \
      DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, int my_rank, int seq) {   \
    for (int rank = 0; rank < num_ranks; rank++) {                                                         \
      unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + rank + seq);              \
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        if (rank == 0) {                                                                                   \
          test_buf[i] = 0;                                                                                 \
        }                                                                                                  \
        seed = ranqd1(seed);                                                                               \
        test_buf[i] += DataType(seed % blockDim.x) / DataType(blockDim.x);                                 \
      }                                                                                                    \
    }                                                                                                      \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) {   \
      assert(abs(float(result_buf[i]) - float(test_buf[i])) < 1e-3 * num_ranks);                           \
    }                                                                                                      \
  }

TEST_DATA_ALL_REDUCE(float16, __half)
TEST_DATA_ALL_REDUCE(float32, float)
TEST_DATA_ALL_REDUCE(int32, int)

#define TEST_DATA_REDUCE_SCATTER(FuncNameType, DataType)                                                   \
  extern "C" __global__ void __launch_bounds__(1024, 1) test_data_reduce_scatter_##FuncNameType(           \
      DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, int my_rank, int seq) {   \
    int nem_elems_per_rank = num_elems / num_ranks;                                                        \
    int offset = nem_elems_per_rank * my_rank;                                                             \
    for (int rank = 0; rank < num_ranks; rank++) {                                                         \
      unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + rank + seq);              \
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        if (rank == 0) {                                                                                   \
          test_buf[i] = 0;                                                                                 \
        }                                                                                                  \
        seed = ranqd1(seed);                                                                               \
        test_buf[i] += DataType(seed % blockDim.x) / DataType(blockDim.x);                                 \
      }                                                                                                    \
    }                                                                                                      \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) {   \
      if (i >= offset && i < offset + nem_elems_per_rank) {                                                \
        assert(abs(float(result_buf[i]) - float(test_buf[i])) < 1e-3 * num_ranks);                         \
      }                                                                                                    \
    }                                                                                                      \
  }

TEST_DATA_REDUCE_SCATTER(float16, __half)
TEST_DATA_REDUCE_SCATTER(float32, float)
TEST_DATA_REDUCE_SCATTER(int32, int)