// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <assert.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_bf16.h>
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

// `split_mask` groups ranks together: group_size = split_mask + 1, group_id = rank / group_size.
// Data is seeded by group_id so that all ranks within a group produce the same fill, and ranks
// in different groups produce different fills. With split_mask == 0 this reduces to per-rank
// seeding (group_id == rank).
#define FILL_DATA(FuncNameType, DataType)                                                                  \
  extern "C" __global__ void __launch_bounds__(1024, 1)                                                    \
      fill_data_##FuncNameType(DataType* input_buf, size_t num_elems, int rank, int seq, int split_mask) { \
    int seed_rank = rank / (split_mask + 1);                                                               \
    unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + seed_rank + seq);           \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) {   \
      seed = ranqd1(seed);                                                                                 \
      input_buf[i] = DataType(seed % blockDim.x) / DataType(blockDim.x);                                   \
    }                                                                                                      \
  }

FILL_DATA(bfloat16, __nv_bfloat16)
FILL_DATA(float16, __half)
FILL_DATA(float32, float)
FILL_DATA(int32, int)

#define TEST_DATA_ALL_GATHER(FuncNameType, DataType)                                                                 \
  extern "C" __global__ void __launch_bounds__(1024, 1)                                                              \
      test_data_all_gather_##FuncNameType(DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, \
                                          int my_rank, int seq, int split_mask) {                                    \
    for (int rank = 0; rank < num_ranks; rank++) {                                                                   \
      size_t rank_offset = rank * num_elems;                                                                         \
      int seed_rank = rank / (split_mask + 1);                                                                       \
      unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + seed_rank + seq);                   \
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) {           \
        seed = ranqd1(seed);                                                                                         \
        test_buf[rank_offset + i] = DataType(seed % blockDim.x) / DataType(blockDim.x);                              \
        assert(result_buf[rank_offset + i] == test_buf[rank_offset + i]);                                            \
      }                                                                                                              \
    }                                                                                                                \
  }

TEST_DATA_ALL_GATHER(bfloat16, __nv_bfloat16)
TEST_DATA_ALL_GATHER(float16, __half)
TEST_DATA_ALL_GATHER(float32, float)
TEST_DATA_ALL_GATHER(int32, int)

#define TEST_DATA_ALL_REDUCE(FuncNameType, DataType, Eps)                                                            \
  extern "C" __global__ void __launch_bounds__(1024, 1)                                                              \
      test_data_all_reduce_##FuncNameType(DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, \
                                          int my_rank, int seq, int split_mask) {                                    \
    for (int rank = 0; rank < num_ranks; rank++) {                                                                   \
      int seed_rank = rank / (split_mask + 1);                                                                       \
      unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + seed_rank + seq);                   \
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) {           \
        if (rank == 0) {                                                                                             \
          test_buf[i] = 0;                                                                                           \
        }                                                                                                            \
        seed = ranqd1(seed);                                                                                         \
        test_buf[i] += DataType(seed % blockDim.x) / DataType(blockDim.x);                                           \
      }                                                                                                              \
    }                                                                                                                \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) {             \
      float expected = float(test_buf[i]);                                                                           \
      float result = float(result_buf[i]);                                                                           \
      float tol = Eps * num_ranks * (1.0f + abs(expected));                                                          \
      assert(abs(result - expected) <= tol);                                                                         \
    }                                                                                                                \
  }

TEST_DATA_ALL_REDUCE(bfloat16, __nv_bfloat16, 7.8125e-3f)
TEST_DATA_ALL_REDUCE(float16, __half, 9.765625e-4f)
TEST_DATA_ALL_REDUCE(float32, float, 1.1920929e-7f)
TEST_DATA_ALL_REDUCE(int32, int, 0.0f)

#define TEST_DATA_REDUCE_SCATTER(FuncNameType, DataType, Eps)                                              \
  extern "C" __global__ void __launch_bounds__(1024, 1)                                                    \
      test_data_reduce_scatter_##FuncNameType(DataType* result_buf, DataType* test_buf, size_t num_elems,  \
                                              int num_ranks, int my_rank, int seq, int split_mask) {       \
    int nem_elems_per_rank = num_elems / num_ranks;                                                        \
    int offset = nem_elems_per_rank * my_rank;                                                             \
    for (int rank = 0; rank < num_ranks; rank++) {                                                         \
      int seed_rank = rank / (split_mask + 1);                                                             \
      unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + seed_rank + seq);         \
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
        float expected = float(test_buf[i]);                                                               \
        float result = float(result_buf[i - offset]);                                                      \
        float tol = Eps * num_ranks * (1.0f + abs(expected));                                              \
        assert(abs(result - expected) <= tol);                                                             \
      }                                                                                                    \
    }                                                                                                      \
  }

TEST_DATA_REDUCE_SCATTER(bfloat16, __nv_bfloat16, 7.8125e-3f)
TEST_DATA_REDUCE_SCATTER(float16, __half, 9.765625e-4f)
TEST_DATA_REDUCE_SCATTER(float32, float, 1.1920929e-7f)
TEST_DATA_REDUCE_SCATTER(int32, int, 0.0f)

#define TEST_DATA_ALL_TO_ALL(FuncNameType, DataType)                                                                 \
  extern "C" __global__ void __launch_bounds__(1024, 1)                                                              \
      test_data_all_to_all_##FuncNameType(DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, \
                                          int my_rank, int seq, int split_mask) {                                    \
    int nem_elems_per_rank = num_elems / num_ranks;                                                                  \
    int offset = nem_elems_per_rank * my_rank;                                                                       \
    for (int rank = 0; rank < num_ranks; rank++) {                                                                   \
      size_t rank_offset = rank * nem_elems_per_rank;                                                                \
      int seed_rank = rank / (split_mask + 1);                                                                       \
      unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + seed_rank + seq);                   \
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) {           \
        seed = ranqd1(seed);                                                                                         \
        if (i >= my_rank * nem_elems_per_rank && i < (my_rank + 1) * nem_elems_per_rank) {                           \
          test_buf[rank_offset + i - offset] = DataType(seed % blockDim.x) / DataType(blockDim.x);                   \
          assert(result_buf[rank_offset + i - offset] == test_buf[rank_offset + i - offset]);                        \
        }                                                                                                            \
      }                                                                                                              \
    }                                                                                                                \
  }

TEST_DATA_ALL_TO_ALL(bfloat16, __nv_bfloat16)
TEST_DATA_ALL_TO_ALL(float16, __half)
TEST_DATA_ALL_TO_ALL(float32, float)
TEST_DATA_ALL_TO_ALL(int32, int)

// Sendrecv verification: receive from the prev group in the ring.
// fill_data seeds by group_id (rank / (split_mask + 1)); the receiver in group g expects the
// data produced by group (g - 1 + num_groups) % num_groups, so we recompute that seed here.
#define TEST_DATA_SEND_RECV(FuncNameType, DataType)                                                                 \
  extern "C" __global__ void __launch_bounds__(1024, 1)                                                             \
      test_data_send_recv_##FuncNameType(DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, \
                                         int my_rank, int seq, int split_mask) {                                    \
    int group_size = split_mask + 1;                                                                                \
    int num_groups = num_ranks / group_size;                                                                        \
    int my_group_id = my_rank / group_size;                                                                         \
    int prev_group_id = (my_group_id - 1 + num_groups) % num_groups;                                                \
    unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + prev_group_id + seq);                \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) {            \
      seed = ranqd1(seed);                                                                                          \
      test_buf[i] = DataType(seed % blockDim.x) / DataType(blockDim.x);                                             \
      assert(result_buf[i] == test_buf[i]);                                                                         \
    }                                                                                                               \
  }

TEST_DATA_SEND_RECV(float16, __half)
TEST_DATA_SEND_RECV(float32, float)
TEST_DATA_SEND_RECV(int32, int)
