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
        assert(abs(float(result_buf[i - offset]) - float(test_buf[i])) < 1e-3 * num_ranks);                \
      }                                                                                                    \
    }                                                                                                      \
  }

TEST_DATA_REDUCE_SCATTER(float16, __half)
TEST_DATA_REDUCE_SCATTER(float32, float)
TEST_DATA_REDUCE_SCATTER(int32, int)

#define TEST_DATA_ALL_TO_ALL(FuncNameType, DataType)                                                       \
  extern "C" __global__ void __launch_bounds__(1024, 1) test_data_all_to_all_##FuncNameType(               \
      DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, int my_rank, int seq) {   \
    int nem_elems_per_rank = num_elems / num_ranks;                                                        \
    int offset = nem_elems_per_rank * my_rank;                                                             \
    for (int rank = 0; rank < num_ranks; rank++) {                                                         \
      size_t rank_offset = rank * nem_elems_per_rank;                                                      \
      unsigned int seed = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + rank + seq);              \
      for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elems; i += blockDim.x * gridDim.x) { \
        seed = ranqd1(seed);                                                                               \
        if (i >= my_rank * nem_elems_per_rank && i < (my_rank + 1) * nem_elems_per_rank) {                 \
          test_buf[rank_offset + i - offset] = DataType(seed % blockDim.x) / DataType(blockDim.x);         \
          assert(result_buf[rank_offset + i - offset] == test_buf[rank_offset + i - offset]);              \
        }                                                                                                  \
      }                                                                                                    \
    }                                                                                                      \
  }

TEST_DATA_ALL_TO_ALL(float16, __half)
TEST_DATA_ALL_TO_ALL(float32, float)
TEST_DATA_ALL_TO_ALL(int32, int)

/*#define TEST_DATA_SENDRECV(FuncNameType, DataType)                                                          \
  extern "C" __global__ void __launch_bounds__(1024, 1) test_data_sendrecv_##FuncNameType(                  \
      DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, int my_rank, int seq) {    \
                                                                                                             \
    /* Ring semantics: receive from prev rank */                                                             \
/*    int peer_rank = (my_rank - 1 + num_ranks) % num_ranks;                                                   \
                                                                                                             \
    unsigned int seed =                                                                                      \
        (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + peer_rank + seq);                             \
                                                                                                             \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;                                                   \
         i < num_elems;                                                                                      \
         i += blockDim.x * gridDim.x) {                                                                      \
      seed = ranqd1(seed);                                                                                   \
      test_buf[i] = DataType(seed % blockDim.x) / DataType(blockDim.x);                                      \
                                                                                                             \
      /* Optional: print first few mismatches */                                                             \
/*      if (result_buf[i] != test_buf[i] && blockIdx.x == 0 && threadIdx.x == 0 && i < 8) {                    \
        printf("MISMATCH rank=%d peer=%d i=%zu result=%f expected=%f\n",                                     \
               my_rank, peer_rank, i, (float)result_buf[i], (float)test_buf[i]);                             \
      }                                                                                                      \
                                                                                                             \
      assert(result_buf[i] == test_buf[i]);                                                                  \
    }                                                                                                        \
  }*/


/*#define TEST_DATA_SENDRECV(FuncNameType, DataType)                                                        \
  extern "C" __global__ void __launch_bounds__(1024, 1) test_data_sendrecv_##FuncNameType(                \
      DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, int my_rank, int seq) {  \
                                                                                                           \
    int prev_rank = (my_rank - 1 + num_ranks) % num_ranks;                                                 \
    int next_rank = (my_rank + 1) % num_ranks;                                                             \
    int self_rank = my_rank;                                                                               \
                                                                                                           \
    unsigned int seed_prev =                                                                               \
        (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + prev_rank + seq);                           \
    unsigned int seed_next =                                                                               \
        (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + next_rank + seq);                           \
    unsigned int seed_self =                                                                               \
        (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + self_rank + seq);                           \
                                                                                                           \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;                                                 \
         i < num_elems;                                                                                    \
         i += blockDim.x * gridDim.x) {                                                                    \
                                                                                                           \
      seed_prev = ranqd1(seed_prev);                                                                       \
      seed_next = ranqd1(seed_next);                                                                       \
      seed_self = ranqd1(seed_self);                                                                       \
                                                                                                           \
      DataType exp_prev = DataType(seed_prev % blockDim.x) / DataType(blockDim.x);                         \
      DataType exp_next = DataType(seed_next % blockDim.x) / DataType(blockDim.x);                         \
      DataType exp_self = DataType(seed_self % blockDim.x) / DataType(blockDim.x);                         \
                                                                                                           \
      /* For compatibility: avoid %zu formatting quirks on device */                                        \
/*      unsigned long long ii = (unsigned long long)i;                                                       \
                                                                                                           \
      if (result_buf[i] != exp_prev) {                                                                     \
        /* Print only a few mismatches to avoid flooding */                                                 \
/*        if (blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 192) && ii < 256ULL) {                  \
          printf("sendrecv-mismatch rank=%d nranks=%d i=%llu result=%f exp_prev(from %d)=%f "              \
                 "exp_next(from %d)=%f exp_self(from %d)=%f\n",                                            \
                 my_rank, num_ranks, ii,                                                                   \
                 (float)result_buf[i],                                                                     \
                 prev_rank, (float)exp_prev,                                                               \
                 next_rank, (float)exp_next,                                                               \
                 self_rank, (float)exp_self);                                                              \
        }                                                                                                  \
      }                                                                                                    \
                                                                                                           \
      test_buf[i] = exp_prev;                                                                              \
      assert(result_buf[i] == test_buf[i]);                                                                \
    }                                                                                                      \
  }
*/


#define TEST_DATA_SENDRECV(FuncNameType, DataType)                                                        \
  extern "C" __global__ void __launch_bounds__(1024, 1) test_data_sendrecv_##FuncNameType(                \
      DataType* result_buf, DataType* test_buf, size_t num_elems, int num_ranks, int my_rank, int seq) {  \
                                                                                                           \
    /* Expected ring semantics (if your algorithm is ring-prev). */                                        \
    int prev_rank = (my_rank - 1 + num_ranks) % num_ranks;                                                 \
    int next_rank = (my_rank + 1) % num_ranks;                                                             \
    int self_rank = my_rank;                                                                               \
                                                                                                           \
    /* Thread identity and stride must match fill_data_* generation pattern. */                            \
    const unsigned long long tid =                                                                        \
        (unsigned long long)(blockIdx.x * blockDim.x + threadIdx.x);                                       \
    const unsigned long long stride =                                                                      \
        (unsigned long long)(blockDim.x * gridDim.x);                                                      \
                                                                                                           \
    for (unsigned long long i = tid; i < (unsigned long long)num_elems; i += stride) {                    \
                                                                                                           \
      /* Compute how many iterations this thread advanced before reaching i. */                            \
      unsigned long long k = (i - tid) / stride;                                                           \
                                                                                                           \
      /* Helper lambda: compute expected value for a given sender rank r at element i for this thread. */  \
      auto expected_for_rank = [&](int r) -> DataType {                                                    \
        unsigned int s = (unsigned int)(tid + (unsigned long long)r + (unsigned long long)seq);            \
        /* fill_data does: seed=ranqd1(seed) once per element visited.                                     \
           For the k-th visited element, apply ranqd1 (k+1) times. */                                      \
        for (unsigned long long step = 0; step < k + 1; ++step) {                                          \
          s = ranqd1(s);                                                                                   \
        }                                                                                                  \
        return DataType(s % blockDim.x) / DataType(blockDim.x);                                            \
      };                                                                                                   \
                                                                                                           \
      DataType exp_prev = expected_for_rank(prev_rank);                                                    \
      DataType exp_next = expected_for_rank(next_rank);                                                    \
      DataType exp_self = expected_for_rank(self_rank);                                                    \
                                                                                                           \
      /* Store expected(prev) in test_buf for the assert (keeps compatibility with your current check). */ \
      test_buf[i] = exp_prev;                                                                              \
                                                                                                           \
      if (result_buf[i] != test_buf[i]) {                                                                  \
        /* Try to identify which rank's stream matches the observed result. */                             \
        int matched = -1;                                                                                  \
        for (int r = 0; r < num_ranks; ++r) {                                                              \
          DataType exp_r = expected_for_rank(r);                                                           \
          if (result_buf[i] == exp_r) {                                                                    \
            matched = r;                                                                                   \
            break;                                                                                          \
          }                                                                                                \
        }                                                                                                  \
                                                                                                           \
        /* Print only a small number of mismatches to avoid log spam. */                                   \
        if (blockIdx.x == 0 && (threadIdx.x == 0 || threadIdx.x == 160) && i < 256ULL) {                   \
          printf("sendrecv-mismatch rank=%d nranks=%d i=%llu result=%f "                                   \
                 "exp_prev(from %d)=%f exp_next(from %d)=%f exp_self(from %d)=%f matched_sender=%d\n",     \
                 my_rank, num_ranks, i,                                                                    \
                 (float)result_buf[i],                                                                     \
                 prev_rank, (float)exp_prev,                                                               \
                 next_rank, (float)exp_next,                                                               \
                 self_rank, (float)exp_self,                                                               \
                 matched);                                                                                 \
        }                                                                                                  \
                                                                                                           \
        assert(result_buf[i] == test_buf[i]);                                                              \
      }                                                                                                    \
    }                                                                                                      \
  }


/*
#define TEST_DATA_SENDRECV(FuncNameType, DataType)                                      \
extern "C" __global__ void __launch_bounds__(1024, 1)                                  \
test_data_sendrecv_##FuncNameType(                                                     \
    DataType* result_buf,                                                              \
    DataType* test_buf,                                                                \
    size_t num_elems,                                                                  \
    int num_ranks,                                                                     \
    int my_rank,                                                                       \
    int seq) {                                                                         \
                                                                                       \
  int prev_rank = (my_rank - 1 + num_ranks) % num_ranks;                               \
  int next_rank = (my_rank + 1) % num_ranks;                                           \
                                                                                       \
  unsigned int seed_prev =                                                             \
      (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + prev_rank + seq);         \
  unsigned int seed_next =                                                             \
      (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x + next_rank + seq);         \
                                                                                       \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;                               \
       i < num_elems;                                                                  \
       i += blockDim.x * gridDim.x) {                                                   \
                                                                                       \
    seed_prev = ranqd1(seed_prev);                                                     \
    seed_next = ranqd1(seed_next);                                                     \
                                                                                       \
    DataType exp_prev = DataType(seed_prev % blockDim.x) / DataType(blockDim.x);       \
    DataType exp_next = DataType(seed_next % blockDim.x) / DataType(blockDim.x);       \
                                                                                       \
    if (result_buf[i] != exp_prev) {                                                   \
      if (blockIdx.x == 0 && threadIdx.x == 0 && i < 8) {                              \
        printf("***rank=%d i=%zu result=%f prev(from %d)=%f next(from %d)=%f\n",          \
               my_rank, i, (float)result_buf[i],                                      \
               prev_rank, (float)exp_prev,                                            \
               next_rank, (float)exp_next);                                           \
      }                                                                                \
    }                                                                                  \
                                                                                       \
    test_buf[i] = exp_prev;                                                           \
    assert(result_buf[i] == test_buf[i]);                                              \
  }                                                                                    \
}
*/
TEST_DATA_SENDRECV(float16, __half)
TEST_DATA_SENDRECV(float32, float)
TEST_DATA_SENDRECV(int32, int)
