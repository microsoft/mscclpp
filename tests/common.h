/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_TESTS_COMMON_H_
#define MSCCLPP_TESTS_COMMON_H_

#include "mscclpp.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <unistd.h>

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include <mpi.h>
#endif // MSCCLPP_USE_MPI_FOR_TESTS

#define CUDACHECK(cmd)                                                                                                 \
  do {                                                                                                                 \
    cudaError_t err = cmd;                                                                                             \
    if (err != cudaSuccess) {                                                                                          \
      char hostname[1024];                                                                                             \
      getHostName(hostname, 1024);                                                                                     \
      printf("%s: Test CUDA failure %s:%d '%s'\n", hostname, __FILE__, __LINE__, cudaGetErrorString(err));             \
      return testCudaError;                                                                                            \
    }                                                                                                                  \
  } while (0)

// Propagate errors up
#define MSCCLPPCHECK(cmd)                                                                                              \
  do {                                                                                                                 \
    mscclppResult_t res = cmd;                                                                                         \
    if (res != mscclppSuccess && res != mscclppInProgress) {                                                           \
      char hostname[1024];                                                                                             \
      getHostName(hostname, 1024);                                                                                     \
      printf("%s: Failure at %s:%d -> %s\n", hostname, __FILE__, __LINE__, mscclppGetErrorString(res));                \
      return testMcclppError;                                                                                          \
    }                                                                                                                  \
  } while (0);

// Relay errors up and trace
#define TESTCHECK(cmd)                                                                                                 \
  do {                                                                                                                 \
    testResult_t r = cmd;                                                                                              \
    if (r != testSuccess) {                                                                                            \
      char hostname[1024];                                                                                             \
      getHostName(hostname, 1024);                                                                                     \
      printf(" .. %s pid %d: Test failure %s:%d\n", hostname, getpid(), __FILE__, __LINE__);                           \
      return r;                                                                                                        \
    }                                                                                                                  \
  } while (0)

typedef enum
{
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testMcclppError = 3,
  testTimeout = 4,
  testNumResults = 5
} testResult_t;

struct testColl
{
  const char name[20];
  void (*getCollByteCount)(size_t* sendcount, size_t* recvcount, size_t* paramcount, size_t* sendInplaceOffset,
                           size_t* recvInplaceOffset, size_t count, int nranks);
  testResult_t (*initData)(struct testArgs* args, int in_place);
  void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(void* sendbuff, void* recvbuff, int nranksPerNode, size_t count, mscclppComm_t comm,
                          cudaStream_t stream, int kernel_num);
};

struct testEngine
{
  void (*getBuffSize)(size_t* sendcount, size_t* recvcount, size_t count, int nranks);
  // We can add more parameters for other communication primitives
  testResult_t (*runTest)(struct testArgs* args);
};

extern struct testEngine mscclppTestEngine;

struct testArgs
{
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int totalProcs;
  int proc;
  int gpuNum;
  int localRank;
  int nranksPerNode;
  int kernel_num;
  void* sendbuff;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void* recvbuff;
  size_t recvInplaceOffset;
  mscclppComm_t comm;
  cudaStream_t stream;

  void* expected;
  size_t expectedBytes;
  int error;
  double bw;
  int bw_count;

  int reportErrors;

  struct testColl* collTest;
};

typedef testResult_t (*entryFunc_t)(struct testArgs* args);
struct testWorker
{
  entryFunc_t func;
  struct testArgs args;
};

// Provided by common.cu
extern testResult_t TimeTest(struct testArgs* args);

static void getHostName(char* hostname, int maxlen)
{
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

inline void print_usage(const char* prog)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  printf("usage: %s IP:PORT [rank nranks]\n", prog);
#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

#define PRINT                                                                                                          \
  if (is_main_thread)                                                                                                  \
  printf

#endif // MSCCLPP_TESTS_COMMON_H_