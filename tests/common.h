/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_TESTS_COMMON_H_
#define MSCCLPP_TESTS_COMMON_H_

#include "mscclpp.h"

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

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testMcclppError = 3,
  testTimeout = 4,
  testNumResults = 5
} testResult_t;

struct testColl {
  const char name[20];
  void (*getCollByteCount)(
      size_t *sendcount, size_t *recvcount, size_t *paramcount,
      size_t *sendInplaceOffset, size_t *recvInplaceOffset,
      size_t count, int nranks);
  testResult_t (*initData)(struct threadArgs* args, int in_place);
  void (*getBw)(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks);
  testResult_t (*runColl)(void* sendbuff, void* recvbuff, int nranksPerNode, size_t count, mscclppComm_t comm,
                          cudaStream_t stream);
};

struct testEngine
{
  void (*getBuffSize)(size_t* sendcount, size_t* recvcount, size_t count, int nranks);
  // We can add more parameters for other communication primitives
  testResult_t (*runTest)(struct threadArgs* args);
};

extern struct testEngine mscclppTestEngine;

struct threadArgs
{
  size_t nbytes;
  size_t minbytes;
  size_t maxbytes;
  size_t stepbytes;
  size_t stepfactor;

  int totalProcs;
  int proc;
  int nThreads;
  int thread;
  int nGpus;
  int* gpus;
  int localRank;
  int nranksPerNode;
  void** sendbuffs;
  size_t sendBytes;
  size_t sendInplaceOffset;
  void** recvbuffs;
  size_t recvInplaceOffset;
  mscclppComm_t comm;
  cudaStream_t stream;

  void** expected;
  size_t expectedBytes;
  int* errors;
  double* bw;
  int* bw_count;

  int reportErrors;

  struct testColl* collTest;
};

typedef testResult_t (*threadFunc_t)(struct threadArgs* args);
struct testThread
{
  pthread_t thread;
  threadFunc_t func;
  struct threadArgs args;
};

// Provided by common.cu
extern testResult_t TimeTest(struct threadArgs* args);

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

inline void parse_arguments(int argc, const char* argv[], const char** ip_port, int* rank, int* world_size)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc != 2 && argc != 4) {
    print_usage(argv[0]);
    exit(-1);
  }
  *ip_port = argv[1];
  if (argc == 4) {
    *rank = atoi(argv[2]);
    *world_size = atoi(argv[3]);
  } else {
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, world_size);
  }
#else
  if (argc != 4) {
    print_usage(argv[0]);
    exit(-1);
  }
  *ip_port = argv[1];
  *rank = atoi(argv[2]);
  *world_size = atoi(argv[3]);
#endif
}

#define PRINT                                                                                                          \
  if (is_main_thread)                                                                                                  \
  printf

#endif // MSCCLPP_TESTS_COMMON_H_