/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "cuda_runtime.h"
#include "common.h"

#define ALIGN 4

void AllGatherGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount, size_t* sendInplaceOffset,
                               size_t* recvInplaceOffset, size_t count, int nranks)
{
  size_t base = (count / (ALIGN * nranks)) * ALIGN;
  *sendcount = base;
  *recvcount = base * nranks;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t AllGatherInitData(struct threadArgs* args, int in_place) {
  size_t sendcount = args->sendBytes;
  size_t recvcount = args->expectedBytes;
  int nranks = args->totalProcs;

  CUDACHECK(cudaSetDevice(args->gpus[0]));
  int rank = args->proc;
  CUDACHECK(cudaMemset(args->recvbuffs[0], 0, args->expectedBytes));
  void* data = in_place ? ((char*)args->recvbuffs[0]) + rank * args->sendBytes : args->sendbuffs[0];
  // TESTCHECK(InitData(data, sendcount, 0, type, ncclSum, 33 * rep + rank, 1, 0));
  // for (int j = 0; j < nranks; j++) {
  //   TESTCHECK(
  //     InitData((char*)args->expected[0] + args->sendBytes * j, sendcount, 0, type, ncclSum, 33 * rep + j, 1, 0));
  // }
  CUDACHECK(cudaDeviceSynchronize());

  return testSuccess;
}

void AllGatherGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks) {
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1))/((double)nranks);
  *busBw = baseBw * factor;
}

testResult_t AllGatherRunColl(void* sendbuff, void* recvbuff, size_t count) {
  // NCCLCHECK(ncclAllGather(sendbuff, recvbuff, count, type, comm, stream));
  return testSuccess;
}

struct testColl allGatherTest = {
  "AllGather",
  AllGatherGetCollByteCount,
  AllGatherInitData,
  AllGatherGetBw,
  AllGatherRunColl
};

void AllGatherGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  AllGatherGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t AllGatherRunTest(struct threadArgs* args) {
  args->collTest = &allGatherTest;

  TESTCHECK(TimeTest(args));
  return testSuccess;
}

struct testEngine allGatherEngine = {
  AllGatherGetBuffSize,
  AllGatherRunTest
};

#pragma weak mscclppTestEngine=allGatherEngine