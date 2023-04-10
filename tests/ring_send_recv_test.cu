#include "common.h"
#include "comm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#define BLOCK_THREADS_NUM 256

#define ALIGN 4

__global__ void initKernel(int* dataDst, int dataCount)
{
  for (size_t i = threadIdx.x; i < dataCount; i += blockDim.x) {
    dataDst[i] = i % 256;
  }
}

__constant__ mscclppDevConn_t sendConnConst;
__constant__ mscclppDevConn_t recvConnConst;

__global__ void kernel(bool root, size_t dataSize)
{
  mscclppDevConn_t sendConn = sendConnConst;
  mscclppDevConn_t recvConn = recvConnConst;

  if (root)
  {
    sendConn.putDirect(0, dataSize, threadIdx.x, blockDim.x);
    // make sure all the threads have put their data
    __syncthreads();
    if (threadIdx.x == 0){
      sendConn.signalDirect();
      recvConn.waitDirectSingal();
    }
  }
  else
  {
    if (threadIdx.x == 0) {
      recvConn.waitDirectSingal();
    }
    // make sure we get the latest data
    __syncthreads();
    sendConn.putDirect(0, dataSize, threadIdx.x, blockDim.x);
    __syncthreads();
    if (threadIdx.x == 0) {
      sendConn.signalDirect();
    }
  }
}

testResult_t resetData(int* dataDst, size_t dataCount, bool isRoot)
{
  if (isRoot) {
    initKernel<<<1, BLOCK_THREADS_NUM>>>(dataDst, dataCount);
  } else {
    CUDACHECK(cudaMemset(dataDst, 0, dataCount * sizeof(int)));
  }
  return testSuccess;
}

void RingSendRecvGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount, size_t* sendInplaceOffset,
                               size_t* recvInplaceOffset, size_t count, int nranks)
{
  size_t base = (count / ALIGN) * ALIGN;
  *sendcount = base;
  *recvcount = base;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t RingSendRecvInitData(struct testArgs* args, int in_place)
{
  size_t recvcount = args->expectedBytes / sizeof(int);

  CUDACHECK(cudaSetDevice(args->gpuNum));
  int rank = args->proc;
  CUDACHECK(cudaMemset(args->recvbuff, 0, args->expectedBytes));
  resetData((int*)args->recvbuff, recvcount, rank == 0);

  int* dataHost = new int[recvcount];
  for (size_t i = 0; i < recvcount; i++) {
    dataHost[i] = i % 256;
  }
  CUDACHECK(cudaMemcpy(args->expected, dataHost, recvcount * sizeof(int), cudaMemcpyHostToDevice));
  delete dataHost;
  CUDACHECK(cudaDeviceSynchronize());
  MSCCLPPCHECK(mscclppBootstrapBarrier(args->comm));
  return testSuccess;
}

void RingSendRecvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks)
{
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1)) / ((double)nranks);
  *busBw = baseBw * factor;
}

testResult_t RingSendRecvRunColl(void* sendbuff, void* recvbuff, int nranksPerNode, size_t count, mscclppComm_t comm,
                              cudaStream_t stream, int kernel_num)
{
  kernel<<<1, BLOCK_THREADS_NUM, 0, stream>>>(comm->rank == 0, count);
  return testSuccess;
}

struct testColl ringSendRecvTest = {"RingSendRecvTest", RingSendRecvGetCollByteCount, RingSendRecvInitData,
                                    RingSendRecvGetBw, RingSendRecvRunColl};

void RingSendRecvGetBuffSize(size_t* sendcount, size_t* recvcount, size_t count, int nranks)
{
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  RingSendRecvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count,
                               nranks);
}

testResult_t RingSendRecvRunTest(struct testArgs* args)
{
  args->collTest = &ringSendRecvTest;
  int rank = args->proc, worldSize = args->totalProcs;

  mscclppDevConn_t *sendDevConn;
  mscclppDevConn_t *recvDevConn;
  MSCCLPPCHECK(mscclppGetDeviceConnection(args->comm, (rank + 1) % worldSize, 0, &sendDevConn));
  MSCCLPPCHECK(mscclppGetDeviceConnection(args->comm, (rank - 1 + worldSize) % worldSize, 0, &recvDevConn));
  CUDACHECK(cudaMemcpyToSymbol(sendConnConst, sendDevConn, sizeof(mscclppDevConn_t)));
  CUDACHECK(cudaMemcpyToSymbol(recvConnConst, recvDevConn, sizeof(mscclppDevConn_t)));
  TESTCHECK(TimeTest(args));
  return testSuccess;
}

struct testEngine ringSendRecvTestEngine = {RingSendRecvGetBuffSize, RingSendRecvRunTest};

#pragma weak mscclppTestEngine = ringSendRecvTestEngine