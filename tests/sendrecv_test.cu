#include "comm.h"
#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#define BLOCK_THREADS_NUM 128

#define ALIGN 4

__constant__ mscclppDevConn_t sendConnConst;
__constant__ mscclppDevConn_t recvConnConst;

inline int getSendTag(int rank, int peer)
{
  return rank < peer ? 0 : 1;
}

inline int getRecvTag(int rank, int peer)
{
  return rank < peer ? 1 : 0;
}

__global__ void kernel(size_t dataSize, int rank)
{
  mscclppDevConn_t sendConn = sendConnConst;
  mscclppDevConn_t recvConn = recvConnConst;

  sendConn.putDirect(blockIdx.x *sizeof(int), dataSize, threadIdx.x, blockDim.x);
  // if (sendConn.remoteRank == 0 && threadIdx.x == 0) {
  //   printf("\nrank is %d, send data to rank %d, tag is %d, localBuff is %p, size is %ld\n", rank, sendConn.remoteRank,
  //          sendConn.tag, sendConn.localBuff, dataSize / sizeof(int));
  //   for (size_t i = 0; i < dataSize / sizeof(int); i++) {
  //     printf("%d ", ((int*)sendConn.localBuff)[i]);
  //   }
  //   printf("\n");
  //   printf("recv from rank %d, local buff is %p\n", recvConn.remoteRank, recvConn.localBuff);
  // }
  __syncthreads();
  if (threadIdx.x == 0) {
    sendConn.signalDirect();
    recvConn.waitDirect();
  }
  // __syncthreads();
}

void SendRecvGetCollByteCount(size_t* sendcount, size_t* recvcount, size_t* paramcount, size_t* sendInplaceOffset,
                              size_t* recvInplaceOffset, size_t count, int nranks)
{
  size_t base = (count / ALIGN) * ALIGN;
  *sendcount = base;
  *recvcount = base;
  *sendInplaceOffset = base;
  *recvInplaceOffset = 0;
  *paramcount = base;
}

testResult_t SendRecvInitData(struct testArgs* args, int in_place)
{
  size_t sendCount = args->sendBytes / sizeof(int);
  size_t recvCount = args->expectedBytes / sizeof(int);
  size_t maxCount = std::max(sendCount, recvCount);

  int rank = args->proc;
  CUDACHECK(cudaMemset(args->sendbuff, 0, args->sendBytes));
  std::vector<int> dataHost(maxCount, rank);
  CUDACHECK(cudaMemcpy(args->sendbuff, dataHost.data(), sendCount * sizeof(int), cudaMemcpyHostToDevice));

  int recvPeerRank = (rank - 1 + args->totalProcs) % args->totalProcs;
  for (size_t i = 0; i < recvCount; i++) {
    dataHost[i] = recvPeerRank;
  }
  CUDACHECK(cudaMemcpy(args->expected, dataHost.data(), recvCount * sizeof(int), cudaMemcpyHostToDevice));
  MSCCLPPCHECK(mscclppBootstrapBarrier(args->comm));

  return testSuccess;
}

void SendRecvGetBw(size_t count, int typesize, double sec, double* algBw, double* busBw, int nranks)
{
  double baseBw = (double)(count * typesize * nranks) / 1.0E9 / sec;

  *algBw = baseBw;
  double factor = ((double)(nranks - 1)) / ((double)nranks);
  *busBw = baseBw * factor;
}

testResult_t SendRecvRunColl(void* sendbuff, void* recvbuff, int nranksPerNode, size_t count, mscclppComm_t comm,
                             cudaStream_t stream, int kernel_num)
{
  kernel<<<1, BLOCK_THREADS_NUM, 0, stream>>>(count, comm->rank);
  return testSuccess;
}

struct testColl sendRecvTest = {"SendRecvTest", SendRecvGetCollByteCount, SendRecvInitData, SendRecvGetBw,
                                SendRecvRunColl};

void SendRecvGetBuffSize(size_t* sendcount, size_t* recvcount, size_t count, int nranks)
{
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  SendRecvGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, count, nranks);
}

testResult_t SendRecvSetupConnections(struct testArgs* args)
{
  int rank = args->proc;
  int worldSize = args->totalProcs;
  int ranksPerNode = args->nranksPerNode;
  int thisNode = rank / ranksPerNode;
  int localRank = rank % ranksPerNode;
  std::string ibDevStr = "mlx5_ib" + std::to_string(localRank);
  int sendToRank = (rank + 1) % worldSize;
  int recvFromRank = (rank - 1 + worldSize) % worldSize;
  std::array<int, 2> ranks = {sendToRank, recvFromRank};

  for (int i = 0; i < 2; i++) {
    int r = ranks[i];
    const char* ibDev = r / ranksPerNode == thisNode ? nullptr : ibDevStr.c_str();
    mscclppTransport_t transportType = ibDev == nullptr ? mscclppTransportP2P : mscclppTransportIB;
    void* buff = (i == 0) ? args->sendbuff : args->recvbuff;
    int tag = (i == 0) ? getSendTag(rank, r) : getRecvTag(rank, r);
    MSCCLPPCHECK(mscclppConnect(args->comm, r, tag, buff, args->maxbytes, transportType, ibDev));
  }
  MSCCLPPCHECK(mscclppConnectionSetup(args->comm));

  return testSuccess;
}

testResult_t SendRecvRunTest(struct testArgs* args)
{
  args->collTest = &sendRecvTest;
  int rank = args->proc, worldSize = args->totalProcs;

  // only support out-of-place for sendrecv test
  args->in_place = 0;

  mscclppDevConn_t* sendDevConn;
  mscclppDevConn_t* recvDevConn;
  MSCCLPPCHECK(mscclppGetDeviceConnection(args->comm, (rank + 1) % worldSize, getSendTag(rank, (rank + 1) % worldSize),
                                          &sendDevConn));
  MSCCLPPCHECK(mscclppGetDeviceConnection(args->comm, (rank - 1 + worldSize) % worldSize,
                                          getRecvTag(rank, (rank - 1 + worldSize) % worldSize), &recvDevConn));
  CUDACHECK(cudaMemcpyToSymbol(sendConnConst, sendDevConn, sizeof(mscclppDevConn_t)));
  CUDACHECK(cudaMemcpyToSymbol(recvConnConst, recvDevConn, sizeof(mscclppDevConn_t)));
  TESTCHECK(TimeTest(args));
  return testSuccess;
}

struct testEngine sendRecvTestEngine = {SendRecvGetBuffSize, SendRecvRunTest, SendRecvSetupConnections, nullptr};

#pragma weak mscclppTestEngine = sendRecvTestEngine
