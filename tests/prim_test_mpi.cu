#include "mscclpp.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "prims/primitives.h"
// Check CUDA RT calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        printf("Cuda failure '%s'", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(false)

__global__ void kernel(mscclppDevConn_t devConns, int rank, int world_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // Set my data
    volatile int *data = (volatile int *)devConns[rank].localBuff;
    volatile int *flag = (volatile int *)devConns[rank].localFlag;
    data[rank] = rank;

    // Inform that the data is set
    *flag = 1;

    for (int i = 0; i < (world_size - 1) * 2; ++i) {
      mscclppDevConn_t devConn = &devConns[i];
      int tag = devConn->tag;
      int rankRecv = tag / world_size;
      int rankSend = tag % world_size;

      if (rankRecv != rank) continue;

      volatile int *remoteData = (volatile int *)devConn->remoteBuff;
      volatile int *remoteFlag = (volatile int *)devConn->remoteFlag;

      // Wait until the remote data is set
      while (*remoteFlag != 1) {}

      // Read remote data
      data[rankSend] = remoteData[rankSend];
    }
  }
}

__global__ void test_send_ll(float *data_src, char *recvbuff,
                             uint64_t *sendConnHead, int size)
{
    using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int sendPeers[2] = {0, -1};
    int recvPeers[2] = {0, -1};
    ncclDevChannelPeer peerInfo;
    peerInfo.send[0].buffs[NCCL_PROTO_LL] = recvbuff;
    peerInfo.send[0].head = sendConnHead;
    peerInfo.send[0].step = 0;
    // peerInfo.recv[0].buffs[NCCL_PROTO_LL] = recvbuff;
    // peerInfo.recv[0].head = sendConnHead;
    // peerInfo.recv[0].step = 0;
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, sendPeers, recvPeers, data_src, NULL, &peerInfo,
        ncclDevSum, 0);
    prims.send(0, size);
    return;
}

__global__ void test_recv_ll(float *data_dst, char *recvbuff,
                             uint64_t *sendConnHead, int size)
{
    using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int sendPeers[2] = {0, -1};
    int recvPeers[2] = {0, -1};
    ncclDevChannelPeer peerInfo;
    // peerInfo.send[0].buffs[NCCL_PROTO_LL] = recvbuff;
    peerInfo.send[0].head = sendConnHead;
    peerInfo.send[0].step = 0;
    peerInfo.recv[0].buffs[NCCL_PROTO_LL] = recvbuff;
    // peerInfo.recv[0].head = sendConnHead;
    peerInfo.recv[0].step = 0;
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, sendPeers, recvPeers, NULL, data_dst, &peerInfo,
        ncclDevSum, 0);
    prims.recv(0, size);
    return;
}

void print_usage(const char *prog)
{
  printf("usage: %s IP:PORT\n", prog);
}

int main(int argc, const char *argv[])
{
  if (argc != 2) {
    print_usage(argv[0]);
    return -1;
  }

  MPI_Init(NULL, NULL);

  int rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  mscclppComm_t comm;
  const char *ip_port = argv[1];
  mscclppCommInitRank(&comm, world_size, rank, ip_port);

  int *data_d;
  int *flag_d;
  CUDACHECK(cudaMalloc(&data_d, sizeof(int) * world_size));
  CUDACHECK(cudaMalloc(&flag_d, sizeof(int)));
  printf("-------- buf: %p, flag: %p\n", data_d, flag_d);

  mscclppResult_t res;

  // Read from all other ranks
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    int tag = rank * world_size + r;
    res = mscclppConnect(comm, rank, r, data_d,sizeof(int) * world_size, flag_d, tag, mscclppTransportP2P);
    if (res != mscclppSuccess) {
      printf("mscclppConnect failed\n");
      return -1;
    }
  }
  // Let others read from me
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    int tag = r * world_size + rank;
    res = mscclppConnect(comm, r, rank, data_d,sizeof(int) * world_size , flag_d, tag, mscclppTransportP2P);
    if (res != mscclppSuccess) {
      printf("mscclppConnect failed\n");
      return -1;
    }
  }
  res = mscclppConnectionSetup(comm);
  if (res != mscclppSuccess) {
    printf("mscclppConnectionSetup failed\n");
    return -1;
  }

  mscclppDevConn_t devConns;
  mscclppGetDevConns(comm, &devConns);

  kernel<<<1, 1>>>(devConns, rank, world_size);
  CUDACHECK(cudaDeviceSynchronize());

  int *buf = (int *)calloc(world_size, sizeof(int));
  if (buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  CUDACHECK(cudaMemcpy(buf, data_d, sizeof(int) * world_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < world_size; ++i) {
    if (buf[i] != i) {
      printf("wrong data: %d, expected %d\n", buf[i], i);
      return -1;
    }
  }

  res = mscclppCommDestroy(comm);
  if (res != mscclppSuccess) {
    printf("mscclppDestroy failed\n");
    return -1;
  }

  MPI_Finalize();

  printf("Succeeded! %d\n", rank);
  return 0;
}
