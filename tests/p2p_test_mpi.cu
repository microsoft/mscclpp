#include "mscclpp.h"
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MSCCLPPCHECK(call) do { \
  mscclppResult_t res = call; \
  if (res != mscclppSuccess && res != mscclppInProgress) { \
    /* Print the back trace*/ \
    printf("Failure at %s:%d -> %d", __FILE__, __LINE__, res);    \
    return res; \
  } \
} while (0);

// Check CUDA RT calls
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        printf("Cuda failure '%s'", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
} while(false)

__constant__ mscclppDevConn_t constDevConns[8];

__global__ void kernel(int rank, int world_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // Set my data
    volatile int *data = (volatile int *)constDevConns[(rank+1) % world_size].localBuff;
    volatile int *flag = (volatile int *)constDevConns[(rank+1) % world_size].localFlag;
    data[rank] = rank;

    // Inform that the data is set
    *flag = 1;

    for (int i = 0; i < world_size; ++i) {
      if (i == rank) continue;
      mscclppDevConn_t* devConn = &constDevConns[i];
      // int tag = devConn->tag;
      // int rankRecv = tag / world_size;
      // int rankSend = tag % world_size;

      // if (rankRecv != rank) continue;

      volatile int *remoteData = (volatile int *)devConn->remoteBuff;
      volatile int *remoteFlag = (volatile int *)devConn->remoteFlag;
      // printf("i = %d ptr1 %p, ptr2 %p\n", i,remoteData, remoteFlag);

      // Wait until the remote data is set
      while (*remoteFlag != 1) {}

      // Read remote data
      data[i] = remoteData[i];
    }
  }
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
  CUDACHECK(cudaSetDevice(rank % 8));
  printf("Starting rank %d of %d\n", rank, world_size);

  mscclppComm_t comm;
  const char *ip_port = argv[1];
  mscclppCommInitRank(&comm, world_size, rank, ip_port);

  int *data_d;
  int *flag_d;
  CUDACHECK(cudaMalloc(&data_d, sizeof(int) * world_size));
  CUDACHECK(cudaMalloc(&flag_d, sizeof(int)));
  printf("-------- buf: %p, flag: %p\n", data_d, flag_d);

  mscclppResult_t res;

  mscclppDevConn_t devConns[8];
  // Read from all other ranks
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    int tag = 0;
    MSCCLPPCHECK(mscclppConnect(comm, &devConns[r], r, data_d, flag_d, tag, mscclppTransportP2P));
  }

  MSCCLPPCHECK(mscclppConnectionSetup(comm));
  CUDACHECK(cudaMemcpyToSymbol(constDevConns, devConns, sizeof(mscclppDevConn_t) * world_size));
  
  
  kernel<<<1, 1>>>(rank, world_size);
  
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

  MSCCLPPCHECK(mscclppCommDestroy(comm));

  MPI_Finalize();

  printf("Succeeded! %d\n", rank);
  return 0;
}
