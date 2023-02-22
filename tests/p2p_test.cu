#include "mscclpp.h"
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
#include "mpi.h"
#endif // MSCCLPP_USE_MPI_FOR_TESTS
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

#define RANKS_PER_NODE 8
#define TEST_CONN_TYPE 1 // 0: P2P(for local)+IB(for remote), 1: IB-Only

// Check CUDA RT calls
#define CUDACHECK(cmd) do {                                   \
    cudaError_t err = cmd;                                    \
    if( err != cudaSuccess ) {                                \
        printf("%s:%d Cuda failure '%s'", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(false)

__global__ void kernel(mscclppDevConn_t devConns, int rank, int world_size)
{
  int warpId = threadIdx.x / 32;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;
  mscclppDevConn_t devConn = &devConns[(remoteRank < rank) ? remoteRank : remoteRank - 1];
  volatile int *data = (volatile int *)devConn->localBuff;
  volatile int *localFlag = devConn->localFlag;
  volatile int *remoteFlag = devConn->remoteFlag;
  volatile uint64_t *trig = (volatile uint64_t *)devConn->trigger;

  if (threadIdx.x == 0) {
    // Set my data and flag
    *(data + rank) = rank + 1;
    __threadfence_system();
    *localFlag = 1;
  }
  __syncthreads();

  // Each warp receives data from different ranks
  if (threadIdx.x % 32 == 0) {
    if (devConn->remoteBuff == NULL) { // IB
      // Trigger sending data and flag
      uint64_t dataOffset = rank * sizeof(int);
      uint64_t dataSize = sizeof(int);
      *trig = (dataOffset << 32) + dataSize;

      // Wait until the proxy have sent my data and flag
      while (*trig != 0) {}

      // Wait for receiving data from remote rank
      while (*remoteFlag != 1) {}
    } else { // P2P
      // Directly read data
      volatile int *remoteData = (volatile int *)devConn->remoteBuff;

      // Wait until the remote data is set
      while (*remoteFlag != 1) {}

      // Read remote data
      data[remoteRank] = remoteData[remoteRank];
    }
  }
}

int rankToLocalRank(int rank)
{
  return rank % RANKS_PER_NODE;
}

int rankToNode(int rank)
{
  return rank / RANKS_PER_NODE;
}

void print_usage(const char *prog)
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  printf("usage: %s IP:PORT [rank nranks]\n", prog);
#else
  printf("usage: %s IP:PORT rank nranks\n", prog);
#endif
}

int main(int argc, const char *argv[])
{
#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc != 2 && argc != 4) {
    print_usage(argv[0]);
    return -1;
  }
  const char *ip_port = argv[1];
  int rank;
  int world_size;
  if (argc == 4) {
    rank = atoi(argv[2]);
    world_size = atoi(argv[3]);
  } else {
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  }
#else
  if (argc != 4) {
    print_usage(argv[0]);
    return -1;
  }
  const char *ip_port = argv[1];
  int rank = atoi(argv[2]);
  int world_size = atoi(argv[3]);
#endif
  int localRank = rankToLocalRank(rank);
  int thisNode = rankToNode(rank);

  mscclppComm_t comm;
  mscclppResult_t res = mscclppCommInitRank(&comm, world_size, rank, ip_port);
  if (res != mscclppSuccess) {
    printf("mscclppCommInitRank failed\n");
    return -1;
  }

  CUDACHECK(cudaSetDevice(localRank));

  int *data_d;
  int *flag_d;
  CUDACHECK(cudaMalloc(&data_d, sizeof(int) * world_size));
  CUDACHECK(cudaMalloc(&flag_d, sizeof(int)));
  CUDACHECK(cudaMemset(data_d, 0, sizeof(int) * world_size));
  CUDACHECK(cudaMemset(flag_d, 0, sizeof(int)));

  std::string ibDevStr = "mlx5_ib" + std::to_string(localRank);

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    mscclppTransport_t transportType = mscclppTransportIB;
    const char *ibDev = ibDevStr.c_str();
#if (TEST_CONN_TYPE == 0) // P2P+IB
    if (rankToNode(r) == thisNode) {
      transportType = mscclppTransportP2P;
      ibDev = NULL;
    }
#endif
    // Connect with all other ranks
    res = mscclppConnect(comm, r, data_d, sizeof(int) * world_size, flag_d, 0, transportType, ibDev);
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

  res = mscclppProxyLaunch(comm);
  if (res != mscclppSuccess) {
    printf("mscclppProxyLaunch failed\n");
    return -1;
  }

  mscclppDevConn_t devConns;
  mscclppGetDevConns(comm, &devConns);

  kernel<<<1, 32 * (world_size - 1)>>>(devConns, rank, world_size);
  CUDACHECK(cudaDeviceSynchronize());

  res = mscclppProxyStop(comm);
  if (res != mscclppSuccess) {
    printf("mscclppProxyStop failed\n");
    return -1;
  }

  // Read results from GPU
  int *buf = (int *)calloc(world_size, sizeof(int));
  if (buf == nullptr) {
    printf("calloc failed\n");
    return -1;
  }
  CUDACHECK(cudaMemcpy(buf, data_d, sizeof(int) * world_size, cudaMemcpyDeviceToHost));

  bool failed = false;
  for (int i = 0; i < world_size; ++i) {
    if (buf[i] != i + 1) {
      printf("rank: %d, wrong data: %d, expected %d\n", rank, buf[i], i + 1);
      failed = true;
    }
  }
  if (failed) {
    return -1;
  }

  res = mscclppCommDestroy(comm);
  if (res != mscclppSuccess) {
    printf("mscclppDestroy failed\n");
    return -1;
  }

#ifdef MSCCLPP_USE_MPI_FOR_TESTS
  if (argc == 2) {
    MPI_Finalize();
  }
#endif
  printf("Succeeded! %d\n", rank);
  return 0;
}
