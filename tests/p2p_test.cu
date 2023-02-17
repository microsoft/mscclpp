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
        printf("Cuda failure '%s'", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(false)

__global__ void kernel(mscclppDevConn_t devConns, int rank, int world_size)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // Get sending data and send flag
    volatile int *data;
    for (int i = 0; i < (world_size - 1) * 2; ++i) {
      mscclppDevConn_t devConn = &devConns[i];
      int tag = devConn->tag;
      int rankSend = tag % world_size;
      if (rankSend == rank) { // I am a sender
        data = (volatile int *)devConn->localBuff;
        // We are sending the same data to all peers, so just break here
        break;
      }
    }

    // Set my data
    *data = rank + 1;

    // Set send flags to inform all peers that the data is ready
    for (int i = 0; i < (world_size - 1) * 2; ++i) {
      mscclppDevConn_t devConn = &devConns[i];
      int tag = devConn->tag;
      int rankSend = tag % world_size;
      if (rankSend == rank) { // I am a sender
        *((volatile int *)devConn->localFlag) = 1;
      }
    }

    // Read data from all other peers
    for (int i = 0; i < (world_size - 1) * 2; ++i) {
      mscclppDevConn_t devConn = &devConns[i];
      int tag = devConn->tag;
      int rankSend = tag % world_size;
      int rankRecv = tag / world_size;
      if (rankRecv == rank) { // I am a receiver
        if (devConn->remoteBuff == NULL) { // IB
          volatile int *localFlag = (volatile int *)devConn->localFlag;

          // Wait until the data comes in via proxy
          while (*localFlag != 1) {}
        } else { // P2P
          volatile int *remoteData = (volatile int *)devConn->remoteBuff;
          volatile int *remoteFlag = (volatile int *)devConn->remoteFlag;

          // Wait until the remote data is set
          while (*remoteFlag != 1) {}

          // Read remote data
          data[rankSend] = remoteData[rankSend];
        }
      }
    }

    // Wait until the proxy have sent my data to all peers
    for (int i = 0; i < (world_size - 1) * 2; ++i) {
      mscclppDevConn_t devConn = &devConns[i];
      int tag = devConn->tag;
      int rankSend = tag % world_size;
      if (rankSend == rank) { // I am a sender
        volatile int *flag = (volatile int *)devConn->localFlag;
        while (*flag == 1) {}
      }
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

  mscclppComm_t comm;
  mscclppResult_t res = mscclppCommInitRank(&comm, world_size, rank, ip_port);
  if (res != mscclppSuccess) {
    printf("mscclppCommInitRank failed\n");
    return -1;
  }

  int *data_d;
  int *send_flags_d;
  int *recv_flags_d;
  CUDACHECK(cudaMalloc(&data_d, sizeof(int) * world_size));
  CUDACHECK(cudaHostAlloc(&send_flags_d, sizeof(int) * (world_size - 1), cudaHostAllocMapped));
  CUDACHECK(cudaHostAlloc(&recv_flags_d, sizeof(int) * (world_size - 1), cudaHostAllocMapped));

  CUDACHECK(cudaMemset(data_d, 0, sizeof(int) * world_size));
  // CUDACHECK(cudaMemcpy(data_d, tmp, sizeof(int) * 2, cudaMemcpyHostToDevice));
  // printf("rank %d CPU: setting data at %p\n", rank, data_d + rank);
  memset(send_flags_d, 0, sizeof(int) * (world_size - 1));
  memset(recv_flags_d, 0, sizeof(int) * (world_size - 1));

  int localRank = rankToLocalRank(rank);
  int thisNode = rankToNode(rank);
  std::string ibDev = "mlx5_ib" + std::to_string(localRank);

  // Read from all other ranks
  int idx = 0;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    int tag = rank * world_size + r;
#if (TEST_CONN_TYPE == 0) // P2P+IB
    int node = rankToNode(r);
    if (node == thisNode) {
      res = mscclppConnect(comm, rank, r, data_d + r, sizeof(int), recv_flags_d + idx, tag, mscclppTransportP2P);
    } else {
      res = mscclppConnect(comm, rank, r, data_d + r, sizeof(int), recv_flags_d + idx, tag, mscclppTransportIB, ibDev.c_str());
    }
#else // (TEST_CONN_TYPE == 1) // IB-Only
    res = mscclppConnect(comm, rank, r, data_d + r, sizeof(int), recv_flags_d + idx, tag, mscclppTransportIB, ibDev.c_str());
#endif
    if (res != mscclppSuccess) {
      printf("mscclppConnect failed\n");
      return -1;
    }
    ++idx;
  }
  // Let others read from me
  idx = 0;
  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    int tag = r * world_size + rank;
#if (TEST_CONN_TYPE == 0) // P2P+IB
    int node = rankToNode(r);
    if (node == thisNode) {
      res = mscclppConnect(comm, r, rank, data_d + rank, sizeof(int), send_flags_d + idx, tag, mscclppTransportP2P);
    } else {
      res = mscclppConnect(comm, r, rank, data_d + rank, sizeof(int), send_flags_d + idx, tag, mscclppTransportIB, ibDev.c_str());
    }
#else // (TEST_CONN_TYPE == 1) // IB-Only
    res = mscclppConnect(comm, r, rank, data_d + rank, sizeof(int), send_flags_d + idx, tag, mscclppTransportIB, ibDev.c_str());
#endif
    if (res != mscclppSuccess) {
      printf("mscclppConnect failed\n");
      return -1;
    }
    ++idx;
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

  kernel<<<1, 1>>>(devConns, rank, world_size);
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
