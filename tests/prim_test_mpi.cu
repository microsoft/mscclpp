#include "mpi.h"
#include "mscclpp.h"
#include "prims_ll.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
// Check CUDA RT calls
#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t err = cmd;                                                 \
        if (err != cudaSuccess) {                                              \
            printf("Cuda failure '%s'", cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (false)

__device__ void test_send_ll(void *data_src, void *recvbuff, void *sendConnHead,
                             int size)
{
    // using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    // Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
    //     tid, nthreads, ncclDevSum, 0);
    // prims.send(0, size);
    return;
}

__device__ void test_recv_ll(void *data_dst, void *recvbuff, void *sendConnHead,
                             int size)
{
    // using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    // Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
    //     tid, nthreads, ncclDevSum, 0);
    // prims.recv(0, size);
    return;
}

__global__ void kernel(mscclppDevConn_t devConns, int rank, int world_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid == 0) {
    //   // Set my data
    //   volatile int *data = (volatile int *)devConns[rank].localBuff;
    //   volatile int *flag = (volatile int *)devConns[rank].localFlag;
    //   data[rank] = rank;

    //   // Inform that the data is set
    //   *flag = 1;

    //   for (int i = 0; i < (world_size - 1) * 2; ++i) {
    //     mscclppDevConn_t devConn = &devConns[i];
    //     int tag = devConn->tag;
    //     int rankRecv = tag / world_size;
    //     int rankSend = tag % world_size;

    //     if (rankRecv != rank) continue;

    //     volatile int *remoteData = (volatile int *)devConn->remoteBuff;
    //     volatile int *remoteFlag = (volatile int *)devConn->remoteFlag;

    //     // Wait until the remote data is set
    //     while (*remoteFlag != 1) {}

    //     // Read remote data
    //     data[rankSend] = remoteData[rankSend];
    //   }
    // }
    if (rank == 0) {
        test_send_ll(devConns[0].localBuff, devConns[0].remoteBuff,
                     devConns[0].remoteFlag, 1);
    } else {
        test_recv_ll(devConns[0].localBuff, devConns[0].remoteBuff,
                     devConns[0].remoteFlag, 1);
    }
}

void print_usage(const char *prog) { printf("usage: %s IP:PORT\n", prog); }

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

    int *data_src;
    int *data_dst;
    int elem_num = 1024;
    int data_size = sizeof(float) * elem_num;
    int *flag_d;
    CUDACHECK(cudaMalloc(&data_src, data_size));
    CUDACHECK(cudaMalloc(&flag_d, sizeof(int)));

    mscclppResult_t res;

    // Read from all other ranks
    for (int r = 0; r < world_size; ++r) {
        if (r == rank)
            continue;
        int tag = rank * world_size + r;
        res = mscclppConnect(comm, rank, r, data_src, data_size,
                             flag_d, tag, mscclppTransportP2P);
        if (res != mscclppSuccess) {
            printf("mscclppConnect failed\n");
            return -1;
        }
    }
    // Let others read from me
    for (int r = 0; r < world_size; ++r) {
        if (r == rank)
            continue;
        int tag = r * world_size + rank;
        res = mscclppConnect(comm, r, rank, data_src, data_size,
                             flag_d, tag, mscclppTransportP2P);
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
    CUDACHECK(cudaMemcpy(buf, data_dst, data_size,
                         cudaMemcpyDeviceToHost));


    res = mscclppCommDestroy(comm);
    if (res != mscclppSuccess) {
        printf("mscclppDestroy failed\n");
        return -1;
    }

    MPI_Finalize();

    printf("Succeeded! %d\n", rank);
    return 0;
}
