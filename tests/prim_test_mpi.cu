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

__global__ void test_send_ll(void *data_src, void *recvbuff,
                             void *sendConnHeadPtr, int size)
{
    // using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    Primitives_LL<float> prims(tid, nthreads, 0, 0);
    prims.sendConnHeadPtr = (volatile uint64_t *)sendConnHeadPtr;
    prims.data_src = (float *)data_src;
    prims.recvBuff = (ncclLLFifoLine *)recvbuff;
    prims.send(0, size);
    return;
}

__global__ void test_recv_ll(void *data_dst, void *recvbuff,
                             void *sendConnHeadPtr, int size)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    Primitives_LL<float> prims(tid, nthreads, 0, 0);
    prims.sendConnHeadPtr = (volatile uint64_t *)sendConnHeadPtr;
    prims.data_dst = (float *)data_dst;
    prims.recvBuff = (ncclLLFifoLine *)recvbuff;
    prims.recv(0, size);
    return;
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

    float *data_src;
    float *data_dst;
    char *recvbuff;
    int elem_num = 1024;
    int data_size = sizeof(float) * elem_num;
    int *flag_d;
    CUDACHECK(cudaMalloc(&data_src, data_size));
    float *h_data_src = (float *)malloc(data_size);
    for (int i = 0; i < elem_num; ++i) {
        h_data_src[i] = i % 23;
    }
    CUDACHECK(
        cudaMemcpy(data_src, h_data_src, data_size, cudaMemcpyHostToDevice));
    // mscclppBootStrapAllGather(comm, data_src, data_size);
    CUDACHECK(cudaMalloc(&data_dst, data_size));
    CUDACHECK(cudaMalloc(&recvbuff, 2 * data_size));
    CUDACHECK(cudaMalloc(&flag_d, sizeof(int)));

    mscclppResult_t res;
    int tag = 0;
    int peer = (rank + 1) % world_size;
    if (rank == 0) {
        mscclppConnect(comm, peer, rank, data_src, data_size, flag_d, tag,
                       mscclppTransportP2P);
    } else {
        mscclppConnect(comm, peer, rank, recvbuff, data_size, flag_d, tag,
                       mscclppTransportP2P);
    }
    res = mscclppConnectionSetup(comm);
    if (res != mscclppSuccess) {
        printf("mscclppConnectionSetup failed\n");
        return -1;
    }

    mscclppDevConn_t devConns;
    mscclppGetDevConns(comm, &devConns);
    if (rank == 0) {
        test_send_ll<<<1, 32>>>(devConns[0].localBuff, devConns[0].remoteBuff,
                                devConns[0].localFlag, data_size);
    }
    if (rank == 1) {
        test_recv_ll<<<1, 32>>>(data_dst, devConns[0].localBuff,
                                devConns[0].remoteFlag, data_size);
    }
    CUDACHECK(cudaDeviceSynchronize());
    if (rank == 1) {
        float *h_data_dst = (float *)malloc(data_size);
        CUDACHECK(cudaMemcpy(h_data_dst, data_dst, data_size,
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < elem_num; ++i) {
            if (h_data_dst[i] != 1.0 * (i % 23)) {
                printf("data_dst[%d] = %f, expected %f", i, h_data_dst[i],
                       (i % 23));
                return -1;
            }
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
