#include "mpi.h"
#include "mscclpp.h"
#include "prims_ll.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MSCCLPPCHECK(call)                                                     \
    do {                                                                       \
        mscclppResult_t res = call;                                            \
        if (res != mscclppSuccess && res != mscclppInProgress) {               \
            /* Print the back trace*/                                          \
            printf("Failure at %s:%d -> %d", __FILE__, __LINE__, res);         \
            return res;                                                        \
        }                                                                      \
    } while (0);

// Check CUDA RT calls
#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t err = cmd;                                                 \
        if (err != cudaSuccess) {                                              \
            printf("Cuda failure '%s'", cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (false)
#define STEPLINES 4096
__global__ void ring_all_reduce(mscclppDevConn_t devConns, int rank, int nranks,
                                void *data_src, void *data_dst, void *recvBuff,
                                int elem_num)
{
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    Primitives_LL<float> prims(tid, nthreads, 0, 0, STEPLINES);
    // devConns[0] is the connection to the next GPU and devConns[1] is the
    // connection to the previous GPU
    prims.data_src = (float *)data_src;
    prims.data_dst = (float *)data_dst;
    prims.sendBuff = (ncclLLFifoLine *)devConns[0].remoteBuff;
    prims.recvBuff = (ncclLLFifoLine *)recvBuff;
    prims.sendConnHeadPtr = (volatile uint64_t *)devConns[0].localFlag;
    prims.recvConnHeadPtr = (volatile uint64_t *)devConns[1].remoteFlag;
    int ChunkSize = elem_num / nranks;

    ssize_t offset;
    int nelem = ChunkSize;
    int chunk;

    // step 0: push data to next GPU
    chunk = (rank + nranks - 1) % nranks;
    offset = chunk * ChunkSize;
    // nelem = min(ChunkSize, size - offset);
    prims.send(offset, nelem);
    // k-2 steps: reduce and copy to next GPU
    for (int j = 2; j < nranks; ++j) {
        chunk = (rank + nranks - j) % nranks;
        offset = chunk * ChunkSize;
        // nelem = min(ChunkSize, size - offset);
        prims.recvReduceSend(offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = rank + 0;
    offset = chunk * ChunkSize;
    // nelem = min(ChunkSize, size - offset);
    prims.recvReduceCopySend(offset, offset, nelem,
                         /*postOp=*/true);
    // prims.recv(offset, nelem,
    //                       /*postOp=*/true);
    // return;
    // k-2 steps: copy to next GPU
    for (int j = 1; j < nranks - 1; ++j) {
        chunk = (rank + nranks - j) % nranks;
        offset = chunk * ChunkSize;
        // nelem = min(ChunkSize, size - offset);
        prims.recvCopySend(offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = (rank + 1) % nranks;
    offset = chunk * ChunkSize;
    // nelem = min(ChunkSize, size - offset);
    prims.recv(offset, nelem);
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
    MSCCLPPCHECK(mscclppCommInitRank(&comm, world_size, rank, ip_port));
    int device_id;
    cudaSetDevice(rank);
    cudaGetDevice(&device_id);

    // printf("Current CUDA device ID: %d\n", device_id);

    float *data_src;
    float *data_dst;
    char *recvbuff;
    int elem_num = 1024;

    int data_size = sizeof(float) * elem_num;

    int *sendConnhead;
    // dummy_flag is not used in LL protocol, just for compatibility
    int *dummy_flag;
    CUDACHECK(cudaMalloc(&data_src, data_size));
    float *h_data_src = (float *)malloc(data_size);
    for (int i = 0; i < elem_num; ++i) {
        h_data_src[i] = i % 23;
    }
    CUDACHECK(
        cudaMemcpy(data_src, h_data_src, data_size, cudaMemcpyHostToDevice));
    // mscclppBootStrapAllGather(comm, data_src, data_size);
    CUDACHECK(cudaMalloc(&data_dst, data_size));
    CUDACHECK(cudaMalloc(&recvbuff, NCCL_STEPS * STEPLINES));
    CUDACHECK(cudaMalloc(&sendConnhead, sizeof(int)));
    CUDACHECK(cudaMalloc(&dummy_flag, sizeof(int)));
    mscclppResult_t res;
    int tag = 0;
    int rank_next = (rank + 1) % world_size;
    int rank_prev = (rank + world_size - 1) % world_size;
    // printf("rank: %d, rank_next: %d, rank_prev: %d\n", rank, rank_next,
    //        rank_prev);
    // in the ring all reduce, we need to connect to the next and previous GPU
    MSCCLPPCHECK(mscclppConnect(comm, rank_next, rank, recvbuff, data_size,
                                sendConnhead, tag, mscclppTransportP2P));
    MSCCLPPCHECK(mscclppConnect(comm, rank, rank_prev, recvbuff, data_size,
                                sendConnhead, tag, mscclppTransportP2P));

    MSCCLPPCHECK(mscclppConnectionSetup(comm));

    mscclppDevConn_t devConns;
    MSCCLPPCHECK(mscclppGetDevConns(comm, &devConns));
    cudaPointerAttributes attributes;
    CUDACHECK(cudaPointerGetAttributes(&attributes, devConns[1].localBuff));
    // printf("devConns[0].remoteBuff: %p located at %d\n", devConns[1].remoteBuff,
    //        attributes.device);
    // printf("data_src: %p, data_dst: %p, recvbuff %p sendConnhead %p\n",
    //        data_src, data_dst, recvbuff, sendConnhead);
    ring_all_reduce<<<1, 32>>>(devConns, rank, world_size, data_src, data_dst,
                               recvbuff, elem_num);
    CUDACHECK(cudaDeviceSynchronize());
    float *h_data_dst = (float *)malloc(data_size);
    CUDACHECK(
        cudaMemcpy(h_data_dst, data_dst, data_size, cudaMemcpyDeviceToHost));
    for (int i = 0; i < elem_num; ++i) {
        if (h_data_dst[i] != world_size * (i % 23)) {
            printf("data_dst[%d] = %f, expected %f", i, h_data_dst[i],
                   1.0 * world_size * (i % 23));
            return -1;
        }
    }
    MSCCLPPCHECK(mscclppCommDestroy(comm));
    MPI_Finalize();

    printf("Succeeded! %d\n", rank);
    return 0;
}
