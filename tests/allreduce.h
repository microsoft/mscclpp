#include "prims_ll.h"

__device__ void run_allreduce()
{
    const int tid = threadIdx.x;
    const int nthreads = args->nWarps * WARP_SIZE;
    int ringIx = ring->index;
    const ssize_t chunkSize =
        int(Proto::calcBytePerStep() / sizeof(T) *
            (Proto::Id == NCCL_PROTO_SIMPLE ? ALLREDUCE_CHUNKSTEPS : 1));
    const int nranks = ncclShmem.comm.nRanks;
    const ssize_t loopSize = nChannels * nranks * chunkSize;
    const ssize_t size = args->count;

    int minChunkSize;
    // if (Proto::Id == NCCL_PROTO_LL)
    minChunkSize = nthreads * (Proto::calcBytePerGrain() / sizeof(T));
    // if (Proto::Id == NCCL_PROTO_LL128) {
    //     // We should not need the final /2 but it makes performance much,
    //     much
    //     // smoother. Might be a bug somewhere.
    //     minChunkSize = nthreads * (Proto::calcBytePerGrain() / sizeof(T)) /
    //     2;
    // }

    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, &ring->prev, &ring->next, args->sendbuff, args->recvbuff,
        args->redOpArg);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        ssize_t realChunkSize;
        realChunkSize =
            min(chunkSize,
                divUp(size - gridOffset, nChannels * nranks * minChunkSize) *
                    minChunkSize);
        realChunkSize = int(realChunkSize);

        auto calcOffset = [&] __device__(int chunk) -> ssize_t {
            return gridOffset + (chunk * nChannels + bid) * realChunkSize;
        };
        auto modRanks = [&] __device__(int r) -> int {
            return r - (r >= nranks ? nranks : 0);
        };

        ssize_t offset;
        int nelem;
        int chunk;

        // step 0: push data to next GPU
        chunk = modRanks(ringIx + nranks - 1);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size - offset);
        prims.send(offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j = 2; j < nranks; ++j) {
            chunk = modRanks(ringIx + nranks - j);
            offset = calcOffset(chunk);
            nelem = min(realChunkSize, size - offset);
            prims.recvReduceSend(offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data and push to the next GPU
        chunk = ringIx + 0;
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size - offset);
        prims.directRecvReduceCopySend(offset, offset, offset, nelem,
                                       /*postOp=*/true);

        // k-2 steps: copy to next GPU
        for (int j = 1; j < nranks - 1; ++j) {
            chunk = modRanks(ringIx + nranks - j);
            offset = calcOffset(chunk);
            nelem = min(realChunkSize, size - offset);
            prims.directRecvCopySend(offset, offset, nelem);
        }

        // Make final copy from buffer to dest.
        chunk = modRanks(ringIx + 1);
        offset = calcOffset(chunk);
        nelem = min(realChunkSize, size - offset);
        prims.directRecv(offset, nelem);
    }
}