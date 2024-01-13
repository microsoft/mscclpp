// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "common.cuh"

__device__ mscclpp::DeviceSyncer deviceSyncer;

// -------------------------------------------
// AllReduce3
// -------------------------------------------

extern "C" __global__ void __launch_bounds__(1024, 1)
    allreduce3(mscclpp::SimpleProxyChannelDeviceHandle* fstRoundChans,
               mscclpp::SimpleProxyChannelDeviceHandle* sndRoundChans, TYPE* buff, TYPE* scratch, int rank,
               int worldSize, size_t nelems) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));

  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int remoteSendRank = (rank + 1) % worldSize;
  int remoteRecvRank = (rank + worldSize - 1) % worldSize;
  int peerSendId = (remoteSendRank < rank) ? remoteSendRank : remoteSendRank - 1;
  int peerRecvId = (remoteRecvRank < rank) ? remoteRecvRank : remoteRecvRank - 1;

  mscclpp::SimpleProxyChannelDeviceHandle& devFstSendChan = fstRoundChans[peerSendId];
  mscclpp::SimpleProxyChannelDeviceHandle& devFstRecvChan = fstRoundChans[peerRecvId];
  mscclpp::SimpleProxyChannelDeviceHandle& devSndSendChan = sndRoundChans[peerSendId];
  mscclpp::SimpleProxyChannelDeviceHandle& devSndRecvChan = sndRoundChans[peerRecvId];

  // Step 1
  size_t chunkIndex = (rank + worldSize - 1) % worldSize;
  size_t chunkNelem = nelems / worldSize;
  size_t offset = chunkIndex * chunkNelem * sizeof(int);
  if (isComm) {
    if (chunkNelem > 1) {
      devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
    }
  }

  // Step 2 ~ Step n-1
  for (int step = 2; step < worldSize; ++step) {
    if (isComm) {
      if (chunkNelem > 1) {
        devFstRecvChan.wait();
        devFstSendChan.flush();
      }
      devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(int), (chunkNelem - chunkNelem / 2) * sizeof(int));
    }
    deviceSyncer.sync(gridDim.x);

    // Reduce
    chunkIndex = (rank + worldSize - step) % worldSize;
    offset = chunkIndex * chunkNelem * sizeof(int);
    int* dst = (int*)((char*)buff + offset);
    int* src = (int*)((char*)scratch + offset);
    vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem / 2);

    if (isComm) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
      if (chunkNelem > 1) {
        devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
      }
    }
    deviceSyncer.sync(gridDim.x);

    dst += chunkNelem / 2;
    src += chunkNelem / 2;
    vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem - chunkNelem / 2);
  }

  // Step n
  if (isComm) {
    if (chunkNelem > 1) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
    }
    devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(int), (chunkNelem - chunkNelem / 2) * sizeof(int));
  }
  deviceSyncer.sync(gridDim.x);

  offset = rank * chunkNelem * sizeof(int);
  int* dst = (int*)((char*)buff + offset);
  int* src = (int*)((char*)scratch + offset);
  vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem / 2);

  if (isComm) {
    devFstRecvChan.wait();
    devFstSendChan.flush();
    if (chunkNelem > 1) {
      devSndSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
    }
  }
  deviceSyncer.sync(gridDim.x);

  dst += chunkNelem / 2;
  src += chunkNelem / 2;
  vectorSum((TYPE*)dst, (TYPE*)src, chunkNelem - chunkNelem / 2);

  if (isComm) {
    if (chunkNelem > 1) {
      devSndRecvChan.wait();
      devSndSendChan.flush();
    }
    devSndSendChan.putWithSignalAndFlush(offset + chunkNelem / 2 * sizeof(int),
                                         (chunkNelem - chunkNelem / 2) * sizeof(int));
  }

  // Step n+1 ~ Step 2n-2
  for (int i = 1; i < worldSize - 1; ++i) {
    if (isComm) {
      devSndRecvChan.wait();
    }
    deviceSyncer.sync(gridDim.x);

    // Copy
    chunkIndex = (rank + worldSize - i) % worldSize;
    if (isComm) {
      devSndSendChan.putWithSignalAndFlush(chunkIndex * chunkNelem * sizeof(int), chunkNelem * sizeof(int));
    }
  }

  // Final receive
  if (isComm) {
    devSndRecvChan.wait();
  }
}
