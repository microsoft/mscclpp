#include <cassert>
#include <mscclpp/concurrency.hpp>
#include <vector>

#include "common.hpp"

#define ALIGN 4
#define BLOCKS_PER_PEER 1

__constant__ mscclpp::channel::SimpleDeviceChannel constDevFstRoundChans[16];
__constant__ mscclpp::channel::SimpleDeviceChannel constDevSndRoundChans[16];

__constant__ mscclpp::channel::DirectChannel constDirChans[16];

// TODO(chhwang): need an interface for this.
static void* resultBuff = nullptr;
int* inputBuff;
int* scratchBuff;

struct Chunk {
  size_t offset;
  size_t size;
};

__host__ __device__ Chunk getChunk(size_t dataCount, size_t numChunks, size_t chunkIdx) {
  size_t remainder = dataCount % numChunks;
  size_t smallChunkSize = dataCount / numChunks;
  size_t largeChunkSize = smallChunkSize + 1;
  size_t numRemainedLargeChunks = chunkIdx < remainder ? remainder - chunkIdx : 0;
  size_t offset = (remainder - numRemainedLargeChunks) * largeChunkSize +
                  (chunkIdx > remainder ? chunkIdx - remainder : 0) * smallChunkSize;
  return Chunk{offset, chunkIdx < remainder ? largeChunkSize : smallChunkSize};
}

__forceinline__ __device__ void vectorSum(int* dst, int* src, size_t nElem) {
  size_t nInt4 = nElem / 4;
  size_t nLastInts = nElem % 4;
  int4* dst4 = (int4*)dst;
  int4* src4 = (int4*)src;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nInt4; i += blockDim.x * gridDim.x) {
    dst4[i].w += src4[i].w;
    dst4[i].x += src4[i].x;
    dst4[i].y += src4[i].y;
    dst4[i].z += src4[i].z;
  }
  if (nLastInts > 0) {
    int* dstLast = dst + nInt4 * 4;
    int* srcLast = src + nInt4 * 4;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nLastInts; i += blockDim.x * gridDim.x) {
      dstLast[i] += srcLast[i];
    }
  }
}

__device__ void vectorSumSingleBlock(int* dst, int* src, size_t nElem) {
  for (int i = threadIdx.x; i < nElem; i += blockDim.x) {
    dst[i] += src[i];
  }
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

__device__ void allreduce0(int* buff, int* scratch, int rank, int worldSize, size_t nelems, size_t scratchDataCount) {
  int peerId = blockIdx.x / BLOCKS_PER_PEER;
  int isComm = (threadIdx.x == 0) && (blockIdx.x % BLOCKS_PER_PEER == 0);
  int remoteRank = (peerId < rank) ? peerId : peerId + 1;

  // 1st communication phase: send data to the scratch buffer of the peer associated with this block
  mscclpp::channel::SimpleDeviceChannel& devFstRoundChan = constDevFstRoundChans[peerId];
  Chunk toPeerChunk = getChunk(nelems, worldSize, remoteRank);
  // Now we need to figure out the offset of this chunk in the scratch buffer of the destination.
  // The destination will have allocated a scratch buffer of size numPeers() * toPeerChunk.size and
  // inside that each of the destination's peers send to the nth chunk, where n is the index of the
  // source peer from the destination's perspective.
  size_t dstOffset = (rank < remoteRank ? rank : rank - 1) * toPeerChunk.size;
  if (isComm) {
    // Write data to the peer
    devFstRoundChan.putWithSignalAndFlush(dstOffset * sizeof(int), toPeerChunk.offset * sizeof(int),
                                          toPeerChunk.size * sizeof(int));
    // Wait for data from the peer
    devFstRoundChan.wait();
  }

  deviceSyncer.sync(gridDim.x);

  // Local reduction: every block reduces a slice of each chunk in the scratch buffer into the user buffer
  mscclpp::channel::SimpleDeviceChannel& devSndRoundChan = constDevSndRoundChans[peerId];
  Chunk rankChunk = getChunk(nelems, worldSize, rank);
  int* chunk = buff + rankChunk.offset;
  int numPeers = gridDim.x / BLOCKS_PER_PEER;
  int numBlocks = gridDim.x;
  Chunk blockUserChunk = getChunk(rankChunk.size, numBlocks, blockIdx.x);
  size_t scratchDataCountPerPeer = scratchDataCount / numPeers;
  Chunk blockScratchChunk = getChunk(scratchDataCountPerPeer, numBlocks, blockIdx.x);
  for (int peerIdx = 0; peerIdx < numPeers; ++peerIdx) {
    int* scratchChunk = scratch + peerIdx * scratchDataCountPerPeer;
    vectorSumSingleBlock(chunk + blockUserChunk.offset, scratchChunk + blockScratchChunk.offset,
                         blockScratchChunk.size);
  }

  deviceSyncer.sync(gridDim.x);

  // 2nd communication phase: send the now reduced data between the user buffers
  Chunk collectionChunk = getChunk(nelems, worldSize, rank);
  if (isComm) {
    // Write data to the peer
    devSndRoundChan.putWithSignalAndFlush(collectionChunk.offset * sizeof(int), collectionChunk.offset * sizeof(int),
                                          collectionChunk.size * sizeof(int));
    // Wait for data from the peer
    devSndRoundChan.wait();
  }
}

__device__ void allreduce1(int* buff, int* scratch, int rank, int worldSize, size_t nelems, size_t scratchDataCount) {
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int remoteSendRank = (rank + 1) % worldSize;
  int remoteRecvRank = (rank + worldSize - 1) % worldSize;
  int peerSendId = (remoteSendRank < rank) ? remoteSendRank : remoteSendRank - 1;
  int peerRecvId = (remoteRecvRank < rank) ? remoteRecvRank : remoteRecvRank - 1;

  mscclpp::channel::SimpleDeviceChannel& devFstSendChan = constDevFstRoundChans[peerSendId];
  mscclpp::channel::SimpleDeviceChannel& devFstRecvChan = constDevFstRoundChans[peerRecvId];
  mscclpp::channel::SimpleDeviceChannel& devSndSendChan = constDevSndRoundChans[peerSendId];
  mscclpp::channel::SimpleDeviceChannel& devSndRecvChan = constDevSndRoundChans[peerRecvId];

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
    vectorSum(dst, src, chunkNelem / 2);

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
    vectorSum(dst, src, chunkNelem - chunkNelem / 2);
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
  vectorSum(dst, src, chunkNelem / 2);

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
  vectorSum(dst, src, chunkNelem - chunkNelem / 2);

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

__device__ void allreduce2(int* buff, int* scratch, void* result, int rank, int worldSize, size_t nelems) {
  int chanIdx = blockIdx.x / BLOCKS_PER_PEER;
  int numPeers = worldSize - 1;
  size_t nPkts = nelems / 2;  // 2 elems per packet, assume nelems is even
  size_t pktBytes = nPkts * sizeof(mscclpp::channel::ChannelPacket);
  mscclpp::channel::DirectChannel devDirChan = constDirChans[chanIdx];
  uint32_t flag = (uint32_t)devDirChan.epochGetLocal() + 1;  // +1 as flag should be non-zero
  size_t srcOffset =
      ((blockIdx.x % BLOCKS_PER_PEER) * nelems * sizeof(int) / BLOCKS_PER_PEER);  // offset for this block
  size_t dstOffset = ((flag & 1) ? 0 : pktBytes * numPeers) +                     // double buffering
                     ((chanIdx < rank ? rank - 1 : rank) * pktBytes) +            // offset for this rank
                     (srcOffset * 2);  // offset for this block: twice of srcOffset because 2 elems per packet

  devDirChan.putPacket(dstOffset, srcOffset, nelems / BLOCKS_PER_PEER * sizeof(int), threadIdx.x, blockDim.x, flag);

  int2* src = (int2*)buff;
  int2* res = (int2*)result;  // cumulate into here
  mscclpp::channel::ChannelPacket* tmpPtr =
      (mscclpp::channel::ChannelPacket*)scratch + ((flag & 1) ? 0 : nPkts * numPeers);  // double buffering
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * gridDim.x) {
    int x = 0;
    int y = 0;
    for (int peerIdx = 0; peerIdx < numPeers / 2; ++peerIdx) {
      mscclpp::channel::ChannelPacket* pkt0 = tmpPtr + 2 * peerIdx * nPkts;
      mscclpp::channel::ChannelPacket* pkt1 = tmpPtr + (2 * peerIdx + 1) * nPkts;
      uint2 data0 = pkt0[idx].read(flag);
      uint2 data1 = pkt1[idx].read(flag);
      x += (int)data0.x;
      y += (int)data0.y;
      x += (int)data1.x;
      y += (int)data1.y;
    }
    if (numPeers & 1) {
      mscclpp::channel::ChannelPacket* pkt = tmpPtr + (numPeers - 1) * nPkts;
      uint2 data = pkt[idx].read(flag);
      x += (int)data.x;
      y += (int)data.y;
    }
    res[idx].x = src[idx].x + x;
    res[idx].y = src[idx].y + y;
  }

  if (threadIdx.x == 0 && (blockIdx.x % BLOCKS_PER_PEER) == 0) {
    devDirChan.epochIncrement();
  }
}

__global__ void kernel(int* buff, int* scratch, void* result, int rank, int worldSize, size_t nelems,
                       size_t scratchDataCount, int kernel) {
  if (kernel == 0)
    allreduce0(buff, scratch, rank, worldSize, nelems, scratchDataCount);
  else if (kernel == 1)
    allreduce1(buff, scratch, rank, worldSize, nelems, scratchDataCount);
  else if (kernel == 2)
    allreduce2(buff, scratch, result, rank, worldSize, nelems);
}

class AllReduceTestColl : public BaseTestColl {
 public:
  AllReduceTestColl() = default;
  ~AllReduceTestColl() = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
};

void AllReduceTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int kernelNum = args.kernelNum;
  const int nPeers = worldSize - 1;
  const Chunk chunk = getChunk(paramCount_, worldSize, rank);
  const size_t scratchDataCount = chunk.size * nPeers;
  const int nBlocks = (kernelNum == 1) ? 24 : nPeers * BLOCKS_PER_PEER;
  kernel<<<nBlocks, 1024, 0, stream>>>(inputBuff, scratchBuff, resultBuff, rank, worldSize, paramCount_,
                                       scratchDataCount, kernelNum);
}

void AllReduceTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  assert(sendBuff.size() == 1);
  const int rank = args.rank;
  const int worldSize = args.totalRanks;
  std::vector<int> dataHost(std::max(sendCount_, recvCount_), rank);
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = worldSize * (worldSize - 1) / 2;
  }
  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void AllReduceTestColl::getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) {
  double baseBw = (double)(paramCount_ * typeSize_) / 1.0E9 / deltaSec;
  algBw = baseBw;
  double factor = (2 * (double)(worldSize_ - 1)) / ((double)worldSize_);
  busBw = baseBw * factor;
}

void AllReduceTestColl::setupCollTest(size_t size) {
  size_t count = size / typeSize_;
  size_t base = (count / ALIGN) * ALIGN;
  sendCount_ = base;
  recvCount_ = base;
  paramCount_ = base;
  recvCount_ = base;

  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

class AllReduceTestEngine : public BaseTestEngine {
 public:
  AllReduceTestEngine(const TestArgs& args);
  ~AllReduceTestEngine() = default;

  void allocateBuffer() override;
  void setupConnections() override;

  bool isUsePacket() const;
  bool isInPlace() const;

  std::vector<void*> getSendBuff() override;
  void* getRecvBuff() override;
  void* getScratchBuff() override;

 private:
  void* getExpectedBuff() override;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int> scratchBuff_;
  std::shared_ptr<int> resultBuff_;
  std::shared_ptr<int[]> expectedBuff_;
};

AllReduceTestEngine::AllReduceTestEngine(const TestArgs& args) : BaseTestEngine(args) { inPlace_ = isInPlace(); }

bool AllReduceTestEngine::isUsePacket() const { return (args_.kernelNum == 2); }

bool AllReduceTestEngine::isInPlace() const { return (args_.kernelNum != 2); }

void AllReduceTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  // Use 4x buffer size for packet-based allreduce: 2x as packets need twice the data size, and 2x for double-buffering
  size_t scale = (isUsePacket() ? (4 * (args_.totalRanks - 1)) : 1);
  scratchBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int) * scale);
  resultBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);

  inputBuff = sendBuff_.get();
  scratchBuff = scratchBuff_.get();
  // TODO(chhwang): need a new interface for this.
  resultBuff = resultBuff_.get();
}

void AllReduceTestEngine::setupConnections() {
  if (isUsePacket()) {
    std::vector<mscclpp::channel::DirectChannel> dirChannels;

    setupMeshConnections(dirChannels, sendBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes);

    assert(dirChannels.size() < sizeof(constDirChans) / sizeof(mscclpp::channel::DirectChannel));
    CUDATHROW(cudaMemcpyToSymbol(constDirChans, dirChannels.data(),
                                 sizeof(mscclpp::channel::DirectChannel) * dirChannels.size()));
  } else {
    std::vector<mscclpp::channel::SimpleDeviceChannel> fstRoundChannels;
    std::vector<mscclpp::channel::SimpleDeviceChannel> sndRoundChannels;

    // Send data from local sendBuff to remote scratchBuff (out-of-place)
    setupMeshConnections(fstRoundChannels, sendBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes);
    assert(fstRoundChannels.size() < sizeof(constDevFstRoundChans) / sizeof(mscclpp::channel::SimpleDeviceChannel));
    CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, fstRoundChannels.data(),
                                 sizeof(mscclpp::channel::SimpleDeviceChannel) * fstRoundChannels.size()));

    // Send data from local sendBuff to remote sendBuff (in-place)
    setupMeshConnections(sndRoundChannels, sendBuff_.get(), args_.maxBytes);
    assert(sndRoundChannels.size() < sizeof(constDevSndRoundChans) / sizeof(mscclpp::channel::SimpleDeviceChannel));
    CUDATHROW(cudaMemcpyToSymbol(constDevSndRoundChans, sndRoundChannels.data(),
                                 sizeof(mscclpp::channel::SimpleDeviceChannel) * sndRoundChannels.size()));
  }
}

std::vector<void*> AllReduceTestEngine::getSendBuff() { return {sendBuff_.get()}; }

void* AllReduceTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* AllReduceTestEngine::getRecvBuff() { return isInPlace() ? sendBuff_.get() : resultBuff_.get(); }

void* AllReduceTestEngine::getScratchBuff() { return scratchBuff_.get(); }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllReduceTestEngine>(args);
}

std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllReduceTestColl>(); }
