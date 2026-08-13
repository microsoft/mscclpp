// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Bulk-copy usage patterns, shaped after the expert-parallel dispatch and combine kernels.
//
// These are deliberately simplified: fixed token counts, one contribution per rank pair, and plain
// arithmetic instead of quantization or top-k routing. What they preserve is the communication
// shape, which is what the API has to support:
//
//   Dispatch      local buffer -> shared (transform) -> peer buffer.  Push, staged through shared
//                 memory, pipelined so a tile is refilled while its store is still in flight.
//   Combine       every peer's buffer -> shared -> reduce -> local output.  Pull, many sources into
//                 one barrier, double-buffered across chunks.
//   CombineReduce the same reduction expressed as a push, with the copy engine accumulating
//                 directly into the peer's memory, so nothing is ever read back.
//
// Note where the channels are and are not used. Bulk data moves through raw peer pointers, exactly
// as the expert-parallel kernels do; the channels carry only signal/wait. That split is the reason
// the bulk primitives are pointer-based rather than channel methods.

#include <algorithm>
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <vector>

#include "mp_unit_tests.hpp"

void BulkPatternTest::SetUp() {
  if (gEnv->nRanksPerNode < 2) {
    SKIP_TEST();
  }
  setNumRanksToUse(gEnv->nRanksPerNode);
  CommunicatorTestBase::SetUp();
  rank = gEnv->rank;
  worldSize = numRanksToUse;
}

void BulkPatternTest::TearDown() {
  syncHandles.clear();
  syncChannels.clear();
  peerPools.clear();
  remotePoolMemories.clear();
  CommunicatorTestBase::TearDown();
}

void BulkPatternTest::setupPeerPools(void* pool, size_t poolBytes) {
  const mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc;
  mscclpp::RegisteredMemory localMem = communicator->registerMemory(pool, poolBytes, transport);

  std::vector<std::shared_future<mscclpp::Connection>> connFutures(worldSize);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> memFutures(worldSize);
  for (int r = 0; r < worldSize; ++r) {
    if (r == rank) continue;
    connFutures[r] = communicator->connect(mscclpp::Transport::CudaIpc, r);
    communicator->sendMemory(localMem, r);
    memFutures[r] = communicator->recvMemory(r);
  }

  peerPools.assign(worldSize, nullptr);
  peerPools[rank] = pool;
  for (int r = 0; r < worldSize; ++r) {
    if (r == rank) continue;
    mscclpp::RegisteredMemory remote = memFutures[r].get();
    peerPools[r] = remote.data();
    remotePoolMemories.push_back(remote);
    syncChannels.emplace_back(communicator->buildSemaphore(connFutures[r].get(), r).get());
  }
  for (const auto& chan : syncChannels) syncHandles.push_back(mscclpp::deviceHandle(chan));
  registeredMemories.push_back(localMem);
}

#if defined(MSCCLPP_DEVICE_CUDA)

namespace {

constexpr int kNumTokens = 16;
constexpr int kHidden = 1024;                              // floats per token
constexpr uint32_t kTokenBytes = kHidden * sizeof(float);  // 4 KB
constexpr uint32_t kChunkBytes = 1024;                     // staging granularity
constexpr int kChunksPerToken = kTokenBytes / kChunkBytes;
constexpr int kChunkFloats = kChunkBytes / sizeof(float);
constexpr int kStages = 2;

// Distinct, exactly representable value for (source rank, destination rank, token, element).
MSCCLPP_HOST_DEVICE_INLINE float payload(int src, int dst, int token, int elem) {
  return (float)(src * 1000 + dst * 100 + token * 10 + (elem % 10));
}

// Offset in floats of the [src][token] row within a receive pool.
MSCCLPP_HOST_DEVICE_INLINE int64_t rowOffset(int src, int token) {
  return ((int64_t)src * kNumTokens + token) * kHidden;
}

}  // namespace

// Signal every peer, then wait on every peer.
//
// Launched as its own kernel rather than folded into the pattern kernels: the ranks have to agree on
// a point that every block has passed, and kernel completion already provides exactly that. Doing it
// inside a kernel would need a cooperative grid launch.
__global__ void kernelPeerBarrier(mscclpp::BaseMemoryChannelDeviceHandle* chans, int nPeers) {
  if ((int)threadIdx.x < nPeers) {
    chans[threadIdx.x].signal();
    chans[threadIdx.x].wait();
  }
}

namespace {

// Peer pool pointers and synchronization handles, staged in device memory for the kernels.
struct DeviceState {
  std::shared_ptr<DeviceHandle<mscclpp::BaseMemoryChannel>> chans;
  std::shared_ptr<void*> pools;
  int nPeers;
};

DeviceState uploadPeerState(const std::vector<DeviceHandle<mscclpp::BaseMemoryChannel>>& handles,
                            const std::vector<void*>& pools) {
  DeviceState state;
  state.nPeers = (int)handles.size();
  state.chans = mscclpp::detail::gpuCallocShared<DeviceHandle<mscclpp::BaseMemoryChannel>>(handles.size());
  MSCCLPP_CUDATHROW(cudaMemcpy(state.chans.get(), handles.data(),
                               handles.size() * sizeof(DeviceHandle<mscclpp::BaseMemoryChannel>),
                               cudaMemcpyHostToDevice));
  state.pools = mscclpp::detail::gpuCallocShared<void*>(pools.size());
  MSCCLPP_CUDATHROW(cudaMemcpy(state.pools.get(), pools.data(), pools.size() * sizeof(void*), cudaMemcpyHostToDevice));
  return state;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Dispatch: push each local token to every peer, staged and transformed in shared memory.
//
// Per chunk the leader loads into a tile, the block transforms it in place, and the leader pushes it
// out. bulkStoreWaitSource() lets the next load start while the store is still travelling, which is
// the reason that entry point exists separately from bulkStoreWait().
// ------------------------------------------------------------------------------------------------
__global__ void kernelDispatch([[maybe_unused]] const float* localTokens, [[maybe_unused]] void** peerPools,
                               [[maybe_unused]] int rank, [[maybe_unused]] int worldSize) {
#if MSCCLPP_BULK_AVAILABLE
  __shared__ alignas(128) uint8_t tile[kChunkBytes];
  __shared__ mscclpp::BulkBarrier barrier;

  if (threadIdx.x == 0) barrier.init();
  __syncthreads();

  uint32_t phase = 0;  // initialized once; wait() advances it on every chunk
  bool storePending = false;

  // One block per (destination, token) pair, strided over the grid.
  const int totalRows = worldSize * kNumTokens;
  for (int row = blockIdx.x; row < totalRows; row += gridDim.x) {
    const int dst = row / kNumTokens;
    const int token = row % kNumTokens;
    const float* srcRow = localTokens + (int64_t)token * kHidden;
    auto* dstRow = reinterpret_cast<uint8_t*>(peerPools[dst]) + rowOffset(rank, token) * sizeof(float);

    for (int c = 0; c < kChunksPerToken; ++c) {
      const uint32_t off = c * kChunkBytes;
      if (threadIdx.x == 0) {
        // The previous chunk's store has released the tile by now; make sure before refilling.
        if (storePending) mscclpp::bulkStoreWaitSource<0>();
        barrier.arriveAndExpect(kChunkBytes);
        mscclpp::bulkLoad(tile, reinterpret_cast<const uint8_t*>(srcRow) + off, kChunkBytes, barrier);
        barrier.wait(phase);
      }
      __syncthreads();
      mscclpp::bulkFence();  // loaded data -> generic reads

      // Stand-in for the quantize/scale step a real dispatch performs while the token is staged.
      float* staged = reinterpret_cast<float*>(tile);
      for (int i = threadIdx.x; i < kChunkFloats; i += blockDim.x) staged[i] += (float)(dst * 100);

      __syncthreads();
      mscclpp::bulkFence();  // generic writes -> the store's read of the tile
      if (threadIdx.x == 0) {
        mscclpp::bulkStore(dstRow + off, tile, kChunkBytes);
        mscclpp::bulkStoreCommit();
        storePending = true;
      }
      __syncthreads();
    }
  }

  // Every push must have landed before any peer is told the data is ready.
  if (threadIdx.x == 0) {
    mscclpp::bulkStoreWait<0>();
    barrier.invalidate();
  }
#endif
}

TEST(BulkPatternTest, Dispatch) {
  if (gEnv->rank >= numRanksToUse) return;
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }

  const size_t poolFloats = (size_t)worldSize * kNumTokens * kHidden;
  std::shared_ptr<float> pool = mscclpp::GpuBuffer<float>(poolFloats).memory();
  std::shared_ptr<float> tokens = mscclpp::GpuBuffer<float>((size_t)kNumTokens * kHidden).memory();
  MSCCLPP_CUDATHROW(cudaMemset(pool.get(), 0, poolFloats * sizeof(float)));

  // Local tokens carry payload(rank, 0, token, elem); the kernel adds dst * 100 while staged, so the
  // value that lands on rank dst is payload(rank, dst, token, elem).
  std::vector<float> host((size_t)kNumTokens * kHidden);
  for (int t = 0; t < kNumTokens; ++t)
    for (int h = 0; h < kHidden; ++h) host[(size_t)t * kHidden + h] = payload(gEnv->rank, 0, t, h);
  MSCCLPP_CUDATHROW(cudaMemcpy(tokens.get(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));

  setupPeerPools(pool.get(), poolFloats * sizeof(float));

  DeviceState dev = uploadPeerState(syncHandles, peerPools);

  kernelDispatch<<<worldSize * kNumTokens, 128>>>(tokens.get(), dev.pools.get(), rank, worldSize);
  // Kernel completion is the grid-wide point after which every push has landed; only then is it safe
  // to tell the peers, and to read what they pushed here.
  kernelPeerBarrier<<<1, 32>>>(dev.chans.get(), dev.nPeers);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  std::vector<float> out(poolFloats);
  MSCCLPP_CUDATHROW(cudaMemcpy(out.data(), pool.get(), poolFloats * sizeof(float), cudaMemcpyDeviceToHost));
  for (int src = 0; src < worldSize; ++src) {
    for (int t = 0; t < kNumTokens; ++t) {
      for (int h = 0; h < kHidden; h += 97) {  // sparse check keeps the assertion count sane
        EXPECT_EQ(out[rowOffset(src, t) + h], payload(src, rank, t, h));
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Combine: pull this rank's row from every peer into shared memory and reduce it.
//
// This is the pattern the barrier design exists for. All contributors of a chunk are issued against
// one barrier and waited on once, and two stages let the next chunk's loads be issued before the
// current one is reduced. The two barriers are set up with relaxedInit() under a single bulkFence()
// rather than paying a fence each.
// ------------------------------------------------------------------------------------------------
__global__ void kernelCombine([[maybe_unused]] float* output, [[maybe_unused]] void** peerPools,
                              [[maybe_unused]] int rank, [[maybe_unused]] int worldSize) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t shared[];
  // Layout: [stage][contributor][chunk] tiles, then the per-stage barriers.
  uint8_t* tiles = shared;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(shared + (size_t)kStages * worldSize * kChunkBytes);
  auto tile = [&](int stage, int contributor) {
    return tiles + ((size_t)stage * worldSize + contributor) * kChunkBytes;
  };

  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) barriers[s].relaxedInit();
    mscclpp::bulkFence();  // one fence publishes both barriers to the async proxy
  }
  __syncthreads();

  uint32_t phase[kStages] = {0, 0};

  auto issue = [&](int stage, int token, int chunk) {
    barriers[stage].arriveAndExpect(kChunkBytes * worldSize);  // whole batch, before any arrival lands
    for (int src = 0; src < worldSize; ++src) {
      const auto* row = reinterpret_cast<const uint8_t*>(peerPools[src]) + rowOffset(rank, token) * sizeof(float);
      mscclpp::bulkLoad(tile(stage, src), row + (size_t)chunk * kChunkBytes, kChunkBytes, barriers[stage]);
    }
  };

  for (int token = blockIdx.x; token < kNumTokens; token += gridDim.x) {
    if (threadIdx.x == 0) issue(0, token, 0);
    for (int c = 0; c < kChunksPerToken; ++c) {
      const int stage = c % kStages;
      if (threadIdx.x == 0) {
        if (c + 1 < kChunksPerToken) issue((c + 1) % kStages, token, c + 1);  // prefetch next chunk
        barriers[stage].wait(phase[stage]);
      }
      __syncthreads();
      mscclpp::bulkFence();

      float* out = output + (int64_t)token * kHidden + (int64_t)c * kChunkFloats;
      for (int i = threadIdx.x; i < kChunkFloats; i += blockDim.x) {
        float acc = 0.0f;
        for (int src = 0; src < worldSize; ++src) acc += reinterpret_cast<const float*>(tile(stage, src))[i];
        out[i] = acc;
      }
      __syncthreads();  // this stage's tiles are consumed before they are reissued
    }
  }

  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) barriers[s].invalidate();
  }
#endif
}

TEST(BulkPatternTest, Combine) {
  if (gEnv->rank >= numRanksToUse) return;
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }

  const size_t poolFloats = (size_t)worldSize * kNumTokens * kHidden;
  std::shared_ptr<float> pool = mscclpp::GpuBuffer<float>(poolFloats).memory();
  std::shared_ptr<float> output = mscclpp::GpuBuffer<float>((size_t)kNumTokens * kHidden).memory();

  // Each rank publishes, for every destination, the contribution that destination will pull.
  std::vector<float> host(poolFloats);
  for (int dst = 0; dst < worldSize; ++dst)
    for (int t = 0; t < kNumTokens; ++t)
      for (int h = 0; h < kHidden; ++h) host[rowOffset(dst, t) + h] = payload(gEnv->rank, dst, t, h);
  MSCCLPP_CUDATHROW(cudaMemcpy(pool.get(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
  MSCCLPP_CUDATHROW(cudaMemset(output.get(), 0, (size_t)kNumTokens * kHidden * sizeof(float)));

  setupPeerPools(pool.get(), poolFloats * sizeof(float));

  DeviceState dev = uploadPeerState(syncHandles, peerPools);

  const size_t sharedBytes = (size_t)kStages * worldSize * kChunkBytes + kStages * sizeof(mscclpp::BulkBarrier);
  kernelPeerBarrier<<<1, 32>>>(dev.chans.get(), dev.nPeers);  // every pool is filled before any pull
  kernelCombine<<<kNumTokens, 128, sharedBytes>>>(output.get(), dev.pools.get(), rank, worldSize);
  kernelPeerBarrier<<<1, 32>>>(dev.chans.get(), dev.nPeers);  // no rank tears down while peers read
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  std::vector<float> out((size_t)kNumTokens * kHidden);
  MSCCLPP_CUDATHROW(cudaMemcpy(out.data(), output.get(), out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  for (int t = 0; t < kNumTokens; ++t) {
    for (int h = 0; h < kHidden; h += 97) {
      float expected = 0.0f;
      for (int src = 0; src < worldSize; ++src) expected += payload(src, rank, t, h);
      EXPECT_EQ(out[(size_t)t * kHidden + h], expected);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// CombineReduce: the same reduction, expressed as a push.
//
// Instead of every rank pulling and summing, each rank accumulates its own contribution straight
// into the destination's accumulator with bulkReduceStore(). The copy engine performs the add at the
// destination, so the accumulator is never read back across the link and no rank stages anyone
// else's data. Cross-device atomicity is guaranteed only by PTX ISA 9.3's system-scope semantics;
// with earlier supported toolkits this pattern exercises empirically observed behavior rather than a
// portable guarantee. The reverse link direction stays idle.
// ------------------------------------------------------------------------------------------------
__global__ void kernelCombineReduce([[maybe_unused]] const float* contribution, [[maybe_unused]] void** peerPools,
                                    [[maybe_unused]] int rank, [[maybe_unused]] int worldSize) {
#if MSCCLPP_BULK_AVAILABLE
  __shared__ alignas(128) uint8_t tile[kChunkBytes];
  __shared__ mscclpp::BulkBarrier barrier;

  if (threadIdx.x == 0) barrier.init();
  __syncthreads();
  uint32_t phase = 0;

  const int totalRows = worldSize * kNumTokens;
  for (int row = blockIdx.x; row < totalRows; row += gridDim.x) {
    const int dst = row / kNumTokens;
    const int token = row % kNumTokens;
    const auto* srcRow = reinterpret_cast<const uint8_t*>(contribution + rowOffset(dst, token));
    // Every rank accumulates into the same [rank-independent] row of the destination.
    auto* dstRow = reinterpret_cast<uint8_t*>(peerPools[dst]) + rowOffset(0, token) * sizeof(float);

    for (int c = 0; c < kChunksPerToken; ++c) {
      const uint32_t off = c * kChunkBytes;
      if (threadIdx.x == 0) {
        barrier.arriveAndExpect(kChunkBytes);
        mscclpp::bulkLoad(tile, srcRow + off, kChunkBytes, barrier);
        barrier.wait(phase);
        mscclpp::bulkFence();  // staged data -> the reduction's read of the tile
        mscclpp::bulkReduceStore<float>(dstRow + off, tile, kChunkBytes);
        mscclpp::bulkStoreCommit();
        mscclpp::bulkStoreWaitSource<0>();  // the tile can be refilled once its source read completes
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    mscclpp::bulkStoreWait<0>();  // every accumulate must land before the kernel completes
    barrier.invalidate();
  }
#endif
}

TEST(BulkPatternTest, CombineReduce) {
  if (gEnv->rank >= numRanksToUse) return;
  if (!mscclpp::isBulkSupported()) {
    SKIP_TEST() << "Bulk asynchronous copy requires compute capability 9.0 or higher.";
    return;
  }

  const size_t poolFloats = (size_t)worldSize * kNumTokens * kHidden;
  std::shared_ptr<float> pool = mscclpp::GpuBuffer<float>(poolFloats).memory();
  std::shared_ptr<float> contribution = mscclpp::GpuBuffer<float>(poolFloats).memory();
  MSCCLPP_CUDATHROW(cudaMemset(pool.get(), 0, poolFloats * sizeof(float)));  // accumulator starts at zero

  std::vector<float> host(poolFloats);
  for (int dst = 0; dst < worldSize; ++dst)
    for (int t = 0; t < kNumTokens; ++t)
      for (int h = 0; h < kHidden; ++h) host[rowOffset(dst, t) + h] = payload(gEnv->rank, dst, t, h);
  MSCCLPP_CUDATHROW(cudaMemcpy(contribution.get(), host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));

  setupPeerPools(pool.get(), poolFloats * sizeof(float));

  DeviceState dev = uploadPeerState(syncHandles, peerPools);

  kernelPeerBarrier<<<1, 32>>>(dev.chans.get(), dev.nPeers);  // every accumulator is zeroed first
  kernelCombineReduce<<<worldSize * kNumTokens, 128>>>(contribution.get(), dev.pools.get(), rank, worldSize);
  kernelPeerBarrier<<<1, 32>>>(dev.chans.get(), dev.nPeers);  // every contribution has landed
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  std::vector<float> out(poolFloats);
  MSCCLPP_CUDATHROW(cudaMemcpy(out.data(), pool.get(), poolFloats * sizeof(float), cudaMemcpyDeviceToHost));
  for (int t = 0; t < kNumTokens; ++t) {
    for (int h = 0; h < kHidden; h += 97) {
      float expected = 0.0f;
      for (int src = 0; src < worldSize; ++src) expected += payload(src, rank, t, h);
      EXPECT_EQ(out[rowOffset(0, t) + h], expected);
    }
  }
}

#endif  // defined(MSCCLPP_DEVICE_CUDA)
