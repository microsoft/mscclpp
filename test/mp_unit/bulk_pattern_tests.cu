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

#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <numeric>
#include <vector>

#include "mp_unit_tests.hpp"

#if defined(MSCCLPP_DEVICE_CUDA)
#include <cooperative_groups.h>
#endif

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

class BulkEpMeshPerfTest : public BulkPatternTest {
 protected:
  void SetUp() override {
    setNumRanksToUse(gEnv->worldSize);
    CommunicatorTestBase::SetUp();
    rank = gEnv->rank;
    worldSize = numRanksToUse;
  }
};

namespace {
constexpr int EpRowBytes = 14336;
constexpr int EpRowsPerPeer = 64;
constexpr int EpRowWorkers = 8;
constexpr int EpRowWords = EpRowBytes / sizeof(uint32_t);

__global__ void kernelEpTmaMesh(void** pools, mscclpp::BaseMemoryChannelDeviceHandle* channels, int rank, int ranks,
                                bool read, int iterations, uint64_t* durations) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t tiles[];
  __shared__ mscclpp::BulkBarrier barriers[EpRowWorkers];
  __shared__ uint64_t start;
  const int warp = threadIdx.x / 32;
  const bool owner = threadIdx.x % 32 == 0;
  const size_t halfBytes = static_cast<size_t>(ranks) * EpRowsPerPeer * EpRowBytes;
  auto grid = cooperative_groups::this_grid();
  if (owner) barriers[warp].init();
  grid.sync();
  uint32_t phase = 0;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (blockIdx.x == 0 && threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    grid.sync();
    if (owner) {
      for (int row = blockIdx.x * EpRowWorkers + warp; row < (ranks - 1) * EpRowsPerPeer;
           row += gridDim.x * EpRowWorkers) {
        const int peerIndex = row / EpRowsPerPeer;
        const int peer = peerIndex < rank ? peerIndex : peerIndex + 1;
        const int token = row % EpRowsPerPeer;
        const size_t localOffset = static_cast<size_t>(peer * EpRowsPerPeer + token) * EpRowBytes;
        const size_t remoteOffset = static_cast<size_t>(rank * EpRowsPerPeer + token) * EpRowBytes;
        const auto* source =
            static_cast<const uint8_t*>(pools[read ? peer : rank]) + (read ? remoteOffset : localOffset);
        auto* destination =
            static_cast<uint8_t*>(pools[read ? rank : peer]) + halfBytes + (read ? localOffset : remoteOffset);
        auto* tile = tiles + warp * EpRowBytes;
        barriers[warp].arriveAndExpect(EpRowBytes);
        mscclpp::bulkLoad(tile, source, EpRowBytes, barriers[warp]);
        barriers[warp].wait(phase);
        mscclpp::bulkFence();
        // Read mode materializes every received word in local HBM, not an unused shared tile.
        mscclpp::bulkStore(destination, tile, EpRowBytes);
        mscclpp::bulkStoreCommit();
        mscclpp::bulkStoreWait<0>();
      }
    }
    // Full destination completion precedes publication; no rank overwrites another rank's rows.
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x < ranks - 1) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait(-1);
    }
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      uint64_t end;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
      durations[iteration] = end - start;
    }
  }
  if (owner) barriers[warp].invalidate();
#endif
}
}  // namespace

PERF_TEST(BulkEpMeshPerfTest, EpTmaRowMeshBandwidth) {
  int supported = mscclpp::isBulkSupported(), allSupported = 0;
  int cooperative = 0, device = 0;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device));
  supported &= cooperative && worldSize > 1 && worldSize <= 32 &&
               communicator->bootstrap()->getNranksPerIpcDomain() == worldSize;
  MPI_Allreduce(&supported, &allSupported, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!allSupported) {
    SKIP_TEST() << "Requires one full CUDA IPC/fabric domain, bulk copy and cooperative launch.";
  }
  constexpr int Blocks = 128;
  constexpr int Iterations = 1000;
  constexpr size_t SharedBytes = EpRowWorkers * EpRowBytes;
  MSCCLPP_CUDATHROW(cudaFuncSetAttribute(kernelEpTmaMesh, cudaFuncAttributeMaxDynamicSharedMemorySize, SharedBytes));
  int blocksPerSm = 0, sms = 0;
  MSCCLPP_CUDATHROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernelEpTmaMesh, 256, SharedBytes));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  int resident = blocksPerSm * sms >= Blocks, allResident = 0;
  MPI_Allreduce(&resident, &allResident, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!allResident) {
    SKIP_TEST() << "The fixed 128-CTA, eight-row-worker shape must fit concurrently on every GPU.";
  }
  const size_t halfWords = static_cast<size_t>(worldSize) * EpRowsPerPeer * EpRowWords;
  const size_t halfBytes = halfWords * sizeof(uint32_t);
  auto pool = mscclpp::GpuBuffer<uint32_t>(2 * halfWords).memory();
  std::vector<uint32_t> source(halfWords), received(halfWords);
  for (size_t i = 0; i < halfWords; ++i)
    source[i] = static_cast<uint32_t>(rank + 1) * 0x9e3779b9u ^ static_cast<uint32_t>(i);
  MSCCLPP_CUDATHROW(cudaMemcpy(pool.get(), source.data(), halfBytes, cudaMemcpyHostToDevice));
  setupPeerPools(pool.get(), 2 * halfBytes);
  auto state = uploadPeerState(syncHandles, peerPools);
  auto deviceDurations = mscclpp::GpuBuffer<uint64_t>(Iterations).memory();
  std::vector<uint64_t> durations(Iterations), allDurations(rank == 0 ? Iterations * worldSize : 0);
  std::ofstream samples;
  if (gEnv->args.count("perf_output")) {
    samples.open(gEnv->args["perf_output"] + ".rank" + std::to_string(rank) + ".csv");
    samples << "read,repetition,round,total_ns\n";
  }
  cudaEvent_t begin, end;
  MSCCLPP_CUDATHROW(cudaEventCreate(&begin));
  MSCCLPP_CUDATHROW(cudaEventCreate(&end));
  auto launch = [&](bool read, int iterations) {
    auto** pools = state.pools.get();
    auto* channels = state.chans.get();
    auto* times = deviceDurations.get();
    void* arguments[] = {&pools, &channels, &rank, &worldSize, &read, &iterations, &times};
    MSCCLPP_CUDATHROW(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernelEpTmaMesh), dim3(Blocks), dim3(256),
                                                  arguments, SharedBytes));
  };
  for (bool read : {false, true}) {
    for (int repetition = 0; repetition < 3; ++repetition) {
      MSCCLPP_CUDATHROW(cudaMemset(pool.get() + halfWords, 0, halfBytes));
      MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
      communicator->bootstrap()->barrier();
      launch(read, 16);
      MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
      communicator->bootstrap()->barrier();
      MSCCLPP_CUDATHROW(cudaEventRecord(begin));
      launch(read, Iterations);
      MSCCLPP_CUDATHROW(cudaEventRecord(end));
      MSCCLPP_CUDATHROW(cudaEventSynchronize(end));
      float milliseconds = 0;
      MSCCLPP_CUDATHROW(cudaEventElapsedTime(&milliseconds, begin, end));
      communicator->bootstrap()->barrier();
      MSCCLPP_CUDATHROW(cudaMemcpy(received.data(), pool.get() + halfWords, halfBytes, cudaMemcpyDeviceToHost));
      uint64_t errors = 0, allErrors = 0;
      for (size_t i = 0; i < halfWords; ++i) {
        const int peer = i / (EpRowsPerPeer * EpRowWords);
        const size_t sourceWord =
            static_cast<size_t>(rank) * EpRowsPerPeer * EpRowWords + i % (EpRowsPerPeer * EpRowWords);
        const uint32_t expected =
            peer == rank ? 0 : static_cast<uint32_t>(peer + 1) * 0x9e3779b9u ^ static_cast<uint32_t>(sourceWord);
        errors += received[i] != expected;
      }
      MPI_Allreduce(&errors, &allErrors, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
      EXPECT_EQ(allErrors, 0u);
      MSCCLPP_CUDATHROW(
          cudaMemcpy(durations.data(), deviceDurations.get(), Iterations * sizeof(uint64_t), cudaMemcpyDeviceToHost));
      for (int i = 0; i < Iterations; ++i)
        if (samples) samples << read << ',' << repetition << ',' << i << ',' << durations[i] << '\n';
      MPI_Gather(durations.data(), Iterations, MPI_UINT64_T, allDurations.data(), Iterations, MPI_UINT64_T, 0,
                 MPI_COMM_WORLD);
      double eventUs = milliseconds * 1000.0 / Iterations, sumEventUs = 0;
      MPI_Reduce(&eventUs, &sumEventUs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      if (rank == 0) {
        std::sort(allDurations.begin(), allDurations.end());
        const double mean = std::accumulate(allDurations.begin(), allDurations.end(), 0.0) / allDurations.size() / 1000;
        const size_t bytes = static_cast<size_t>(worldSize - 1) * EpRowsPerPeer * EpRowBytes;
        std::cout << std::setprecision(9) << "[NVL_TMA_MESH] read=" << read << " repetition=" << repetition
                  << " ranks=" << worldSize << " ipc_domain=" << communicator->bootstrap()->getNranksPerIpcDomain()
                  << " blocks=" << Blocks << " threads=256 workers=8 row_bytes=" << EpRowBytes
                  << " rows_per_peer=" << EpRowsPerPeer << " bytes=" << bytes
                  << " event_mean_us=" << sumEventUs / worldSize << " round_mean_us=" << mean
                  << " p50_us=" << allDurations[allDurations.size() / 2] / 1000.0
                  << " p95_us=" << allDurations[allDurations.size() * 95 / 100] / 1000.0
                  << " GBps=" << bytes / (sumEventUs / worldSize) / 1000 << " errors=" << allErrors << std::endl;
        ::mscclpp::test::reportPerfResult(std::string(read ? "read" : "write") + " rep" + std::to_string(repetition),
                                          bytes / (sumEventUs / worldSize) / 1000, "GB/s");
      }
    }
  }
  MSCCLPP_CUDATHROW(cudaEventDestroy(begin));
  MSCCLPP_CUDATHROW(cudaEventDestroy(end));
}

#endif  // defined(MSCCLPP_DEVICE_CUDA)
