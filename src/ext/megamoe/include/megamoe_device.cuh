// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_DEVICE_CUH_
#define MSCCLPP_EXT_MEGAMOE_DEVICE_CUH_

#include <cutlass/fast_math.h>

#include <cstddef>
#include <cstdint>
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/packet_device.hpp>
#include <type_traits>

#include "megamoe_collective.cuh"
#include "megamoe_kernel.hpp"

#if defined(__CUDA_ARCH__) && (!defined(__CUDA_ARCH_FAMILY_SPECIFIC__) || __CUDA_ARCH_FAMILY_SPECIFIC__ != 1000)
#error "Native MegaMoE requires an SM100-family compilation target"
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

constexpr int W4Threads = W4NumWarps * 32;
constexpr int W4TmemColumns = 512;
constexpr int EntryRegisters = 128;
constexpr int TransferRegisters = 32;
constexpr int64_t SpinLimit = 1000000000;

constexpr bool validW4RegisterBudget(int epilogueWarps, int epilogueRegisters, int transferRegisters) {
  int epilogueThreads = epilogueWarps * 32;
  constexpr int NonEpilogueComputeThreads = 128;
  int dispatchThreads = W4Threads - epilogueThreads - NonEpilogueComputeThreads;
  return dispatchThreads >= 0 && epilogueThreads * epilogueRegisters + NonEpilogueComputeThreads * transferRegisters +
                                         dispatchThreads * TransferRegisters <=
                                     W4Threads * EntryRegisters;
}

static_assert(validW4RegisterBudget(W4EpilogueWarps, W4EpilogueRegisters, W4TransferRegisters));

struct Control {
  uint64_t epoch;
  int tokenBlocks;
  uint32_t residentGridSyncState;
};

// Routing exchanges a rank's routes in chunks of one route per kernel thread; even an empty batch has one chunk.
constexpr int RouteChunk = W4Threads;

__host__ __device__ constexpr int routeChunks(int routes) {
  return routes > RouteChunk ? routes / RouteChunk + (routes % RouteChunk != 0) : 1;
}

struct alignas(16) Route {
  int rank;
  int token;
  int slot;
  float weight;
};

struct alignas(8) TokenBlock {
  int16_t expert;
  int16_t rows;
  int16_t groupOffset;
  int16_t groupShift;
};

static_assert(sizeof(TokenBlock) == sizeof(uint2));

struct Task {
  bool fc1;
  int block;
  int m;
  TokenBlock tokens;
};

struct Workspace {
  Control* control;
  int* counts;
  int* starts;
  int* cursors;
  int* inputReady;
  int* hiddenReady;
  int* inputChunkReady;
  Route* routes;
  TokenBlock* blocks;
  uint8_t* quantizedInput;
  uint8_t* inputScale;
  uint8_t* quantizedHidden;
  uint8_t* hiddenScale;
  int poolRows;
};

template <class Tiles>
__host__ __device__ constexpr int w4StorageRow(int row) {
  // Block-scaled MMA requires an even TMEM scale-column address. N32 tiles
  // therefore occupy alternating halves of a 64-row activation/scale pitch.
  return row / Tiles::N * Tiles::TokenStride + row % Tiles::N;
}

__host__ __device__ constexpr int w4SourceScaleStride(int hidden) { return (hidden / 32 + 15) / 16 * 16; }

template <class Types>
__host__ __device__ constexpr int w4InputChunks(int hidden) {
  return (hidden + Types::DispatchChunk - 1) / Types::DispatchChunk;
}

constexpr int W4ReadyCounterStride = 32;

template <class Types>
__device__ __forceinline__ int* w4InputChunkCounter(const Workspace& w, int hidden, int block, int chunk) {
  int chunks = w4InputChunks<Types>(hidden);
  // The last chunk also publishes full-row readiness.
  return chunk == chunks - 1 ? w.inputReady + size_t(block) * W4ReadyCounterStride
                             : w.inputChunkReady + (size_t(block) * (chunks - 1) + chunk) * W4ReadyCounterStride;
}

template <class Types, bool Borrowed = false, ActivationDtype ActDtype = ActivationDtype::Mxfp8>
struct KernelParameters {
  template <class T>
  using Resource = std::conditional_t<Borrowed, const T&, T>;
  using Collective = Types;
  using Tiles = typename Types::Tiles;
  static constexpr int ThreadCount = Types::NumWarps * 32;
  static constexpr ActivationDtype Activation = ActDtype;
  NativeConfig config;
  Resource<SymmetricLayout> symmetric;
  Resource<Workspace> workspace;
  void* local;
  const uint64_t* peers;
  Resource<cutlass::FastDivmod> fc1TaskDivisor;
  Resource<cutlass::FastDivmod> fc2TaskDivisor;
  Resource<typename Types::Mainloop::Params> fc1;
  Resource<typename Types::Mainloop::Params> fc2;
};

using W4A8Parameters = KernelParameters<W4A8CollectiveTypes>;

template <ActivationDtype ActDtype = ActivationDtype::Mxfp8, class P>
__device__ __forceinline__ auto w4a8ParameterView(const P& p, NativeConfig config) {
  using View = KernelParameters<typename P::Collective, true, ActDtype>;
  return View{config, p.symmetric, p.workspace, p.local, p.peers, p.fc1TaskDivisor, p.fc2TaskDivisor, p.fc1, p.fc2};
}

using W4A8KernelEntry = void (*)(W4A8Parameters, int, __bfloat16*, uint32_t*, const int32_t*, const float*);
W4A8KernelEntry w4a8KernelEntry(const NativeConfig& config, bool capturing = false);

template <class Types>
struct W4DispatchStorageT {
  static constexpr int RequiredScaleStageBytes = (Types::DispatchChunk / 32 + 15) / 16 * 16;
  static constexpr int ScaleStageBytes = RequiredScaleStageBytes < 128 ? 128 : RequiredScaleStageBytes;
  alignas(128) uint8_t tiles[Types::DispatchWarps][Types::DispatchStages][Types::DispatchChunk];
  BulkBarrier barriers[Types::DispatchWarps][Types::DispatchStages];
  alignas(128) uint8_t scales[Types::DispatchWarps][Types::DispatchStages][ScaleStageBytes];
  uint32_t arrivals;
  uint32_t publishedPhase;
};

struct EmptyStorage {};

// FC2 BF16 rows are padded by 16 bytes so transposing stmatrix row writes are bank-conflict free.
constexpr int PackedRowStride = 128 + 8;
constexpr int PackedFp8RowStride = 128 + 16;

template <int Tokens>
struct alignas(128) EpilogueStorage {
  union {
    float scratch[64 * Tokens];
    __bfloat16 packed[Tokens * PackedRowStride];
    uint8_t packedFp8[Tokens * PackedFp8RowStride];
  };
};

// Routing scratch words ahead of the cached routes: two per-expert arrays, three per-rank arrays, a batch of
// RouteSegmentBatch segment bounds and bases, and three CTA words, rounded up to an even count so the 8-byte cached
// routes stay aligned.
constexpr int RouteSegmentBatch = 32;

__host__ __device__ constexpr size_t routingScratchHeaderInts(int localExperts, int worldSize) {
  return (2 * size_t(localExperts) + 3 * size_t(worldSize) + 2 * RouteSegmentBatch + 3 + 1) / 2 * 2;
}

template <class Types>
struct alignas(1024) W4A8SharedStorageT {
  // Routing finishes before GEMM uses these buffers; pipeline state and dispatch flags remain separate.
  union {
    typename Types::Mainloop::TensorStorage tensors;
    int routing[sizeof(typename Types::Mainloop::TensorStorage) / sizeof(int)];
  };
  typename Types::Load::SharedStorage mainloop;
  std::conditional_t<Types::SplitPipelines, typename Types::Load::SharedStorage, EmptyStorage> activationLoad;
  typename Types::Accumulate::SharedStorage accumulated;
  uint32_t tmem;
  cutlass::arch::ClusterBarrier tmemReady;
  EpilogueStorage<Types::EpilogueTokens> epilogue;
  W4DispatchStorageT<Types> dispatch;
};

using W4A8SharedStorage = W4A8SharedStorageT<W4A8CollectiveTypes>;

template <class T>
__host__ __device__ T* at(void* base, size_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base) + offset);
}

template <class T, auto Scope>
__device__ void waitAtLeast(T* address, T value) {
  POLL_MAYBE_JAILBREAK((atomicLoad<T, Scope>(address, memoryOrderAcquire) < value), SpinLimit);
}

// Sense-reversing barrier for a complete resident 1D grid. Every CTA must call each round, CTA0 must participate,
// and the zero-initialized state cannot be shared by concurrent grids.
MSCCLPP_DEVICE_INLINE void syncResidentGrid(uint32_t* state, int ctas) {
  constexpr uint32_t Phase = 1u << 31;
  __syncthreads();
  if (ctas == 1) return;
  if (threadIdx.x == 0) {
    uint32_t increment = blockIdx.x == 0 ? Phase - uint32_t(ctas - 1) : 1u;
    uint32_t phase = atomicFetchAdd<uint32_t, scopeDevice>(state, increment, memoryOrderRelease) & Phase;
    POLL_MAYBE_JAILBREAK((atomicLoad<uint32_t, scopeDevice>(state, memoryOrderAcquire) & Phase) == phase, SpinLimit);
  }
  __syncthreads();
}

template <class T, class P>
__device__ T* peerAt(const P& p, int rank, size_t offset) {
  return at<T>(reinterpret_cast<void*>(p.peers[rank]), offset);
}

template <class P>
__device__ __forceinline__ void signalAndWait(const P& p, int peer) {
  BaseMemoryChannelDeviceHandle channel{
      MemoryDevice2DeviceSemaphoreDeviceHandle{at<uint64_t>(p.local, p.symmetric.peerSignals) + peer,
                                               peerAt<uint64_t>(p, peer, p.symmetric.peerSignals) + p.config.rank,
                                               at<uint64_t>(p.local, p.symmetric.expectedPeerSignals) + peer}};
  channel.signal();
  channel.wait(SpinLimit);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_DEVICE_CUH_
