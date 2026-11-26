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

#ifndef WARP_SIZE
#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

constexpr int W4Threads = W4NumWarps * WARP_SIZE;
constexpr int W4TmemColumns = 512;
constexpr int EntryRegisters = 128;
constexpr int TransferRegisters = 32;
constexpr int LocalThreads = W8A16WarpSchedule<true>::NumWarps * WARP_SIZE;
constexpr int LocalEntryRegisters = 168;
constexpr int LocalComputeRegisters = 232;
static_assert(256 * LocalComputeRegisters + (LocalThreads - 256) * TransferRegisters <=
              LocalThreads * LocalEntryRegisters);
constexpr int LocalTokenAlignment = 64;
constexpr int64_t SpinLimit = 1000000000;

constexpr bool validW4RegisterBudget(int epilogueWarps, int epilogueRegisters, int transferRegisters) {
  int epilogueThreads = epilogueWarps * WARP_SIZE;
  constexpr int NonEpilogueComputeThreads = 128;
  int dispatchThreads = W4Threads - epilogueThreads - NonEpilogueComputeThreads;
  return dispatchThreads >= 0 && epilogueThreads * epilogueRegisters + NonEpilogueComputeThreads * transferRegisters +
                                         dispatchThreads * TransferRegisters <=
                                     W4Threads * EntryRegisters;
}

static_assert(validW4RegisterBudget(W4EpilogueWarps, W4EpilogueRegisters, W4TransferRegisters));

struct Control {
  uint64_t epoch;
  int nTokenTiles;
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

struct alignas(8) TokenTile {
  int16_t expertIdx;
  int16_t nTokens;
  int16_t tileIdxInGroup;
  // Packs the tile-group shift in the low bits and the W4A8 weight L2 eviction flag in bit 8.
  int16_t packedFlags;

  static constexpr int GroupShiftFieldMask = 0x7f;
  static constexpr int16_t WeightEvictNormalFlag = int16_t(1 << 8);

  __host__ __device__ static constexpr TokenTile make(int expertIdx, int nTokens, int tileIdxInGroup = 0,
                                                      int groupShift = 0, bool weightEvictNormal = false) {
    return TokenTile{int16_t(expertIdx), int16_t(nTokens), int16_t(tileIdxInGroup),
                     int16_t(groupShift | (weightEvictNormal ? WeightEvictNormalFlag : 0))};
  }

  __host__ __device__ constexpr int groupShift() const { return int(packedFlags) & GroupShiftFieldMask; }

  __host__ __device__ constexpr bool weightEvictNormal() const { return (packedFlags & WeightEvictNormalFlag) != 0; }
};

static_assert(sizeof(TokenTile) == sizeof(uint2));

struct Task {
  bool fc1;
  int block;
  int m;
  TokenTile tokens;
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
  TokenTile* tokenTiles;
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

constexpr int MxfpScaleBlockSize = 32;
constexpr int BulkCopyAlignment = 16;

__host__ __device__ constexpr int alignedMxfpScaleBytes(int elemBytes) {
  return (elemBytes / MxfpScaleBlockSize + BulkCopyAlignment - 1) / BulkCopyAlignment * BulkCopyAlignment;
}

__host__ __device__ constexpr int w4SourceScaleStride(int hidden) { return alignedMxfpScaleBytes(hidden); }

template <class Types>
__host__ __device__ constexpr int w4InputChunks(int hidden) {
  constexpr int ElemBytes = Types::WeightMxfp4 ? sizeof(uint8_t) : sizeof(__bfloat16);
  return (hidden * ElemBytes + Types::DispatchChunk - 1) / Types::DispatchChunk;
}

constexpr int W4ReadyCounterStride = 32;

template <class Types>
MSCCLPP_DEVICE_INLINE int* w4InputChunkCounter(const Workspace& workspace, int hidden, int block, int chunk) {
  int chunks = w4InputChunks<Types>(hidden);
  // The last chunk also publishes full-row readiness.
  return chunk == chunks - 1
             ? workspace.inputReady + size_t(block) * W4ReadyCounterStride
             : workspace.inputChunkReady + (size_t(block) * (chunks - 1) + chunk) * W4ReadyCounterStride;
}

template <class Types>
struct W4DispatchStorageT {
  static constexpr int NumStages =
      (Types::WeightMxfp4 && Types::DispatchStages == 1 && Types::DispatchWarps == 2 && Types::DispatchChunk >= 4096)
          ? 2
          : Types::DispatchStages;
  static constexpr int RequiredScaleStageBytes = alignedMxfpScaleBytes(Types::DispatchChunk);
  static constexpr int ScaleStageBytes = RequiredScaleStageBytes < 128 ? 128 : RequiredScaleStageBytes;
  alignas(128) uint8_t tiles[Types::DispatchWarps][NumStages][Types::DispatchChunk];
  BulkBarrier barriers[Types::DispatchWarps][NumStages];
  alignas(128) uint8_t scales[Types::DispatchWarps][NumStages][ScaleStageBytes];
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

template <class Types>
struct alignas(1024) W8A16SharedStorageT {
  union {
    typename Types::Mainloop::TensorStorage tensors;
    int routing[sizeof(typename Types::Mainloop::TensorStorage) / sizeof(int)];
  };
  typename Types::LoadA::SharedStorage mainloopA;
  typename Types::LoadB::SharedStorage mainloopB;
  typename Types::Transform::SharedStorage transformed;
  typename Types::Accumulate::SharedStorage accumulated;
  uint32_t tmem;
  cutlass::arch::ClusterBarrier tmemReady;
  EpilogueStorage<Types::EpilogueTokens> epilogue;
  std::conditional_t<Types::LocalExpert, EmptyStorage, W4DispatchStorageT<Types>> dispatch;
};

enum class OutputOffloadOp : uint32_t {
  Store = 0u,
  Noop = 1u,
  Stop = 2u,
};

struct alignas(8) OutputOffloadTask {
  uint64_t packed;

  static constexpr uint64_t OpFieldMask = 0xfu;
  static constexpr int NumTokensShift = 4;
  static constexpr uint64_t NumTokensFieldMask = 0xfffu;
  static constexpr int MTileIdxShift = 16;
  static constexpr uint64_t MTileIdxFieldMask = 0xffffu;
  static constexpr int BaseRowIdxShift = 32;

  __host__ __device__ static constexpr OutputOffloadTask make(OutputOffloadOp op, int baseRowIdx = 0, int nTokens = 0,
                                                              int mTileIdx = 0) {
    return OutputOffloadTask{(uint64_t(uint32_t(baseRowIdx)) << BaseRowIdxShift) |
                             ((uint64_t(uint32_t(mTileIdx)) & MTileIdxFieldMask) << MTileIdxShift) |
                             ((uint64_t(uint32_t(nTokens)) & NumTokensFieldMask) << NumTokensShift) |
                             (uint64_t(op) & OpFieldMask)};
  }

  __host__ __device__ static constexpr OutputOffloadTask makeStore(int baseRowIdx, int nTokens, int mTileIdx) {
    return make(OutputOffloadOp::Store, baseRowIdx, nTokens, mTileIdx);
  }

  __host__ __device__ static constexpr OutputOffloadTask makeNoop() { return make(OutputOffloadOp::Noop); }

  __host__ __device__ static constexpr OutputOffloadTask makeStop() { return make(OutputOffloadOp::Stop); }

  __host__ __device__ constexpr OutputOffloadOp op() const { return OutputOffloadOp(uint32_t(packed & OpFieldMask)); }

  __host__ __device__ constexpr int nTokens() const { return int((packed >> NumTokensShift) & NumTokensFieldMask); }

  __host__ __device__ constexpr int mTileIdx() const { return int((packed >> MTileIdxShift) & MTileIdxFieldMask); }

  __host__ __device__ constexpr int baseRowIdx() const { return int(uint32_t(packed >> BaseRowIdxShift)); }
};

static_assert(sizeof(OutputOffloadTask) == sizeof(uint64_t));

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
  OutputOffloadTask outputOffloadTask;
};

using W4A8SharedStorage = W4A8SharedStorageT<W4A8CollectiveTypes>;

template <class Types, bool Borrowed = false, ActivationDtype ActDtype = ActivationDtype::Mxfp8>
struct KernelParameters {
  template <class T>
  using Resource = std::conditional_t<Borrowed, const T&, T>;
  using Collective = Types;
  using Tiles = typename Types::Tiles;
  using SharedStorage = std::conditional_t<Types::WeightMxfp4, W4A8SharedStorageT<Types>, W8A16SharedStorageT<Types>>;
  static constexpr int ThreadCount = Types::NumWarps * WARP_SIZE;
  static constexpr bool WeightMxfp4 = Types::WeightMxfp4;
  static constexpr bool LocalExpert = Types::LocalExpert;
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
using W8A16Parameters = KernelParameters<W8A16CollectiveTypesT<false>, false, ActivationDtype::Bf16>;
using W8A16LocalParameters = KernelParameters<W8A16CollectiveTypesT<true>, false, ActivationDtype::Bf16>;

template <ActivationDtype ActDtype = ActivationDtype::Mxfp8, class KernelParams>
MSCCLPP_DEVICE_INLINE auto w4a8ParameterView(const KernelParams& params, NativeConfig config) {
  using View = KernelParameters<typename KernelParams::Collective, true, ActDtype>;
  return View{config,       params.symmetric,      params.workspace,      params.local,
              params.peers, params.fc1TaskDivisor, params.fc2TaskDivisor, params.fc1,
              params.fc2};
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE auto w8a16ParameterView(const KernelParams& params, NativeConfig config) {
  return w4a8ParameterView<ActivationDtype::Bf16>(params, config);
}

struct KernelThreadLocation {
  int warpIdx;
  int laneIdx;
  int ctaIdx;
  int clusterIdx;
};

template <int ClusterSize>
MSCCLPP_DEVICE_INLINE KernelThreadLocation makeKernelThreadLocation() {
  return KernelThreadLocation{int(threadIdx.x / WARP_SIZE), int(threadIdx.x % WARP_SIZE), int(blockIdx.x % ClusterSize),
                              int(blockIdx.x / ClusterSize)};
}

// Notify a consumer stream that CTA 0 has entered the kernel, not that output is ready.
MSCCLPP_DEVICE_INLINE void signalKernelEntry(uint32_t* kernelEntrySignal) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && kernelEntrySignal) {
    atomicStore<uint32_t, scopeDevice>(kernelEntrySignal, 1, memoryOrderRelease);
  }
}

template <int Hidden, int Intermediate, ActivationDtype ActDtype, class KernelParams>
MSCCLPP_DEVICE_INLINE auto makeW4A8KernelView(const KernelParams& parameters) {
  NativeConfig configuration = parameters.config;
  configuration.hidden = Hidden;
  configuration.intermediate = Intermediate;
  return w4a8ParameterView<ActDtype>(parameters, configuration);
}

template <int Hidden, int Intermediate, class KernelParams>
MSCCLPP_DEVICE_INLINE auto makeW8A16KernelView(const KernelParams& parameters) {
  NativeConfig configuration = parameters.config;
  configuration.hidden = Hidden;
  configuration.intermediate = Intermediate;
  return w8a16ParameterView(parameters, configuration);
}

using W4A8KernelEntry = void (*)(W4A8Parameters, int, __bfloat16*, uint32_t*, const void*, const uint8_t*,
                                 const int32_t*, const float*);
W4A8KernelEntry w4a8KernelEntry(const NativeConfig& config, bool capturing = false);

template <int LocalMode = 0>
using W8A16KernelEntry = void (*)(std::conditional_t<LocalMode != 0, W8A16LocalParameters, W8A16Parameters>, int,
                                  __bfloat16*, uint32_t*, const void*, const int32_t*, const float*);

template <int LocalMode = 0>
W8A16KernelEntry<LocalMode> w8a16KernelEntry(const NativeConfig& config);

// Routing scratch words ahead of the cached routes: two per-expert arrays, three per-rank arrays, a batch of
// RouteSegmentBatch segment bounds and bases, and three CTA words, rounded up to an even count so the 8-byte cached
// routes stay aligned.
constexpr int RouteSegmentBatch = WARP_SIZE;

__host__ __device__ constexpr size_t routingScratchHeaderInts(int localExperts, int worldSize) {
  return (2 * size_t(localExperts) + 3 * size_t(worldSize) + 2 * RouteSegmentBatch + 3 + 1) / 2 * 2;
}

template <class T>
__host__ __device__ T* at(void* base, size_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base) + offset);
}

template <class T, auto Scope>
__device__ void waitAtLeast(T* address, T value) {
  POLL_MAYBE_JAILBREAK((atomicLoad<T, Scope>(address, memoryOrderAcquire) < value), SpinLimit);
}

MSCCLPP_DEVICE_INLINE void signalReadyCounter(int* counter, uint32_t count = 1u) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(count) : "memory");
}

MSCCLPP_DEVICE_INLINE void waitReadyCounter(int* counter, int expectedCount) {
  waitAtLeast<int, scopeDevice>(counter, expectedCount);
}

MSCCLPP_DEVICE_INLINE int atomicAddShared(int* counter, int value) {
  int old;
  // An explicit shared address avoids the generic-address atomic instruction.
  uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(counter));
  asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(address), "r"(value) : "memory");
  return old;
}

MSCCLPP_DEVICE_INLINE void addSharedRelease(uint32_t* counter, uint32_t value) {
  uint32_t sharedAddr = static_cast<uint32_t>(__cvta_generic_to_shared(counter));
  asm volatile("red.release.cta.shared.add.u32 [%0], %1;" ::"r"(sharedAddr), "r"(value) : "memory");
}

MSCCLPP_DEVICE_INLINE void resetSharedRelaxed(uint32_t* counter) {
  uint32_t sharedAddr = static_cast<uint32_t>(__cvta_generic_to_shared(counter));
  asm volatile("atom.relaxed.cta.shared.exch.b32 _, [%0], 0;" ::"r"(sharedAddr) : "memory");
}

MSCCLPP_DEVICE_INLINE void fenceAsyncProxyGlobal() { asm volatile("fence.proxy.async.global;" ::: "memory"); }

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

template <class T, class KernelParams>
__device__ T* peerAt(const KernelParams& params, int rank, size_t offset) {
  return at<T>(reinterpret_cast<void*>(params.peers[rank]), offset);
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE void signalAndWait(const KernelParams& params, int peer) {
  BaseMemoryChannelDeviceHandle channel{MemoryDevice2DeviceSemaphoreDeviceHandle{
      at<uint64_t>(params.local, params.symmetric.peerSignals) + peer,
      peerAt<uint64_t>(params, peer, params.symmetric.peerSignals) + params.config.rank,
      at<uint64_t>(params.local, params.symmetric.expectedPeerSignals) + peer}};
  channel.signal();
  channel.wait(SpinLimit);
}

// Synchronizes the resident grid and waits until all peer ranks have completed their partial-output stores for this
// epoch. Returns false for follower CTAs without downstream work so they can exit early after the grid barrier.
template <class KernelParams>
MSCCLPP_DEVICE_INLINE bool syncPeerPartialOutputs(const KernelParams& params, bool ctaHasDownstreamWork) {
  traceKernel(KernelTracePhase::OutputJoin, true);
  syncResidentGrid(&params.workspace.control->residentGridSyncState, gridDim.x);
  if (blockIdx.x > 0 && !ctaHasDownstreamWork) {
    traceKernel(KernelTracePhase::OutputJoin, false);
    return false;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *at<uint64_t>(params.local, params.symmetric.epoch) = params.workspace.control->epoch;
  }
  if (threadIdx.x < params.config.worldSize) {
    // Only CTA0 consumes the peer notification; followers observe the same epoch.
    if (blockIdx.x == 0) {
      signalAndWait(params, threadIdx.x);
    } else if (threadIdx.x != params.config.rank) {
      waitAtLeast<uint64_t, scopeSystem>(at<uint64_t>(params.local, params.symmetric.peerSignals) + threadIdx.x,
                                         params.workspace.control->epoch);
    }
  }
  __syncthreads();
  traceKernel(KernelTracePhase::OutputJoin, false);
  return true;
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_DEVICE_CUH_
