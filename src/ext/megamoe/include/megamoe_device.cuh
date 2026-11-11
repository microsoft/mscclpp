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
constexpr int LocalThreads = W8A16WarpSchedule<true>::NumWarps * 32;
constexpr int LocalEntryRegisters = 168;
constexpr int LocalComputeRegisters = 232;
static_assert(256 * LocalComputeRegisters + (LocalThreads - 256) * TransferRegisters <=
              LocalThreads * LocalEntryRegisters);
constexpr int LocalTokenAlignment = 64;
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
  // Fused shared expert with in-kernel static-FP8 quantization: one sense-reversing arrival per CTA. Its own cache
  // line keeps these atomics off the routing grid barrier.
  alignas(128) uint32_t sharedQuantizeSyncState;
};

// Routing exchanges a rank's routes in chunks of one route per kernel thread; even an empty batch has one chunk.
constexpr int RouteChunk = W4Threads;

__host__ __device__ constexpr int routeChunks(int routes) {
  return routes > RouteChunk ? routes / RouteChunk + (routes % RouteChunk != 0) : 1;
}

// Small batches quantize the BF16 input inside the kernel's routing instead of a separate launch: CTAs 1..tokens each
// quantize one token while CTA ctas - worldSize - 1 stages the source route weights. MXFP8 always fuses; static FP8
// fuses only with MSCCLPP_MEGAMOE_W4_FUSE_STATIC_QUANTIZE. The host launch and the kernel must agree on this predicate.
__host__ __device__ constexpr bool fusesInputQuantize(ActivationDtype activation, int hidden, int worldSize, int tokens,
                                                      int topK, int ctas) {
  return (activation == ActivationDtype::Mxfp8 || (W4FuseStaticQuantize && activation == ActivationDtype::Fp8Static)) &&
         hidden == 4096 && worldSize > 1 && worldSize <= 64 && tokens > 0 && tokens <= 64 &&
         tokens * topK <= RouteChunk && ctas >= worldSize + 2 + tokens;
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
  int8_t groupOffset;
  int8_t groupShift;
  uint8_t cacheWeights;
  uint8_t reserved;
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
  const uint8_t* fc1Weights;
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
  constexpr int ElemBytes = Types::WeightMxfp4 ? sizeof(uint8_t) : sizeof(__bfloat16);
  return (hidden * ElemBytes + Types::DispatchChunk - 1) / Types::DispatchChunk;
}

constexpr int W4ReadyCounterStride = 32;

template <class Types>
__device__ __forceinline__ int* w4InputChunkCounter(const Workspace& w, int hidden, int block, int chunk) {
  int chunks = w4InputChunks<Types>(hidden);
  // The last chunk also publishes full-row readiness.
  return chunk == chunks - 1 ? w.inputReady + size_t(block) * W4ReadyCounterStride
                             : w.inputChunkReady + (size_t(block) * (chunks - 1) + chunk) * W4ReadyCounterStride;
}

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

template <int DispatchWarps, int DispatchStages, int DispatchChunk, int ScaleStageBytes>
struct W4DispatchBuffersT {
  alignas(128) uint8_t tiles[DispatchWarps][DispatchStages][DispatchChunk];
  alignas(128) uint8_t scales[DispatchWarps][DispatchStages][ScaleStageBytes];
};

template <class Types>
struct W4SharedDispatchStateT {
  BulkBarrier barriers[Types::DispatchWarps][Types::DispatchStages];
  uint32_t arrivals;
  uint32_t publishedPhase;
  uint32_t dispatchDone;
};

template <class Types>
struct alignas(1024) W4A8SharedStorageT {
  static constexpr int SharedBlockCapacity = 64;
  static constexpr int SharedExpertCapacity = 32;
  static constexpr int RequiredScaleStageBytes = (Types::DispatchChunk / 32 + 15) / 16 * 16;
  static constexpr int ScaleStageBytes = RequiredScaleStageBytes < 128 ? 128 : RequiredScaleStageBytes;
  static constexpr bool OverlapEpilogueAndDispatch = (Types::DispatchWarps * Types::DispatchStages > 2);
  // Routing finishes before GEMM uses these buffers; pipeline state and dispatch flags remain separate.
  union {
    typename Types::Mainloop::TensorStorage tensors;
    int routing[sizeof(typename Types::Mainloop::TensorStorage) / sizeof(int)];
  };
  // The local W8A8 kernel (LocalExpert) has no dispatch role and carries no dispatch buffers.
  std::conditional_t<
      OverlapEpilogueAndDispatch || Types::LocalExpert, EmptyStorage,
      W4DispatchBuffersT<Types::DispatchWarps, Types::DispatchStages, Types::DispatchChunk, ScaleStageBytes>>
      separateDispatchBuffers;
  union {
    EpilogueStorage<Types::EpilogueTokens> epilogue;
    std::conditional_t<
        OverlapEpilogueAndDispatch && !Types::LocalExpert,
        W4DispatchBuffersT<Types::DispatchWarps, Types::DispatchStages, Types::DispatchChunk, ScaleStageBytes>,
        EmptyStorage>
        sharedDispatchBuffers;
  };
  typename Types::Load::SharedStorage mainloop;
  std::conditional_t<Types::SplitPipelines, typename Types::Load::SharedStorage, EmptyStorage> activationLoad;
  typename Types::Accumulate::SharedStorage accumulated;
  uint32_t tmem;
  int16_t expertFirstTile[SharedExpertCapacity];
  int16_t expertRows[SharedExpertCapacity];
  TokenBlock blocks[SharedBlockCapacity];
  std::conditional_t<Types::LocalExpert, EmptyStorage, W4SharedDispatchStateT<Types>> dispatch;
  // Routed kernels order TMEM allocation through the routing grid barrier; the local W8A8 kernel has no routing and
  // keeps a cluster barrier. Placed last so routed member offsets are unchanged.
  std::conditional_t<Types::LocalExpert, cutlass::arch::ClusterBarrier, EmptyStorage> tmemReady;
};

using W4A8SharedStorage = W4A8SharedStorageT<W4A8CollectiveTypes>;

// Fused routed + shared-expert storage. `routed` keeps the routed kernel's layout at identical offsets because the
// routed roles locate the token-block cache, dispatch state and epilogue through P::SharedStorage. The shared expert
// reuses the GEMM tensor buffers, which stream its FC1 during routing, so the routing scratch instead overlays the
// dispatch buffers and the epilogue staging: neither is used before this CTA's routing completes.
struct alignas(1024) W4A8FusedSharedStorage {
  using Routed = W4A8SharedStorage;
  using SharedTensorStorage = W8A8LocalCollectiveTypes::Mainloop::TensorStorage;
  static_assert(std::is_same_v<W4A8CollectiveTypes::Load, W8A8LocalCollectiveTypes::Load>);
  static_assert(std::is_same_v<W4A8CollectiveTypes::Accumulate, W8A8LocalCollectiveTypes::Accumulate>);
  static_assert(sizeof(SharedTensorStorage) <= sizeof(W4A8CollectiveTypes::Mainloop::TensorStorage));
  static constexpr size_t RoutingOffset = offsetof(Routed, separateDispatchBuffers);
  static constexpr size_t RoutingInts = (offsetof(Routed, mainloop) - RoutingOffset) / sizeof(int);
  struct RoutingOverlay {
    uint8_t skipped[RoutingOffset];
    int routing[RoutingInts];
  };
  union {
    Routed routed;
    RoutingOverlay routingOverlay;
  };
  cutlass::arch::ClusterBarrier tmemReady;
  cutlass::arch::ClusterBarrier computeReady;
  cutlass::arch::ClusterBarrier phaseReady;
  cutlass::arch::ClusterBarrier routingReady;

  __device__ SharedTensorStorage& sharedTensors() { return reinterpret_cast<SharedTensorStorage&>(routed.tensors); }
};

template <class Types, bool Borrowed = false, ActivationDtype ActDtype = ActivationDtype::Mxfp8>
struct KernelParameters {
  template <class T>
  using Resource = std::conditional_t<Borrowed, const T&, T>;
  using Collective = Types;
  using Tiles = typename Types::Tiles;
  using SharedStorage = std::conditional_t<Types::WeightMxfp4, W4A8SharedStorageT<Types>, W8A16SharedStorageT<Types>>;
  static constexpr int ThreadCount = Types::NumWarps * 32;
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
// Routed parameter layouts stay unchanged: only the local W8A8 kernel carries per-matrix weight scales (fp8_nv),
// which its epilogue multiplies into the static-FP8 dequantization. They are 1 for block-scaled MXFP8 weights.
struct W8A8LocalParameters : KernelParameters<W8A8LocalCollectiveTypes, false, ActivationDtype::Fp8Static> {
  float fc1WeightScale;
  float fc2WeightScale;
};

struct W4A8FusedParameters {
  W4A8Parameters routed;
  W8A8LocalParameters shared;
  // Shared FC1 starts this long after kernel entry so its HBM stream does not slow the latency-bound routing exchange.
  uint32_t sharedFc1DelayNs;
};

template <ActivationDtype ActDtype = ActivationDtype::Mxfp8, class P>
__device__ __forceinline__ auto w4a8ParameterView(const P& p, NativeConfig config) {
  using View = KernelParameters<typename P::Collective, true, ActDtype>;
  return View{config, p.symmetric, p.workspace, p.local, p.peers, p.fc1TaskDivisor, p.fc2TaskDivisor, p.fc1, p.fc2};
}

using W4A8KernelEntry = void (*)(W4A8Parameters, int, __bfloat16*, uint32_t*, const int32_t*, const float*,
                                 const __bfloat16*);
W4A8KernelEntry w4a8KernelEntry(const NativeConfig& config, bool capturing = false);
using W4A8FusedKernelEntry = void (*)(W4A8FusedParameters, int, __bfloat16*, uint32_t*, const int32_t*, const float*,
                                      __bfloat16*, const __bfloat16*, const __bfloat16*);
W4A8FusedKernelEntry w4a8FusedKernelEntry(const NativeConfig& config, bool capturing = false);

template <int LocalMode = 0>
using W8A16KernelEntry = void (*)(std::conditional_t<LocalMode != 0, W8A16LocalParameters, W8A16Parameters>, int,
                                  __bfloat16*, uint32_t*, const int32_t*, const float*);

template <int LocalMode = 0>
W8A16KernelEntry<LocalMode> w8a16KernelEntry();

using W8A8LocalKernelEntry = void (*)(W8A8LocalParameters, int, __bfloat16*);
W8A8LocalKernelEntry w8a8LocalKernelEntry();

// Routing scratch words ahead of the cached routes: two per-expert arrays, three per-rank arrays, a batch of
// RouteSegmentBatch segment bounds and bases, and three CTA words, rounded up to an even count so the 8-byte cached
// routes stay aligned.
constexpr int RouteSegmentBatch = 32;

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
__device__ __forceinline__ void signalPeer(const P& p, int peer) {
  const uint64_t epoch = p.workspace.control->epoch;
  if (peer != p.config.rank) {
    auto* remote = peerAt<uint64_t>(p, peer, p.symmetric.peerSignals) + p.config.rank;
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(remote), "l"(epoch) : "memory");
  } else {
    at<uint64_t>(p.local, p.symmetric.peerSignals)[peer] = epoch;
  }
  at<uint64_t>(p.local, p.symmetric.expectedPeerSignals)[peer] = epoch;
}

template <class P>
__device__ __forceinline__ void signalAndWait(const P& p, int peer) {
  signalPeer(p, peer);
  if (peer != p.config.rank) {
    waitAtLeast<uint64_t, scopeSystem>(at<uint64_t>(p.local, p.symmetric.peerSignals) + peer,
                                       p.workspace.control->epoch);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_DEVICE_CUH_
