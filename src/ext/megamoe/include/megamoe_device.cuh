// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_DEVICE_CUH_
#define MSCCLPP_EXT_MEGAMOE_DEVICE_CUH_

#include <cstddef>
#include <cstdint>
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <type_traits>

#include "megamoe_collective.cuh"
#include "megamoe_kernel.hpp"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#error "Native MegaMoE requires an SM100a compilation target"
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

constexpr int Threads = WarpSchedule<false>::NumWarps * 32;
constexpr int W4Threads = W4NumWarps * 32;
constexpr int EntryRegisters = 128;
constexpr int ComputeRegisters = 224;
constexpr int TransferRegisters = 32;
// Reconfiguration redistributes the CTA's entry allocation, not the entire SM register file.
static_assert(256 * ComputeRegisters + (Threads - 256) * TransferRegisters <= Threads * EntryRegisters);
static_assert(128 * ComputeRegisters + 128 * W4TransferRegisters + (W4Threads - 256) * TransferRegisters <=
              W4Threads * EntryRegisters);
constexpr int LocalThreads = WarpSchedule<true>::NumWarps * 32;
constexpr int LocalEntryRegisters = 168;
constexpr int LocalComputeRegisters = 232;
static_assert(256 * LocalComputeRegisters + (LocalThreads - 256) * TransferRegisters <=
              LocalThreads * LocalEntryRegisters);
constexpr int LocalTokenAlignment = 64;
constexpr int EpilogueTokens = 32;
constexpr int DispatchChunkBytes = 2048;
constexpr int DispatchWarpCount = 4;
constexpr int SmallRoutingSlots = 2 * Threads;
constexpr int SmallRoutingExperts = 128;
constexpr int RoutingTagTicketBits = 13;
constexpr uint32_t RoutingTagTicketMask = (1u << RoutingTagTicketBits) - 1;
constexpr uint32_t InvalidRoutingTag = 0xffffffffu;

// Bounded encoding; larger native configurations retain the two-scan planner.
__host__ __device__ constexpr bool useRoutingTags(const NativeConfig& c) {
  return c.weightMxfp4 && size_t(c.worldSize) * c.maxTokens * c.topK <= (1u << RoutingTagTicketBits) &&
         c.numExperts / c.worldSize <= 16;
}
constexpr int64_t SpinLimit = 1000000000;

struct Control {
  DeviceSyncer gridBarrier;
  uint64_t epoch;
  int tokenBlocks;
  int completedCtas;
  // Private routing-plan state, independent of grid/output completion counters.
  int planningArrivals;
  alignas(8) uint64_t planningReadyEpoch;
};
static_assert(offsetof(Control, planningReadyEpoch) % 8 == 0);

struct Route {
  int rank;
  int token;
  int slot;
  float weight;
};

struct TokenBlock {
  int expert;
  int rows;
};

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
  uint32_t* routingTags;
  int* inputReady;
  int* hiddenReady;
  int* inputChunkReady;
  Route* routes;
  TokenBlock* blocks;
  __bfloat16* input;
  __bfloat16* hidden;
  uint8_t* quantizedInput;
  uint8_t* inputScale;
  uint8_t* quantizedHidden;
  uint8_t* hiddenScale;
  int poolRows;
};

__host__ __device__ constexpr int w4StorageRow(int row) {
  // Block-scaled MMA requires an even TMEM scale-column address. N32 tiles
  // therefore occupy alternating halves of a 64-row activation/scale pitch.
  return row / W4TileN * W4TokenStride + row % W4TileN;
}

__host__ __device__ constexpr int w4SourceScaleStride(int hidden) { return (hidden / 32 + 15) / 16 * 16; }

__host__ __device__ constexpr int w4InputChunks(int hidden) { return (hidden + W4DispatchChunk - 1) / W4DispatchChunk; }

__device__ __forceinline__ int* w4InputChunkCounter(const Workspace& w, int hidden, int block, int chunk) {
  int chunks = w4InputChunks(hidden);
  // The last chunk also publishes full-row readiness.
  return chunk == chunks - 1 ? w.inputReady + block : w.inputChunkReady + size_t(block) * (chunks - 1) + chunk;
}

template <class Types, bool E5M2, bool Local, bool Mxfp4 = false>
struct KernelParameters {
  static_assert(!Mxfp4 || (!E5M2 && !Local));
  using Collective = Types;
  using Tiles = TilePolicy<Local, Mxfp4>;
  static constexpr int ThreadCount = Local ? LocalThreads : (Mxfp4 ? W4Threads : Threads);
  static constexpr bool WeightE5M2 = E5M2;
  static constexpr bool WeightMxfp4 = Mxfp4;
  static constexpr bool LocalExpert = Local;
  NativeConfig config;
  SymmetricLayout symmetric;
  Workspace workspace;
  void* local;
  const uint64_t* peers;
  typename Types::Mainloop::Params fc1;
  typename Types::Mainloop::Params fc2;
};

template <bool E5M2, bool Local = false>
using Parameters = KernelParameters<CollectiveTypes<E5M2, Local>, E5M2, Local>;

template <bool E5M2, int LocalMode>
using KernelEntry = void (*)(Parameters<E5M2, (LocalMode != 0)>, int, __bfloat16*, uint32_t*);

// Resolve CUDA's translation-unit-local template launch stubs in the kernel's own TU.
template <bool E5M2, int LocalMode>
KernelEntry<E5M2, LocalMode> kernelEntry();

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
using W4A8Parameters = KernelParameters<W4A8CollectiveTypes, false, false, true>;

using W4A8KernelEntry = void (*)(W4A8Parameters, int, __bfloat16*, uint32_t*);
W4A8KernelEntry w4a8KernelEntry();
#endif

struct DispatchStorage {
  alignas(128) uint8_t tiles[DispatchWarpCount][2][DispatchChunkBytes];
  BulkBarrier barriers[DispatchWarpCount][2];
};

constexpr int W4ScaleChunkBytes = 256;
struct W4DispatchStorage {
  alignas(128) uint8_t tiles[W4DispatchWarps][2][W4DispatchChunk];
  BulkBarrier barriers[W4DispatchWarps][2];
  alignas(128) uint8_t scales[W4DispatchWarps][W4ScaleChunkBytes];
};

struct NoDispatchStorage {};

struct RoutingStorage {
  int counts[SmallRoutingExperts];
  int starts[SmallRoutingExperts];
  uint64_t peers[72];
  int tokenCounts[72];
};

template <int Tokens>
union alignas(128) EpilogueStorage {
  float scratch[128 * (Tokens + 1)];
  __bfloat16 packed[Tokens * 128];
  RoutingStorage routing;
  static_assert(sizeof(RoutingStorage) <= sizeof(scratch));
};

template <bool E5M2, bool Local = false>
struct alignas(1024) SharedStorage {
  typename CollectiveTypes<E5M2, Local>::Mainloop::TensorStorage tensors;
  typename CollectiveTypes<E5M2, Local>::LoadA::SharedStorage loadA;
  typename CollectiveTypes<E5M2, Local>::LoadB::SharedStorage loadB;
  typename CollectiveTypes<E5M2, Local>::Transform::SharedStorage transformed;
  typename CollectiveTypes<E5M2, Local>::Accumulate::SharedStorage accumulated;
  uint32_t tmem;
  EpilogueStorage<EpilogueTokens> epilogue;
  std::conditional_t<Local, NoDispatchStorage, DispatchStorage> dispatch;
};

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
struct alignas(1024) W4A8SharedStorage {
  W4A8CollectiveTypes::Mainloop::TensorStorage tensors;
  W4A8CollectiveTypes::Load::SharedStorage mainloop;
  std::conditional_t<W4SplitPipelines, W4A8CollectiveTypes::Load::SharedStorage, NoDispatchStorage> activationLoad;
  W4A8CollectiveTypes::Accumulate::SharedStorage accumulated;
  uint32_t tmem;
  EpilogueStorage<W4EpilogueTokens> epilogue;
  W4DispatchStorage dispatch;
};
#endif

template <class T>
__host__ __device__ T* at(void* base, size_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base) + offset);
}

template <class T, auto Scope>
__device__ void waitAtLeast(T* address, T value) {
  POLL_MAYBE_JAILBREAK((atomicLoad<T, Scope>(address, memoryOrderAcquire) < value), SpinLimit);
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
