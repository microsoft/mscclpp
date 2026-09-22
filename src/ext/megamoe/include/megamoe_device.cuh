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
constexpr int EntryRegisters = 128;
constexpr int ComputeRegisters = 224;
constexpr int TransferRegisters = 32;
// Reconfiguration redistributes the CTA's entry allocation, not the entire SM register file.
static_assert(256 * ComputeRegisters + (Threads - 256) * TransferRegisters <= Threads * EntryRegisters);
constexpr int LocalThreads = WarpSchedule<true>::NumWarps * 32;
constexpr int LocalEntryRegisters = 168;
constexpr int LocalComputeRegisters = 232;
static_assert(256 * LocalComputeRegisters + (LocalThreads - 256) * TransferRegisters <=
              LocalThreads * LocalEntryRegisters);
constexpr int LocalTokenAlignment = 64;
constexpr int EpilogueTokens = 32;
constexpr int ScratchStride = EpilogueTokens + 1;
constexpr int DispatchChunkBytes = 2048;
constexpr int DispatchWarpCount = 4;
constexpr int SmallRoutingSlots = 2 * Threads;
constexpr int SmallRoutingExperts = 128;
constexpr int64_t SpinLimit = 1000000000;

struct Control {
  DeviceSyncer gridBarrier;
  uint64_t epoch;
  int tokenBlocks;
};

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
  int* inputReady;
  int* hiddenReady;
  Route* routes;
  TokenBlock* blocks;
  __bfloat16* input;
  __bfloat16* hidden;
  int poolRows;
};

template <bool E5M2, bool Local = false>
struct Parameters {
  static constexpr bool WeightE5M2 = E5M2;
  static constexpr bool LocalExpert = Local;
  NativeConfig config;
  SymmetricLayout symmetric;
  Workspace workspace;
  void* local;
  const uint64_t* peers;
  typename CollectiveTypes<E5M2, Local>::Mainloop::Params fc1;
  typename CollectiveTypes<E5M2, Local>::Mainloop::Params fc2;
};

template <bool E5M2, int LocalMode>
using KernelEntry = void (*)(Parameters<E5M2, (LocalMode != 0)>, int, __bfloat16*, uint32_t*);

// Resolve CUDA's translation-unit-local template launch stubs in the kernel's own TU.
template <bool E5M2, int LocalMode>
KernelEntry<E5M2, LocalMode> kernelEntry();

struct DispatchStorage {
  alignas(128) uint8_t tiles[DispatchWarpCount][2][DispatchChunkBytes];
  BulkBarrier barriers[DispatchWarpCount][2];
};

struct NoDispatchStorage {};

struct RoutingStorage {
  int counts[SmallRoutingExperts];
  int starts[SmallRoutingExperts];
  uint64_t peers[72];
  int tokenCounts[72];
};
static_assert(sizeof(RoutingStorage) <= 128 * ScratchStride * sizeof(float));

template <bool E5M2, bool Local = false>
struct alignas(1024) SharedStorage {
  typename CollectiveTypes<E5M2, Local>::Mainloop::TensorStorage tensors;
  typename CollectiveTypes<E5M2, Local>::LoadA::SharedStorage loadA;
  typename CollectiveTypes<E5M2, Local>::LoadB::SharedStorage loadB;
  typename CollectiveTypes<E5M2, Local>::Transform::SharedStorage transformed;
  typename CollectiveTypes<E5M2, Local>::Accumulate::SharedStorage accumulated;
  uint32_t tmem;
  union alignas(128) {
    float scratch[128 * ScratchStride];
    __bfloat16 packed[EpilogueTokens * 128];
    RoutingStorage routing;
  } epilogue;
  std::conditional_t<Local, NoDispatchStorage, DispatchStorage> dispatch;
};

template <class T>
__host__ __device__ T* at(void* base, size_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base) + offset);
}

template <class T, auto Scope>
__device__ void waitAtLeast(T* address, T value) {
  POLL_MAYBE_JAILBREAK((atomicLoad<T, Scope>(address, memoryOrderAcquire) < value), SpinLimit);
}

template <bool E5M2, class T>
__device__ T* peerAt(const Parameters<E5M2>& p, int rank, size_t offset) {
  return at<T>(reinterpret_cast<void*>(p.peers[rank]), offset);
}

template <bool E5M2>
__device__ __forceinline__ void signalAndWait(const Parameters<E5M2>& p, int peer) {
  BaseMemoryChannelDeviceHandle channel{
      MemoryDevice2DeviceSemaphoreDeviceHandle{at<uint64_t>(p.local, p.symmetric.peerSignals) + peer,
                                               peerAt<E5M2, uint64_t>(p, peer, p.symmetric.peerSignals) + p.config.rank,
                                               at<uint64_t>(p.local, p.symmetric.expectedPeerSignals) + peer}};
  channel.signal();
  channel.wait(SpinLimit);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_DEVICE_CUH_
