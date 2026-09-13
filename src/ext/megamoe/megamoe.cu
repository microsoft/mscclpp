// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include "megamoe_collective.cuh"
#include "megamoe_kernel.hpp"
#include "megamoe_mma.cuh"

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
#error "Native MegaMoE requires an SM100a compilation target"
#endif

namespace mscclpp::megamoe {
namespace detail {

constexpr int Threads = 512;
constexpr int EntryRegisters = 128;
constexpr int ComputeRegisters = 224;
constexpr int TransferRegisters = 32;
// Reconfiguration redistributes the CTA's entry allocation, not the entire SM register file.
static_assert(256 * ComputeRegisters + (Threads - 256) * TransferRegisters <= Threads * EntryRegisters);
constexpr int LocalThreads = 384;
constexpr int LocalEntryRegisters = 168;
constexpr int LocalComputeRegisters = 232;
static_assert(256 * LocalComputeRegisters + (LocalThreads - 256) * TransferRegisters <=
              LocalThreads * LocalEntryRegisters);
constexpr int TileN = 32;
constexpr int LocalTileN = 128;
constexpr int LocalTokenAlignment = 64;
constexpr int EpilogueTokens = 32;
constexpr int ScratchStride = EpilogueTokens + 1;
constexpr int DispatchChunkBytes = 2048;
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

size_t aligned(size_t n, size_t alignment = 256) { return (n + alignment - 1) / alignment * alignment; }

bool isLocalExpert(const NativeConfig& c) { return c.worldSize == 1 && c.numExperts == 1 && c.topK == 1; }

size_t appendRegion(size_t& bytes, size_t regionBytes) {
  size_t offset = aligned(bytes);
  if (regionBytes > std::numeric_limits<size_t>::max() - offset) {
    throw std::overflow_error("MegaMoE workspace size overflows size_t");
  }
  bytes = offset + regionBytes;
  return offset;
}

template <class T>
__host__ __device__ T* at(void* base, size_t offset) {
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base) + offset);
}

Workspace workspaceLayout(const NativeConfig& c, void* base, size_t& bytes) {
  const size_t experts = c.numExperts / c.worldSize;
  const size_t routes = size_t(c.worldSize) * c.maxTokens * c.topK;
  const bool local = isLocalExpert(c);
  const int tileN = local ? LocalTileN : TileN;
  const size_t rows =
      local ? aligned(c.maxTokens, LocalTokenAlignment) : aligned(routes + experts * (tileN - 1), tileN);
  if (rows > size_t(std::numeric_limits<int>::max()) ||
      (rows + tileN - 1) / tileN * ((2 * size_t(c.intermediate) + 255) / 256 + (size_t(c.hidden) + 255) / 256) >
          size_t(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("MegaMoE routing workspace exceeds 32-bit tile indexing");
  }
  Workspace w{};
  bytes = 0;
  w.control = at<Control>(base, appendRegion(bytes, sizeof(Control)));
  if (local) {
    w.hiddenReady = at<int>(base, appendRegion(bytes, (rows + tileN - 1) / tileN * sizeof(int)));
  } else {
    w.counts = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
    w.starts = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
    w.cursors = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
    w.inputReady = at<int>(base, appendRegion(bytes, rows / TileN * sizeof(int)));
    w.hiddenReady = at<int>(base, appendRegion(bytes, rows / TileN * sizeof(int)));
    w.routes = at<Route>(base, appendRegion(bytes, rows * sizeof(Route)));
    w.blocks = at<TokenBlock>(base, appendRegion(bytes, rows / TileN * sizeof(TokenBlock)));
    w.input = at<__bfloat16>(base, appendRegion(bytes, rows * c.hidden * sizeof(__bfloat16)));
  }
  w.hidden = at<__bfloat16>(base, appendRegion(bytes, rows * c.intermediate * sizeof(__bfloat16)));
  w.poolRows = int(rows);
  bytes = aligned(bytes);
  return w;
}

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

struct DispatchStorage {
  alignas(128) uint8_t tiles[4][2][DispatchChunkBytes];
  BulkBarrier barriers[4][2];
};

struct NoDispatchStorage {};

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
  } epilogue;
  std::conditional_t<Local, NoDispatchStorage, DispatchStorage> dispatch;
};

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

template <bool E5M2>
__device__ int routeExpert(const Parameters<E5M2>& p, int rank, int token, int slot) {
  if (token >= *peerAt<E5M2, int>(p, rank, p.symmetric.tokenCount)) return -1;
  int id = peerAt<E5M2, int>(p, rank, p.symmetric.topkIds)[token * p.config.topK + slot];
  if (id < -1 || id >= p.config.numExperts) {
    printf("MegaMoE rank %d: invalid expert %d from rank %d, token %d, slot %d\n", p.config.rank, id, rank, token,
           slot);
    __trap();
  }
  int localExperts = p.config.numExperts / p.config.worldSize;
  return id >= p.config.rank * localExperts && id < (p.config.rank + 1) * localExperts
             ? id - p.config.rank * localExperts
             : -1;
}

template <bool E5M2>
__device__ void prepareRoutes(const Parameters<E5M2>& p, int tokens) {
  const auto& c = p.config;
  const auto& w = p.workspace;
  int thread = blockIdx.x * Threads + threadIdx.x;
  int stride = gridDim.x * Threads;
  if (thread == 0) {
    w.control->epoch = ++*at<uint64_t>(p.local, p.symmetric.epoch);
    *at<int>(p.local, p.symmetric.tokenCount) = tokens;
  }
  for (int e = thread; e < c.numExperts / c.worldSize; e += stride) {
    w.counts[e] = 0;
    w.cursors[e] = 0;
  }
  for (int r = thread; r < w.poolRows; r += stride) w.routes[r].rank = -1;
  for (int b = thread; b < w.poolRows / TileN; b += stride) {
    w.inputReady[b] = 0;
    w.hiddenReady[b] = 0;
  }
  auto partial = at<__bfloat16>(p.local, p.symmetric.partialOutput);
  for (size_t i = thread; i < size_t(tokens) * c.topK * c.hidden; i += stride) partial[i] = __bfloat16(0.0f);
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);

  // The grid barrier joins all staging/reset writes before this system release.
  if (blockIdx.x == 0 && threadIdx.x < c.worldSize) {
    signalAndWait(p, threadIdx.x);
  }
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);

  int slots = c.worldSize * c.maxTokens * c.topK;
  for (int i = thread; i < slots; i += stride) {
    int rank = i / (c.maxTokens * c.topK);
    int token = i / c.topK % c.maxTokens;
    int expert = routeExpert(p, rank, token, i % c.topK);
    if (expert >= 0) atomicFetchAdd<int, scopeDevice>(w.counts + expert, 1, memoryOrderRelaxed);
  }
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);

  if (thread == 0) {
    int block = 0;
    for (int e = 0; e < c.numExperts / c.worldSize; ++e) {
      w.starts[e] = block * TileN;
      for (int row = 0; row < w.counts[e]; row += TileN)
        w.blocks[block++] = TokenBlock{e, min(int(TileN), w.counts[e] - row)};
    }
    w.control->tokenBlocks = block;
  }
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);

  for (int i = thread; i < slots; i += stride) {
    int rank = i / (c.maxTokens * c.topK);
    int token = i / c.topK % c.maxTokens;
    int slot = i % c.topK;
    int expert = routeExpert(p, rank, token, slot);
    if (expert >= 0) {
      int row = w.starts[expert] + atomicFetchAdd<int, scopeDevice>(w.cursors + expert, 1, memoryOrderRelaxed);
      float weight = peerAt<E5M2, float>(p, rank, p.symmetric.topkWeights)[token * c.topK + slot];
      w.routes[row] = Route{rank, token, slot, weight};
    }
  }
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);
}

template <bool E5M2>
__device__ __forceinline__ void dispatchTokens(const Parameters<E5M2>& p, SharedStorage<E5M2>& s) {
#if MSCCLPP_BULK_AVAILABLE
  const auto& w = p.workspace;
  const int localWarp = threadIdx.x / 32 - 8;
  if (threadIdx.x % 32 == 0) {
    auto& barriers = s.dispatch.barriers[localWarp];
    barriers[0].relaxedInit();
    barriers[1].relaxedInit();
    bulkFence();
    uint32_t phases[2] = {0, 0};
    int bytes = p.config.hidden * sizeof(__bfloat16);
    int chunks = 1 + (bytes - 1) / DispatchChunkBytes;
    for (int row = blockIdx.x * 4 + localWarp; row < w.control->tokenBlocks * TileN; row += gridDim.x * 4) {
      Route route = w.routes[row];
      if (route.rank < 0) continue;
      auto src = reinterpret_cast<const uint8_t*>(peerAt<E5M2, __bfloat16>(p, route.rank, p.symmetric.input) +
                                                  size_t(route.token) * p.config.hidden);
      auto dst = reinterpret_cast<uint8_t*>(w.input + size_t(row) * p.config.hidden);
      auto load = [&](int chunk) {
        int stage = chunk % 2;
        int size = min(int(DispatchChunkBytes), bytes - chunk * DispatchChunkBytes);
        barriers[stage].arriveAndExpect(size);
        bulkLoad(s.dispatch.tiles[localWarp][stage], src + chunk * DispatchChunkBytes, size, barriers[stage]);
      };
      load(0);
      if (chunks > 1) load(1);
      for (int chunk = 0; chunk < chunks; ++chunk) {
        int stage = chunk % 2;
        int size = min(int(DispatchChunkBytes), bytes - chunk * DispatchChunkBytes);
        barriers[stage].wait(phases[stage], SpinLimit);
        // No generic shared-memory access between these two async-proxy copies.
        bulkStore(dst + chunk * DispatchChunkBytes, s.dispatch.tiles[localWarp][stage], size);
        bulkStoreCommit();
        if (chunk + 2 < chunks) {
          bulkStoreWaitSource();
          load(chunk + 2);
        }
      }
      bulkStoreWait();
      atomicFetchAdd<int, scopeDevice>(w.inputReady + row / TileN, 1, memoryOrderRelease);
    }
    barriers[0].invalidate();
    barriers[1].invalidate();
  }
#endif
}

struct Task {
  bool fc1;
  int block;
  int m;
  TokenBlock tokens;
};

__device__ __forceinline__ float swiglu(float gate, float up, float probability) {
  float product, exponent, denominator, inverse, activated, weighted;
  // Preserve the reference's (up * gate) * sigmoid(gate) * probability association.
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(up), "f"(gate));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(exponent) : "f"(gate), "f"(-1.4426950408889634f));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(denominator) : "f"(exponent));
  asm("add.rn.f32 %0, %1, 0f3f800000;" : "=f"(denominator) : "f"(denominator));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(denominator));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(activated) : "f"(product), "f"(inverse));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(weighted) : "f"(activated), "f"(probability));
  return weighted;
}

template <bool E5M2, bool Local>
__device__ Task taskAt(const Parameters<E5M2, Local>& p, int ordinal, int tokens, int hidden, int intermediate) {
  constexpr int KernelTileN = Local ? LocalTileN : TileN;
  int fc1Tiles = intermediate / 128;
  int fc2Tiles = (hidden + 255) / 256;
  int tokenBlocks = Local ? (tokens + KernelTileN - 1) / KernelTileN : p.workspace.control->tokenBlocks;
  int fc1Tasks = tokenBlocks * fc1Tiles;
  bool fc1 = ordinal < fc1Tasks;
  int local = fc1 ? ordinal : ordinal - fc1Tasks;
  int tiles = fc1 ? fc1Tiles : fc2Tiles;
  int block = local / tiles;
  TokenBlock tokenBlock;
  if constexpr (Local) {
    tokenBlock = TokenBlock{0, min(int(KernelTileN), tokens - block * KernelTileN)};
  } else {
    tokenBlock = p.workspace.blocks[block];
  }
  return Task{fc1, block, local % tiles, tokenBlock};
}

template <bool E5M2, int LocalMode, class Accumulators>
__device__ __forceinline__ void epilogue(
    const Parameters<E5M2, (LocalMode != 0)>& p, const Task& task, SharedStorage<E5M2, (LocalMode != 0)>& s,
    typename CollectiveTypes<E5M2, (LocalMode != 0)>::Accumulate& pipeline,
    typename CollectiveTypes<E5M2, (LocalMode != 0)>::Accumulate::PipelineState& state, Accumulators accumulators,
    __bfloat16* directOutput, int hidden, int intermediate) {
#if MSCCLPP_BULK_AVAILABLE
  using namespace cute;
  constexpr int KernelTileN = LocalMode ? LocalTileN : TileN;
  auto matrix = coalesce(accumulators(_, _, _, state.index()));
  CUTE_STATIC_ASSERT_V(size<0>(matrix) == _128{});
  CUTE_STATIC_ASSERT_V(size<1>(matrix) == Int<KernelTileN>{});
  pipeline.consumer_wait(state);
  for (int tokenOffset = 0; tokenOffset < task.tokens.rows; tokenOffset += EpilogueTokens) {
    int validRows = min(int(EpilogueTokens), task.tokens.rows - tokenOffset);
    auto acc =
        local_tile(matrix, make_shape(_128{}, Int<EpilogueTokens>{}), make_coord(0, tokenOffset / EpilogueTokens));
    auto copyOp = make_tmem_copy(SM100_TMEM_LOAD_32dp32b8x{}, acc);
    auto threadCopy = copyOp.get_slice(threadIdx.x);
    auto source = threadCopy.partition_S(acc);
    auto identity = make_identity_tensor(make_shape(_128{}, Int<EpilogueTokens>{}));
    auto coordinates = threadCopy.partition_D(identity);
    auto values = make_tensor<float>(shape(coordinates));
    copy(copyOp, source, values);
    cutlass::arch::fence_view_async_tmem_load();
    if (tokenOffset + EpilogueTokens >= task.tokens.rows) pipeline.consumer_release(state);
    if (task.fc1) {
      CUTE_UNROLL
      for (int i = 0; i < size(values); ++i) {
        auto coord = coordinates(i);
        s.epilogue.scratch[(int(get<0>(coord)) % 128) * ScratchStride + int(get<1>(coord))] = values(i);
      }
      cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      __bfloat16 folded[EpilogueTokens / 2];
      CUTE_UNROLL
      for (int j = 0; j < EpilogueTokens / 2; ++j) {
        int i = threadIdx.x + j * 128;
        int token = i / 64;
        int feature = i % 64;
        if (token < validRows) {
          int gateRow = feature / 16 * 32 + feature % 16;
          float gate = s.epilogue.scratch[gateRow * ScratchStride + token];
          float up = s.epilogue.scratch[(gateRow + 16) * ScratchStride + token];
          if (p.config.gateUpClamp >= 0) {
            gate = fminf(gate, p.config.gateUpClamp);
            up = fmaxf(-p.config.gateUpClamp, fminf(up, p.config.gateUpClamp));
          }
          int row = task.block * KernelTileN + tokenOffset + token;
          float probability = 1.0f;
          bool active = true;
          if constexpr (LocalMode == 1) {
            int id = at<int>(p.local, p.symmetric.topkIds)[row];
            if (id < -1 || id > 0) {
              printf("MegaMoE local expert: invalid routing ID %d for token %d\n", id, row);
              __trap();
            }
            active = id == 0;
            if (active) probability = at<float>(p.local, p.symmetric.topkWeights)[row];
          } else if constexpr (LocalMode == 0) {
            probability = p.workspace.routes[row].weight;
          }
          folded[j] = active ? __bfloat16(swiglu(gate, up, probability)) : __bfloat16(0.0f);
          if constexpr (LocalMode != 0) {
            int column = task.m * 128 + (blockIdx.x % 2) * 64 + feature;
            p.workspace.hidden[size_t(row) * intermediate + column] = folded[j];
          }
        }
      }
      if constexpr (LocalMode != 0) {
        // Publish ordinary global stores to the async proxy before FC2's TMA load.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        continue;
      }
      // Every FP32 reader must finish before the same storage becomes a BF16 tile.
      cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      CUTE_UNROLL
      for (int j = 0; j < EpilogueTokens / 2; ++j) {
        int i = threadIdx.x + j * 128;
        if (i < validRows * 64) s.epilogue.packed[i] = folded[j];
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < size(values); ++i) {
        auto coord = coordinates(i);
        bool active = true;
        if constexpr (LocalMode == 1) {
          int row = task.block * KernelTileN + tokenOffset + int(get<1>(coord));
          if (int(get<1>(coord)) < validRows) active = at<int>(p.local, p.symmetric.topkIds)[row] == 0;
        }
        s.epilogue.packed[int(get<1>(coord)) * 128 + int(get<0>(coord)) % 128] =
            active ? __bfloat16(values(i)) : __bfloat16(0.0f);
      }
    }
    bulkFence();
    cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    if constexpr (LocalMode != 0) {
      if (!task.fc1 && (reinterpret_cast<uintptr_t>(directOutput) & 15) != 0) {
        int feature = task.m * 256 + (blockIdx.x % 2) * 128;
        if (feature < hidden) {
          for (int token = threadIdx.x / 32; token < validRows; token += 4) {
            size_t offset = size_t(task.block * KernelTileN + tokenOffset + token) * hidden + feature;
            auto* destination = directOutput + offset;
            auto* source = s.epilogue.packed + token * 128;
            // Preserve support for contiguous BF16 views with unaligned storage offsets.
            mscclpp::detail::copy<__bfloat16>(destination, source, 128, threadIdx.x % 32, 32);
          }
        }
        cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        continue;
      }
    }
    if (threadIdx.x % 32 == 0) {
      int width = task.fc1 ? 64 : 128;
      int feature = task.m * width * 2 + (blockIdx.x % 2) * width;
      for (int token = threadIdx.x / 32; token < validRows; token += 4) {
        if (task.fc1) {
          auto output =
              p.workspace.hidden + size_t(task.block * KernelTileN + tokenOffset + token) * intermediate + feature;
          bulkStore(output, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
        } else if (feature < hidden) {
          if constexpr (LocalMode != 0) {
            size_t offset = size_t(task.block * KernelTileN + tokenOffset + token) * hidden + feature;
            bulkStore(directOutput + offset, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
          } else {
            Route route = p.workspace.routes[task.block * TileN + tokenOffset + token];
            auto output = peerAt<E5M2, __bfloat16>(p, route.rank, p.symmetric.partialOutput);
            size_t offset = (size_t(route.token) * p.config.topK + route.slot) * p.config.hidden + feature;
            bulkStore(output + offset, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
          }
        }
        bulkStoreCommit();
        bulkStoreWait<4>();
      }
      if constexpr (LocalMode != 0) {
        if (task.fc1)
          bulkStoreWait();
        else
          bulkStoreWaitSource();
      } else {
        bulkStoreWait();
      }
    }
    cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }
  ++state;
  if (task.fc1 && threadIdx.x == 0)
    atomicFetchAdd<int, scopeDevice>(p.workspace.hiddenReady + task.block, 1, memoryOrderRelease);
#endif
}

template <bool E5M2, int LocalMode = 0>
__global__ __launch_bounds__((LocalMode ? LocalThreads : Threads),
                             1) void megaMoe(__grid_constant__ const Parameters<E5M2, (LocalMode != 0)> p, int tokens,
                                             __bfloat16* output, uint32_t* startSignal) {
  using namespace cute;
  using Types = CollectiveTypes<E5M2, (LocalMode != 0)>;
  using Mainloop = typename Types::Mainloop;
  using LoadA = typename Types::LoadA;
  using LoadB = typename Types::LoadB;
  using Transform = typename Types::Transform;
  using Accumulate = typename Types::Accumulate;
  extern __shared__ __align__(1024) char storage[];
  constexpr bool Local = LocalMode != 0;
  constexpr int KernelTileN = Local ? LocalTileN : TileN;
  constexpr int TransformWarp = Local ? 8 : 12;
  constexpr int RoleRegisters = Local ? LocalComputeRegisters : ComputeRegisters;
  constexpr int RestoreRegisters = Local ? LocalEntryRegisters : EntryRegisters;
  auto& s = *reinterpret_cast<SharedStorage<E5M2, Local>*>(storage);
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int cta = blockIdx.x % 2;
  int cluster = blockIdx.x / 2;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  if (blockIdx.x == 0 && threadIdx.x == 0 && startSignal)
    atomicStore<uint32_t, scopeDevice>(startSignal, 1, memoryOrderRelease);
  if constexpr (Local) {
    if (tokens == 0) return;
  } else {
    prepareRoutes(p, tokens);
  }

  typename LoadA::Params aParams{};
  aParams.role =
      warp == 5 ? LoadA::ThreadCategory::Producer
                : (warp >= TransformWarp ? LoadA::ThreadCategory::Consumer : LoadA::ThreadCategory::NonParticipant);
  aParams.is_leader = lane == 0;
  aParams.num_consumers = 128;
  aParams.transaction_bytes = Mainloop::TmaTransactionBytes_A;
  aParams.initializing_warp = 5;
  LoadA aPipeline(s.loadA, aParams, ClusterShape{}, cutlass::McastDirection::kRow, true_type{}, false_type{});
  typename LoadB::Params bParams{};
  bParams.role = warp == 6 ? LoadB::ThreadCategory::Producer
                           : (warp == 4 ? LoadB::ThreadCategory::Consumer : LoadB::ThreadCategory::NonParticipant);
  bParams.is_leader = lane == 0 && cta == 0 && warp == 6;
  bParams.num_consumers = 32;
  bParams.transaction_bytes = Mainloop::TmaTransactionBytes_B;
  bParams.initializing_warp = 6;
  LoadB bPipeline(s.loadB, bParams, ClusterShape{}, cutlass::McastDirection::kCol, true_type{}, false_type{});
  typename Transform::Params tParams{};
  tParams.role = warp >= TransformWarp
                     ? Transform::ThreadCategory::Producer
                     : (warp == 4 ? Transform::ThreadCategory::Consumer : Transform::ThreadCategory::NonParticipant);
  tParams.consumer_arv_count = 1;
  tParams.producer_arv_count = 256;
  tParams.initializing_warp = TransformWarp;
  Transform tPipeline(s.transformed, tParams, ClusterShape{}, true_type{}, false_type{});
  typename Accumulate::Params cParams{};
  cParams.role = warp == 4
                     ? Accumulate::ThreadCategory::Producer
                     : (warp < 4 ? Accumulate::ThreadCategory::Consumer : Accumulate::ThreadCategory::NonParticipant);
  cParams.producer_arv_count = 1;
  cParams.consumer_arv_count = 256;
  cParams.initializing_warp = 0;
  Accumulate cPipeline(s.accumulated, cParams, ClusterShape{}, true_type{}, false_type{});
  cutlass::arch::fence_barrier_init();
  cute::cluster_sync();
  aPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster(), cutlass::McastDirection::kRow);
  bPipeline.init_masks(ClusterShape{}, cutlass::McastDirection::kCol);
  tPipeline.init_masks(ClusterShape{});
  cPipeline.init_masks(ClusterShape{});
  cute::TMEM::Allocator2Sm allocator;
  if (warp == 4) allocator.allocate(512, &s.tmem);
  __syncthreads();
  cute::cluster_sync();

  Mainloop fc1(p.fc1, ClusterShape{}, cta);
  Mainloop fc2(p.fc2, ClusterShape{}, cta);
  auto acc = Mainloop::TiledMma::make_fragment_C(append(fc1.partition_accumulator_shape(), _2{}));
  acc.data() = s.tmem;
  int tokenBlocks = Local ? (tokens + KernelTileN - 1) / KernelTileN : p.workspace.control->tokenBlocks;
  int tasks = tokenBlocks * (intermediate / 128 + (hidden + 255) / 256);
  int localExperts = p.config.numExperts / p.config.worldSize;
  ProblemShape shape1{2 * intermediate, p.workspace.poolRows, hidden, localExperts};
  ProblemShape shape2{hidden, p.workspace.poolRows, intermediate, localExperts};

  // Keep register reconfiguration inside each disjoint role branch. Merging
  // before dispatch makes ptxas constrain the compute roles to the smaller budget.
  if (warp < 4) {
    cutlass::arch::warpgroup_reg_alloc<RoleRegisters>();
    typename Accumulate::PipelineState state;
    for (int i = cluster; i < tasks; i += gridDim.x / 2)
      epilogue<E5M2, LocalMode>(p, taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate), s, cPipeline, state, acc,
                                output, hidden, intermediate);
#if MSCCLPP_BULK_AVAILABLE
    if constexpr (Local) {
      if (threadIdx.x % 32 == 0) bulkStoreWait();
    }
#endif
    cutlass::arch::warpgroup_reg_dealloc<RestoreRegisters>();
  } else if (warp >= TransformWarp) {
    cutlass::arch::warpgroup_reg_alloc<RoleRegisters>();
    typename LoadA::PipelineState aState;
    auto tState = cutlass::make_producer_start_state<Transform>();
    auto inputs = fc1.transform_init(p.fc1, shape1, acc, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / 2) {
      Task task = taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate);
      transformWeights<E5M2, Local>(aPipeline, aState, tPipeline, tState, inputs,
                                    task.fc1 ? hidden / 128 : intermediate / 128);
    }
    tPipeline.producer_tail(tState);
    cutlass::arch::warpgroup_reg_dealloc<RestoreRegisters>();
  } else {
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    if constexpr (!Local) {
      if (warp >= 8) {
        dispatchTokens(p, s);
      }
    }
    if (warp == 5 && lane == 0) {
      auto state = cutlass::make_producer_start_state<LoadA>();
      auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
      auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
      for (int i = cluster; i < tasks; i += gridDim.x / 2) {
        Task task = taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate);
        auto coord = make_coord(task.m * 2 + cta, task.block, 0, task.tokens.expert);
        auto iterator = cute::make_coord_iterator(0, task.fc1 ? hidden / 128 : intermediate / 128);
        auto result = task.fc1 ? fc1.load_A(p.fc1, aPipeline, state, load1, coord, iterator, hidden / 128)
                               : fc2.load_A(p.fc2, aPipeline, state, load2, coord, iterator, intermediate / 128);
        state = get<0>(result);
      }
      aPipeline.producer_tail(state);
    } else if (warp == 6 && lane == 0) {
      auto state = cutlass::make_producer_start_state<LoadB>();
      auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
      auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
      for (int i = cluster; i < tasks; i += gridDim.x / 2) {
        Task task = taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate);
        if constexpr (Local) {
          if (!task.fc1) waitAtLeast<int, scopeDevice>(p.workspace.hiddenReady + task.block, 2 * (intermediate / 128));
        } else {
          waitAtLeast<int, scopeDevice>((task.fc1 ? p.workspace.inputReady : p.workspace.hiddenReady) + task.block,
                                        task.fc1 ? task.tokens.rows : 2 * (intermediate / 128));
        }
        auto coord = make_coord(task.m * 2 + cta, task.block, 0, task.tokens.expert);
        auto iterator = cute::make_coord_iterator(0, task.fc1 ? hidden / 128 : intermediate / 128);
        auto result = task.fc1 ? fc1.load_B(p.fc1, bPipeline, state, load1, coord, iterator, hidden / 128)
                               : fc2.load_B(p.fc2, bPipeline, state, load2, coord, iterator, intermediate / 128);
        state = get<0>(result);
      }
      bPipeline.producer_tail(state);
    } else if (warp == 4 && cta == 0) {
      typename LoadB::PipelineState bState;
      typename Transform::PipelineState tState;
      auto cState = cutlass::make_producer_start_state<Accumulate>();
      auto inputs = fc1.mma_init(acc, s.tensors);
      for (int i = cluster; i < tasks; i += gridDim.x / 2) {
        Task task = taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate);
        auto result = mmaTiles<E5M2, Local>(bPipeline, bState, tPipeline, tState, cPipeline, cState, acc, inputs,
                                            task.fc1 ? hidden / 128 : intermediate / 128);
        bState = get<0>(result);
        tState = get<1>(result);
        cState = get<2>(result);
      }
      cPipeline.producer_tail(cState);
    }
    cutlass::arch::warpgroup_reg_alloc<RestoreRegisters>();
  }
  __syncthreads();
  cute::cluster_sync();
  if (warp == 4) {
    allocator.release_allocation_lock();
    allocator.free(s.tmem, 512);
  }
  if constexpr (!Local) {
    p.workspace.control->gridBarrier.sync(gridDim.x, SpinLimit);
    // Bulk stores have landed; CTA/grid joins transfer their completion to these publishers.
    if (blockIdx.x == 0 && threadIdx.x < p.config.worldSize) {
      signalAndWait(p, threadIdx.x);
    }
    p.workspace.control->gridBarrier.sync(gridDim.x, SpinLimit);
    auto partial = at<uint16_t>(p.local, p.symmetric.partialOutput);
    for (size_t i = blockIdx.x * Threads + threadIdx.x; i < size_t(tokens) * p.config.hidden;
         i += gridDim.x * Threads) {
      size_t token = i / p.config.hidden;
      int feature = i % p.config.hidden;
      float sum = 0.0f;
      for (int slot = 0; slot < p.config.topK; ++slot) {
        uint16_t bits;
        auto address = partial + (token * p.config.topK + slot) * p.config.hidden + feature;
        asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(bits) : "l"(address) : "memory");
        sum += float(mscclpp::bit_cast<__bfloat16>(bits));
      }
      output[i] = __bfloat16(sum);
    }
  }
}

__global__ void packWeightsKernel(NativeConfig c, PackedWeights src, PackedWeights dst) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  int experts = c.numExperts / c.worldSize;
  size_t fc1Size = size_t(experts) * 2 * c.intermediate * c.hidden;
  size_t fc2Size = size_t(experts) * c.hidden * c.intermediate;
  for (size_t i = idx; i < fc1Size; i += stride) {
    int k = i % c.hidden;
    size_t row = i / c.hidden;
    int m = row % (2 * c.intermediate);
    int canonical = (m / 32 * 16 + m % 16) + (m % 32 >= 16 ? c.intermediate : 0);
    dst.fc1[i] = src.fc1[(row / (2 * c.intermediate) * 2 * c.intermediate + canonical) * c.hidden + k];
  }
  for (size_t i = idx; i < fc2Size; i += stride) dst.fc2[i] = src.fc2[i];
  for (size_t i = idx; i < fc1Size / 32; i += stride) {
    int m = i % (2 * c.intermediate);
    int k = i / (2 * c.intermediate) % (c.hidden / 32);
    size_t expert = i / (size_t(2) * c.intermediate * (c.hidden / 32));
    int canonical = (m / 32 * 16 + m % 16) + (m % 32 >= 16 ? c.intermediate : 0);
    dst.fc1Scale[i] = src.fc1Scale[(expert * 2 * c.intermediate + canonical) * (c.hidden / 32) + k];
  }
  for (size_t i = idx; i < fc2Size / 32; i += stride) {
    int m = i % c.hidden;
    int k = i / c.hidden % (c.intermediate / 32);
    size_t expert = i / (size_t(c.hidden) * (c.intermediate / 32));
    dst.fc2Scale[i] = src.fc2Scale[(expert * c.hidden + m) * (c.intermediate / 32) + k];
  }
}

template <bool E5M2, bool Local = false>
Parameters<E5M2, Local> makeParameters(const NativeConfig& c, void* symmetric, const uint64_t* peers, void* workspace,
                                       const PackedWeights& weights) {
  using namespace cute;
  using Mainloop = typename CollectiveTypes<E5M2, Local>::Mainloop;
  using Weight = typename CollectiveTypes<E5M2, Local>::Weight;
  using Scale = typename CollectiveTypes<E5M2, Local>::Scale;
  using Activation = typename CollectiveTypes<E5M2, Local>::Activation;
  Parameters<E5M2, Local> p{};
  p.config = c;
  p.symmetric = getSymmetricLayout(c);
  p.local = symmetric;
  p.peers = peers;
  size_t bytes;
  p.workspace = workspaceLayout(c, workspace, bytes);
  int experts = c.numExperts / c.worldSize;
  auto make = [&](int m, int k, int rows, uint8_t* weight, uint8_t* scale, __bfloat16* input) {
    ProblemShape shape{m, rows, k, experts};
    typename Mainloop::Arguments args{};
    args.ptr_A = reinterpret_cast<const Weight*>(weight);
    args.dA = make_stride(int64_t(k), _1{}, int64_t(m) * k);
    args.ptr_B = reinterpret_cast<const Activation*>(input);
    args.dB = make_stride(int64_t(k), _1{}, int64_t(0));
    args.ptr_S = reinterpret_cast<const Scale*>(scale);
    args.layout_S = ScaleConfig::tile_atom_to_shape_scale(make_shape(m, k, experts));
    if (!Mainloop::can_implement(shape, args)) throw std::invalid_argument("MegaMoE TMA input layout is unsupported");
    return Mainloop::to_underlying_arguments(shape, args, nullptr);
  };
  p.fc1 = make(2 * c.intermediate, c.hidden, p.workspace.poolRows, weights.fc1, weights.fc1Scale,
               isLocalExpert(c) ? at<__bfloat16>(symmetric, p.symmetric.input) : p.workspace.input);
  p.fc2 = make(c.hidden, c.intermediate, p.workspace.poolRows, weights.fc2, weights.fc2Scale, p.workspace.hidden);
  return p;
}

}  // namespace detail

struct KernelPlan {
  std::variant<detail::Parameters<false>, detail::Parameters<true>, detail::Parameters<false, true>,
               detail::Parameters<true, true>>
      params;
  int ctas;
  int device;
  size_t sharedBytes = 0;
  bool localExpert = false;
};

void validateNativeConfig(const NativeConfig& c) {
  if (c.worldSize < 1 || c.worldSize > 72 || c.rank < 0 || c.rank >= c.worldSize)
    throw std::invalid_argument("MegaMoE requires 1 <= worldSize <= 72 and rank in [0, worldSize)");
  if (c.maxTokens < 1 || c.hidden < 128 || c.intermediate < 128 || c.hidden % 128 || c.intermediate % 128)
    throw std::invalid_argument("MegaMoE requires positive capacity and H/I divisible by 128");
  if (c.hidden > std::numeric_limits<int>::max() / 2 || c.intermediate > std::numeric_limits<int>::max() / 2)
    throw std::invalid_argument("MegaMoE H/I exceed 32-bit indexing");
  if (c.numExperts < 1 || c.numExperts % c.worldSize || c.topK < 1 || c.topK > 32 || c.topK > c.numExperts)
    throw std::invalid_argument("MegaMoE requires evenly partitioned experts and 1 <= topK <= min(32, experts)");
  if (c.smMargin < 0 || !std::isfinite(c.gateUpClamp))
    throw std::invalid_argument("MegaMoE smMargin must be nonnegative and gateUpClamp must be finite");
  if (size_t(c.worldSize) * c.maxTokens * c.topK > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("MegaMoE routing capacity exceeds 32-bit indexing");
}

SymmetricLayout getSymmetricLayout(const NativeConfig& c) {
  validateNativeConfig(c);
  SymmetricLayout layout{};
  const size_t inputTokens =
      detail::isLocalExpert(c) ? detail::aligned(c.maxTokens, detail::LocalTokenAlignment) : size_t(c.maxTokens);
  layout.input = detail::appendRegion(layout.bytes, inputTokens * c.hidden * sizeof(__bfloat16));
  layout.topkIds = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * sizeof(int));
  layout.topkWeights = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * sizeof(float));
  layout.partialOutput =
      detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * c.hidden * sizeof(__bfloat16));
  layout.epoch = detail::appendRegion(layout.bytes, sizeof(uint64_t));
  layout.peerSignals = detail::appendRegion(layout.bytes, size_t(c.worldSize) * sizeof(uint64_t));
  layout.expectedPeerSignals = detail::appendRegion(layout.bytes, size_t(c.worldSize) * sizeof(uint64_t));
  layout.tokenCount = detail::appendRegion(layout.bytes, sizeof(int));
  layout.bytes = detail::aligned(layout.bytes);
  return layout;
}

size_t getPrivateWorkspaceBytes(const NativeConfig& c) {
  validateNativeConfig(c);
  size_t bytes;
  detail::workspaceLayout(c, nullptr, bytes);
  return bytes;
}

void packNativeWeights(const NativeConfig& c, const PackedWeights& source, const PackedWeights& destination,
                       cudaStream_t stream) {
  validateNativeConfig(c);
  if (!source.fc1 || !source.fc1Scale || !source.fc2 || !source.fc2Scale || !destination.fc1 || !destination.fc1Scale ||
      !destination.fc2 || !destination.fc2Scale)
    throw std::invalid_argument("MegaMoE weight buffers must be non-null");
  detail::packWeightsKernel<<<256, 256, 0, stream>>>(c, source, destination);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

std::shared_ptr<KernelPlan> createKernelPlan(const NativeConfig& c, void* symmetric, const uint64_t* peers,
                                             void* workspace, const PackedWeights& weights) {
  validateNativeConfig(c);
  if (!symmetric || !peers || !workspace || !weights.fc1 || !weights.fc1Scale || !weights.fc2 || !weights.fc2Scale)
    throw std::invalid_argument("MegaMoE plan requires live workspaces, peer addresses, and packed weights");
  auto plan = std::make_shared<KernelPlan>();
  MSCCLPP_CUDATHROW(cudaGetDevice(&plan->device));
  cudaDeviceProp properties{};
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&properties, plan->device));
  if (properties.major != 10 || properties.minor != 0)
    throw std::invalid_argument("Native MegaMoE currently requires an SM100 GPU");
  if (c.smMargin > properties.multiProcessorCount - 2)
    throw std::invalid_argument("MegaMoE smMargin must leave at least two SMs");
  plan->ctas = (properties.multiProcessorCount - c.smMargin) / 2 * 2;
  plan->localExpert = detail::isLocalExpert(c);
  if (plan->localExpert && c.weightE5M2) {
    plan->params = detail::makeParameters<true, true>(c, symmetric, peers, workspace, weights);
  } else if (plan->localExpert) {
    plan->params = detail::makeParameters<false, true>(c, symmetric, peers, workspace, weights);
  } else if (c.weightE5M2) {
    plan->params = detail::makeParameters<true>(c, symmetric, peers, workspace, weights);
  } else {
    plan->params = detail::makeParameters<false>(c, symmetric, peers, workspace, weights);
  }
  std::visit(
      [&](const auto& params) {
        constexpr bool E5M2 = std::decay_t<decltype(params)>::WeightE5M2;
        auto configure = [&]<int LocalMode>(std::integral_constant<int, LocalMode>) {
          constexpr bool Local = LocalMode != 0;
          constexpr int Entry = Local ? detail::LocalEntryRegisters : detail::EntryRegisters;
          auto kernel = detail::megaMoe<E5M2, LocalMode>;
          plan->sharedBytes = std::max(plan->sharedBytes, sizeof(detail::SharedStorage<E5M2, Local>));
          cudaFuncAttributes attributes{};
          MSCCLPP_CUDATHROW(cudaFuncGetAttributes(&attributes, kernel));
          if (attributes.numRegs < Entry)
            throw std::runtime_error("MegaMoE entry register allocation is too small for warpgroup reconfiguration");
          MSCCLPP_CUDATHROW(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(plan->sharedBytes)));
          cudaLaunchAttribute attribute{};
          attribute.id = cudaLaunchAttributeClusterDimension;
          attribute.val.clusterDim = {2, 1, 1};
          cudaLaunchConfig_t launch{};
          launch.gridDim = dim3(plan->ctas);
          launch.blockDim = dim3(Local ? detail::LocalThreads : detail::Threads);
          launch.dynamicSmemBytes = plan->sharedBytes;
          launch.attrs = &attribute;
          launch.numAttrs = 1;
          int clusters = 0;
          MSCCLPP_CUDATHROW(cudaOccupancyMaxActiveClusters(&clusters, kernel, &launch));
          plan->ctas = std::min(plan->ctas, clusters * 2);
          if (plan->ctas < 2) throw std::runtime_error("MegaMoE cannot keep a two-CTA cluster resident");
        };
        if constexpr (std::decay_t<decltype(params)>::LocalExpert) {
          configure(std::integral_constant<int, 1>{});
          configure(std::integral_constant<int, 2>{});
        } else {
          configure(std::integral_constant<int, 0>{});
        }
      },
      plan->params);
  return plan;
}

int kernelPlanCtaCount(const KernelPlan& plan) { return plan.ctas; }
size_t kernelPlanSharedBytes(const KernelPlan& plan) { return plan.sharedBytes; }

namespace {
void launchPlan(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream,
                uint32_t* startSignal, bool unweightedShared) {
  if (!plan) throw std::invalid_argument("MegaMoE kernel plan is null");
  if (unweightedShared && !plan->localExpert)
    throw std::invalid_argument("Shared forward requires a single local expert");
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != plan->device) throw std::invalid_argument("MegaMoE must launch on the device owning its workspace");
  std::visit(
      [&](const auto& params) {
        if (tokens < 0 || tokens > params.config.maxTokens || (tokens && !output))
          throw std::invalid_argument("MegaMoE token count or output pointer is invalid");
        constexpr bool E5M2 = std::decay_t<decltype(params)>::WeightE5M2;
        if (plan->localExpert && tokens) {
          const size_t tokenBlocks = (tokens + detail::LocalTileN - 1) / detail::LocalTileN;
          MSCCLPP_CUDATHROW(cudaMemsetAsync(params.workspace.hiddenReady, 0, tokenBlocks * sizeof(int), stream));
        }
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim = {2, 1, 1};
        cudaLaunchConfig_t launch{};
        launch.gridDim = dim3(plan->ctas);
        launch.blockDim = dim3(plan->localExpert ? detail::LocalThreads : detail::Threads);
        launch.dynamicSmemBytes = plan->sharedBytes;
        launch.stream = stream;
        launch.attrs = &attribute;
        launch.numAttrs = 1;
        auto run = [&]<int LocalMode>(std::integral_constant<int, LocalMode>) {
          MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&launch, detail::megaMoe<E5M2, LocalMode>, params, tokens,
                                               static_cast<__bfloat16*>(output), startSignal));
        };
        if constexpr (std::decay_t<decltype(params)>::LocalExpert) {
          if (unweightedShared) {
            run(std::integral_constant<int, 2>{});
          } else
            run(std::integral_constant<int, 1>{});
        } else {
          run(std::integral_constant<int, 0>{});
        }
      },
      plan->params);
}
}  // namespace

void launchNativeMegaMoe(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream,
                         uint32_t* startSignal) {
  launchPlan(plan, tokens, output, stream, startSignal, false);
}

void launchNativeSharedExpert(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream) {
  launchPlan(plan, tokens, output, stream, nullptr, true);
}

}  // namespace mscclpp::megamoe
