// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_BUFFERED_TIMING_HPP_
#define MSCCLPP_EP_BUFFERED_TIMING_HPP_

#include <cstddef>
#include <cstdint>

namespace mscclpp {
namespace ep {

constexpr int TimingOperations = 2;
constexpr int TimingMaxBlocks = 130;
constexpr int TimingMaxThreads = 512;
constexpr int TimingValues = 8;

enum class TimingKind : uint32_t {
  DispatchBlock,
  DispatchSend,
  DispatchNotify,
  DispatchQueue,
  CombineBlock,
  CombineSend,
  CombineDrain,
  CombineRecv,
  CombineReady,
  DispatchPost,
  DispatchBatch,
  DispatchChunk,
  DispatchSubmit,
  DispatchSubmitCounts,
  DispatchRecv,
  DispatchTopology,
  DispatchStage
};

struct TimingRecord {
  uint64_t invocation;
  uint64_t beginNs;
  uint64_t endNs;
  uint64_t values[TimingValues];
  uint32_t epoch;
  TimingKind kind;
  int thread;
  int peer;
  int queue;
};

struct alignas(128) TimingBlock {
  unsigned long long attempted;
  uint64_t invocations[TimingMaxThreads];
};

struct BufferedTiming {
  TimingBlock blocks[TimingOperations * TimingMaxBlocks];
  TimingRecord* records;
  unsigned int capacityPerBlock;
  unsigned int submitEvery = 0;
  bool submitCounts = false;
};

inline size_t timingRecordIndex(int operation, int block, unsigned int slot, unsigned int capacity) {
  return (static_cast<size_t>(operation) * TimingMaxBlocks + block) * capacity + slot;
}

#if defined(__CUDACC__) && defined(MSCCLPP_EP_GPUNETIO_TIMING)
__device__ __forceinline__ uint64_t timingNow() {
  uint64_t timestamp;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timestamp)::"memory");
  return timestamp;
}

struct TimingWriter {
  BufferedTiming* buffer = nullptr;
  uint64_t invocation = 0;
  uint32_t epoch = 0;
  int operation = 0;

  __host__ __device__ TimingWriter() = default;

  __host__ __device__ TimingWriter(BufferedTiming* storage, int op, uint32_t capturedEpoch)
      : buffer(storage), epoch(capturedEpoch), operation(op) {
#if defined(__CUDA_ARCH__)
    if (buffer == nullptr || blockIdx.x >= TimingMaxBlocks || threadIdx.x >= TimingMaxThreads) {
      buffer = nullptr;
      return;
    }
    invocation = ++buffer->blocks[operation * TimingMaxBlocks + blockIdx.x].invocations[threadIdx.x];
#endif
  }

  __device__ bool traceSubmit() const {
    return buffer != nullptr && !buffer->submitCounts && buffer->submitEvery != 0 &&
           (invocation - 1) % buffer->submitEvery == 0;
  }

  __device__ bool countSubmit() const { return buffer != nullptr && buffer->submitCounts; }

  __device__ void recordSubmitCounts(const uint64_t* values, int peer, int queue) const {
    record(TimingKind::DispatchSubmitCounts, 0, 0, peer, queue, values[0], values[1], values[2], values[3], values[4],
           values[5], values[6], values[7]);
  }

  __device__ void recordSubmit(const uint64_t* stages, uint64_t begin, uint64_t end, int peer, int queue) const {
    const auto elapsed = [&](int first, int last) -> uint64_t {
      return stages[first] != 0 && stages[last] != 0 ? stages[last] - stages[first] : 0;
    };
    record(TimingKind::DispatchSubmit, begin, end, peer, queue, stages[8], elapsed(0, 1),
           elapsed(stages[1] != 0 ? 1 : 0, 2), elapsed(2, 3), elapsed(3, 4), elapsed(4, 5), elapsed(1, 6), stages[9]);
  }

  __device__ void record(TimingKind kind, uint64_t begin, uint64_t end, int peer = -1, int queue = -1,
                         uint64_t first = 0, uint64_t second = 0, uint64_t third = 0, uint64_t fourth = 0,
                         uint64_t fifth = 0, uint64_t sixth = 0, uint64_t seventh = 0, uint64_t eighth = 0) const {
    if (buffer == nullptr) return;
    if (buffer->submitCounts && kind != TimingKind::DispatchSubmitCounts) return;
    const int block = operation * TimingMaxBlocks + blockIdx.x;
    const auto slot = atomicAdd(&buffer->blocks[block].attempted, 1ULL);
    if (slot >= buffer->capacityPerBlock) return;
    auto& output = buffer->records[static_cast<size_t>(block) * buffer->capacityPerBlock + slot];
    output = {invocation,
              begin,
              end,
              {first, second, third, fourth, fifth, sixth, seventh, eighth},
              epoch,
              kind,
              static_cast<int>(threadIdx.x),
              peer,
              queue};
  }
};
#endif

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_BUFFERED_TIMING_HPP_