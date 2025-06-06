// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_COMMON_HPP_
#define MSCCLPP_EXECUTION_COMMON_HPP_

#include <mscclpp/memory_channel.hpp>
#include <mscclpp/nvls.hpp>
#include <mscclpp/port_channel.hpp>

namespace mscclpp {

constexpr int MAX_CHANNEL = 16;
constexpr int MAX_CHANNEL_PER_OPERATION = 8;
constexpr int MAX_OPERATION = 64;
constexpr int INPUT_BUFFER_MEMORY_ID = 0;
constexpr int OUTPUT_BUFFER_MEMORY_ID = 1;
constexpr int SCRATCH_BUFFER_MEMORY_ID = 2;
constexpr int MAX_RESERVED_MEMORY_IDS = 3;

constexpr int MAX_DEVICE_SYNCERS = 16;
constexpr int MAX_DEVICE_SEMAPHORES = 16;

constexpr uint32_t PREDFINED_SCRATCH_SIZE = 1 << 26;  // 64 MB

enum class BufferType : uint8_t {
  NONE,
  INPUT,
  OUTPUT,
  SCRATCH,
};

enum class ChannelType : uint8_t {
  NONE,
  MEMORY,
  PORT,
  SWITCH,
};

// NOTE(chhwang): any modification here requires corresponding updates in `tools/npkit/npkit_trace_generator.py`.
enum class OperationType : uint8_t {
  NOP,
  BARRIER,
  PUT,
  PUT_PACKET,
  READ_PUT_PACKET,
  PUT_WITH_SIGNAL,
  PUT_WITH_SIGNAL_AND_FLUSH,
  GET,
  COPY,
  COPY_PACKET,
  TRANSFORM_TO_PACKET,
  SIGNAL,
  WAIT,
  FLUSH,
  REDUCE,
  REDUCE_PACKET,
  REDUCE_SEND,
  REDUCE_SEND_PACKET,
  READ_REDUCE_COPY,
  READ_REDUCE_COPY_SEND,
  MULTI_LOAD_REDUCE_STORE,
  RELAXED_SIGNAL,
  RELAXED_WAIT,
  PIPELINE,
  SEM_RELEASE,
  SEM_ACQUIRE,
};

struct Channels {
  mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel> memoryChannels[MAX_CHANNEL];
  mscclpp::DeviceHandle<mscclpp::BasePortChannel> portChannels[MAX_CHANNEL];
  mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer> nvlsChannels[MAX_CHANNEL];
};

struct RemoteBuffers {
  BufferType remoteBufferTypes[MAX_CHANNEL];
  void* remoteBuffersViaMemoryChannel[MAX_CHANNEL];
  MemoryId remoteBuffersViaPortChannel[MAX_CHANNEL];
};

union BufferRef {
  uint8_t id;
  BufferType type;
};

struct Operation {
  OperationType type;
  ChannelType channelType;
  union {
    BufferRef inputBufferRefs[MAX_CHANNEL_PER_OPERATION];
    struct {
      uint8_t nvlsInputIndex;
      BufferType nvlsInputBufferType;
    };
  };
  union {
    BufferRef outputBufferRefs[MAX_CHANNEL_PER_OPERATION];
    struct {
      uint8_t nvlsOutputIndex;
      BufferType nvlsOutputBufferType;
    };
  };

  union {
    struct {
      uint8_t channelIndexes[MAX_CHANNEL_PER_OPERATION];
      uint32_t inputOffsets[MAX_CHANNEL_PER_OPERATION];
      uint32_t outputOffsets[MAX_CHANNEL_PER_OPERATION];
      uint32_t inputBufferSizes[MAX_CHANNEL_PER_OPERATION];
      uint32_t outputBufferSizes[MAX_CHANNEL_PER_OPERATION];

      uint8_t nChannels;
      uint8_t nInputs;
      uint8_t nOutputs;
    };
    struct {
      uint32_t unitSize;
      uint32_t nIterations;
      uint8_t nOperations;
    };
    struct {
      uint32_t deviceSyncerIndex;
      uint32_t nThreadBlocks;
    };
    struct {
      uint32_t deviceSemaphoreIds[MAX_DEVICE_SEMAPHORES];
      uint32_t nDeviceSemaphores;
    };
  };
};

// total size = 2304 + 6400 + 4 + 12(padding) = 8720 bytes
struct __attribute__((aligned(16))) DeviceExecutionPlan {
  uint8_t nMemoryChannels;              // 1 bytes
  uint8_t nPortChannels;                // 1 bytes
  uint16_t nOperations;                 // 2 bytes
  Channels channels;                    // 2304 bytes
  RemoteBuffers remoteBuffers;          // 192 bytes
  Operation operations[MAX_OPERATION];  // 64 * 100 = 6400 bytes
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_COMMON_HPP_
