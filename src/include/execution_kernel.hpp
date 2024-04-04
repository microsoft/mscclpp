// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_KERNEL_HPP_
#define MSCCLPP_EXECUTION_KERNEL_HPP_

#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

namespace mscclpp {

constexpr int MAX_CHANNEL = 16;
constexpr int MAX_CHANNEL_PER_OPERATION = 8;
constexpr int MAX_OPERATION = 64;

enum class BufferType : uint8_t {
  INPUT,
  OUTPUT,
  SCRATCH,
};

enum class ChannelType : uint8_t {
  SM,
  PROXY,
};

enum class OperationType : uint8_t {
  BARRIER,
  PUT,
  GET,
  COPY,
  SIGNAL,
  WAIT,
  FLUSH,
  REDUCE,
  REDUCE_SEND,
  READ_REDUCE,
  READ_REDUCE_SEND,
};

struct Channels {
  mscclpp::DeviceHandle<mscclpp::SmChannel> smChannels[MAX_CHANNEL];
  mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel> proxyChannels[MAX_CHANNEL];
};

struct Operation {
  OperationType type;
  ChannelType channelType;
  BufferType srcBufferType;
  BufferType dstBufferType;
  uint8_t nInputChannels;
  uint8_t nOutputChannels;
  uint8_t inputChannelIndex[MAX_CHANNEL_PER_OPERATION];
  uint8_t outputChannelIndex[MAX_CHANNEL_PER_OPERATION];
  uint32_t inputOffset[MAX_CHANNEL_PER_OPERATION];
  uint32_t outputOffset[MAX_CHANNEL_PER_OPERATION];
  uint32_t srcOffset;
  uint32_t dstOffset;
  uint32_t size;
};

// total size = 1920 + 6400 + 4 + 4(padding) = 8324 bytes
struct DeviceExecutionPlan {
  uint8_t nSmChannels;                  // 1 bytes
  uint8_t nProxyChannels;               // 1 bytes
  uint16_t nOperations;                 // 2 bytes
  Channels channels;                    // 1920 bytes
  Operation operations[MAX_OPERATION];  // 64 * 100 = 6400 bytes
};

class ExecutionKernel {
 public:
  static void launchKernel(int rank, int nthreadblocks, int nthreads, DeviceExecutionPlan* plan, size_t sharedMemSize,
                           cudaStream_t stream);
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_KERNEL_HPP_
