// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_COMMON_HPP_
#define MSCCLPP_EXECUTION_COMMON_HPP_

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
  READ_REDUCE_COPY,
  READ_REDUCE_COPY_SEND,
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
  uint8_t inputChannelIndexes[MAX_CHANNEL_PER_OPERATION];
  uint8_t outputChannelIndexes[MAX_CHANNEL_PER_OPERATION];
  uint32_t inputOffsets[MAX_CHANNEL_PER_OPERATION];
  uint32_t outputOffsets[MAX_CHANNEL_PER_OPERATION];
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

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_COMMON_HPP_
