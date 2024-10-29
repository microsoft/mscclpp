// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_COMMON_HPP_
#define MSCCLPP_EXECUTION_COMMON_HPP_

#include <mscclpp/nvls.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

namespace mscclpp {

constexpr int MAX_CHANNEL = 16;
constexpr int MAX_CHANNEL_PER_OPERATION = 8;
constexpr int MAX_OPERATION = 64;
// input-input, output-output, scratch-scratch
constexpr int MAX_NLVS_CHANNELS = 3;

enum class BufferType : uint8_t {
  NONE,
  INPUT,
  OUTPUT,
  SCRATCH,
};

enum class ChannelType : uint8_t {
  NONE,
  SM,
  PROXY,
};

// NOTE(chhwang): any modification here requires corresponding updates in `tools/npkit/npkit_trace_generator.py`.
enum class OperationType : uint8_t {
  BARRIER,
  PUT,
  PUT_PACKET,
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
};

struct Channels {
  mscclpp::DeviceHandle<mscclpp::SmChannel> smChannels[MAX_CHANNEL];
  mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel> proxyChannels[MAX_CHANNEL];
  mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer> nvlsChannels[MAX_NLVS_CHANNELS];
};

struct Operation {
  OperationType type;
  ChannelType channelType;
  BufferType srcBufferType;
  BufferType dstBufferType;
  uint8_t nInputs;
  uint8_t nOutputs;
  union {
    // For ops which require reading from multiple remote sources
    uint8_t inputChannelIndexes[MAX_CHANNEL_PER_OPERATION];
    // For ops which require reading from multiple local sources
    BufferType inputBufferType;
    uint8_t nvlsInputIndex;
  };
  union {
    // For ops which require writing to multiple remote destinations
    uint8_t outputChannelIndexes[MAX_CHANNEL_PER_OPERATION];
    // For ops which require writing to multiple local destinations
    BufferType outputBufferType;
    uint8_t nvlsOutputIndex;
  };
  uint32_t inputOffsets[MAX_CHANNEL_PER_OPERATION];
  uint32_t outputOffsets[MAX_CHANNEL_PER_OPERATION];
  uint32_t srcOffset;
  uint32_t dstOffset;
  uint32_t size;
};

// TODO(binyli): update it
// total size = 1944 + 6400 + 4 + 4(padding) + 12(align) = 8336 bytes
struct __attribute__((aligned(16))) DeviceExecutionPlan {
  uint8_t nSmChannels;                  // 1 bytes
  uint8_t nProxyChannels;               // 1 bytes
  uint16_t nOperations;                 // 2 bytes
  Channels channels;                    // 1944 bytes
  Operation operations[MAX_OPERATION];  // 64 * 100 = 6400 bytes
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_COMMON_HPP_
