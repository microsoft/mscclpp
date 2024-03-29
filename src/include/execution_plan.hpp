// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_PLAN_HPP_
#define MSCCLPP_EXECUTOR_PLAN_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include <string>

namespace mscclpp {

constexpr int MAX_CHANNEL = 24;
constexpr int MAX_CHANNEL_PER_OPERATION = 8;

enum class OperationType {
  BARRIER,
  PUT,
  GET,
  COPY,
  SIGNAL,
  WAIT,
  FLUSH,
  REDUCE,
  READ_REDUCE_COPY,
  READ_REDUCE_COPY_PUT,
};

enum class ChannelType {
  SM,
  PROXY,
};

enum class BufferType {
  INPUT,
  OUTPUT,
  SCRATCH,
};

struct ChannelInfo {
  BufferType srcBufferType;
  BufferType dstBufferType;
  ChannelType channelType;
  std::vector<int> connectedPeers;
};

struct Channels {
  mscclpp::DeviceHandle<mscclpp::SmChannel> smChannels[MAX_CHANNEL];
  mscclpp::DeviceHandle<mscclpp::ProxyChannel> proxyChannels[MAX_CHANNEL];
};

struct Operation {
  OperationType type;
  ChannelType channelType;
  uint16_t inputChannelIndex[MAX_CHANNEL_PER_OPERATION];
  uint16_t outputChannelIndex[MAX_CHANNEL_PER_OPERATION];
  size_t inputOffset[MAX_CHANNEL_PER_OPERATION];
  size_t outputOffset[MAX_CHANNEL_PER_OPERATION];
  size_t srcOffset;
  size_t dstOffset;
  size_t size;
};

struct DeviceExecutionPlan {
  int nSmChannels;
  int nProxyChannels;
  int nOperations;
  Channels channels;
  Operation operations[1];
};

class ExecutionPlan {
 public:
  ExecutionPlan();
  void loadExecutionPlan(std::ifstream& file);
  std::vector<int> getConnectedPeers(int rank);
  size_t getScratchSize(size_t inputSize);
  std::vector<Operation> getOperations(int rank, int threadblock);
  std::pair<int, int> getThreadBlockChannelRange(int rank, int threadblock, BufferType srcBufferType,
                                                 BufferType dstBufferType, ChannelType channelType);
  ~ExecutionPlan();

 private:
  // operations for [rank][threadblock]
  std::vector<std::vector<Operation>> operations_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_PLAN_HPP_
