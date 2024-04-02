// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_PLAN_HPP_
#define MSCCLPP_EXECUTOR_PLAN_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>
#include <string>
#include <unordered_map>

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
  mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel> proxyChannels[MAX_CHANNEL];
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

struct ExecutionPlan::Impl {
 public:
  Impl(std::ifstream& file);
  ~Impl() = default;

  std::vector<ChannelInfo> getChannelInfos(int rank, ChannelType channelType) const;
  std::vector<ChannelInfo> getChannelInfos(int rank, BufferType bufferType) const;
  std::vector<int> getConnectedPeers(int rank) const;
  std::vector<BufferType> getConnectedBufferTypes(int rank) const;
  size_t getScratchBufferSize(int rank, size_t inputSize) const;
  std::vector<Operation> getOperations(int rank, int threadblock);
  std::pair<int, int> getThreadBlockChannelRange(int rank, int threadblock, BufferType srcBufferType,
                                                 BufferType dstBufferType, ChannelType channelType);
  void loadExecutionPlan(std::ifstream& file);

  // operations for [rank][threadblock]
  std::vector<std::vector<Operation>> operations;
  std::unordered_map<int, std::vector<ChannelInfo>> channelInfos;
  std::string name;
  std::unordered_map<int, uint32_t> inputChunks;
  std::unordered_map<int, uint32_t> outputChunks;
  std::unordered_map<int, uint32_t> scratchChunks;
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_PLAN_HPP_
