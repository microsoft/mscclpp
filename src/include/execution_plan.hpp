// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_PLAN_HPP_
#define MSCCLPP_EXECUTOR_PLAN_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace mscclpp {

enum class BufferType {
  INPUT,
  OUTPUT,
  SCRATCH,
};

enum class ChannelType {
  SM,
  PROXY,
};

struct ChannelKey {
  BufferType srcBufferType;
  BufferType dstBufferType;
  ChannelType channelType;
  bool operator==(const ChannelKey& other) const {
    return srcBufferType == other.srcBufferType && dstBufferType == other.dstBufferType &&
           channelType == other.channelType;
  }
};
}  // namespace mscclpp

namespace std {
template <>
struct hash<mscclpp::ChannelKey> {
  std::size_t operator()(const mscclpp::ChannelKey& key) const {
    return std::hash<int>()(static_cast<int>(key.srcBufferType)) ^
           std::hash<int>()(static_cast<int>(key.dstBufferType)) ^ std::hash<int>()(static_cast<int>(key.channelType));
  }
};
}  // namespace std

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
  uint16_t nInputChannels;
  uint16_t nOutputChannels;
  uint16_t inputChannelIndex[MAX_CHANNEL_PER_OPERATION];
  uint16_t outputChannelIndex[MAX_CHANNEL_PER_OPERATION];
  size_t inputOffset[MAX_CHANNEL_PER_OPERATION];
  size_t outputOffset[MAX_CHANNEL_PER_OPERATION];
  BufferType srcBufferType;
  BufferType dstBufferType;
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
  Impl(std::string planPath);
  ~Impl() = default;

  std::vector<ChannelInfo> getChannelInfos(int rank, ChannelType channelType) const;
  std::vector<ChannelInfo> getChannelInfos(int rank, BufferType bufferType) const;
  std::vector<int> getConnectedPeers(int rank) const;
  std::vector<BufferType> getConnectedBufferTypes(int rank) const;
  size_t getScratchBufferSize(int rank, size_t inputSize) const;
  std::vector<Operation> getOperations(int rank, int threadblock) const;
  int getThreadblockCount(int rank) const;

  void loadExecutionPlan(size_t inputSize);
  void setupChannels(const nlohmann::json& gpus);
  void setupOperations(const nlohmann::json& gpus);

  std::string planPath;
  // operations for [rank][threadblock] = [operations]
  std::unordered_map<int, std::vector<std::vector<Operation>>> operations;
  std::unordered_map<int, std::vector<ChannelInfo>> channelInfos;
  // threadblockChannelMap[rank][threadblock] = [channelIndex]
  std::unordered_map<int, std::vector<std::vector<std::pair<int, ChannelKey>>>> threadblockSMChannelMap;
  std::unordered_map<int, std::vector<std::vector<std::pair<int, ChannelKey>>>> threadblockProxyChannelMap;
  std::string name;
  std::unordered_map<int, uint32_t> inputChunks;
  std::unordered_map<int, uint32_t> outputChunks;
  std::unordered_map<int, uint32_t> scratchChunks;
  size_t chunkSize;
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_PLAN_HPP_
