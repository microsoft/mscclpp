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

enum class BufferType : uint8_t {
  INPUT,
  OUTPUT,
  SCRATCH,
};

enum class ChannelType : uint8_t {
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

constexpr int MAX_CHANNEL = 16;
constexpr int MAX_CHANNEL_PER_OPERATION = 8;
constexpr int MAX_OPERATION = 64;

enum class OperationType : uint8_t {
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
