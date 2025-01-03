// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_PLAN_HPP_
#define MSCCLPP_EXECUTOR_PLAN_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "execution_common.hpp"

namespace mscclpp {

struct ChannelKey {
  BufferType srcBufferType;
  BufferType dstBufferType;
  ChannelType channelType;
  bool operator==(const ChannelKey& other) const {
    return srcBufferType == other.srcBufferType && dstBufferType == other.dstBufferType &&
           channelType == other.channelType;
  }
};

struct NvlsInfo {
  std::vector<int> ranks;
  size_t bufferSize;
  BufferType bufferType;
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

template <>
struct hash<std::pair<int, mscclpp::ChannelType>> {
  std::size_t operator()(const std::pair<int, mscclpp::ChannelType>& key) const {
    std::size_t h1 = std::hash<int>()(key.first);
    std::size_t h2 = std::hash<int>()(static_cast<int>(key.second));
    // Refer hash_combine from boost
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
  }
};
}  // namespace std

namespace mscclpp {

struct ChannelInfo {
  BufferType srcBufferType;
  BufferType dstBufferType;
  ChannelType channelType;
  std::vector<int> connectedPeers;
};

struct ExecutionPlan::Impl {
 public:
  Impl(const std::string planPath);
  ~Impl() = default;

  std::vector<ChannelInfo> getChannelInfos(int rank, ChannelType channelType) const;
  std::vector<ChannelInfo> getChannelInfos(int rank, BufferType bufferType) const;
  std::vector<ChannelInfo> getChannelInfosByDstRank(int rank, BufferType bufferType) const;
  std::vector<ChannelInfo> getUnpairedChannelInfos(int rank, int worldSize, ChannelType channelType);
  std::vector<NvlsInfo> getNvlsInfos(int rank, size_t sendBuffserSize = 0, size_t recvBufferSize = 0) const;
  std::vector<int> getConnectedPeers(int rank) const;
  std::vector<BufferType> getConnectedBufferTypes(int rank) const;
  size_t getScratchBufferSize(int rank, size_t inputSize, size_t outputSize) const;
  size_t getMaxScratchBufferSize(int rank) const;
  std::vector<Operation> getOperations(int rank, int threadblock) const;
  int getThreadblockCount(int rank) const;
  int getNThreadsPerBlock() const;

  void loadExecutionPlan(size_t inputSize, size_t outputSize, size_t contsSrcOffset, size_t constDstOffset);
  void lightLoadExecutionPlan(size_t inputSize, size_t outputSize, size_t contsSrcOffset, size_t constDstOffset);
  void setupChannels(const nlohmann::json& gpus);
  void setupOperations(const nlohmann::json& gpus, size_t contsSrcOffset, size_t constDstOffset);

  void reset();
  void operationsReset();

  std::string name;
  std::string collective;
  const std::string planPath;
  bool isUsingPacket;
  // operations for [rank][threadblock] = [operations]
  std::unordered_map<int, std::vector<std::vector<Operation>>> operations;
  std::unordered_map<int, std::vector<ChannelInfo>> channelInfos;
  std::unordered_map<int, std::vector<ChannelInfo>> channelInfosByDstRank;
  std::unordered_map<std::pair<int, ChannelType>, std::unordered_map<int, int>> channelCountMap;
  // for nvls channels
  std::unordered_map<int, std::vector<NvlsInfo>> nvlsInfos;
  // threadblockChannelMap[rank][threadblock] = [channelIndex, channelKey]
  std::unordered_map<int, std::vector<std::vector<std::pair<int, ChannelKey>>>> threadblockSMChannelMap;
  std::unordered_map<int, std::vector<std::vector<std::pair<int, ChannelKey>>>> threadblockProxyChannelMap;
  std::unordered_map<int, std::vector<std::vector<std::pair<int, ChannelKey>>>> threadblockNvlsChannelMap;
  std::unordered_map<int, uint32_t> inputChunks;
  std::unordered_map<int, uint32_t> outputChunks;
  std::unordered_map<int, uint32_t> scratchChunks;
  std::unordered_map<int, uint32_t> chunkGroups;
  size_t inputSize;
  size_t outputSize;
  int nThreadsPerBlock;
  size_t minMessageSize;
  size_t maxMessageSize;
  bool isInPlace;

 private:
  std::pair<size_t, uint32_t> getSizeAndChunksForRank(int rank, size_t inputSize, size_t outputSize) const;
  size_t getOffset(int rank, size_t inputSize, size_t outputSize, uint32_t chunkIndex, uint32_t alignment = 16) const;
  size_t getNChunkSize(int rank, size_t inputSize, size_t outputSize, uint32_t nChunks,
                       const std::vector<uint32_t> offsets) const;
  size_t getUpperBoundChunkSize(int rank, size_t inputSize, size_t outputSize) const;

  // helper functions to setup the channels
  void parseChannels(
      const nlohmann::json& gpu, std::vector<ChannelInfo>& channelInfos, std::vector<NvlsInfo>& nvlsInfos,
      std::map<std::tuple<int, BufferType, BufferType, ChannelType>, std::vector<int>>& chanConnectedPeersMap,
      int rank);
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_PLAN_HPP_
