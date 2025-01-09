// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"

#include <cassert>
#include <fstream>
#include <set>

namespace {
template <typename T, typename Predicate>
std::vector<T> filter(const std::vector<T>& vec, Predicate pred) {
  std::vector<T> filtered;
  std::copy_if(vec.begin(), vec.end(), std::back_inserter(filtered), pred);
  return filtered;
}

auto getOpType = [](const std::string& str) {
  if (str == "nop") {
    return mscclpp::OperationType::NOP;
  } else if (str == "barrier") {
    return mscclpp::OperationType::BARRIER;
  } else if (str == "put") {
    return mscclpp::OperationType::PUT;
  } else if (str == "pws") {
    return mscclpp::OperationType::PUT_WITH_SIGNAL;
  } else if (str == "pwsf") {
    return mscclpp::OperationType::PUT_WITH_SIGNAL_AND_FLUSH;
  } else if (str == "get") {
    return mscclpp::OperationType::GET;
  } else if (str == "copy") {
    return mscclpp::OperationType::COPY;
  } else if (str == "signal") {
    return mscclpp::OperationType::SIGNAL;
  } else if (str == "wait") {
    return mscclpp::OperationType::WAIT;
  } else if (str == "flush") {
    return mscclpp::OperationType::FLUSH;
  } else if (str == "re") {
    return mscclpp::OperationType::REDUCE;
  } else if (str == "rs") {
    return mscclpp::OperationType::REDUCE_SEND;
  } else if (str == "rrc") {
    return mscclpp::OperationType::READ_REDUCE_COPY;
  } else if (str == "rrcs") {
    return mscclpp::OperationType::READ_REDUCE_COPY_SEND;
  } else if (str == "ppkt") {
    return mscclpp::OperationType::PUT_PACKET;
  } else if (str == "rspkt") {
    return mscclpp::OperationType::REDUCE_SEND_PACKET;
  } else if (str == "cpkt") {
    return mscclpp::OperationType::COPY_PACKET;
  } else if (str == "tpkt") {
    return mscclpp::OperationType::TRANSFORM_TO_PACKET;
  } else if (str == "rpkt") {
    return mscclpp::OperationType::REDUCE_PACKET;
  } else if (str == "glres") {
    return mscclpp::OperationType::MULTI_LOAD_REDUCE_STORE;
  } else {
    throw mscclpp::Error("Invalid operation type", mscclpp::ErrorCode::ExecutorError);
  }
};

auto convertToBufferType = [](const std::string& str) {
  if (str == "i") {
    return mscclpp::BufferType::INPUT;
  } else if (str == "o") {
    return mscclpp::BufferType::OUTPUT;
  } else if (str == "s") {
    return mscclpp::BufferType::SCRATCH;
  } else {
    throw mscclpp::Error("Invalid buffer type", mscclpp::ErrorCode::ExecutorError);
  }
};

auto convertToChannelType = [](const std::string& str) {
  if (str == "sm") {
    return mscclpp::ChannelType::SM;
  } else if (str == "proxy") {
    return mscclpp::ChannelType::PROXY;
  } else if (str == "none") {
    return mscclpp::ChannelType::NONE;
  } else if (str == "nvls") {
    return mscclpp::ChannelType::NVLS;
  } else {
    throw mscclpp::Error("Invalid channel type", mscclpp::ErrorCode::ExecutorError);
  }
};

std::set groupChannelType{mscclpp::ChannelType::NVLS};

}  // namespace

namespace mscclpp {
using json = nlohmann::json;

ExecutionPlan::Impl::Impl(const std::string planPath) : planPath(planPath), isUsingPacket(false) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  this->name = obj["name"];
  this->collective = obj["collective"];
  this->isInPlace = obj["inplace"];
  this->minMessageSize = obj.value("min_message_size", 0);
  this->maxMessageSize = obj.value("max_message_size", std::numeric_limits<uint64_t>::max());
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, ChannelType channelType) const {
  auto pred = [channelType](const ChannelInfo& info) { return info.channelType == channelType; };
  return filter(this->channelInfos.at(rank), pred);
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, BufferType dstBufferType) const {
  auto pred = [dstBufferType](const ChannelInfo& info) { return info.dstBufferType == dstBufferType; };
  return filter(this->channelInfos.at(rank), pred);
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfosByDstRank(int rank, BufferType bufferType) const {
  auto pred = [bufferType](const ChannelInfo& info) { return info.dstBufferType == bufferType; };
  return filter(this->channelInfosByDstRank.at(rank), pred);
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getUnpairedChannelInfos(int rank, int worldSize,
                                                                      ChannelType channelType) {
  std::vector<ChannelInfo> unpaired;
  for (int peer = 0; peer < worldSize; peer++) {
    if (peer == rank) {
      continue;
    }
    if (this->channelCountMap[{rank, channelType}][peer] < this->channelCountMap[{peer, channelType}][rank]) {
      int count = this->channelCountMap[{peer, channelType}][rank] - this->channelCountMap[{rank, channelType}][peer];
      for (int i = 0; i < count; i++) {
        ChannelInfo info;
        info.srcBufferType = BufferType::NONE;
        info.dstBufferType = BufferType::NONE;
        info.channelType = channelType;
        info.connectedPeers.push_back(peer);
        unpaired.push_back(info);
      }
    }
  }
  return unpaired;
}

std::vector<NvlsInfo> ExecutionPlan::Impl::getNvlsInfos(int rank, size_t sendBuffserSize, size_t recvBufferSize) const {
  if (sendBuffserSize == 0 && recvBufferSize == 0) {
    return this->nvlsInfos.at(rank);
  }
  size_t chunkSize = this->getUpperBoundChunkSize(rank, sendBuffserSize, recvBufferSize);
  std::vector<NvlsInfo> infos = this->nvlsInfos.at(rank);
  for (auto& info : infos) {
    info.bufferSize = info.bufferSize * chunkSize;
  }
  return infos;
}

std::vector<int> ExecutionPlan::Impl::getConnectedPeers(int rank) const {
  std::set<int> peers;
  for (const auto& info : this->channelInfos.at(rank)) {
    for (int peer : info.connectedPeers) {
      peers.insert(peer);
    }
  }
  for (const auto& info : this->channelInfosByDstRank.at(rank)) {
    for (int peer : info.connectedPeers) {
      peers.insert(peer);
    }
  }
  return std::vector<int>(peers.begin(), peers.end());
}

std::vector<BufferType> ExecutionPlan::Impl::getConnectedBufferTypes(int rank) const {
  std::set<BufferType> bufferTypes;
  for (const auto& info : this->channelInfos.at(rank)) {
    bufferTypes.insert(info.dstBufferType);
  }
  return std::vector<BufferType>(bufferTypes.begin(), bufferTypes.end());
}

size_t ExecutionPlan::Impl::getScratchBufferSize(int rank, size_t inputSize, size_t outputSize) const {
  size_t sizePerRank = 0;
  if (this->inputChunks.at(rank) != 0)
    sizePerRank = inputSize / this->inputChunks.at(rank);
  else if (this->outputChunks.at(rank) != 0)
    sizePerRank = outputSize / this->outputChunks.at(rank);
  else
    throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);

  if (this->isUsingPacket) {
    return sizePerRank * this->scratchChunks.at(rank) * 2 /* data + flag*/ * 2 /*double buffer*/;
  }
  return sizePerRank * this->scratchChunks.at(rank);
}

size_t ExecutionPlan::Impl::getMaxScratchBufferSize(int rank) const {
  if (this->maxMessageSize == std::numeric_limits<uint64_t>::max()) {
    return std::numeric_limits<size_t>::max();
  }
  size_t sizePerChunk = 0;
  if (this->inputChunks.at(rank) != 0)
    sizePerChunk = maxMessageSize / this->inputChunks.at(rank);
  else if (this->outputChunks.at(rank) != 0)
    sizePerChunk = maxMessageSize / this->outputChunks.at(rank);
  else
    throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);

  return this->getScratchBufferSize(rank, sizePerChunk * this->inputChunks.at(rank),
                                    sizePerChunk * this->outputChunks.at(rank));
}

std::vector<Operation> ExecutionPlan::Impl::getOperations(int rank, int threadblock) const {
  return this->operations.at(rank)[threadblock];
}

int ExecutionPlan::Impl::getThreadblockCount(int rank) const { return this->operations.at(rank).size(); }

int ExecutionPlan::Impl::getNThreadsPerBlock() const { return this->nThreadsPerBlock; }

void ExecutionPlan::Impl::loadExecutionPlan(size_t inputSize, size_t outputSize, size_t contsSrcOffset,
                                            size_t constDstOffset) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  if (this->name != obj["name"]) {
    throw Error("Plan name does not match", ErrorCode::ExecutorError);
  }
  this->collective = obj["collective"];
  std::string protocol = obj["protocol"];
  if (protocol == "LL") {
    this->isUsingPacket = true;
  }
  this->inputSize = inputSize;
  this->outputSize = outputSize;
  this->nThreadsPerBlock = obj.value("num_threads_per_block", 1024);
  this->minMessageSize = obj.value("min_message_size", 0);
  this->maxMessageSize = obj.value("max_message_size", std::numeric_limits<uint64_t>::max());
  this->isInPlace = obj["inplace"];
  const auto& gpus = obj["gpus"];

  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    this->inputChunks[rank] = gpu["inputChunks"];
    this->outputChunks[rank] = gpu["outputChunks"];
    this->scratchChunks[rank] = gpu["scratchChunks"];
    this->chunkGroups[rank] = gpu["chunkGroups"];
  }
  this->setupChannels(gpus);
  this->setupOperations(gpus, contsSrcOffset, constDstOffset);
}

void ExecutionPlan::Impl::lightLoadExecutionPlan(size_t inputSize, size_t outputSize, size_t contsSrcOffset,
                                                 size_t constDstOffset) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  if (this->name != obj["name"]) {
    throw Error("Plan name does not match", ErrorCode::ExecutorError);
  }
  std::string protocol = obj["protocol"];
  if (protocol == "LL") {
    this->isUsingPacket = true;
  }
  const auto& gpus = obj["gpus"];

  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    this->inputChunks[rank] = gpu["inputChunks"];
    this->outputChunks[rank] = gpu["outputChunks"];
    this->scratchChunks[rank] = gpu["scratchChunks"];
    this->chunkGroups[rank] = gpu["chunkGroups"];
  }

  this->inputSize = inputSize;
  this->outputSize = outputSize;
  this->setupOperations(gpus, contsSrcOffset, constDstOffset);
}

void ExecutionPlan::Impl::parseChannels(
    const json& gpu, std::vector<ChannelInfo>& channelInfos, std::vector<NvlsInfo>& nvlsInfos,
    std::map<std::tuple<int, BufferType, BufferType, ChannelType>, std::vector<int>>& chanConnectedPeersMap, int rank) {
  for (const auto& channel : gpu["channels"]) {
    ChannelType chanType = convertToChannelType(channel["type"]);

    if (chanType == ChannelType::NVLS) {
      NvlsInfo info;
      info.bufferType = convertToBufferType(channel["buff"]);
      for (const auto& group : channel["rankGroups"]) {
        info.bufferSize = (int)group["size"];
        info.ranks.clear();
        for (int rank : group["ranks"]) {
          info.ranks.push_back(rank);
        }
        nvlsInfos.push_back(info);
      }
    } else {
      ChannelInfo info;
      info.srcBufferType = convertToBufferType(channel["srcbuff"]);
      info.dstBufferType = convertToBufferType(channel["dstbuff"]);
      info.channelType = convertToChannelType(channel["type"]);
      for (const auto& peer : channel["connectedTo"]) {
        info.connectedPeers.push_back(peer);
        chanConnectedPeersMap[{peer, info.srcBufferType, info.dstBufferType, info.channelType}].push_back(rank);
        this->channelCountMap[{rank, info.channelType}][peer]++;
      }
      channelInfos.push_back(info);
    }
  }
}

// Construct the channel info. Step 1. Flatten SM and PROXY channels into separate vectors.
// Step 2. For each threadblock, construct a vector of channel indexes and keys.
void ExecutionPlan::Impl::setupChannels(const json& gpus) {
  using mapKey = std::tuple<int, BufferType, BufferType, ChannelType>;
  std::map<mapKey, std::vector<int>> chanConnectedPeersMap;
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    std::vector<ChannelInfo> channelInfos;
    std::vector<NvlsInfo> nvlsInfos;
    this->parseChannels(gpu, channelInfos, nvlsInfos, chanConnectedPeersMap, rank);
    this->channelInfos[rank] = channelInfos;
    this->nvlsInfos[rank] = nvlsInfos;
  }

  for (const auto& [key, connectedFrom] : chanConnectedPeersMap) {
    auto [peer, srcBufferType, dstBufferType, channelType] = key;
    ChannelInfo info;
    info.srcBufferType = srcBufferType;
    info.dstBufferType = dstBufferType;
    info.channelType = channelType;
    info.connectedPeers = connectedFrom;
    this->channelInfosByDstRank[peer].push_back(info);
  }

  // setup threadblockChannelMap
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    auto channelTypes = {ChannelType::SM, ChannelType::PROXY, ChannelType::NVLS};
    std::unordered_map<ChannelKey, std::vector<int>> channelMap;
    for (auto channelType : channelTypes) {
      const std::vector<ChannelInfo> channelInfos = this->getChannelInfos(rank, channelType);
      int index = 0;
      if (channelType == ChannelType::NVLS) {
        const std::vector<NvlsInfo> nvlsInfos = this->getNvlsInfos(rank);
        for (const auto& info : nvlsInfos) {
          ChannelKey key = {info.bufferType, info.bufferType, ChannelType::NVLS};
          channelMap[key].push_back(index++);
        }
      } else {
        for (const auto& info : channelInfos) {
          ChannelKey key = {info.srcBufferType, info.dstBufferType, info.channelType};
          for (size_t i = 0; i < info.connectedPeers.size(); i++) {
            channelMap[key].push_back(index++);
          }
        }
      }
    }
    int nthreadblocks = gpu["threadblocks"].size();
    this->threadblockSMChannelMap[rank].resize(nthreadblocks);
    this->threadblockProxyChannelMap[rank].resize(nthreadblocks);
    this->threadblockNvlsChannelMap[rank].resize(nthreadblocks);
    for (const auto& threadblock : gpu["threadblocks"]) {
      for (const auto& channel : threadblock["channels"]) {
        ChannelType channelType = convertToChannelType(channel["ctype"]);
        ChannelKey key = {convertToBufferType(channel["src"]), convertToBufferType(channel["dst"]), channelType};
        for (int id : channel["cids"]) {
          if (channelType == ChannelType::SM) {
            this->threadblockSMChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id], key);
          } else if (channelType == ChannelType::PROXY) {
            this->threadblockProxyChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id], key);
          } else if (channelType == ChannelType::NVLS) {
            this->threadblockNvlsChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id], key);
          }
        }
      }
    }
  }
}

void ExecutionPlan::Impl::setupOperations(const json& gpus, size_t constSrcOffset, size_t constDstOffset) {
  auto getConstOffset = [&](BufferType type) -> size_t {
    switch (type) {
      case BufferType::INPUT:
        return constSrcOffset;
      case BufferType::OUTPUT:
        return constDstOffset;
      case BufferType::SCRATCH:
        return 0;
      default:
        throw Error("Invalid buffer type", ErrorCode::ExecutorError);
    }
  };

  // setup threadblocks and operations
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    for (const auto& threadblock : gpu["threadblocks"]) {
      std::unordered_map<ChannelKey, std::vector<int>> channelIndexes;
      std::vector<Operation> ops;
      int threadblockId = threadblock["id"];
      const auto& smChannels = this->threadblockSMChannelMap[rank][threadblockId];
      const auto& proxyChannels = this->threadblockProxyChannelMap[rank][threadblockId];
      const auto& nvlsChannels = this->threadblockNvlsChannelMap[rank][threadblockId];
      for (size_t i = 0; i < smChannels.size(); i++) {
        const auto& [_, key] = smChannels[i];
        channelIndexes[key].push_back(i);
      }
      for (size_t i = 0; i < proxyChannels.size(); i++) {
        const auto& [_, key] = proxyChannels[i];
        channelIndexes[key].push_back(i);
      }
      for (size_t i = 0; i < nvlsChannels.size(); i++) {
        const auto& [_, key] = nvlsChannels[i];
        channelIndexes[key].push_back(i);
      }
      for (const auto& op : threadblock["ops"]) {
        Operation operation = {};
        std::vector<uint32_t> chunkIndexes;
        operation.type = static_cast<mscclpp::OperationType>(getOpType(op["name"]));
        if (op.contains("ctype")) {
          operation.channelType = convertToChannelType(op["ctype"]);
        }
        if (op.contains("i_cids")) {
          if (operation.channelType == mscclpp::ChannelType::NVLS) {
            BufferType srcBufferType = convertToBufferType(op["srcbuff"]);
            operation.nvlsInputIndex =
                channelIndexes[{srcBufferType, srcBufferType, ChannelType::NVLS}][op["i_cids"][0]["id"]];
            chunkIndexes.push_back((uint32_t)op["srcoff"]);
          } else {
            operation.nInputs = op["i_cids"].size();
            for (int i = 0; i < operation.nInputs; i++) {
              BufferType srcBufferType = convertToBufferType(op["i_buff"]["src"]);
              BufferType dstBufferType = convertToBufferType(op["i_buff"]["dst"]);
              // Get the relevant channel index in rank channelInfos
              operation.inputChannelIndexes[i] =
                  channelIndexes[{srcBufferType, dstBufferType, operation.channelType}][op["i_cids"][i]["id"]];
              operation.inputOffsets[i] =
                  this->getOffset(rank, this->inputSize, this->outputSize, (uint32_t)op["i_cids"][i]["off"]) +
                  getConstOffset(srcBufferType);
              chunkIndexes.push_back((uint32_t)op["i_cids"][i]["off"]);
            }
          }
        }
        // will have either srcs or i_cids
        if (op.contains("srcs")) {
          operation.nInputs = op["srcs"].size();
          operation.inputBufferType = convertToBufferType(op["srcs"][0]["buff"]);
          for (int i = 0; i < operation.nInputs; i++) {
            operation.inputOffsets[i] =
                this->getOffset(rank, this->inputSize, this->outputSize, (uint32_t)op["srcs"][i]["off"]) +
                getConstOffset(operation.inputBufferType);
            chunkIndexes.push_back((uint32_t)op["srcs"][i]["off"]);
          }
        }
        if (op.contains("o_cids")) {
          operation.nOutputs = op["o_cids"].size();
          for (int i = 0; i < operation.nOutputs; i++) {
            if (operation.channelType == mscclpp::ChannelType::NVLS) {
              BufferType dstBufferType = convertToBufferType(op["dstbuff"]);
              operation.nvlsInputIndex =
                  channelIndexes[{dstBufferType, dstBufferType, ChannelType::NVLS}][op["o_cids"][0]["id"]];
              chunkIndexes.push_back((uint32_t)op["dstoff"]);
            } else {
              BufferType srcBufferType = convertToBufferType(op["o_buff"]["src"]);
              BufferType dstBufferType = convertToBufferType(op["o_buff"]["dst"]);
              operation.outputChannelIndexes[i] =
                  channelIndexes[{srcBufferType, dstBufferType, operation.channelType}][op["o_cids"][i]["id"]];
              operation.outputOffsets[i] =
                  this->getOffset(rank, this->inputSize, this->outputSize, (uint32_t)op["o_cids"][i]["off"]) +
                  getConstOffset(dstBufferType);
              chunkIndexes.push_back((uint32_t)op["o_cids"][i]["off"]);
            }
          }
        }
        // will have either dsts or o_cids
        if (op.contains("dsts")) {
          operation.nOutputs = op["dsts"].size();
          operation.outputBufferType = convertToBufferType(op["dsts"][0]["buff"]);
          for (int i = 0; i < operation.nOutputs; i++) {
            operation.outputOffsets[i] =
                this->getOffset(rank, this->inputSize, this->outputSize, (uint32_t)op["dsts"][i]["off"]) +
                getConstOffset(operation.outputBufferType);
            chunkIndexes.push_back((uint32_t)op["dsts"][i]["off"]);
          }
        }
        if (op.contains("srcbuff")) {
          operation.srcBufferType = convertToBufferType(op["srcbuff"]);
        }
        if (op.contains("srcoff")) {
          operation.srcOffset = this->getOffset(rank, this->inputSize, this->outputSize, (uint32_t)op["srcoff"]);
          chunkIndexes.push_back((uint32_t)op["srcoff"]);
        }
        if (op.contains("dstbuff")) {
          operation.dstBufferType = convertToBufferType(op["dstbuff"]);
        }
        if (op.contains("dstoff")) {
          operation.dstOffset = this->getOffset(rank, this->inputSize, this->outputSize, (uint32_t)op["dstoff"]);
          chunkIndexes.push_back((uint32_t)op["dstoff"]);
        }
        if (op.contains("cnt")) {
          operation.size =
              this->getNChunkSize(rank, this->inputSize, this->outputSize, (uint32_t)op["cnt"], chunkIndexes);
        }
        if (op.contains("barrier_id")) {
          operation.deviceSyncerIndex = op["barrier_id"];
        }
        if (op.contains("nthread_blocks")) {
          operation.nThreadBlocks = op["nthread_blocks"];
        }
        ops.push_back(operation);
      }
      this->operations[rank].push_back(ops);
    }
  }
}

std::pair<size_t, uint32_t> ExecutionPlan::Impl::getSizeAndChunksForRank(int rank, size_t inputSize,
                                                                         size_t outputSize) const {
  std::pair<size_t, uint32_t> sizePerRank;
  if (this->inputChunks.at(rank) == 0 && this->outputChunks.at(rank) == 0) {
    throw mscclpp::Error("Output or Input chunks must be greater than 0", mscclpp::ErrorCode::ExecutorError);
  } else if (this->inputChunks.at(rank) != 0 && this->outputChunks.at(rank) != 0) {
    if (inputSize / this->inputChunks.at(rank) != outputSize / this->outputChunks.at(rank))
      throw mscclpp::Error("Size per chunks inconsistent", mscclpp::ErrorCode::ExecutorError);
    else
      sizePerRank = std::make_pair(inputSize, this->inputChunks.at(rank));
  } else if (this->inputChunks.at(rank) != 0) {
    sizePerRank = std::make_pair(inputSize, this->inputChunks.at(rank));
  } else if (this->outputChunks.at(rank) != 0) {
    sizePerRank = std::make_pair(outputSize, this->outputChunks.at(rank));
  }
  return sizePerRank;
}

size_t ExecutionPlan::Impl::getOffset(int rank, size_t inputSize, size_t outputSize, uint32_t chunkIndex,
                                      uint32_t alignment) const {
  if (inputSize % alignment != 0) {
    throw Error("inputSize must be a multiple of alignment", ErrorCode::ExecutorError);
  }

  const int nGroups = this->chunkGroups.at(rank);
  auto rankSizeAndChunks = getSizeAndChunksForRank(rank, inputSize, outputSize);
  uint32_t nChunks = rankSizeAndChunks.second;
  uint32_t nelems = rankSizeAndChunks.first / (alignment * sizeof(uint8_t));
  if (nelems % nGroups != 0) {
    throw Error("Input size must be a multiple of nGroups", ErrorCode::ExecutorError);
  }

  int nelemsPerGroup = nelems / nGroups;
  int nChunksPerGroup = nChunks / nGroups;
  uint32_t minNelems = nelemsPerGroup / nChunksPerGroup;
  uint32_t remainder = nelemsPerGroup % nChunksPerGroup;
  uint32_t groupIdx = chunkIndex / nChunksPerGroup;
  uint32_t chunkIndexInGroup = chunkIndex % nChunksPerGroup;
  uint32_t offset = groupIdx * nelemsPerGroup + chunkIndexInGroup * minNelems +
                    (chunkIndexInGroup % nelemsPerGroup < remainder ? chunkIndexInGroup % nelemsPerGroup : remainder);
  return static_cast<size_t>(offset) * alignment;
}

size_t ExecutionPlan::Impl::getNChunkSize(int rank, size_t inputSize, size_t outputSize, uint32_t nChunks,
                                          const std::vector<uint32_t> chunkIndexes) const {
  size_t nChunkSize = 0;
  for (uint32_t index : chunkIndexes) {
    uint32_t beginOff = getOffset(rank, inputSize, outputSize, index);
    uint32_t endOff = getOffset(rank, inputSize, outputSize, index + nChunks);
    if (nChunkSize == 0) {
      nChunkSize = endOff - beginOff;
    } else if (nChunkSize != endOff - beginOff) {
      throw Error("Inconsistent chunk size", ErrorCode::ExecutorError);
    }
  }
  return nChunkSize;
}

size_t ExecutionPlan::Impl::getUpperBoundChunkSize(int rank, size_t inputSize, size_t outputSize) const {
  size_t nInputChunks = this->inputChunks.at(rank);
  size_t nOutputChunks = this->outputChunks.at(rank);
  size_t inputChunkSize = 0;
  size_t outputChunkSize = 0;
  if (nInputChunks != 0) {
    inputChunkSize = inputSize / nInputChunks;
  }
  if (nOutputChunks != 0) {
    outputChunkSize = outputSize / nOutputChunks;
  }
  return std::max(inputChunkSize, outputChunkSize);
}

void ExecutionPlan::Impl::reset() {
  this->operations.clear();
  this->channelInfos.clear();
  this->nvlsInfos.clear();
  this->threadblockSMChannelMap.clear();
  this->threadblockProxyChannelMap.clear();
  this->threadblockNvlsChannelMap.clear();
  this->inputChunks.clear();
  this->outputChunks.clear();
  this->scratchChunks.clear();
  this->chunkGroups.clear();
}

void ExecutionPlan::Impl::operationsReset() { this->operations.clear(); }

ExecutionPlan::ExecutionPlan(const std::string& planPath) : impl_(std::make_shared<Impl>(planPath)) {}

std::string ExecutionPlan::name() const { return this->impl_->name; }

std::string ExecutionPlan::collective() const { return this->impl_->collective; }

size_t ExecutionPlan::minMessageSize() const { return this->impl_->minMessageSize; }

size_t ExecutionPlan::maxMessageSize() const { return this->impl_->maxMessageSize; }

bool ExecutionPlan::isInPlace() const { return this->impl_->isInPlace; }

}  // namespace mscclpp
