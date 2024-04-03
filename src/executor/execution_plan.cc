// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"

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
    return mscclpp::OperationType::BARRIER;
  } else if (str == "put") {
    return mscclpp::OperationType::PUT;
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
  } else if (str == "rr") {
    return mscclpp::OperationType::READ_REDUCE;
  } else if (str == "rrs") {
    return mscclpp::OperationType::READ_REDUCE_SEND;
  } else {
    throw std::runtime_error("Invalid operation type");
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
    throw std::runtime_error("Invalid buffer type");
  }
};

auto convertToChannelType = [](const std::string& str) {
  if (str == "sm") {
    return mscclpp::ChannelType::SM;
  } else if (str == "proxy") {
    return mscclpp::ChannelType::PROXY;
  } else {
    throw std::runtime_error("Invalid channel type");
  }
};

}  // namespace

namespace mscclpp {
using json = nlohmann::json;

ExecutionPlan::Impl::Impl(std::string planPath) : planPath(planPath) {}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, ChannelType channelType) const {
  auto pred = [channelType](const ChannelInfo& info) { return info.channelType == channelType; };
  return filter(this->channelInfos.at(rank), pred);
}

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, BufferType dstBufferType) const {
  auto pred = [dstBufferType](const ChannelInfo& info) { return info.dstBufferType == dstBufferType; };
  return filter(this->channelInfos.at(rank), pred);
}

std::vector<int> ExecutionPlan::Impl::getConnectedPeers(int rank) const {
  std::set<int> peers;
  for (const auto& info : this->channelInfos.at(rank)) {
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
size_t ExecutionPlan::Impl::getScratchBufferSize(int rank, size_t inputSize) const {
  return inputSize / this->inputChunks.at(rank) * this->scratchChunks.at(rank);
}
std::vector<Operation> ExecutionPlan::Impl::getOperations(int rank, int threadblock) const {
  return this->operations.at(rank)[threadblock];
}

int ExecutionPlan::Impl::getThreadblockCount(int rank) const { return this->operations.at(rank).size(); }

void ExecutionPlan::Impl::loadExecutionPlan(size_t inputSize) {
  std::ifstream file(this->planPath);
  json obj = json::parse(file);
  this->name = obj["name"];
  auto gpus = obj["gpus"];

  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    this->inputChunks[rank] = gpu["inputChunks"];
    this->outputChunks[rank] = gpu["outputChunks"];
    this->scratchChunks[rank] = gpu["scratchChunks"];
  }
  this->setupChannels(gpus);

  uint32_t maxInputChunks = 0;
  for (const auto& [rank, chunks] : this->inputChunks) {
    maxInputChunks = std::max(maxInputChunks, chunks);
  }
  this->chunkSize = inputSize / maxInputChunks;
  this->setupOperations(gpus);
}

void ExecutionPlan::Impl::setupChannels(const json& gpus) {
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    std::vector<ChannelInfo> channelInfos;
    for (const auto& channel : gpu["channels"]) {
      ChannelInfo info;
      info.srcBufferType = convertToBufferType(channel["srcbuff"]);
      info.dstBufferType = convertToBufferType(channel["dstbuff"]);
      info.channelType = convertToChannelType(channel["type"]);
      for (const auto& peer : channel["connectedTo"]) {
        info.connectedPeers.push_back(peer);
      }
      channelInfos.push_back(info);
    }
    this->channelInfos[rank] = channelInfos;
  }

  // setup threadblockChannelMap
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    auto channelTypes = {ChannelType::SM, ChannelType::PROXY};
    std::unordered_map<ChannelKey, std::vector<int>> channelMap;
    for (auto channelType : channelTypes) {
      const std::vector<ChannelInfo> channelInfos = this->getChannelInfos(rank, channelType);
      for (size_t i = 0; i < channelInfos.size(); i++) {
        const ChannelInfo& info = channelInfos[i];
        ChannelKey key = {info.srcBufferType, info.dstBufferType, info.channelType};
        channelMap[key].push_back(i);
      }
    }
    int nthreadblocks = gpu["threadblocks"].size();
    this->threadblockSMChannelMap[rank].resize(nthreadblocks);
    this->threadblockProxyChannelMap[rank].resize(nthreadblocks);
    for (const auto& threadblock : gpu["threadblocks"]) {
      for (const auto& channel : threadblock["channels"]) {
        ChannelType channelType = convertToChannelType(channel["ctype"]);
        ChannelKey key = {convertToBufferType(channel["src"]), convertToBufferType(channel["dst"]), channelType};
        for (int id : channel["cids"]) {
          if (channelType == ChannelType::SM) {
            this->threadblockSMChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id], key);
          } else if (channelType == ChannelType::PROXY) {
            this->threadblockProxyChannelMap[rank][threadblock["id"]].emplace_back(channelMap[key][id], key);
          }
        }
      }
    }
  }
}

void ExecutionPlan::Impl::setupOperations(const json& gpus) {
  // setup threadblocks and operations
  for (const auto& gpu : gpus) {
    int rank = gpu["id"];
    for (const auto& threadblock : gpu["threadblocks"]) {
      std::unordered_map<ChannelKey, std::vector<int>> channelIndexes;
      std::vector<Operation> ops;
      int threadblockId = threadblock["id"];
      const auto& smChannels = this->threadblockSMChannelMap[rank][threadblockId];
      const auto& proxyChannels = this->threadblockProxyChannelMap[rank][threadblockId];
      for (size_t i = 0; i < smChannels.size(); i++) {
        const auto& [_, key] = smChannels[i];
        channelIndexes[key].push_back(i);
      }
      for (size_t i = 0; i < proxyChannels.size(); i++) {
        const auto& [_, key] = proxyChannels[i];
        channelIndexes[key].push_back(i);
      }
      for (const auto& op : threadblock["ops"]) {
        Operation operation = {};
        operation.type = static_cast<mscclpp::OperationType>(getOpType(op["name"]));
        if (op.contains("ctype")) {
          operation.channelType = convertToChannelType(op["ctype"]);
        }
        if (op.contains("i_cids")) {
          operation.nInputChannels = op["i_cids"].size();
        }
        if (op.contains("o_cids")) {
          operation.nOutputChannels = op["o_cids"].size();
        }
        for (int i = 0; i < operation.nInputChannels; i++) {
          BufferType srcBufferType = convertToBufferType(op["i_buff"]["src"]);
          BufferType dstBufferType = convertToBufferType(op["i_buff"]["dst"]);
          operation.inputChannelIndex[i] =
              channelIndexes[{srcBufferType, dstBufferType, operation.channelType}][op["i_cids"][i]["id"]];
          operation.inputOffset[i] = this->chunkSize * (int)op["i_cids"][i]["off"];
        }
        for (int i = 0; i < operation.nOutputChannels; i++) {
          BufferType srcBufferType = convertToBufferType(op["o_buff"]["src"]);
          BufferType dstBufferType = convertToBufferType(op["o_buff"]["dst"]);
          operation.outputChannelIndex[i] =
              channelIndexes[{srcBufferType, dstBufferType, operation.channelType}][op["o_cids"][i]["id"]];
          operation.outputOffset[i] = this->chunkSize * (int)op["o_cids"][i]["off"];
        }
        if (op.contains("srcbuff")) {
          operation.srcBufferType = convertToBufferType(op["srcbuff"]);
        }
        if (op.contains("srcoff")) {
          operation.srcOffset = (int)op["srcoff"] * this->chunkSize;
        }
        if (op.contains("dstbuff")) {
          operation.dstBufferType = convertToBufferType(op["dstbuff"]);
        }
        if (op.contains("dstoff")) {
          operation.dstOffset = (int)op["dstoff"] * this->chunkSize;
        }
        if (op.contains("cnt")) {
          operation.size = this->chunkSize * (int)op["cnt"];
        }
        ops.push_back(operation);
      }
      this->operations[rank].push_back(ops);
    }
  }
}

ExecutionPlan::ExecutionPlan(std::string planPath) : impl_(std::make_shared<Impl>(planPath)) {}

}  // namespace mscclpp
