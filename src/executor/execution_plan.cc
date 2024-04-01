// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

namespace {
template <typename T, typename Predicate>
std::vector<T> filter(const std::vector<T>& vec, Predicate pred) {
  std::vector<T> filtered;
  std::copy_if(vec.begin(), vec.end(), std::back_inserter(filtered), pred);
  return filtered;
}
}  // namespace

namespace mscclpp {
using json = nlohmann::json;

ExecutionPlan::Impl::Impl(std::ifstream& file) { this->loadExecutionPlan(file); }

std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, ChannelType channelType) const {
  auto pred = [channelType](const ChannelInfo& info) { return info.channelType == channelType; };
  return filter(this->channelInfos.at(rank), pred);
}
std::vector<ChannelInfo> ExecutionPlan::Impl::getChannelInfos(int rank, BufferType dstBufferType) const {
  auto pred = [dstBufferType](const ChannelInfo& info) { return info.dstBufferType == dstBufferType; };
  return filter(this->channelInfos.at(rank), pred);
}

std::vector<BufferType> ExecutionPlan::Impl::getConnectedBufferTypes(int rank) const {
  return std::vector<BufferType>();
}
size_t ExecutionPlan::Impl::getScratchBufferSize(int rank, size_t inputSize) const { return 0; };
std::vector<Operation> ExecutionPlan::Impl::getOperations(int rank, int threadblock) {
  return std::vector<Operation>();
}
std::pair<int, int> ExecutionPlan::Impl::getThreadBlockChannelRange(int rank, int threadblock, BufferType srcBufferType,
                                                                    BufferType dstBufferType, ChannelType channelType) {
  return std::make_pair(0, 0);
}

void ExecutionPlan::Impl::loadExecutionPlan(std::ifstream& file) {
  auto convertToBufferType = [](const std::string& str) {
    if (str == "input") {
      return BufferType::INPUT;
    } else if (str == "output") {
      return BufferType::OUTPUT;
    } else if (str == "scratch") {
      return BufferType::SCRATCH;
    } else {
      throw std::runtime_error("Invalid buffer type");
    }
  };
  auto convertToChannelType = [](const std::string& str) {
    if (str == "sm") {
      return ChannelType::SM;
    } else if (str == "proxy") {
      return ChannelType::PROXY;
    } else {
      throw std::runtime_error("Invalid channel type");
    }
  };

  json obj = json::parse(file);
  this->name = obj["name"];
  this->nranksPerNode = obj["nranksPerNode"];
  auto gpus = obj["gpus"];
  for (const auto& gpu : gpus) {
    int rank = gpu["rank"];
    std::vector<ChannelInfo> channelInfos;
    for (const auto& channel : gpu["channels"]) {
      ChannelInfo info;
      info.srcBufferType = convertToBufferType(channel["srcBuffer"]);
      info.dstBufferType = convertToBufferType(channel["dstBuffer"]);
      info.channelType = convertToChannelType(channel["type"]);
      for (const auto& peer : channel["connectedTo"]) {
        info.connectedPeers.push_back(peer);
      }
      channelInfos.push_back(info);
    }
    this->channelInfos[rank] = channelInfos;
  }
}

ExecutionPlan::ExecutionPlan(std::ifstream& file) : impl_(std::make_shared<Impl>(file)) {}

}  // namespace mscclpp
