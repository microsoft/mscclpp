// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"

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

ExecutionPlan::ExecutionPlan(std::ifstream& file) { this->loadExecutionPlan(file); }

std::string ExecutionPlan::getName() const { return this->name_; }

int ExecutionPlan::nranksPerNode() const { return this->nranksPerNode_; }

std::vector<ChannelInfo> ExecutionPlan::getChannelInfos(int rank, ChannelType channelType) const {
  auto pred = [channelType](const ChannelInfo& info) { return info.channelType == channelType; };
  return filter(this->channelInfos_.at(rank), pred);
}

std::vector<ChannelInfo> ExecutionPlan::getChannelInfos(int rank, BufferType dstBufferType) const {
  auto pred = [dstBufferType](const ChannelInfo& info) { return info.dstBufferType == dstBufferType; };
  return filter(this->channelInfos_.at(rank), pred);
}

void ExecutionPlan::loadExecutionPlan(std::ifstream& file) {
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
  this->name_ = obj["name"];
  this->nranksPerNode_ = obj["nranksPerNode"];
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
    this->channelInfos_[rank] = channelInfos;
  }
}

}  // namespace mscclpp
