// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_HPP_
#define MSCCLPP_EXECUTOR_HPP_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "execution_plan.hpp"

namespace mscclpp {

class Executor {
 public:
  Executor(const std::unordered_map<int, mscclpp::Connection> connections);
  template <typename T>
  void execute(std::shared_ptr<T> sendbuff, std::shared_ptr<T> recvBuff, size_t sendBuffSize, size_t recvBuffSize,
               const ExecutionPlan& plan);
  ~Executor();

 private:
  struct Impl;

  std::shared_ptr<Impl> impl_;
};

struct ExecutionContext {
  std::unordered_map<std::pair<BufferType, int>, std::vector<mscclpp::RegisteredMemory>> registeredMemories;
  std::vector<mscclpp::SmChannel> smChannels;
  std::vector<mscclpp::ProxyChannel> proxyChannels;
  std::vector<DeviceExecutionPlan> deviceExecutionPlans;
  std::shared_ptr<char> scratchBuffer;
};

struct ExecutionPlanKey {
  void* sendBuff;
  void* recvBuff;
  size_t sendBuffSize;
  size_t recvBuffSize;
  std::string plan;
};

struct Executor::Impl {
  std::unordered_map<ExecutionPlanKey, ExecutionContext> contexts;
  const std::unordered_map<int, mscclpp::Connection> connections;
  std::shared_ptr<mscclpp::Communicator> comm;

  Impl(const std::unordered_map<int, mscclpp::Connection> connections);
  ExecutionContext setupExecutionContext(int rank, void* sendbuff, void* recvBuff, size_t sendBuffSize,
                                         size_t recvBuffSize, const ExecutionPlan& plan);
  void setupRegisteredMemories(ExecutionContext& context, int rank, const ExecutionPlan& plan);
  void setupChannels(ExecutionContext& context, int rank, const ExecutionPlan& plan);
  void launchKernel();
  ~Impl();
};

}  // namespace mscclpp

namespace std {
template <>
struct hash<mscclpp::ExecutionPlanKey> {
  std::size_t operator()(const mscclpp::ExecutionPlanKey& key) const {
    return std::hash<void*>()(key.sendBuff) ^ std::hash<void*>()(key.recvBuff) ^ std::hash<size_t>()(key.sendBuffSize) ^
           std::hash<size_t>()(key.recvBuffSize) ^ std::hash<std::string>()(key.plan);
  }
};
}  // namespace std

#endif  // MSCCLPP_EXECUTOR_HPP_
