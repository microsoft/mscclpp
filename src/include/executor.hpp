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
struct ExecutionContextKey {
  void* sendBuff;
  void* recvBuff;
  size_t sendBuffSize;
  size_t recvBuffSize;
  std::string plan;
};
}  // namespace mscclpp

namespace std {
template <>
struct hash<std::pair<mscclpp::BufferType, int>> {
  std::size_t operator()(const std::pair<mscclpp::BufferType, int>& key) const {
    return std::hash<int>()(key.second) ^ std::hash<int>()(static_cast<int>(key.first));
  }
};

template <>
struct hash<mscclpp::ExecutionContextKey> {
  std::size_t operator()(const mscclpp::ExecutionContextKey& key) const {
    return std::hash<void*>()(key.sendBuff) ^ std::hash<void*>()(key.recvBuff) ^ std::hash<size_t>()(key.sendBuffSize) ^
           std::hash<size_t>()(key.recvBuffSize) ^ std::hash<std::string>()(key.plan);
  }
};
}  // namespace std

namespace mscclpp {
class Executor {
 public:
  Executor(std::shared_ptr<Communicator> comm, const std::unordered_map<int, mscclpp::Connection> connections);
  void execute(void* sendbuff, void* recvBuff, size_t sendBuffSize, size_t recvBuffSize, const ExecutionPlan& plan);
  ~Executor() = default;

 private:
  struct Impl;

  std::shared_ptr<Impl> impl_;
};

struct ExecutionContext {
  std::unordered_map<std::pair<BufferType, int>, mscclpp::RegisteredMemory> registeredMemories;
  std::vector<std::shared_ptr<mscclpp::SmDevice2DeviceSemaphore>> smSemaphores;
  std::vector<mscclpp::SemaphoreId> proxySemaphores;
  std::vector<mscclpp::SmChannel> smChannels;
  std::vector<mscclpp::SimpleProxyChannel> proxyChannels;
  std::vector<DeviceExecutionPlan> deviceExecutionPlans;
  std::shared_ptr<char> scratchBuffer;
  size_t scratchBufferSize;
};

struct Executor::Impl {
  std::shared_ptr<Communicator> comm;
  const std::unordered_map<int, std::shared_ptr<Connection>> connections;
  std::shared_ptr<ProxyService> proxyService;
  std::unordered_map<ExecutionContextKey, ExecutionContext> contexts;

  Impl(std::shared_ptr<Communicator> comm, const std::unordered_map<int, std::shared_ptr<Connection>> connections);
  ExecutionContext setupExecutionContext(int rank, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                                         size_t recvBufferSize, const ExecutionPlan& plan);
  void setupRegisteredMemories(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize,
                               size_t recvBufferSize, int rank, const ExecutionPlan& plan);
  void setupChannels(ExecutionContext& context, void* sendbuff, void* recvbuff, size_t sendBufferSize, int rank,
                     const ExecutionPlan& plan);
  void launchKernel(ExecutionContext& context);
  ~Impl() = default;
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_HPP_
