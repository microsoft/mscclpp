// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_HPP_
#define MSCCLPP_EXECUTOR_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <unordered_map>
#include <unordered_set>

namespace mscclpp {

enum class DataType {
  INT32,
  UINT32,
  FLOAT16,
  FLOAT32,
  BFLOAT16,
};

enum class PacketType {
  LL8,
  LL16,
};

class ExecutionPlan {
 public:
  ExecutionPlan(const std::string& planPath, int rank);
  ~ExecutionPlan() = default;

  std::string name() const;
  std::string collective() const;
  size_t minMessageSize() const;
  size_t maxMessageSize() const;
  bool isInPlace() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class Executor;
};

struct ExecutionPlanHandle {
  struct Constraint {
    int worldSize;
    int nRanksPerNode;
  };
  std::string id;
  Constraint constraint;
  std::shared_ptr<ExecutionPlan> plan;
  std::unordered_set<std::string> tags;
};

struct ExecutionRequest {
  int worldSize;
  int nRanksPerNode;
  void* inputBuffer;
  void* outputBuffer;
  size_t messageSize;
  std::string collective;
  std::unordered_map<std::string, void*> hints;
};

using ExecutionPlanSelector = std::function<ExecutionPlanHandle(const ExecutionRequest& request)>;
class ExecutionPlanRegistry {
 public:
  static std::shared_ptr<ExecutionPlanRegistry> getInstance();
  void registerExecutionPlan(const std::shared_ptr<ExecutionPlan>& plan);
  std::vector<ExecutionPlanHandle> get(const std::string& collective);
  std::shared_ptr<ExecutionPlanHandle> select(const std::string& collective, int worldSize, int nRanksPerNode,
                                              const void* sendBuffer, void* recvBuffer, size_t messageSize,
                                              std::unordered_map<std::string, void*> hints);
  void setSelector(ExecutionPlanSelector selector);
  void setDefaultSelector(ExecutionPlanSelector selector);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ExecutionPlanRegistry() = default;
};

class Executor {
 public:
  Executor(std::shared_ptr<Communicator> comm);
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  ~Executor();

  void execute(int rank, void* sendbuff, void* recvBuff, size_t sendBuffSize, size_t recvBuffSize, DataType dataType,
               const ExecutionPlan& plan, cudaStream_t stream, PacketType packetType = PacketType::LL16);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_HPP_
