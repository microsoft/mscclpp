// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_HPP_
#define MSCCLPP_EXECUTOR_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <unordered_map>

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

struct ExecutionRequest {
  int worldSize;
  int nRanksPerNode;
  int rank;
  const void* inputBuffer;
  void* outputBuffer;
  size_t messageSize;
  const std::string& collective;
  const std::unordered_map<std::string, std::vector<uint64_t>>& hints;

  bool isInPlace() const;
};

struct ExecutionPlanHandle {
  struct Constraint {
    int worldSize;
    int nRanksPerNode;
  };
  std::string id;
  Constraint constraint;
  std::shared_ptr<ExecutionPlan> plan;
  std::unordered_map<std::string, uint64_t> tags;

  static std::shared_ptr<ExecutionPlanHandle> create(const std::string& id, int worldSize, int nRanksPerNode,
                                                     std::shared_ptr<ExecutionPlan> plan,
                                                     const std::unordered_map<std::string, uint64_t>& tags = {});
  bool match(const ExecutionRequest& request);
};

using ExecutionPlanSelector = std::function<std::shared_ptr<ExecutionPlanHandle>(
    const std::vector<std::shared_ptr<ExecutionPlanHandle>> plans, const ExecutionRequest& request)>;
class ExecutionPlanRegistry {
 public:
  static std::shared_ptr<ExecutionPlanRegistry> getInstance();
  ~ExecutionPlanRegistry();

  void registerPlan(const std::shared_ptr<ExecutionPlanHandle> planHandle);
  std::vector<std::shared_ptr<ExecutionPlanHandle>> getPlans(const std::string& collective);
  std::shared_ptr<ExecutionPlanHandle> get(const std::string& id);
  std::shared_ptr<ExecutionPlanHandle> select(const std::string& collective, int worldSize, int nRanksPerNode, int rank,
                                              const void* sendBuffer, void* recvBuffer, size_t messageSize,
                                              const std::unordered_map<std::string, std::vector<uint64_t>>& hints);
  void setSelector(ExecutionPlanSelector selector);
  void setDefaultSelector(ExecutionPlanSelector selector);
  void loadDefaultPlans(int rank);
  void clear();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ExecutionPlanRegistry();
};

class Executor {
 public:
  Executor(std::shared_ptr<Communicator> comm, std::shared_ptr<char> defaultScratchBuffer = nullptr);
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
