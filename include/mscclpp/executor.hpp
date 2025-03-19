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
  ExecutionPlan(const std::string& planPath);
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
