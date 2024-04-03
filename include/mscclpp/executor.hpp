// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_HPP_
#define MSCCLPP_EXECUTOR_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <unordered_map>

namespace mscclpp {

class ExecutionPlan {
 public:
  ExecutionPlan(std::string planPath);
  ~ExecutionPlan() = default;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class Executor;
};

class Executor {
 public:
  Executor(std::shared_ptr<Communicator> comm, int nranksPerNode);
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  ~Executor();

  void execute(int rank, void* sendbuff, void* recvBuff, size_t sendBuffSize, size_t recvBuffSize, int nthreads,
               const ExecutionPlan& plan);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_HPP_
