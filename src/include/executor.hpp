// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_HPP_
#define MSCCLPP_EXECUTOR_HPP_

#include <memory>
#include <string>

#include "execution_plan.hpp"

namespace mscclpp {

class Executor {
 public:
  Executor();
  template <typename T>
  void execute(std::shared_ptr<T> sendbuff, std::shared_ptr<T> recvBuff, size_t sendBuffSize, size_t recvBuffSize,
               const ExectionPlan& plan);
  ~Executor();

 private:
  struct Impl;

  std::shared_ptr<Impl> impl_;
};

struct Executor::Impl {
  Impl();
  void setupCommnucation(void* sendbuff, void* recvBuff, size_t sendBuffSize, size_t recvBuffSize,
                         const ExecutionPlan& plan);
  void launchKernel();
  ~Impl();
};

}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_HPP_
