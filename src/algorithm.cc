// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>

namespace mscclpp {

class AlgorithmImpl {
 public:
  AlgorithmImpl(std::string name, Algorithm::InitFunc initFunc, Algorithm::KernelFunc kernelFunc,
                Algorithm::ContextInitFunc contextInitFunc, Algorithm::ContextKeyGenFunc contextKeyGenFunc)
      : name_(name),
        initFunc_(initFunc),
        kernelLaunchFunc_(kernelFunc),
        contextInitFunc_(contextInitFunc),
        contextKeyGenFunc_(contextKeyGenFunc) {}
  int launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count, int dtype,
             cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras);

 private:
  std::string name_;
  Algorithm::InitFunc initFunc_;
  Algorithm::KernelFunc kernelLaunchFunc_;
  Algorithm::ContextInitFunc contextInitFunc_;
  Algorithm::ContextKeyGenFunc contextKeyGenFunc_;

  bool initialized_ = false;
  std::unordered_map<AlgorithmCtxKey, std::shared_ptr<AlgorithmCtx>> contexts_;
};

int AlgorithmImpl::launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
                          int dtype, cudaStream_t stream,
                          std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  if (!initialized_) {
    initFunc_(comm);
    initialized_ = true;
  }
  AlgorithmCtxKey ctxKey = contextKeyGenFunc_(input, output, count, dtype);
  auto it = contexts_.find(ctxKey);
  if (it == contexts_.end()) {
    auto ctx = contextInitFunc_(comm, input, output, count, dtype);
    contexts_[ctxKey] = ctx;
  }
  return kernelLaunchFunc_(contexts_[ctxKey], input, output, count, dtype, stream, extras);
}

Algorithm::Algorithm(std::string name, InitFunc initFunc, KernelFunc kernelFunc, ContextInitFunc contextInitFunc,
                     ContextKeyGenFunc contextKeyGenFunc)
    : impl_(std::make_shared<AlgorithmImpl>(name, initFunc, kernelFunc, contextInitFunc, contextKeyGenFunc)) {}

int Algorithm::launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
                      int dtype, cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  return this->impl_->launch(comm, input, output, count, dtype, stream, extras);
}

bool Algorithm::isEmpty() { return !impl_; }

void AlgorithmFactory::registerAlgorithm(const std::string collective, const std::string algoName,
                                         Algorithm algorithm) {
  getInstance()->algoMapByCollective_[collective][algoName] = algorithm;
}

Algorithm AlgorithmFactory::selectAlgorithm(const std::string& collective, size_t messageSize, int nRanksPerNode,
                                            int worldSize) {
  Algorithm algo;
  if (algoSelector_) {
    algo = algoSelector_(algoMapByCollective_, collective, messageSize, nRanksPerNode, worldSize);
  }
  if (algo.isEmpty()) {
    algo = fallbackAlgoSelector_(algoMapByCollective_, collective, messageSize, nRanksPerNode, worldSize);
  }
  return algo;
}

void AlgorithmFactory::setAlgorithmSelector(AlgoSelectFunc selector) { algoSelector_ = selector; }

void AlgorithmFactory::setFallbackAlgorithmSelector(AlgoSelectFunc selector) { fallbackAlgoSelector_ = selector; }

void AlgorithmFactory::destroy() {
  algoMapByCollective_.clear();
  algoSelector_ = nullptr;
  fallbackAlgoSelector_ = nullptr;
}

}  // namespace mscclpp