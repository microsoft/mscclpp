// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>

namespace mscclpp {

class AlgorithmImpl {
 public:
  AlgorithmImpl(Algorithm::KernelFunc kernelFunc, Algorithm::ContextInitFunc contextInitFunc,
                Algorithm::ContextKeyGenFunc contextKeyGenFunc)
      : kernelLaunchFunc(kernelFunc), contextInitFunc(contextInitFunc), contextKeyGenFunc(contextKeyGenFunc) {}
  int launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count, int dtype,
             cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras);

 private:
  Algorithm::KernelFunc kernelLaunchFunc;
  Algorithm::ContextInitFunc contextInitFunc;
  Algorithm::ContextKeyGenFunc contextKeyGenFunc;

  std::unordered_map<AlgorithmCtxKey, std::shared_ptr<AlgorithmCtx>> contexts;
};

int AlgorithmImpl::launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
                          int dtype, cudaStream_t stream,
                          std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  AlgorithmCtxKey ctxKey = contextKeyGenFunc(input, output, count, dtype);
  auto it = contexts.find(ctxKey);
  if (it == contexts.end()) {
    auto ctx = contextInitFunc(comm, input, output, count, dtype);
    contexts[ctxKey] = ctx;
  }
  return kernelLaunchFunc(contexts[ctxKey], input, output, count, dtype, stream, extras);
}

Algorithm::Algorithm(std::string name, KernelFunc kernelFunc, ContextInitFunc contextInitFunc,
                     ContextKeyGenFunc contextKeyGenFunc)
    : name(name), impl(std::make_shared<AlgorithmImpl>(kernelFunc, contextInitFunc, contextKeyGenFunc)) {}

int Algorithm::launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
                      int dtype, cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  return this->impl->launch(comm, input, output, count, dtype, stream, extras);
}

bool Algorithm::isEmpty() { return !impl; }

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