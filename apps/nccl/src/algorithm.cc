// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>

namespace mscclpp {

class AlgorithmImpl {
 public:
  AlgorithmImpl(std::shared_ptr<mscclpp::Communicator> comm, Algorithm::KernelFunc kernelFunc,
                Algorithm::ContextInitFunc contextInitFunc, Algorithm::ContextKeyGenFunc contextKeyGenFunc)
      : comm(comm),
        kernelLaunchFunc(kernelFunc),
        contextInitFunc(contextInitFunc),
        contextKeyGenFunc(contextKeyGenFunc) {}
  ncclResult_t launch(const void* input, void* output, size_t count, ncclDataType_t dtype, cudaStream_t stream,
                      std::unordered_map<std::string, std::shared_ptr<void>>& extras);

 private:
  std::shared_ptr<mscclpp::Communicator> comm;
  Algorithm::KernelFunc kernelLaunchFunc;
  Algorithm::ContextInitFunc contextInitFunc;
  Algorithm::ContextKeyGenFunc contextKeyGenFunc;

  std::unordered_map<AlgorithmCtxKey, std::shared_ptr<AlgorithmCtx>> contexts;
};

ncclResult_t AlgorithmImpl::launch(const void* input, void* output, size_t count, ncclDataType_t dtype,
                                   cudaStream_t stream,
                                   std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  AlgorithmCtxKey ctxKey = contextKeyGenFunc(input, output, count, dtype);
  auto it = contexts.find(ctxKey);
  if (it == contexts.end()) {
    auto ctx = contextInitFunc(comm, input, output, count, dtype);
    contexts[ctxKey] = ctx;
  }
  return kernelLaunchFunc(contexts[ctxKey], input, output, count, dtype, stream, extras);
}

Algorithm::Algorithm(std::shared_ptr<Communicator> comm, std::string name, KernelFunc kernelFunc,
                     ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc)
    : name(name), impl(std::make_shared<AlgorithmImpl>(comm, kernelFunc, contextInitFunc, contextKeyGenFunc)) {}

ncclResult_t Algorithm::launch(const void* input, void* output, size_t count, ncclDataType_t dtype, cudaStream_t stream,
                                std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  return this->impl->launch(input, output, count, dtype, stream, extras);
}

bool Algorithm::isEmpty() { return !impl; }

void AlgorithmFactory::registerAlgorithm(const std::string collective, const std::string algoName,
                                         Algorithm algorithm) {
  getInstance()->algoMapByCollective[collective][algoName] = algorithm;
}

Algorithm AlgorithmFactory::selectAlgorithm(const std::string& collective, size_t messageSizes, const void* input,
                                            void* output) {
  for (const auto& selector : algoSelectors) {
    Algorithm algo = selector(algoMapByCollective, collective, messageSizes, input, output);
    if (!algo.isEmpty()) {
      return algo;
    }
  }
  return Algorithm();
}

void AlgorithmFactory::addAlgorithmSelector(AlgoSelectFunc selector) { algoSelectors.push_back(selector); }

void AlgorithmFactory::destroy() {
  algoMapByCollective.clear();
  algoSelectors.clear();
}

}  // namespace mscclpp