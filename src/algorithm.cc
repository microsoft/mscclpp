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
  cudaError_t launch(void* input, void* output, size_t count, uint32_t dtype, cudaStream_t stream);

 private:
  std::shared_ptr<mscclpp::Communicator> comm;
  Algorithm::KernelFunc kernelLaunchFunc;
  Algorithm::ContextInitFunc contextInitFunc;
  Algorithm::ContextKeyGenFunc contextKeyGenFunc;

  std::unordered_map<AlgorithmCtxKey, std::shared_ptr<AlgorithmCtx>> contexts;
};

cudaError_t AlgorithmImpl::launch(void* input, void* output, size_t count, uint32_t dtype, cudaStream_t stream) {
  AlgorithmCtxKey ctxKey = contextKeyGenFunc(input, output, count, dtype);
  auto it = contexts.find(ctxKey);
  if (it == contexts.end()) {
    auto ctx = contextInitFunc(comm);
    contexts[ctxKey] = ctx;
  }
  return kernelLaunchFunc(contexts[ctxKey], stream);
}

Algorithm::Algorithm(std::shared_ptr<Communicator> comm, std::string name, KernelFunc kernelFunc,
                     ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc)
    : name(name), impl(std::make_shared<AlgorithmImpl>(comm, kernelFunc, contextInitFunc, contextKeyGenFunc)) {}

void Algorithm::launch(void* input, void* output, size_t count, uint32_t dtype, cudaStream_t stream) {
  this->impl->launch(input, output, count, dtype, stream);
}

void AlgorithmFactory::registerAlgorithm(const std::string algoName, const std::string collective,
                                         Algorithm algorithm) {
  AlgorithmKey key = {algoName, collective};
  getInstance()->algoMap.insert({key, algorithm});
}

Algorithm AlgorithmFactory::getAlgorithm(const AlgorithmKey& algoKey) {
  auto it = getInstance()->algoMap.find(algoKey);
  if (it != getInstance()->algoMap.end()) {
    return it->second;
  }
  throw Error("Algorithm not found with name: " + algoKey.name + " for collective " + algoKey.collective,
              ErrorCode::InvalidUsage);
}

}  // namespace mscclpp