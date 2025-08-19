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
  AlgorithmKey key = {collective, algoName};
  getInstance()->algoMap.insert({key, algorithm});
  getInstance()->algoMapByCollective[collective].push_back(algorithm);
}

Algorithm AlgorithmFactory::getAlgorithm(const AlgorithmKey& algoKey) {
  auto it = getInstance()->algoMap.find(algoKey);
  if (it != getInstance()->algoMap.end()) {
    return it->second;
  }
  throw Error("Algorithm not found with name: " + algoKey.name + " for collective " + algoKey.collective,
              ErrorCode::InvalidUsage);
}

Algorithm AlgorithmFactory::selectAlgorithm(const std::string& collective, size_t messageSizes, const void* input,
                                            void* output) {
  if (algoSelector) {
    return algoSelector(this->algoMapByCollective, collective, messageSizes, input, output);
  }
  throw Error("No algorithm selector set", ErrorCode::InvalidUsage);
}

void AlgorithmFactory::setAlgorithmSelector(AlgoSelectFunc selector) { algoSelector = selector; }

bool AlgorithmFactory::hasAlgorithmSelector() const { return algoSelector != nullptr; }

void AlgorithmFactory::destroy() {
  algoMap.clear();
  algoMapByCollective.clear();
  algoSelector = nullptr;
}

}  // namespace mscclpp