// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>

namespace mscclpp {

class Algorithm::Impl {
 public:
  Impl(std::string name, std::string collective, Algorithm::InitFunc initFunc, Algorithm::KernelFunc kernelFunc,
       Algorithm::ContextInitFunc contextInitFunc, Algorithm::ContextKeyGenFunc contextKeyGenFunc)
      : name_(name),
        collective_(collective),
        initFunc_(initFunc),
        kernelLaunchFunc_(kernelFunc),
        contextInitFunc_(contextInitFunc),
        contextKeyGenFunc_(contextKeyGenFunc) {}
  int launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count, DataType dtype,
             cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras);

  std::string name_;
  std::string collective_;
  Algorithm::InitFunc initFunc_;
  Algorithm::KernelFunc kernelLaunchFunc_;
  Algorithm::ContextInitFunc contextInitFunc_;
  Algorithm::ContextKeyGenFunc contextKeyGenFunc_;

  bool initialized_ = false;
  std::unordered_map<AlgorithmCtxKey, std::shared_ptr<AlgorithmCtx>> contexts_;
};

int Algorithm::Impl::launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
                            DataType dtype, cudaStream_t stream,
                            std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  if (!initialized_) {
    initFunc_(comm, extras);
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

Algorithm::Algorithm(std::string name, std::string collective, InitFunc initFunc, KernelFunc kernelFunc,
                     ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc)
    : impl_(std::make_shared<Impl>(name, collective, initFunc, kernelFunc, contextInitFunc, contextKeyGenFunc)) {}

int Algorithm::launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
                      DataType dtype, cudaStream_t stream,
                      std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  return this->impl_->launch(comm, input, output, count, dtype, stream, extras);
}

bool Algorithm::isEmpty() { return !impl_; }

std::string Algorithm::name() const { return impl_->name_; }

std::string Algorithm::collective() const { return impl_->collective_; }

void AlgorithmCollection::registerAlgorithm(const std::string collective, const std::string algoName,
                                            Algorithm algorithm) {
  this->algoMapByCollective_[collective][algoName] = algorithm;
}

Algorithm AlgorithmCollection::selectAlgorithm(const std::string& collective, const void* input, void* output,
                                               size_t messageSize, DataType dtype, int nRanksPerNode, int worldSize) {
  Algorithm algo;
  if (algoSelector_) {
    algo = algoSelector_(algoMapByCollective_, collective, input, output, messageSize, dtype, nRanksPerNode, worldSize);
  }
  if (algo.isEmpty()) {
    algo = fallbackAlgoSelector_(algoMapByCollective_, collective, input, output, messageSize, dtype, nRanksPerNode,
                                 worldSize);
  }
  return algo;
}

std::shared_ptr<AlgorithmCollectionBuilder> AlgorithmCollectionBuilder::getInstance() {
  static std::shared_ptr<AlgorithmCollectionBuilder> instance(new AlgorithmCollectionBuilder());
  return instance;
}

void AlgorithmCollectionBuilder::addAlgorithmBuilder(std::shared_ptr<AlgorithmBuilder> builder) {
  this->algoBuilders_.push_back(builder);
}

void AlgorithmCollectionBuilder::setAlgorithmSelector(AlgoSelectFunc selector) { algoSelector_ = selector; }

void AlgorithmCollectionBuilder::setFallbackAlgorithmSelector(AlgoSelectFunc selector) {
  fallbackAlgoSelector_ = selector;
}

std::shared_ptr<AlgorithmCollection> AlgorithmCollectionBuilder::build() {
  auto collection = std::make_shared<AlgorithmCollection>();
  for (const auto& builder : algoBuilders_) {
    auto algo = builder->build();
    collection->registerAlgorithm(algo.collective(), algo.name(), algo);
  }
  collection->algoSelector_ = algoSelector_;
  collection->fallbackAlgoSelector_ = fallbackAlgoSelector_;
  return collection;
}

}  // namespace mscclpp