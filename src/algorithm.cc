// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>

namespace mscclpp {

class NativeAlgorithm::Impl {
 public:
  Impl(std::string name, std::string collective, NativeAlgorithm::InitFunc initFunc,
       NativeAlgorithm::KernelFunc kernelFunc, NativeAlgorithm::ContextInitFunc contextInitFunc,
       NativeAlgorithm::ContextKeyGenFunc contextKeyGenFunc, size_t minMessageSize, size_t maxMessageSize,
       CollectiveBufferMode bufferMode, std::unordered_map<std::string, uint64_t> tags)
      : name_(name),
        collective_(collective),
        initFunc_(initFunc),
        kernelLaunchFunc_(kernelFunc),
        contextInitFunc_(contextInitFunc),
        contextKeyGenFunc_(contextKeyGenFunc),
        minMessageSize_(minMessageSize),
        maxMessageSize_(maxMessageSize),
        bufferMode_(bufferMode),
        tags_(tags) {}
  int execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count, int dtype,
              cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras);

  std::string name_;
  std::string collective_;
  NativeAlgorithm::InitFunc initFunc_;
  NativeAlgorithm::KernelFunc kernelLaunchFunc_;
  NativeAlgorithm::ContextInitFunc contextInitFunc_;
  NativeAlgorithm::ContextKeyGenFunc contextKeyGenFunc_;
  size_t minMessageSize_;
  size_t maxMessageSize_;
  CollectiveBufferMode bufferMode_;
  std::unordered_map<std::string, uint64_t> tags_;

  bool initialized_ = false;
  std::unordered_map<AlgorithmCtxKey, std::shared_ptr<AlgorithmCtx>> contexts_;
};

int NativeAlgorithm::Impl::execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output,
                                   size_t count, int dtype, cudaStream_t stream,
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

NativeAlgorithm::NativeAlgorithm(std::string name, std::string collective, InitFunc initFunc, KernelFunc kernelFunc,
                                 ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc,
                                 size_t minMessageSize, size_t maxMessageSize, CollectiveBufferMode bufferMode,
                                 std::unordered_map<std::string, uint64_t> tags)
    : impl_(std::make_shared<Impl>(name, collective, initFunc, kernelFunc, contextInitFunc, contextKeyGenFunc,
                                   minMessageSize, maxMessageSize, bufferMode, tags)) {}

int NativeAlgorithm::execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count,
                             int dtype, cudaStream_t stream,
                             std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  return this->impl_->execute(comm, input, output, count, dtype, stream, extras);
}

const std::string& NativeAlgorithm::name() const { return impl_->name_; }

const std::string& NativeAlgorithm::collective() const { return impl_->collective_; }

const std::pair<size_t, size_t>& NativeAlgorithm::messageRange() const {
  static std::pair<size_t, size_t> range;
  range = {impl_->minMessageSize_, impl_->maxMessageSize_};
  return range;
}

const std::unordered_map<std::string, uint64_t>& NativeAlgorithm::tags() const { return impl_->tags_; }

const CollectiveBufferMode& NativeAlgorithm::bufferMode() const { return impl_->bufferMode_; }

void AlgorithmCollection::registerAlgorithm(const std::string collective, const std::string algoName,
                                            std::shared_ptr<Algorithm> algorithm) {
  this->algoMapByCollective_[collective][algoName] = algorithm;
}

std::shared_ptr<Algorithm> AlgorithmCollection::selectAlgorithm(const std::string& collective, const void* input,
                                                                void* output, size_t messageSize, int dtype,
                                                                int nRanksPerNode, int worldSize) {
  std::shared_ptr<Algorithm> algo;
  if (algoSelector_) {
    algo = algoSelector_(algoMapByCollective_, collective, input, output, messageSize, dtype, nRanksPerNode, worldSize);
  }
  if (!algo) {
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
    collection->registerAlgorithm(algo->collective(), algo->name(), algo);
  }
  collection->algoSelector_ = algoSelector_;
  collection->fallbackAlgoSelector_ = fallbackAlgoSelector_;
  return collection;
}

}  // namespace mscclpp