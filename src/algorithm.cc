// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>
#include <mscclpp/executor.hpp>

#include "logger.hpp"

namespace mscclpp {

bool CollectiveRequest::isInPlace() const {
  if (inputBuffer == outputBuffer) return true;
  if (collective == "allgather") {
    size_t rankOffset = rank * messageSize;
    const char* expectedInput = static_cast<const char*>(outputBuffer) + rankOffset;
    return static_cast<const void*>(expectedInput) == inputBuffer;
  }
  return false;
}

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
  int execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
              size_t outputSize, int dtype, cudaStream_t stream,
              std::unordered_map<std::string, std::shared_ptr<void>>& extras);

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
                                   size_t inputSize, size_t outputSize, int dtype, cudaStream_t stream,
                                   std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  if (!initialized_) {
    initFunc_(comm, extras);
    initialized_ = true;
  }
  AlgorithmCtxKey ctxKey = contextKeyGenFunc_(input, output, inputSize, outputSize, dtype);
  auto it = contexts_.find(ctxKey);
  if (it == contexts_.end()) {
    auto ctx = contextInitFunc_(comm, input, output, inputSize, outputSize, dtype);
    contexts_[ctxKey] = ctx;
  }
  return kernelLaunchFunc_(contexts_[ctxKey], input, output, inputSize, outputSize, dtype, stream, extras);
}

NativeAlgorithm::NativeAlgorithm(std::string name, std::string collective, InitFunc initFunc, KernelFunc kernelFunc,
                                 ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc,
                                 size_t minMessageSize, size_t maxMessageSize, CollectiveBufferMode bufferMode,
                                 std::unordered_map<std::string, uint64_t> tags)
    : impl_(std::make_shared<Impl>(name, collective, initFunc, kernelFunc, contextInitFunc, contextKeyGenFunc,
                                   minMessageSize, maxMessageSize, bufferMode, tags)) {}

int NativeAlgorithm::execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output,
                             size_t inputSize, size_t outputSize, int dtype, cudaStream_t stream,
                             std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  return this->impl_->execute(comm, input, output, inputSize, outputSize, dtype, stream, extras);
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

std::shared_ptr<Algorithm> AlgorithmCollection::selectAlgorithm(CollectiveRequest request) {
  std::shared_ptr<Algorithm> algo;
  if (algoSelector_) {
    algo = algoSelector_(algoMapByCollective_, request);
  }
  if (!algo) {
    algo = fallbackAlgoSelector_(algoMapByCollective_, request);
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

DslAlgorithm::DslAlgorithm(std::shared_ptr<ExecutionPlanHandle> planHandle) : planHandle_(planHandle) {}

const std::string& DslAlgorithm::name() const { return planHandle_->plan->name(); }

const std::string& DslAlgorithm::collective() const { return planHandle_->plan->collective(); }

const std::pair<size_t, size_t>& DslAlgorithm::messageRange() const {
  static std::pair<size_t, size_t> range;
  range = {planHandle_->plan->minMessageSize(), planHandle_->plan->maxMessageSize()};
  return range;
}

const std::unordered_map<std::string, uint64_t>& DslAlgorithm::tags() const { return planHandle_->tags; }

const CollectiveBufferMode& DslAlgorithm::bufferMode() const {
  // TODO: need to fix
  static CollectiveBufferMode mode =
      planHandle_->plan->isInPlace() ? CollectiveBufferMode::IN_PLACE : CollectiveBufferMode::OUT_OF_PLACE;
  return mode;
}

int DslAlgorithm::execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output,
                          size_t inputSize, size_t outputSize, int dtype, cudaStream_t stream,
                          std::unordered_map<std::string, std::shared_ptr<void>>& extras) {
  auto executorIt = extras.find("executor");
  if (executorIt == extras.end()) {
    THROW(EXEC, Error, ErrorCode::InvalidUsage, "DslAlgorithm requires 'executor' in extras");
  }
  auto executor = std::static_pointer_cast<Executor>(executorIt->second);
  int rank = comm->bootstrap()->getRank();
  DataType dataType = static_cast<DataType>(dtype);
  switch (dataType) {
    case DataType::FLOAT16:
      executor->execute(rank, (half*)input, (half*)output, inputSize, outputSize, mscclpp::DataType::FLOAT16,
                        *planHandle_->plan, stream);
      break;
    case DataType::FLOAT32:
      executor->execute(rank, (float*)input, (float*)output, inputSize, outputSize, mscclpp::DataType::FLOAT32,
                        *planHandle_->plan, stream);
      break;
    case DataType::BFLOAT16:
      executor->execute(rank, (__bfloat16*)input, (__bfloat16*)output, inputSize, outputSize,
                        mscclpp::DataType::BFLOAT16, *planHandle_->plan, stream);
      break;
#if defined(__FP8_TYPES_EXIST__)
    case DataType::FP8_E4M3:
      executor->execute(rank, (__fp8_e4m3*)input, (__fp8_e4m3*)output, inputSize, outputSize,
                        mscclpp::DataType::FP8_E4M3, *planHandle_->plan, stream);
      break;
    case DataType::FP8_E5M2:
      executor->execute(rank, (__fp8_e5m2*)input, (__fp8_e5m2*)output, inputSize, outputSize,
                        mscclpp::DataType::FP8_E5M2, *planHandle_->plan, stream);
      break;
#endif
    case DataType::INT32:
    case DataType::UINT32:
      executor->execute(rank, (int*)input, (int*)output, inputSize, outputSize, mscclpp::DataType::UINT32,
                        *planHandle_->plan, stream);
      break;
    default:
      WARN(EXEC, "Unsupported data type %d in DslAlgorithm", static_cast<int>(dataType));
      return 4;  // TODO: need to fix
  }
  return 0;
}

std::shared_ptr<ExecutionPlanHandle> DslAlgorithm::planHandle() const { return planHandle_; }

DslAlgorithmBuilder::DslAlgorithmBuilder(std::shared_ptr<ExecutionPlanHandle> planHandle) : planHandle_(planHandle) {}

std::shared_ptr<Algorithm> DslAlgorithmBuilder::build() { return std::make_shared<DslAlgorithm>(planHandle_); }

}  // namespace mscclpp