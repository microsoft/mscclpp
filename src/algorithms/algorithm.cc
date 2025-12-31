// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <filesystem>
#include <mscclpp/algorithm.hpp>

#include "algorithms/allgather/allgather_fullmesh.hpp"
#include "algorithms/allgather/allgather_fullmesh2.hpp"
#include "algorithms/allreduce/allreduce_allpair_packet.hpp"
#include "algorithms/allreduce/allreduce_fullmesh.hpp"
#include "algorithms/allreduce/allreduce_nvls.hpp"
#include "algorithms/allreduce/allreduce_nvls_packet.hpp"
#include "algorithms/allreduce/allreduce_nvls_with_copy.hpp"
#include "algorithms/allreduce/allreduce_nvls_with_copy2.hpp"
#include "algorithms/allreduce/allreduce_packet.hpp"
#include "algorithms/allreduce/allreduce_rsag.hpp"
#include "algorithms/allreduce/allreduce_rsag_pipeline.hpp"
#include "algorithms/utils.hpp"
#include "logger.hpp"

namespace mscclpp {

CollectiveBufferMode CollectiveRequest::bufferMode() const {
  if (inputBuffer == outputBuffer) return CollectiveBufferMode::IN_PLACE;
  if (collective == "allgather") {
    size_t rankOffset = rank * messageSize;
    const char* expectedInput = static_cast<const char*>(outputBuffer) + rankOffset;
    if (static_cast<const void*>(expectedInput) == inputBuffer) {
      return CollectiveBufferMode::IN_PLACE;
    }
    return CollectiveBufferMode::OUT_OF_PLACE;
  }
  return CollectiveBufferMode::OUT_OF_PLACE;
}

NativeAlgorithm::NativeAlgorithm(std::string name, std::string collective, InitFunc initFunc, KernelFunc kernelFunc,
                                 ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc,
                                 size_t minMessageSize, size_t maxMessageSize, CollectiveBufferMode bufferMode,
                                 std::unordered_map<std::string, uint64_t> tags, Constraint constraint)
    : name_(name),
      collective_(collective),
      initFunc_(initFunc),
      kernelLaunchFunc_(kernelFunc),
      contextInitFunc_(contextInitFunc),
      contextKeyGenFunc_(contextKeyGenFunc),
      minMessageSize_(minMessageSize),
      maxMessageSize_(maxMessageSize),
      bufferMode_(bufferMode),
      tags_(tags),
      constraint_(constraint) {}

CommResult NativeAlgorithm::execute(std::shared_ptr<Communicator> comm, const void* input, void* output,
                                    size_t inputSize, size_t outputSize, DataType dtype, ReduceOp op,
                                    cudaStream_t stream, std::shared_ptr<Executor>, int nBlocks, int nThreadsPerBlock,
                                    const std::unordered_map<std::string, uintptr_t>& extras) {
  if (!initialized_) {
    initFunc_(comm);
    initialized_ = true;
  }
  AlgorithmCtxKey ctxKey = contextKeyGenFunc_(input, output, inputSize, outputSize, dtype);
  auto it = contexts_.find(ctxKey);
  if (it == contexts_.end()) {
    auto ctx = contextInitFunc_(comm, input, output, inputSize, outputSize, dtype);
    contexts_[ctxKey] = ctx;
  }
  return kernelLaunchFunc_(contexts_[ctxKey], input, output, inputSize, outputSize, dtype, op, stream, nBlocks,
                           nThreadsPerBlock, extras);
}

const std::string& NativeAlgorithm::name() const { return name_; }

const std::string& NativeAlgorithm::collective() const { return collective_; }

const std::pair<size_t, size_t>& NativeAlgorithm::messageRange() const {
  static std::pair<size_t, size_t> range;
  range = {minMessageSize_, maxMessageSize_};
  return range;
}

const std::unordered_map<std::string, uint64_t>& NativeAlgorithm::tags() const { return tags_; }

const CollectiveBufferMode& NativeAlgorithm::bufferMode() const { return bufferMode_; }

Algorithm::Constraint NativeAlgorithm::constraint() const { return constraint_; }

void NativeAlgorithm::reset() {
  contexts_.clear();
  initialized_ = false;
}

void AlgorithmCollection::registerAlgorithm(const std::string collective, const std::string algoName,
                                            std::shared_ptr<Algorithm> algorithm) {
  this->algoMapByCollective_[collective][algoName] = algorithm;
}

std::shared_ptr<Algorithm> AlgorithmCollection::selectAlgorithm(const CollectiveRequest& request) {
  std::shared_ptr<Algorithm> algo;
  if (!algoSelector_ && !fallbackAlgoSelector_) {
    THROW(ALGO, Error, ErrorCode::InvalidUsage, "No algorithm selector is set in AlgorithmCollection.");
  }
  if (algoSelector_) {
    algo = algoSelector_(algoMapByCollective_, request);
  }
  if (!algo) {
    algo = fallbackAlgoSelector_(algoMapByCollective_, request);
  }
  return algo;
}

void AlgorithmCollection::extend(const AlgorithmCollection& other) {
  for (const auto& [collective, algoMap] : other.algoMapByCollective_) {
    for (const auto& [algoName, algorithm] : algoMap) {
      this->registerAlgorithm(collective, algoName, algorithm);
    }
  }
}

std::vector<std::shared_ptr<Algorithm>> AlgorithmCollection::getAllAlgorithms() const {
  std::vector<std::shared_ptr<Algorithm>> allAlgos;
  for (const auto& [collective, algoMap] : algoMapByCollective_) {
    for (const auto& [algoName, algorithm] : algoMap) {
      allAlgos.push_back(algorithm);
    }
  }
  return allAlgos;
}

std::unordered_map<std::string, std::shared_ptr<Algorithm>> AlgorithmCollection::getAlgorithmsByCollective(
    const std::string& collective) const {
  auto it = algoMapByCollective_.find(collective);
  if (it != algoMapByCollective_.end()) {
    return it->second;
  } else {
    return {};
  }
}

std::shared_ptr<AlgorithmCollectionBuilder> AlgorithmCollectionBuilder::gAlgorithmCollectionBuilder_;
std::shared_ptr<AlgorithmCollectionBuilder> AlgorithmCollectionBuilder::getInstance() {
  if (!gAlgorithmCollectionBuilder_) {
    gAlgorithmCollectionBuilder_ = std::shared_ptr<AlgorithmCollectionBuilder>(new AlgorithmCollectionBuilder());
  }
  return gAlgorithmCollectionBuilder_;
}

void AlgorithmCollectionBuilder::addAlgorithmBuilder(std::shared_ptr<AlgorithmBuilder> builder) {
  this->algoBuilders_.push_back(builder);
}

AlgorithmCollection AlgorithmCollectionBuilder::buildDefaultAlgorithms(uintptr_t scratchBuffer,
                                                                       size_t scratchBufferSize, int rank) {
  auto nativeCollection = buildDefaultNativeAlgorithms(scratchBuffer, scratchBufferSize);
  auto dslCollection = buildDefaultDslAlgorithms(rank);
  nativeCollection.extend(dslCollection);
  return nativeCollection;
}

AlgorithmCollection AlgorithmCollectionBuilder::buildDefaultNativeAlgorithms(uintptr_t scratchBuffer,
                                                                             size_t scratchBufferSize) {
  AlgorithmCollection collection;
  auto allreduceAllpairPkt =
      std::make_shared<algorithm::AllreduceAllpairPacket>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceAllpairPkt->collective(), allreduceAllpairPkt->name(), allreduceAllpairPkt);
  auto allreduceNvlsPkt = std::make_shared<algorithm::AllreduceNvlsPacket>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceNvlsPkt->collective(), allreduceNvlsPkt->name(), allreduceNvlsPkt);
  auto allreduceNvlsWithCopyPkt =
      std::make_shared<algorithm::AllreduceNvlsWithCopy>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceNvlsWithCopyPkt->collective(), allreduceNvlsWithCopyPkt->name(),
                               allreduceNvlsWithCopyPkt);
  auto allreduceNvlsWithCopy2Pkt =
      std::make_shared<algorithm::AllreduceNvlsWithCopy2>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceNvlsWithCopy2Pkt->collective(), allreduceNvlsWithCopy2Pkt->name(),
                               allreduceNvlsWithCopy2Pkt);
  auto allreducePkt = std::make_shared<algorithm::AllreducePacket>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreducePkt->collective(), allreducePkt->name(), allreducePkt);
  auto allreduceNvls = std::make_shared<algorithm::AllreduceNvls>()->build();
  collection.registerAlgorithm(allreduceNvls->collective(), allreduceNvls->name(), allreduceNvls);
  auto allreduceFullmesh = std::make_shared<algorithm::AllreduceFullmesh>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceFullmesh->collective(), allreduceFullmesh->name(), allreduceFullmesh);
  auto allreduceRsAg = std::make_shared<algorithm::AllreduceRsAg>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceRsAg->collective(), allreduceRsAg->name(), allreduceRsAg);
  auto allreduceRsAgPipeline =
      std::make_shared<algorithm::AllreduceRsAgPipeline>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceRsAgPipeline->collective(), allreduceRsAgPipeline->name(),
                               allreduceRsAgPipeline);

  auto allgatherFullmesh = std::make_shared<algorithm::AllgatherFullmesh>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allgatherFullmesh->collective(), allgatherFullmesh->name(), allgatherFullmesh);
  auto allgatherFullmesh2 = std::make_shared<algorithm::AllgatherFullmesh2>()->build();
  collection.registerAlgorithm(allgatherFullmesh2->collective(), allgatherFullmesh2->name(), allgatherFullmesh2);
  collection.algoSelector_ = algoSelector_;
  collection.fallbackAlgoSelector_ = fallbackAlgoSelector_;
  return collection;
}

AlgorithmCollection AlgorithmCollectionBuilder::buildDefaultDslAlgorithms(int rank) {
  struct DslAlgoConfig {
    std::string filename;
    std::string collective;
    int nRanksPerNode;
    int worldSize;
    std::unordered_map<std::string, uint64_t> tags;
  };
  static const std::vector<DslAlgoConfig> defaultAlgoConfigs = {
      {"allreduce_2nodes_1K_64K.json", "allreduce", 8, 16, {{"default", 1}}},
      {"allreduce_2nodes_64K_2M.json", "allreduce", 8, 16, {{"default", 1}}}};
  AlgorithmCollection collection;
  collection.algoSelector_ = algoSelector_;
  collection.fallbackAlgoSelector_ = fallbackAlgoSelector_;

  static auto generateFileId = [](const std::string& input) {
    std::hash<std::string> hasher;
    size_t hashValue = hasher(input);
    std::ostringstream oss;
    oss << std::hex << hashValue;
    return oss.str();
  };

  std::string planDir = mscclpp::env()->executionPlanDir;
  if (!std::filesystem::exists(planDir)) {
    INFO(ALGO, "Plan directory does not exist: ", planDir);
    return collection;
  }
  for (const auto& config : defaultAlgoConfigs) {
    std::string planPath = planDir + "/" + config.filename;
    INFO(ALGO, "Loading plan: ", planPath);
    if (!std::filesystem::exists(planPath)) {
      INFO(ALGO, "Plan file does not exist: ", planPath);
      continue;
    }
    std::string planId = generateFileId(planPath);
    auto collectionBuilder = mscclpp::AlgorithmCollectionBuilder::getInstance();
    try {
      auto executionPlan = mscclpp::ExecutionPlan(planPath, rank);
      auto algoBuilder = std::make_shared<mscclpp::DslAlgorithm>(
          planId, executionPlan, config.tags, mscclpp::Algorithm::Constraint{config.worldSize, config.nRanksPerNode});
      collectionBuilder->addAlgorithmBuilder(algoBuilder);
      INFO(ALGO, "Successfully loaded plan: ", planId, " for collective: ", config.collective);
    } catch (const std::exception& e) {
      WARN(ALGO, "Failed to load plan : ", planPath, " ", e.what());
    }
  }
  return collection;
}

void AlgorithmCollectionBuilder::setAlgorithmSelector(AlgoSelectFunc selector) { algoSelector_ = selector; }

void AlgorithmCollectionBuilder::setFallbackAlgorithmSelector(AlgoSelectFunc selector) {
  fallbackAlgoSelector_ = selector;
}

AlgorithmCollection AlgorithmCollectionBuilder::build() {
  AlgorithmCollection collection;
  for (const auto& builder : algoBuilders_) {
    auto algo = builder->build();
    collection.registerAlgorithm(algo->collective(), algo->name(), algo);
  }
  collection.algoSelector_ = algoSelector_;
  collection.fallbackAlgoSelector_ = fallbackAlgoSelector_;
  return collection;
}

void AlgorithmCollectionBuilder::reset() { gAlgorithmCollectionBuilder_.reset(); }

DslAlgorithm::DslAlgorithm(std::string id, ExecutionPlan plan, std::unordered_map<std::string, uint64_t> tags,
                           Constraint constraint)
    : plan_(plan), id_(id), tags_(tags), constraint_(constraint) {}

const std::string& DslAlgorithm::name() const { return plan_.name(); }

const std::string& DslAlgorithm::collective() const { return plan_.collective(); }

const std::pair<size_t, size_t>& DslAlgorithm::messageRange() const {
  static std::pair<size_t, size_t> range;
  range = {plan_.minMessageSize(), plan_.maxMessageSize()};
  return range;
}

const std::unordered_map<std::string, uint64_t>& DslAlgorithm::tags() const { return tags_; }

const CollectiveBufferMode& DslAlgorithm::bufferMode() const {
  // TODO: need to fix
  static CollectiveBufferMode mode =
      plan_.isInPlace() ? CollectiveBufferMode::IN_PLACE : CollectiveBufferMode::OUT_OF_PLACE;
  return mode;
}

Algorithm::Constraint DslAlgorithm::constraint() const { return constraint_; }

CommResult DslAlgorithm::execute(std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
                                 size_t outputSize, DataType dtype, ReduceOp, cudaStream_t stream,
                                 std::shared_ptr<Executor> executor, int, int,
                                 const std::unordered_map<std::string, uintptr_t>&) {
  if (!executor) {
    THROW(EXEC, Error, ErrorCode::InvalidUsage, "Executor is null in DslAlgorithm::execute");
  }
  int rank = comm->bootstrap()->getRank();
  switch (dtype) {
    case DataType::FLOAT16:
      executor->execute(rank, (half*)input, (half*)output, inputSize, outputSize, DataType::FLOAT16, plan_, stream);
      break;
    case DataType::FLOAT32:
      executor->execute(rank, (float*)input, (float*)output, inputSize, outputSize, DataType::FLOAT32, plan_, stream);
      break;
    case DataType::BFLOAT16:
      executor->execute(rank, (__bfloat16*)input, (__bfloat16*)output, inputSize, outputSize, DataType::BFLOAT16, plan_,
                        stream);
      break;
#if defined(__FP8_TYPES_EXIST__)
    case DataType::FP8_E4M3:
      executor->execute(rank, (__fp8_e4m3*)input, (__fp8_e4m3*)output, inputSize, outputSize, DataType::FP8_E4M3, plan_,
                        stream);
      break;
    case DataType::FP8_E5M2:
      executor->execute(rank, (__fp8_e5m2*)input, (__fp8_e5m2*)output, inputSize, outputSize, DataType::FP8_E5M2, plan_,
                        stream);
      break;
#endif
    case DataType::INT32:
    case DataType::UINT32:
      executor->execute(rank, (int*)input, (int*)output, inputSize, outputSize, DataType::UINT32, plan_, stream);
      break;
    default:
      WARN(ALGO, "Unsupported data type: ", static_cast<int>(dtype), " in DslAlgorithm");
      return CommResult::commInvalidArgument;
  }
  return CommResult::commSuccess;
}

std::shared_ptr<Algorithm> DslAlgorithm::build() { return shared_from_this(); }

// TODO: implement this
void DslAlgorithm::reset() {}

}  // namespace mscclpp