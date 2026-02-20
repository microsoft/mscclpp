// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <filesystem>
#include <mscclpp/ext/collectives/algorithm_collection_builder.hpp>

#include "allgather/allgather_fullmesh.hpp"
#include "allgather/allgather_fullmesh_2.hpp"
#include "allreduce/allreduce_allpair_packet.hpp"
#include "allreduce/allreduce_fullmesh.hpp"
#include "allreduce/allreduce_nvls.hpp"
#include "allreduce/allreduce_nvls_packet.hpp"
#include "allreduce/allreduce_nvls_with_copy.hpp"
#include "allreduce/allreduce_nvls_with_copy_2.hpp"
#include "allreduce/allreduce_packet.hpp"
#include "logger.hpp"

namespace mscclpp {
namespace collective {

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
  collection.setSelectors(algoSelector_, fallbackAlgoSelector_);
  return collection;
}

void AlgorithmCollectionBuilder::reset() { gAlgorithmCollectionBuilder_.reset(); }

AlgorithmCollection AlgorithmCollectionBuilder::buildDefaultAlgorithms(uintptr_t scratchBuffer,
                                                                       size_t scratchBufferSize, uintptr_t flagBuffer,
                                                                       size_t flagBufferSize, int rank) {
  auto nativeCollection = buildDefaultNativeAlgorithms(scratchBuffer, scratchBufferSize, flagBuffer, flagBufferSize);
  auto dslCollection = buildDefaultDslAlgorithms(rank);
  nativeCollection.extend(dslCollection);
  nativeCollection.setSelectors(algoSelector_, fallbackAlgoSelector_);
  return nativeCollection;
}

AlgorithmCollection AlgorithmCollectionBuilder::buildDefaultNativeAlgorithms(uintptr_t scratchBuffer,
                                                                             size_t scratchBufferSize,
                                                                             uintptr_t flagBuffer,
                                                                             size_t flagBufferSize) {
  AlgorithmCollection collection;
  auto allreduceAllpairPkt =
      std::make_shared<AllreduceAllpairPacket>(scratchBuffer, scratchBufferSize, flagBuffer, flagBufferSize)->build();
  collection.registerAlgorithm(allreduceAllpairPkt->collective(), allreduceAllpairPkt->name(), allreduceAllpairPkt);
  auto allreduceNvlsPacket =
      std::make_shared<AllreduceNvlsPacket>(scratchBuffer, scratchBufferSize, flagBuffer, flagBufferSize)->build();
  collection.registerAlgorithm(allreduceNvlsPacket->collective(), allreduceNvlsPacket->name(), allreduceNvlsPacket);
  auto allreduceNvlsWithCopy = std::make_shared<AllreduceNvlsWithCopy>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceNvlsWithCopy->collective(), allreduceNvlsWithCopy->name(),
                               allreduceNvlsWithCopy);
  auto allreduceNvlsWithCopy2 = std::make_shared<AllreduceNvlsWithCopy2>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceNvlsWithCopy2->collective(), allreduceNvlsWithCopy2->name(),
                               allreduceNvlsWithCopy2);
  auto allreducePkt =
      std::make_shared<AllreducePacket>(scratchBuffer, scratchBufferSize, flagBuffer, flagBufferSize)->build();
  collection.registerAlgorithm(allreducePkt->collective(), allreducePkt->name(), allreducePkt);
  auto allreduceNvls = std::make_shared<AllreduceNvls>()->build();
  collection.registerAlgorithm(allreduceNvls->collective(), allreduceNvls->name(), allreduceNvls);
  auto allreduceFullmesh = std::make_shared<AllreduceFullmesh>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceFullmesh->collective(), allreduceFullmesh->name(), allreduceFullmesh);

  auto allgatherFullmesh = std::make_shared<AllgatherFullmesh>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allgatherFullmesh->collective(), allgatherFullmesh->name(), allgatherFullmesh);
  auto allgatherFullmesh2 = std::make_shared<AllgatherFullmesh2>()->build();
  collection.registerAlgorithm(allgatherFullmesh2->collective(), allgatherFullmesh2->name(), allgatherFullmesh2);
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

  static auto generateFileId = [](const std::string& input) {
    std::hash<std::string> hasher;
    size_t hashValue = hasher(input);
    std::ostringstream oss;
    oss << std::hex << hashValue;
    return oss.str();
  };

  auto planDir = std::filesystem::path(env()->cacheDir) / "default";
  if (!std::filesystem::exists(planDir)) {
    INFO(ALGO, "Default plan directory does not exist: ", planDir);
    return collection;
  }
  for (const auto& config : defaultAlgoConfigs) {
    auto planPath = planDir / config.filename;
    INFO(ALGO, "Loading plan: ", planPath);
    if (!std::filesystem::exists(planPath)) {
      INFO(ALGO, "Plan file does not exist: ", planPath);
      continue;
    }
    std::string planId = generateFileId(planPath);
    auto collectionBuilder = AlgorithmCollectionBuilder::getInstance();
    try {
      auto executionPlan = ExecutionPlan(planPath, rank);
      auto algoBuilder = std::make_shared<DslAlgorithm>(planId, executionPlan, config.tags,
                                                        Algorithm::Constraint{config.worldSize, config.nRanksPerNode});
      collectionBuilder->addAlgorithmBuilder(algoBuilder);
      INFO(ALGO, "Successfully loaded plan: ", planId, " for collective: ", config.collective);
    } catch (const std::exception& e) {
      WARN(ALGO, "Failed to load plan : ", planPath, " ", e.what());
    }
  }
  return collection;
}

}  // namespace collective
}  // namespace mscclpp