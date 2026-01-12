#include <filesystem>
#include <mscclpp/ext/collectives/default_algorithm_builder.hpp>

#include "allgather/allgather_fullmesh.hpp"
#include "allgather/allgather_fullmesh2.hpp"
#include "allreduce/allreduce_allpair_packet.hpp"
#include "allreduce/allreduce_fullmesh.hpp"
#include "allreduce/allreduce_nvls.hpp"
#include "allreduce/allreduce_nvls_packet.hpp"
#include "allreduce/allreduce_nvls_with_copy.hpp"
#include "allreduce/allreduce_nvls_with_copy2.hpp"
#include "allreduce/allreduce_packet.hpp"
#include "logger.hpp"

namespace mscclpp {
namespace collective {
AlgorithmCollection DefaultAlgorithmBuilder::buildDefaultAlgorithms(uintptr_t scratchBuffer, size_t scratchBufferSize,
                                                                    int rank) {
  auto nativeCollection = buildDefaultNativeAlgorithms(scratchBuffer, scratchBufferSize);
  auto dslCollection = buildDefaultDslAlgorithms(rank);
  nativeCollection.extend(dslCollection);
  return nativeCollection;
}

AlgorithmCollection DefaultAlgorithmBuilder::buildDefaultNativeAlgorithms(uintptr_t scratchBuffer,
                                                                          size_t scratchBufferSize) {
  AlgorithmCollection collection;
  auto allreduceAllpairPkt = std::make_shared<AllreduceAllpairPacket>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceAllpairPkt->collective(), allreduceAllpairPkt->name(), allreduceAllpairPkt);
  auto allreduceNvlsWithCopyPkt = std::make_shared<AllreduceNvlsWithCopy>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceNvlsWithCopyPkt->collective(), allreduceNvlsWithCopyPkt->name(),
                               allreduceNvlsWithCopyPkt);
  auto allreduceNvlsWithCopy2Pkt = std::make_shared<AllreduceNvlsWithCopy2>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceNvlsWithCopy2Pkt->collective(), allreduceNvlsWithCopy2Pkt->name(),
                               allreduceNvlsWithCopy2Pkt);
  auto allreducePkt = std::make_shared<AllreducePacket>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreducePkt->collective(), allreducePkt->name(), allreducePkt);
  auto allreduceNvls = std::make_shared<AllreduceNvls>()->build();
  collection.registerAlgorithm(allreduceNvls->collective(), allreduceNvls->name(), allreduceNvls);
  auto allreduceFullmesh = std::make_shared<AllreduceFullmesh>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allreduceFullmesh->collective(), allreduceFullmesh->name(), allreduceFullmesh);

  auto allgatherFullmesh = std::make_shared<AllgatherFullmesh>(scratchBuffer, scratchBufferSize)->build();
  collection.registerAlgorithm(allgatherFullmesh->collective(), allgatherFullmesh->name(), allgatherFullmesh);
  auto allgatherFullmesh2 = std::make_shared<AllgatherFullmesh2>()->build();
  collection.registerAlgorithm(allgatherFullmesh2->collective(), allgatherFullmesh2->name(), allgatherFullmesh2);
  //   collection.algoSelector_ = algoSelector_;
  //   collection.fallbackAlgoSelector_ = fallbackAlgoSelector_;
  return collection;
}

AlgorithmCollection DefaultAlgorithmBuilder::buildDefaultDslAlgorithms(int rank) {
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
  //   collection.algoSelector_ = algoSelector_;
  //   collection.fallbackAlgoSelector_ = fallbackAlgoSelector_;

  static auto generateFileId = [](const std::string& input) {
    std::hash<std::string> hasher;
    size_t hashValue = hasher(input);
    std::ostringstream oss;
    oss << std::hex << hashValue;
    return oss.str();
  };

  std::string planDir = env()->executionPlanDir;
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