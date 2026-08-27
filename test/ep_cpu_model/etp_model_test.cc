// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Host-side model of the latency EXPERT_MAJOR dispatch/combine data flow under
// expert-tensor parallelism (ETP).
//
// This is not a GPU test: it re-executes, on the CPU and in float32, the exact
// index algebra that the CUDA kernels use, by including the real
// ExpertMap / MoETopology / WorkspaceView / LatencyStorageLayout headers. It
// checks that
//
//   1. the (EP, ETP) routing produces, for every rank, the same final combine
//      output as the etpSize == 1 configuration with identical routing and
//      weights, and as a direct dense reference;
//   2. dispatch never overflows a per-rank slot capacity and every ETP rank of
//      a group receives exactly the same token set;
//   3. the metadata / workspace / symmetric-buffer sizes are large enough for
//      every index the model produces.
//
// Build (needs only CUDA headers, no GPU and no nvcc):
//   CUDA_INCLUDE=/usr/local/cuda/include ./test/ep_cpu_model/run_etp_model_test.sh

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "common/latency.cuh"
#include "config.hpp"
#include "expert_map.hpp"

using mscclpp::ep::EtpDispatchMode;
using mscclpp::ep::EtpRankOrder;
using mscclpp::ep::EtpReduceMode;
using mscclpp::ep::ExpertMap;
using mscclpp::ep::MoETopology;
using mscclpp::ep::WorkspaceView;

namespace {

struct Config {
  int numRanks = 16;
  int etpSize = 4;
  int numExperts = 16;
  int numTopk = 3;
  int hidden = 16;
  int intermediate = 8;  // FFN intermediate dim, sharded across the ETP group
  int tokensPerRank = 12;
  int maxTokensPerRank = 16;
  EtpRankOrder order = EtpRankOrder::EP_MAJOR;
  EtpReduceMode reduceMode = EtpReduceMode::GROUP_REDUCE_SCATTER;
  EtpDispatchMode dispatchMode = EtpDispatchMode::LEADER_SINGLE_SEND;
  // FP8-style payload: the token is sent quantized with one scale per
  // scaleBlockSize hidden elements, and the receiver replicates the scale
  // vector into the expert-major scale output.
  bool quantized = false;
  int scaleBlockSize = 4;

  int numScales() const { return quantized ? hidden / scaleBlockSize : 0; }
};

// Deterministic routing/input shared by every configuration under test.
struct Workload {
  std::vector<std::vector<std::vector<float>>> tokens;   // [rank][token][hidden], quantized payload
  std::vector<std::vector<std::vector<float>>> scales;   // [rank][token][numScales]
  std::vector<std::vector<std::vector<float>>> dequant;  // [rank][token][hidden], tokens * scales
  std::vector<std::vector<std::vector<int>>> topkIdx;    // [rank][token][topk]
  std::vector<std::vector<std::vector<float>>> weights;  // [rank][token][topk]
  std::vector<std::vector<float>> w1;                    // [expert][intermediate * hidden]
  std::vector<std::vector<float>> w2;                    // [expert][intermediate * hidden]
};

/// Split a row into (quantized payload, per-block scales) the way FP8 E4M3
/// dispatch does: one scale per scaleBlockSize elements, payload = value/scale.
void quantizeRow(const std::vector<float>& row, int scaleBlockSize, std::vector<float>& payload,
                 std::vector<float>& scales) {
  const int numScales = static_cast<int>(row.size()) / scaleBlockSize;
  payload.assign(row.size(), 0.0f);
  scales.assign(numScales, 1.0f);
  for (int block = 0; block < numScales; ++block) {
    float maxAbs = 1e-4f;
    for (int i = 0; i < scaleBlockSize; ++i) maxAbs = std::max(maxAbs, std::fabs(row[block * scaleBlockSize + i]));
    const float scale = maxAbs / 8.0f;
    scales[block] = scale;
    for (int i = 0; i < scaleBlockSize; ++i) {
      const int index = block * scaleBlockSize + i;
      payload[index] = std::round(row[index] / scale);  // integer-valued payload
    }
  }
}

Workload makeWorkload(const Config& config, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  Workload workload;
  workload.tokens.resize(config.numRanks);
  workload.scales.resize(config.numRanks);
  workload.dequant.resize(config.numRanks);
  workload.topkIdx.resize(config.numRanks);
  workload.weights.resize(config.numRanks);
  for (int rank = 0; rank < config.numRanks; ++rank) {
    for (int token = 0; token < config.tokensPerRank; ++token) {
      std::vector<float> row(config.hidden);
      for (float& value : row) value = uniform(rng);
      if (config.quantized) {
        std::vector<float> payload;
        std::vector<float> scales;
        quantizeRow(row, config.scaleBlockSize, payload, scales);
        std::vector<float> dequantized(config.hidden);
        for (int i = 0; i < config.hidden; ++i) dequantized[i] = payload[i] * scales[i / config.scaleBlockSize];
        workload.tokens[rank].push_back(payload);
        workload.scales[rank].push_back(scales);
        workload.dequant[rank].push_back(dequantized);
      } else {
        workload.tokens[rank].push_back(row);
        workload.scales[rank].push_back({});
        workload.dequant[rank].push_back(row);
      }

      std::vector<int> experts(config.numExperts);
      for (int expert = 0; expert < config.numExperts; ++expert) experts[expert] = expert;
      std::shuffle(experts.begin(), experts.end(), rng);
      experts.resize(config.numTopk);
      workload.topkIdx[rank].push_back(experts);

      std::vector<float> weight(config.numTopk);
      for (float& value : weight) value = 0.25f + 0.5f * std::fabs(uniform(rng));
      workload.weights[rank].push_back(weight);
    }
  }
  for (int expert = 0; expert < config.numExperts; ++expert) {
    std::vector<float> w1(static_cast<size_t>(config.intermediate) * config.hidden);
    std::vector<float> w2(static_cast<size_t>(config.intermediate) * config.hidden);
    for (float& value : w1) value = uniform(rng);
    for (float& value : w2) value = uniform(rng);
    workload.w1.push_back(w1);
    workload.w2.push_back(w2);
  }
  return workload;
}

/// One expert's FFN restricted to the intermediate rows owned by @p tpIndex.
///
/// Summing this over tpIndex in [0, etpSize) reproduces the dense expert, which
/// is exactly the invariant ETP relies on.
std::vector<float> expertShard(const Config& config, const Workload& workload, int expert,
                               const std::vector<float>& token, int tpIndex, int etpSize) {
  std::vector<float> out(config.hidden, 0.0f);
  const int rowsPerShard = config.intermediate / etpSize;
  const int begin = tpIndex * rowsPerShard;
  const int end = begin + rowsPerShard;
  for (int row = begin; row < end; ++row) {
    float activation = 0.0f;
    for (int h = 0; h < config.hidden; ++h) {
      activation += workload.w1[expert][static_cast<size_t>(row) * config.hidden + h] * token[h];
    }
    activation = activation > 0.0f ? activation : 0.01f * activation;  // leaky ReLU
    for (int h = 0; h < config.hidden; ++h) {
      out[h] += activation * workload.w2[expert][static_cast<size_t>(row) * config.hidden + h];
    }
  }
  return out;
}

/// Dense (single-GPU) reference: sum_k weight_k * expert_k(token).
std::vector<std::vector<std::vector<float>>> denseReference(const Config& config, const Workload& workload) {
  std::vector<std::vector<std::vector<float>>> out(config.numRanks);
  for (int rank = 0; rank < config.numRanks; ++rank) {
    for (int token = 0; token < config.tokensPerRank; ++token) {
      std::vector<float> combined(config.hidden, 0.0f);
      for (int k = 0; k < config.numTopk; ++k) {
        const int expert = workload.topkIdx[rank][token][k];
        const float weight = workload.weights[rank][token][k];
        const std::vector<float> value = expertShard(config, workload, expert, workload.dequant[rank][token], 0, 1);
        for (int h = 0; h < config.hidden; ++h) combined[h] += weight * value[h];
      }
      out[rank].push_back(combined);
    }
  }
  return out;
}

struct Payload {
  std::vector<float> data;
  std::vector<float> scales;
  std::vector<int> topkIdx;
  std::vector<float> topkWeights;
  int srcTokenGlobalIdx = -1;
  bool valid = false;
};

struct Stats {
  int maxSlot = 0;
  int maxMetadataSlot = 0;
  size_t maxRowOffsetIndex = 0;
  int maxExpertOutputRow = 0;
};

/// Run the modelled dispatch + combine for one configuration.
std::vector<std::vector<std::vector<float>>> runConfig(const Config& config, const Workload& workload, Stats& stats) {
  const MoETopology topology(config.numRanks, config.etpSize, config.order);
  const int numRanks = config.numRanks;
  const int maxTokens = config.maxTokensPerRank;
  const int numTopk = config.numTopk;

  std::vector<ExpertMap> maps;
  for (int rank = 0; rank < numRanks; ++rank) {
    maps.emplace_back(topology, config.numExperts, rank, config.dispatchMode);
  }
  const int numLocalExperts = maps[0].numLocalExperts();

  // --- dispatch send -------------------------------------------------------
  // recvPayloads[dst][src][slot], rankCounts[src][dst], expertCounts[src][expert]
  std::vector<std::vector<std::vector<Payload>>> recvPayloads(
      numRanks, std::vector<std::vector<Payload>>(numRanks, std::vector<Payload>(maxTokens)));
  std::vector<std::vector<int>> rankCounts(numRanks, std::vector<int>(numRanks, 0));
  std::vector<std::vector<int>> expertCounts(numRanks, std::vector<int>(config.numExperts, 0));

  for (int src = 0; src < numRanks; ++src) {
    const ExpertMap& map = maps[src];
    for (int token = 0; token < config.tokensPerRank; ++token) {
      for (int k = 0; k < numTopk; ++k) expertCounts[src][workload.topkIdx[src][token][k]]++;
      for (int copyIdx = 0; copyIdx < map.numSendCopies(); ++copyIdx) {
        std::set<int> destinations;
        for (int k = 0; k < numTopk; ++k) {
          destinations.insert(map.destinationRank(workload.topkIdx[src][token][k], copyIdx));
        }
        for (int dst : destinations) {
          const int slot = rankCounts[src][dst]++;
          if (slot >= maxTokens) {
            std::fprintf(stderr, "slot overflow: src=%d dst=%d slot=%d\n", src, dst, slot);
            std::abort();
          }
          stats.maxSlot = std::max(stats.maxSlot, slot);
          Payload& payload = recvPayloads[dst][src][slot];
          payload.data = workload.tokens[src][token];
          payload.scales = workload.scales[src][token];
          payload.topkIdx = workload.topkIdx[src][token];
          payload.topkWeights = workload.weights[src][token];
          payload.srcTokenGlobalIdx = src * maxTokens + token;
          payload.valid = true;
        }
      }
    }
  }

  // --- dispatch metadata ---------------------------------------------------
  // metadataRank[dst][src] and metadataExpert[dst][slot] as written by the
  // sender into its destinations' LL8 headers.
  const int metadataSlots = maps[0].metadataExpertSlots();
  std::vector<std::vector<int>> metadataRank(numRanks, std::vector<int>(numRanks, -1));
  std::vector<std::vector<int>> metadataExpert(numRanks, std::vector<int>(metadataSlots, -1));
  for (int src = 0; src < numRanks; ++src) {
    const ExpertMap& map = maps[src];
    for (int dst = 0; dst < numRanks; ++dst) {
      if (map.leaderInGroupOf(dst) != dst) continue;
      metadataRank[dst][src] = rankCounts[src][dst];
    }
    for (int expert = 0; expert < config.numExperts; ++expert) {
      const int slot = map.metadataExpertSlot(src, map.localExpert(expert));
      if (slot >= metadataSlots) {
        std::fprintf(stderr, "metadata slot overflow: %d >= %d\n", slot, metadataSlots);
        std::abort();
      }
      stats.maxMetadataSlot = std::max(stats.maxMetadataSlot, slot);
      for (int copyIdx = 0; copyIdx < map.numSendCopies(); ++copyIdx) {
        metadataExpert[map.destinationRank(expert, copyIdx)][slot] = expertCounts[src][expert];
      }
    }
  }

  // --- dispatch receive ----------------------------------------------------
  const int slotsPerExpert = numRanks * maxTokens;
  // expertInput[rank][localExpert * slotsPerExpert + row]
  std::vector<std::vector<std::vector<float>>> expertInput(
      numRanks, std::vector<std::vector<float>>(static_cast<size_t>(numLocalExperts) * slotsPerExpert));
  std::vector<std::vector<int>> expertInputExpert(
      numRanks, std::vector<int>(static_cast<size_t>(numLocalExperts) * slotsPerExpert, -1));
  // Expert-major scale output, indexed exactly like
  // dispatchRecvExpertMajorOutput(): (localExpert * numScales + s) * slots + row.
  const int numScales = config.numScales();
  std::vector<std::vector<float>> expertInputScales(
      numRanks, std::vector<float>(static_cast<size_t>(numLocalExperts) * numScales * slotsPerExpert, 0.0f));
  // (rank, rowOffset) -> (sourceRank, sourceToken), used to check the replication.
  std::vector<std::map<int, std::pair<int, int>>> rowSource(numRanks);
  // rowOffsets[rank][combineRowOffsetIndex(src, slot, lane)]
  std::vector<std::vector<int>> rowOffsets(
      numRanks, std::vector<int>(static_cast<size_t>(numRanks) * maxTokens * numTopk, -1));
  std::vector<std::vector<int>> receivedTokens(numRanks);  // for the group-consistency check

  for (int rank = 0; rank < numRanks; ++rank) {
    const ExpertMap& map = maps[rank];
    std::vector<std::vector<int>> layoutOffset(numLocalExperts, std::vector<int>(numRanks, 0));
    for (int localExpert = 0; localExpert < numLocalExperts; ++localExpert) {
      int offset = 0;
      for (int src = 0; src < numRanks; ++src) {
        const int peer = map.sourcePeer(src);
        const int count = metadataExpert[peer][map.metadataExpertSlot(src, localExpert)];
        layoutOffset[localExpert][src] = offset;
        offset += count < 0 ? 0 : count;
      }
    }
    std::vector<std::vector<int>> copied(numRanks, std::vector<int>(numLocalExperts, 0));
    for (int src = 0; src < numRanks; ++src) {
      const int peer = map.sourcePeer(src);
      const int count = metadataRank[peer][src];
      if (count < 0) {
        std::fprintf(stderr, "rank %d has no rank count for source %d (peer %d)\n", rank, src, peer);
        std::abort();
      }
      for (int slot = 0; slot < count; ++slot) {
        const Payload& payload = recvPayloads[peer][src][slot];
        if (!payload.valid) {
          std::fprintf(stderr, "rank %d read an empty payload from %d slot %d\n", rank, src, slot);
          std::abort();
        }
        receivedTokens[rank].push_back(payload.srcTokenGlobalIdx);
        for (int lane = 0; lane < numTopk; ++lane) {
          const int expert = payload.topkIdx[lane];
          const size_t offsetIndex =
              WorkspaceView::combineRowOffsetIndex(src, slot, numTopk, maxTokens, lane);
          stats.maxRowOffsetIndex = std::max(stats.maxRowOffsetIndex, offsetIndex);
          if (!map.ownsExpert(expert)) {
            rowOffsets[rank][offsetIndex] = -1;
            continue;
          }
          const int localExpert = map.localExpert(expert);
          const int row = layoutOffset[localExpert][src] + copied[src][localExpert]++;
          const int rowOffset = localExpert * slotsPerExpert + row;
          stats.maxExpertOutputRow = std::max(stats.maxExpertOutputRow, rowOffset);
          expertInput[rank][rowOffset] = payload.data;
          expertInputExpert[rank][rowOffset] = expert;
          rowOffsets[rank][offsetIndex] = rowOffset;
          rowSource[rank][rowOffset] = {src, payload.srcTokenGlobalIdx - src * maxTokens};
          for (int scaleIdx = 0; scaleIdx < numScales; ++scaleIdx) {
            const size_t scaleOffset =
                (static_cast<size_t>(localExpert) * numScales + scaleIdx) * slotsPerExpert + row;
            if (scaleOffset >= expertInputScales[rank].size()) {
              std::fprintf(stderr, "scale index overflow: rank=%d offset=%zu\n", rank, scaleOffset);
              std::abort();
            }
            expertInputScales[rank][scaleOffset] = payload.scales[scaleIdx];
          }
        }
      }
    }
  }

  // Every ETP rank of a group must have received exactly the same token set.
  //
  // Corollary used by test_latency_multirank.py: a per-rank reconstruction of a
  // dispatched row exists on exactly etpSize ranks, so summing those
  // reconstructions across the world counts each row etpSize times unless one
  // rank per group is selected. Getting that wrong makes the *reference*
  // etpSize x too large (measured: max diff 474 at ETP=2, 1422 at ETP=4).
  for (int rank = 0; rank < numRanks; ++rank) {
    const int peer = topology.rankOf(topology.epIndex(rank), 0);
    std::multiset<int> mine(receivedTokens[rank].begin(), receivedTokens[rank].end());
    std::multiset<int> theirs(receivedTokens[peer].begin(), receivedTokens[peer].end());
    if (mine != theirs) {
      std::fprintf(stderr, "rank %d and group peer %d received different token sets\n", rank, peer);
      std::abort();
    }
  }
  for (int src = 0; src < numRanks; ++src) {
    for (int token = 0; token < config.tokensPerRank; ++token) {
      const int globalToken = src * maxTokens + token;
      int holders = 0;
      for (int rank = 0; rank < numRanks; ++rank) {
        holders += static_cast<int>(std::count(receivedTokens[rank].begin(), receivedTokens[rank].end(), globalToken));
      }
      const std::set<int> groups = [&] {
        std::set<int> result;
        for (int k = 0; k < config.numTopk; ++k) {
          result.insert(maps[src].group(workload.topkIdx[src][token][k]));
        }
        return result;
      }();
      const int expected = static_cast<int>(groups.size()) * config.etpSize;
      if (holders != expected) {
        std::fprintf(stderr, "token %d of rank %d is held by %d ranks, expected %d\n", token, src, holders, expected);
        std::abort();
      }
    }
  }

  // Every replicated scale must belong to the row's own source token: this is
  // what a scale vector resolved against the wrong rank would break.
  for (int rank = 0; rank < numRanks; ++rank) {
    const ExpertMap& map = maps[rank];
    for (const auto& entry : rowSource[rank]) {
      const int row = entry.first;
      const int localExpert = row / slotsPerExpert;
      const int rowInExpert = row % slotsPerExpert;
      const int src = entry.second.first;
      const int token = entry.second.second;
      for (int scaleIdx = 0; scaleIdx < numScales; ++scaleIdx) {
        const size_t scaleOffset =
            (static_cast<size_t>(localExpert) * numScales + scaleIdx) * slotsPerExpert + rowInExpert;
        if (expertInputScales[rank][scaleOffset] != workload.scales[src][token][scaleIdx]) {
          std::fprintf(stderr, "rank %d row %d has the wrong FP8 scale for source %d token %d block %d\n", rank, row,
                       src, token, scaleIdx);
          std::abort();
        }
      }
      (void)map;
    }
  }

  // --- expert compute (ETP-sharded FFN) ------------------------------------
  // The expert input is dequantized with the replicated per-block scales, so a
  // mismatched scale vector shows up as a wrong expert output.
  std::vector<std::vector<std::vector<float>>> expertOutput(
      numRanks, std::vector<std::vector<float>>(static_cast<size_t>(numLocalExperts) * slotsPerExpert));
  std::map<std::pair<int, int>, std::vector<std::vector<float>>> groupDequantized;  // (epIndex, srcTokenGlobalIdx)
  for (int rank = 0; rank < numRanks; ++rank) {
    for (size_t row = 0; row < expertInput[rank].size(); ++row) {
      if (expertInputExpert[rank][row] < 0) continue;
      std::vector<float> dequantized = expertInput[rank][row];
      if (numScales > 0) {
        const int localExpert = static_cast<int>(row) / slotsPerExpert;
        const int rowInExpert = static_cast<int>(row) % slotsPerExpert;
        for (int h = 0; h < config.hidden; ++h) {
          const int scaleIdx = h / config.scaleBlockSize;
          const size_t scaleOffset =
              (static_cast<size_t>(localExpert) * numScales + scaleIdx) * slotsPerExpert + rowInExpert;
          dequantized[h] *= expertInputScales[rank][scaleOffset];
        }
      }
      const auto& source = rowSource[rank].at(static_cast<int>(row));
      const std::pair<int, int> key{topology.epIndex(rank), source.first * maxTokens + source.second};
      groupDequantized[key].push_back(dequantized);
      expertOutput[rank][row] = expertShard(config, workload, expertInputExpert[rank][row], dequantized,
                                            topology.tpIndex(rank), config.etpSize);
    }
  }

  // Every ETP rank of a group must dequantize a shared token identically.
  for (const auto& entry : groupDequantized) {
    for (size_t i = 1; i < entry.second.size(); ++i) {
      if (entry.second[i] != entry.second[0]) {
        std::fprintf(stderr, "EP group %d dequantized source token %d differently across ETP ranks\n",
                     entry.first.first, entry.first.second);
        std::abort();
      }
    }
  }

  // --- combine -------------------------------------------------------------
  // combineRecv[dstRank][senderRank * maxTokens + token]
  std::vector<std::vector<std::vector<float>>> combineRecv(
      numRanks, std::vector<std::vector<float>>(static_cast<size_t>(numRanks) * maxTokens,
                                                std::vector<float>(config.hidden, 0.0f)));
  std::vector<std::vector<bool>> combineRecvValid(
      numRanks, std::vector<bool>(static_cast<size_t>(numRanks) * maxTokens, false));
  // Staging buffer of the group reduce-scatter, indexed like etpStageRow().
  std::vector<std::vector<std::vector<float>>> etpStage(
      numRanks, std::vector<std::vector<float>>(static_cast<size_t>(numRanks) * maxTokens,
                                                std::vector<float>(config.hidden, 0.0f)));
  std::vector<std::vector<bool>> etpStageValid(
      numRanks, std::vector<bool>(static_cast<size_t>(numRanks) * maxTokens, false));

  const bool groupReduceScatter = topology.isEtpEnabled() && config.reduceMode == EtpReduceMode::GROUP_REDUCE_SCATTER;
  for (int rank = 0; rank < numRanks; ++rank) {
    const ExpertMap& map = maps[rank];
    for (int src = 0; src < numRanks; ++src) {
      const int peer = map.sourcePeer(src);
      const int count = metadataRank[peer][src];
      for (int slot = 0; slot < count; ++slot) {
        const Payload& payload = recvPayloads[peer][src][slot];
        std::vector<float> reduced(config.hidden, 0.0f);
        for (int lane = 0; lane < numTopk; ++lane) {
          const int rowOffset =
              rowOffsets[rank][WorkspaceView::combineRowOffsetIndex(src, slot, numTopk, maxTokens, lane)];
          if (rowOffset < 0) continue;
          const float weight = payload.topkWeights[lane];
          for (int h = 0; h < config.hidden; ++h) reduced[h] += weight * expertOutput[rank][rowOffset][h];
        }
        const int sourceTokenIdx = payload.srcTokenGlobalIdx - src * maxTokens;
        if (groupReduceScatter) {
          const int leader = map.groupLeaderFor(src);
          const size_t row =
              (static_cast<size_t>(topology.tpIndex(rank)) * topology.epSize + topology.epIndex(src)) * maxTokens +
              sourceTokenIdx;
          etpStage[leader][row] = reduced;
          etpStageValid[leader][row] = true;
        } else {
          combineRecv[src][static_cast<size_t>(rank) * maxTokens + sourceTokenIdx] = reduced;
          combineRecvValid[src][static_cast<size_t>(rank) * maxTokens + sourceTokenIdx] = true;
        }
      }
    }
  }

  if (groupReduceScatter) {
    for (int rank = 0; rank < numRanks; ++rank) {
      const ExpertMap& map = maps[rank];
      for (int src = 0; src < numRanks; ++src) {
        if (map.groupLeaderFor(src) != rank) continue;  // only the leader owns this source
        const int count = metadataRank[rank][src];
        for (int slot = 0; slot < count; ++slot) {
          const Payload& payload = recvPayloads[rank][src][slot];
          const int sourceTokenIdx = payload.srcTokenGlobalIdx - src * maxTokens;
          std::vector<float> summed(config.hidden, 0.0f);
          for (int contributor = 0; contributor < config.etpSize; ++contributor) {
            const size_t row =
                (static_cast<size_t>(contributor) * topology.epSize + topology.epIndex(src)) * maxTokens +
                sourceTokenIdx;
            if (!etpStageValid[rank][row]) {
              std::fprintf(stderr, "missing staged partial: leader=%d src=%d tp=%d\n", rank, src, contributor);
              std::abort();
            }
            for (int h = 0; h < config.hidden; ++h) summed[h] += etpStage[rank][row][h];
          }
          combineRecv[src][static_cast<size_t>(rank) * maxTokens + sourceTokenIdx] = summed;
          combineRecvValid[src][static_cast<size_t>(rank) * maxTokens + sourceTokenIdx] = true;
        }
      }
    }
  }

  // Source-side reduction over the distinct partial senders of each token.
  const int numContributors =
      topology.isEtpEnabled() && config.reduceMode == EtpReduceMode::SOURCE_SIDE ? config.etpSize : 1;
  std::vector<std::vector<std::vector<float>>> output(config.numRanks);
  for (int rank = 0; rank < numRanks; ++rank) {
    const ExpertMap& map = maps[rank];
    for (int token = 0; token < config.tokensPerRank; ++token) {
      std::vector<float> combined(config.hidden, 0.0f);
      std::set<int> seenPartialRanks;
      for (int lane = 0; lane < numTopk; ++lane) {
        const int partialRank = map.leaderRank(workload.topkIdx[rank][token][lane]);
        if (!seenPartialRanks.insert(partialRank).second) continue;  // warp-level rank dedup
        for (int contributor = 0; contributor < numContributors; ++contributor) {
          const int contributorRank =
              numContributors == 1 ? partialRank : topology.rankOf(topology.epIndex(partialRank), contributor);
          const size_t row = static_cast<size_t>(contributorRank) * maxTokens + token;
          if (!combineRecvValid[rank][row]) {
            std::fprintf(stderr, "missing partial: rank=%d token=%d from=%d\n", rank, token, contributorRank);
            std::abort();
          }
          for (int h = 0; h < config.hidden; ++h) combined[h] += combineRecv[rank][row][h];
        }
      }
      output[rank].push_back(combined);
    }
  }
  return output;
}

void expectClose(const std::vector<std::vector<std::vector<float>>>& lhs,
                 const std::vector<std::vector<std::vector<float>>>& rhs, const char* what, float tolerance = 1e-4f) {
  double worst = 0.0;
  for (size_t rank = 0; rank < lhs.size(); ++rank) {
    for (size_t token = 0; token < lhs[rank].size(); ++token) {
      for (size_t h = 0; h < lhs[rank][token].size(); ++h) {
        worst = std::max<double>(worst, std::fabs(lhs[rank][token][h] - rhs[rank][token][h]));
      }
    }
  }
  std::printf("  %-58s max abs diff = %.3e\n", what, worst);
  if (!(worst <= tolerance)) {
    std::fprintf(stderr, "FAILED: %s exceeds tolerance %g\n", what, tolerance);
    std::abort();
  }
}

void checkBufferSizes(const Config& config, const Stats& stats) {
  const MoETopology topology(config.numRanks, config.etpSize, config.order);
  const size_t workspace = WorkspaceView::numBytes(config.numRanks, config.numExperts, config.maxTokensPerRank,
                                                   config.numTopk, topology.epSize);
  std::vector<uint8_t> storage(workspace, 0);
  const ExpertMap map(topology, config.numExperts, 0, config.dispatchMode);
  WorkspaceView view(storage.data(), map, config.maxTokensPerRank, config.numTopk);
  const auto* base = reinterpret_cast<const uint8_t*>(storage.data());
  const auto* rowOffsetEnd = reinterpret_cast<const uint8_t*>(
      view.dispatchCombineRowOffsets_ + stats.maxRowOffsetIndex + 1);
  if (rowOffsetEnd > base + workspace) {
    std::fprintf(stderr, "workspace overflow: row offsets end past the workspace\n");
    std::abort();
  }
  const size_t symmetricBytes = mscclpp::ep::latencyStorageSize(
      config.maxTokensPerRank, config.hidden, config.numRanks, config.numExperts, config.numTopk,
      mscclpp::ep::DispatchLayout::EXPERT_MAJOR, mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE, topology,
      config.reduceMode, config.dispatchMode);
  const mscclpp::ep::LatencyStorageLayout layout(nullptr, config.maxTokensPerRank, config.hidden, config.numRanks,
                                                 config.numExperts, config.numTopk,
                                                 mscclpp::ep::DispatchLayout::EXPERT_MAJOR,
                                                 mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE, topology,
                                                 config.reduceMode, config.dispatchMode);
  const size_t expertRowBytes = static_cast<size_t>(stats.maxExpertOutputRow + 1) * config.hidden * sizeof(uint16_t);
  if (expertRowBytes > layout.dispatchOutputBytes_) {
    std::fprintf(stderr, "expert-major output overflow: %zu > %zu\n", expertRowBytes, layout.dispatchOutputBytes_);
    std::abort();
  }
  if (mscclpp::ep::dispatchMetadataExpertSlots(topology, config.numExperts, config.dispatchMode) <=
      stats.maxMetadataSlot) {
    std::fprintf(stderr, "metadata region too small\n");
    std::abort();
  }
  if (config.etpSize > 1 && config.reduceMode == EtpReduceMode::GROUP_REDUCE_SCATTER) {
    if (layout.etpReduceBufferBytes_ <
        static_cast<size_t>(config.numRanks) * config.maxTokensPerRank * config.hidden * sizeof(uint16_t)) {
      std::fprintf(stderr, "ETP staging buffer too small\n");
      std::abort();
    }
  } else if (layout.etpReduceBufferBytes_ != 0) {
    std::fprintf(stderr, "ETP staging buffer allocated when it should not be\n");
    std::abort();
  }
  std::printf("  buffer sizes ok (symmetric %zu B, workspace %zu B)\n", symmetricBytes, workspace);
}

const char* quantName(bool quantized) { return quantized ? "fp8-style" : "bf16-style"; }

void runMatrix(const Config& baseline);

const char* reduceName(EtpReduceMode mode) {
  switch (mode) {
    case EtpReduceMode::SOURCE_SIDE:
      return "SOURCE_SIDE";
    case EtpReduceMode::GROUP_REDUCE_SCATTER:
      return "GROUP_REDUCE_SCATTER";
    default:
      return "GROUP_NVLS";
  }
}

}  // namespace

int main() {
  for (bool quantized : {false, true}) {
    Config baseline;  // 16 ranks, ETP=4
    baseline.quantized = quantized;
    runMatrix(baseline);
  }
  std::printf("all ETP model checks passed\n");
  return 0;
}

namespace {

void runMatrix(const Config& baseline) {
  const Workload workload = makeWorkload(baseline, 1234u);

  Config plain = baseline;
  plain.etpSize = 1;
  Stats plainStats;
  std::printf("[%s] EP=%d ETP=1 (reference run)\n", quantName(baseline.quantized), plain.numRanks);
  const auto plainOutput = runConfig(plain, workload, plainStats);
  checkBufferSizes(plain, plainStats);
  const auto reference = denseReference(plain, workload);
  expectClose(plainOutput, reference, "etpSize=1 vs dense single-GPU reference");

  for (int etpSize : {2, 4, 8}) {
    for (EtpRankOrder order : {EtpRankOrder::EP_MAJOR, EtpRankOrder::TP_MAJOR}) {
      for (EtpReduceMode reduceMode : {EtpReduceMode::GROUP_REDUCE_SCATTER, EtpReduceMode::SOURCE_SIDE}) {
        for (EtpDispatchMode dispatchMode :
             {EtpDispatchMode::LEADER_SINGLE_SEND, EtpDispatchMode::DUPLICATE_SEND}) {
          Config config = baseline;
          config.etpSize = etpSize;
          config.order = order;
          config.reduceMode = reduceMode;
          config.dispatchMode = dispatchMode;
          if (config.intermediate % etpSize != 0) continue;
          std::printf("[%s] EP=%d ETP=%d order=%s reduce=%s dispatch=%s\n", quantName(config.quantized),
                      config.numRanks / etpSize, etpSize,
                      order == EtpRankOrder::EP_MAJOR ? "EP_MAJOR" : "TP_MAJOR", reduceName(reduceMode),
                      dispatchMode == EtpDispatchMode::LEADER_SINGLE_SEND ? "LEADER" : "DUPLICATE");
          Stats stats;
          const auto output = runConfig(config, workload, stats);
          checkBufferSizes(config, stats);
          expectClose(output, plainOutput, "ETP output vs etpSize=1 output");
          expectClose(output, reference, "ETP output vs dense single-GPU reference");
        }
      }
    }
  }
}

}  // namespace
