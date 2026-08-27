// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/ext/ep/types.hpp>
#include <set>

#include "../framework.hpp"

using mscclpp::ep::EtpRankOrder;
using mscclpp::ep::MoETopology;

TEST(EpTopology, DefaultIsPlainExpertParallel) {
  const MoETopology topology(16, 1);
  EXPECT_EQ(topology.numRanks, 16);
  EXPECT_EQ(topology.epSize, 16);
  EXPECT_EQ(topology.etpSize, 1);
  EXPECT_FALSE(topology.isEtpEnabled());
  for (int rank = 0; rank < 16; ++rank) {
    EXPECT_EQ(topology.epIndex(rank), rank);
    EXPECT_EQ(topology.tpIndex(rank), 0);
    EXPECT_EQ(topology.rankOf(rank, 0), rank);
  }
  // With etpSize == 1 the leader of an expert is its owning rank.
  const int numExpertsPerGroup = topology.numExpertsPerGroup(64);
  EXPECT_EQ(numExpertsPerGroup, 4);
  for (int expert = 0; expert < 64; ++expert) {
    EXPECT_EQ(topology.expertGroup(expert, numExpertsPerGroup), expert / 4);
    EXPECT_EQ(topology.localExpert(expert, numExpertsPerGroup), expert % 4);
    for (int src = 0; src < 16; ++src) {
      EXPECT_EQ(topology.leaderRank(expert, numExpertsPerGroup, topology.tpIndex(src)), expert / 4);
    }
  }
}

TEST(EpTopology, EpMajorGridIsABijection) {
  const MoETopology topology(16, 4);
  EXPECT_EQ(topology.epSize, 4);
  EXPECT_EQ(topology.etpSize, 4);
  EXPECT_TRUE(topology.isEtpEnabled());
  std::set<int> seen;
  for (int ep = 0; ep < topology.epSize; ++ep) {
    for (int tp = 0; tp < topology.etpSize; ++tp) {
      const int rank = topology.rankOf(ep, tp);
      EXPECT_GE(rank, 0);
      EXPECT_LT(rank, 16);
      EXPECT_EQ(topology.epIndex(rank), ep);
      EXPECT_EQ(topology.tpIndex(rank), tp);
      seen.insert(rank);
    }
  }
  EXPECT_EQ(seen.size(), 16u);
  // EP_MAJOR keeps the TP group contiguous.
  EXPECT_EQ(topology.rankOf(1, 2), 6);
}

TEST(EpTopology, TpMajorGridIsABijection) {
  const MoETopology topology(16, 4, EtpRankOrder::TP_MAJOR);
  std::set<int> seen;
  for (int ep = 0; ep < topology.epSize; ++ep) {
    for (int tp = 0; tp < topology.etpSize; ++tp) {
      const int rank = topology.rankOf(ep, tp);
      EXPECT_EQ(topology.epIndex(rank), ep);
      EXPECT_EQ(topology.tpIndex(rank), tp);
      seen.insert(rank);
    }
  }
  EXPECT_EQ(seen.size(), 16u);
  // TP_MAJOR keeps the EP group contiguous.
  EXPECT_EQ(topology.rankOf(1, 2), 9);
}

TEST(EpTopology, LeaderSelectionIsBalancedAndInGroup) {
  const MoETopology topology(16, 4);
  const int numExperts = 8;
  const int numExpertsPerGroup = topology.numExpertsPerGroup(numExperts);
  EXPECT_EQ(numExpertsPerGroup, 2);
  for (int expert = 0; expert < numExperts; ++expert) {
    std::set<int> leaders;
    for (int src = 0; src < 16; ++src) {
      const int leader = topology.leaderRank(expert, numExpertsPerGroup, topology.tpIndex(src));
      // The leader is in the expert's group ...
      EXPECT_EQ(topology.epIndex(leader), topology.expertGroup(expert, numExpertsPerGroup));
      // ... and shares the source's tpIndex, so the a2a is a permutation.
      EXPECT_EQ(topology.tpIndex(leader), topology.tpIndex(src));
      EXPECT_TRUE(topology.ownsExpert(leader, expert, numExpertsPerGroup));
      leaders.insert(leader);
    }
    // Every group member is the leader for exactly numRanks/etpSize sources.
    EXPECT_EQ(leaders.size(), static_cast<size_t>(topology.etpSize));
  }
}

TEST(EpTopology, ExpertOwnershipIsGroupWide) {
  const MoETopology topology(16, 4);
  const int numExperts = 8;
  const int numExpertsPerGroup = topology.numExpertsPerGroup(numExperts);
  for (int expert = 0; expert < numExperts; ++expert) {
    int owners = 0;
    for (int rank = 0; rank < 16; ++rank) {
      if (topology.ownsExpert(rank, expert, numExpertsPerGroup)) ++owners;
    }
    EXPECT_EQ(owners, topology.etpSize);
  }
}
