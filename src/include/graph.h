/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MSCCLPP_GRAPH_H_
#define MSCCLPP_GRAPH_H_

#include "mscclpp.h"
#include "devcomm.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#include <sched.h>

// mscclppResult_t mscclppTopoCudaPath(int cudaDev, char** path);

// struct mscclppTopoSystem;
// // Build the topology
// mscclppResult_t mscclppTopoGetSystem(struct mscclppComm* comm, struct mscclppTopoSystem** system);
// mscclppResult_t mscclppTopoSortSystem(struct mscclppTopoSystem* system);
// mscclppResult_t mscclppTopoPrint(struct mscclppTopoSystem* system);

// mscclppResult_t mscclppTopoComputePaths(struct mscclppTopoSystem* system, struct mscclppComm* comm);
// void mscclppTopoFree(struct mscclppTopoSystem* system);
// mscclppResult_t mscclppTopoTrimSystem(struct mscclppTopoSystem* system, struct mscclppComm* comm);
// mscclppResult_t mscclppTopoComputeP2pChannels(struct mscclppComm* comm);
// mscclppResult_t mscclppTopoGetNvbGpus(struct mscclppTopoSystem* system, int rank, int* nranks, int** ranks);
// int mscclppTopoPathAllNVLink(struct mscclppTopoSystem* system);

// // Query topology
// mscclppResult_t mscclppTopoGetNetDev(struct mscclppComm* comm, int rank, struct mscclppTopoGraph* graph, int channelId, int peerRank, int* net, int* proxyRank);
// mscclppResult_t mscclppTopoCheckP2p(struct mscclppTopoSystem* system, int64_t id1, int64_t id2, int* p2p, int *read, int* intermediateRank);
// mscclppResult_t mscclppTopoCheckGdr(struct mscclppTopoSystem* topo, int64_t busId, int netDev, int read, int* useGdr);
// mscclppResult_t mscclppTopoNeedFlush(struct mscclppTopoSystem* system, int64_t busId, int* flush);
// mscclppResult_t mscclppTopoCheckNet(struct mscclppTopoSystem* system, int64_t id1, int64_t id2, int* net);
// int mscclppPxnDisable(struct mscclppComm* comm);
// mscclppResult_t mscclppTopoGetPxnRanks(struct mscclppComm* comm, int** intermediateRanks, int* nranks);
// mscclppResult_t mscclppTopoGetLocalRank(struct mscclppTopoSystem* system, int rank, int* localRank);

// // Find CPU affinity
// mscclppResult_t mscclppTopoGetCpuAffinity(struct mscclppTopoSystem* system, int rank, cpu_set_t* affinity);

// #define MSCCLPP_TOPO_CPU_ARCH_X86 1
// #define MSCCLPP_TOPO_CPU_ARCH_POWER 2
// #define MSCCLPP_TOPO_CPU_ARCH_ARM 3
// #define MSCCLPP_TOPO_CPU_VENDOR_INTEL 1
// #define MSCCLPP_TOPO_CPU_VENDOR_AMD 2
// #define MSCCLPP_TOPO_CPU_VENDOR_ZHAOXIN 3
// #define MSCCLPP_TOPO_CPU_TYPE_BDW 1
// #define MSCCLPP_TOPO_CPU_TYPE_SKL 2
// #define MSCCLPP_TOPO_CPU_TYPE_YONGFENG 1
// mscclppResult_t mscclppTopoCpuType(struct mscclppTopoSystem* system, int* arch, int* vendor, int* model);
// mscclppResult_t mscclppTopoGetNetCount(struct mscclppTopoSystem* system, int* count);
// mscclppResult_t mscclppTopoGetNvsCount(struct mscclppTopoSystem* system, int* count);
// mscclppResult_t mscclppTopoGetLocalNet(struct mscclppTopoSystem* system, int rank, int* id);

#define MSCCLPP_TOPO_MAX_NODES 256

// // Init search. Needs to be done before calling mscclppTopoCompute
// mscclppResult_t mscclppTopoSearchInit(struct mscclppTopoSystem* system);

// #define MSCCLPP_TOPO_PATTERN_BALANCED_TREE 1   // Spread NIC traffic between two GPUs (Tree parent + one child on first GPU, second child on second GPU)
// #define MSCCLPP_TOPO_PATTERN_SPLIT_TREE 2      // Spread NIC traffic between two GPUs (Tree parent on first GPU, tree children on the second GPU)
// #define MSCCLPP_TOPO_PATTERN_TREE 3            // All NIC traffic going to/from the same GPU
// #define MSCCLPP_TOPO_PATTERN_RING 4            // Ring
struct mscclppTopoGraph {
  // Input / output
  int id; // ring : 0, tree : 1, collnet : 2
  int pattern;
  int crossNic;
  int collNet;
  int minChannels;
  int maxChannels;
  // Output
  int nChannels;
  float bwIntra;
  float bwInter;
  float latencyInter;
  int typeIntra;
  int typeInter;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS*MSCCLPP_TOPO_MAX_NODES];
  int inter[MAXCHANNELS*2];
};
// mscclppResult_t mscclppTopoCompute(struct mscclppTopoSystem* system, struct mscclppTopoGraph* graph);

// mscclppResult_t mscclppTopoPrintGraph(struct mscclppTopoSystem* system, struct mscclppTopoGraph* graph);
// mscclppResult_t mscclppTopoDumpGraphs(struct mscclppTopoSystem* system, int ngraphs, struct mscclppTopoGraph** graphs);

// struct mscclppTopoRanks {
//   int ringRecv[MAXCHANNELS];
//   int ringSend[MAXCHANNELS];
//   int ringPrev[MAXCHANNELS];
//   int ringNext[MAXCHANNELS];
//   int treeToParent[MAXCHANNELS];
//   int treeToChild0[MAXCHANNELS];
//   int treeToChild1[MAXCHANNELS];
// };

// mscclppResult_t mscclppTopoPreset(struct mscclppComm* comm,
//     struct mscclppTopoGraph* treeGraph, struct mscclppTopoGraph* ringGraph, struct mscclppTopoGraph* collNetGraph,
//     struct mscclppTopoRanks* topoRanks);

// mscclppResult_t mscclppTopoPostset(struct mscclppComm* comm, int* firstRanks, int* treePatterns,
//     struct mscclppTopoRanks** allTopoRanks, int* rings, struct mscclppTopoGraph* collNetGraph);

// mscclppResult_t mscclppTopoTuneModel(struct mscclppComm* comm, int minCompCap, int maxCompCap, struct mscclppTopoGraph* treeGraph, struct mscclppTopoGraph* ringGraph, struct mscclppTopoGraph* collNetGraph);
// #include "info.h"
// mscclppResult_t mscclppTopoGetAlgoTime(struct mscclppInfo* info, int algorithm, int protocol, int numPipeOps, float* time);

#endif
