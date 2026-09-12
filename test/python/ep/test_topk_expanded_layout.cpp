// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "config.hpp"

using namespace mscclpp::ep;
using namespace mscclpp::ep::low_latency;

namespace {

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

size_t alignBytes(size_t bytes) {
  return ((bytes + BufferAlignmentBytes - 1) / BufferAlignmentBytes) * BufferAlignmentBytes;
}

void requireNoExpandedPointers(const Layout& layout) {
  require(layout.expandedSendIds_ == nullptr, "unexpected expanded send IDs");
  require(layout.expandedSendWeights_ == nullptr, "unexpected expanded send weights");
  require(layout.expandedSyncFlags_ == nullptr, "unexpected expanded sync flags");
  require(layout.expandedSyncEpoch_ == nullptr, "unexpected expanded sync epoch");
  require(layout.expandedCounts_ == nullptr, "unexpected expanded receiver counts");
  require(layout.expandedCountStaging_ == nullptr, "unexpected expanded sender counts");
}

void checkLayout(int ranks, int capacity, int topk, int hidden) {
  const int experts = ranks * 4;
  const PayloadView<Bf16> bf16(hidden, topk);
  const PayloadView<Fp8E4M3, float> fp8(hidden, topk, 128);
  const size_t sourceRows = static_cast<size_t>(capacity) * topk;
  const size_t rows = static_cast<size_t>(ranks) * sourceRows;
  const size_t tokenBytes = static_cast<size_t>(ranks) * capacity * hidden * sizeof(Bf16);
  const size_t metadataBytes = alignBytes(static_cast<size_t>(ranks + experts) * sizeof(uint64_t));
  const size_t dispatchBytes =
      metadataBytes + static_cast<size_t>(ranks) * capacity * alignBytes(std::max(bf16.numBytes_, fp8.numBytes_));
  const size_t idsOffset = alignBytes(static_cast<size_t>(ranks + experts) * sizeof(mscclpp::LL8Packet));
  const size_t weightsOffset = idsOffset + alignBytes(rows * sizeof(int));
  const size_t tokenOffset = weightsOffset + alignBytes(rows * sizeof(float));
  const size_t slotStride =
      alignBytes(static_cast<size_t>(hidden) * sizeof(Bf16) + topk * (sizeof(int) + sizeof(float)));
  const size_t stagingBytes = alignBytes(static_cast<size_t>(GpuNetIoStagingSlots) * slotStride);
  const size_t flagsBytes = alignBytes(static_cast<size_t>(ranks) * GpuNetIoMaxQpsPerPeer * sizeof(uint64_t));

  // As in the original layout unit, inspect synthetic addresses only: no GPU
  // allocation, pointer dereference, kernel launch, or runtime protocol is involved.
  constexpr uintptr_t base = 0x100000;
  const auto offset = [](const void* pointer) { return reinterpret_cast<uintptr_t>(pointer) - base; };
  require(rankMajorTopkIdsOffset(ranks, experts) == idsOffset, "public ID offset changed");
  require(rankMajorTopkWeightsOffset(ranks, experts, capacity, topk) == weightsOffset, "public weight offset changed");
  require(rankMajorTokenOffset(ranks, experts, capacity, topk) == tokenOffset, "public token offset changed");

  // Independent pre-expanded formulas guard both expert-major and compact rank-major
  // allocation sizes and every legacy region, including explicit false and defaults.
  for (bool rankMajor : {false, true}) {
    const size_t combineBytes =
        rankMajor ? tokenBytes : static_cast<size_t>(experts) * capacity * hidden * sizeof(Bf16);
    const size_t expectedRecv = alignBytes(std::max({dispatchBytes, tokenOffset + tokenBytes, combineBytes}));
    const size_t expectedTotal = 2 * expectedRecv + stagingBytes + 2 * flagsBytes + alignBytes(tokenBytes);
    const Layout legacy(reinterpret_cast<void*>(base), capacity, hidden, ranks, experts, topk, rankMajor);
    const Layout explicitLegacy(reinterpret_cast<void*>(base), capacity, hidden, ranks, experts, topk, rankMajor,
                                false);
    const Layout sizing(nullptr, capacity, hidden, ranks, experts, topk, rankMajor);
    for (const Layout* layout : {&legacy, &explicitLegacy}) {
      require(layout->recvBufferBytes_ == expectedRecv, "legacy receive allocation changed");
      require(layout->totalBytes_ == expectedTotal, "legacy total allocation changed");
      require(layout->gpuNetIoSlotStride_ == slotStride, "legacy staging stride changed");
      require(offset(layout->dispatchRecvBuffer_) == 0, "legacy dispatch base changed");
      require(offset(layout->combineRecvBuffer_) == expectedRecv, "legacy combine base changed");
      require(offset(layout->rankMajorTopkIdsBuffer_) == idsOffset, "legacy IDs moved");
      require(offset(layout->rankMajorTopkWeightsBuffer_) == weightsOffset, "legacy weights moved");
      require(offset(layout->rankMajorTokenBuffer_) == tokenOffset, "legacy tokens moved");
      require(layout->rankMajorExpertOutputBuffer_ == layout->combineRecvBuffer_, "legacy expert output moved");
      require(layout->rankMajorExpertOutputBuffer_ != layout->rankMajorTokenBuffer_,
              "legacy expert output aliases tokens");
      require(offset(layout->gpuNetIoStagingBuffer_) == 2 * expectedRecv, "legacy staging moved");
      require(offset(layout->gpuNetIoFlagsBuffer_) == 2 * expectedRecv + stagingBytes, "legacy dispatch flags moved");
      require(offset(layout->gpuNetIoCombineFlagsBuffer_) == 2 * expectedRecv + stagingBytes + flagsBytes,
              "legacy combine flags moved");
      require(offset(layout->gpuNetIoCombineLandingBuffer_) == 2 * expectedRecv + stagingBytes + 2 * flagsBytes,
              "legacy landing moved");
      requireNoExpandedPointers(*layout);
    }
    require(sizing.recvBufferBytes_ == expectedRecv && sizing.totalBytes_ == expectedTotal, "legacy sizing differs");
    requireNoExpandedPointers(sizing);
    require(symmetricBufferSize(capacity, hidden, ranks, experts, topk, rankMajor) == expectedTotal,
            "legacy size helper changed");
    require(symmetricBufferSize(capacity, hidden, ranks, experts, topk, rankMajor, false) == expectedTotal,
            "explicit legacy size helper changed");
    if (!rankMajor) {
      const Layout defaults(nullptr, capacity, hidden, ranks, experts, topk);
      require(defaults.recvBufferBytes_ == expectedRecv && defaults.totalBytes_ == expectedTotal,
              "layout defaults changed");
      require(symmetricBufferSize(capacity, hidden, ranks, experts, topk) == expectedTotal, "size defaults changed");
      requireNoExpandedPointers(defaults);
    }
  }

  const Layout expanded(reinterpret_cast<void*>(base), capacity, hidden, ranks, experts, topk, true, true);
  const Layout expandedSizing(nullptr, capacity, hidden, ranks, experts, topk, true, true);
  const size_t expandedTokenBytes = tokenBytes * topk;
  const size_t expandedRecv =
      alignBytes(std::max({dispatchBytes, tokenOffset + expandedTokenBytes, expandedTokenBytes}));
  const size_t expandedStagingBytes =
      alignBytes(static_cast<size_t>(std::max(GpuNetIoStagingSlots, capacity)) * slotStride);
  require(expanded.recvBufferBytes_ == expandedRecv, "expanded receive allocation differs");
  require(offset(expanded.dispatchRecvBuffer_) == 0, "expanded dispatch base differs");
  require(offset(expanded.combineRecvBuffer_) == expandedRecv, "expanded second receive region moved");
  require(offset(expanded.rankMajorTopkIdsBuffer_) == idsOffset, "expanded public IDs moved");
  require(offset(expanded.rankMajorTopkWeightsBuffer_) == weightsOffset, "expanded public weights moved");
  require(offset(expanded.rankMajorTokenBuffer_) == tokenOffset, "expanded public tokens moved");
  const void* expertOutput = expanded.rankMajorExpertOutputBuffer_;
  require(expertOutput == expanded.rankMajorTokenBuffer_, "expanded expert output must alias tokens");
  require(expertOutput != expanded.combineRecvBuffer_, "expanded expert output uses compact region");
  require(tokenOffset + expandedTokenBytes <= expandedRecv, "expanded token rows overflow");
  require(expanded.gpuNetIoSlotStride_ == slotStride, "expanded staging stride changed");
  require(static_cast<size_t>(capacity) * slotStride <= expandedStagingBytes, "per-token source staging overflows");

  // Walk the tail in registration order. Exact offsets plus aligned sizes prove
  // non-overlap and retain all per-destination metadata until sends complete.
  size_t cursor = 2 * expandedRecv;
  const auto region = [&](void* pointer, size_t bytes) {
    require(offset(pointer) == cursor, "expanded region offset differs");
    require(offset(pointer) % BufferAlignmentBytes == 0, "expanded region is not 128-byte aligned");
    require(cursor + bytes <= expanded.totalBytes_, "expanded region exceeds registered allocation");
    cursor += alignBytes(bytes);
  };
  region(expanded.gpuNetIoStagingBuffer_, expandedStagingBytes);
  region(expanded.gpuNetIoFlagsBuffer_, flagsBytes);
  region(expanded.gpuNetIoCombineFlagsBuffer_, flagsBytes);
  region(expanded.gpuNetIoCombineLandingBuffer_, expandedTokenBytes);
  region(expanded.expandedSendIds_, rows * sizeof(int));
  region(expanded.expandedSendWeights_, rows * sizeof(float));
  region(expanded.expandedSyncFlags_, static_cast<size_t>(ranks) * sizeof(uint64_t));
  region(expanded.expandedSyncEpoch_, sizeof(uint64_t));
  region(expanded.expandedCounts_, static_cast<size_t>(ranks) * sizeof(int));
  region(expanded.expandedCountStaging_, static_cast<size_t>(ranks) * sizeof(int));
  require(cursor == expanded.totalBytes_, "expanded total allocation differs");
  require(expanded.totalBytes_ % BufferAlignmentBytes == 0, "expanded total is not aligned");
  require(expandedSizing.recvBufferBytes_ == expandedRecv && expandedSizing.totalBytes_ == cursor,
          "expanded null-base sizing differs");
  requireNoExpandedPointers(expandedSizing);
  require(symmetricBufferSize(capacity, hidden, ranks, experts, topk, true, true) == cursor,
          "expanded size helper differs");

  // Check the last peer/token/slot against the next region, not merely one source's staging image.
  const size_t lastEntry =
      static_cast<size_t>(ranks - 1) * sourceRows + static_cast<size_t>(capacity - 1) * topk + topk - 1;
  require(lastEntry + 1 == rows, "last expanded slot differs");
  require(offset(expanded.expandedSendIds_) + (lastEntry + 1) * sizeof(int) <= offset(expanded.expandedSendWeights_),
          "last peer IDs overlap weights");
  require(
      offset(expanded.expandedSendWeights_) + (lastEntry + 1) * sizeof(float) <= offset(expanded.expandedSyncFlags_),
      "last peer weights overlap synchronization");
  require(offset(expanded.expandedCounts_) + static_cast<size_t>(ranks) * sizeof(int) <=
              offset(expanded.expandedCountStaging_),
          "receiver counts overlap sender counts");
}

size_t checkOriginalShapes() {
  size_t cases = 0;
  for (int ranks : {1, 2, 8, 32, 64}) {
    for (int capacity : {1, 8, 128, 1024}) {
      for (int topk : {1, 2, 8, 9}) {
        for (int hidden : {4096, 7168, 9216}) {
          checkLayout(ranks, capacity, topk, hidden);
          ++cases;
        }
      }
    }
  }
  require(cases == 240, "original shape coverage changed");
  return cases;
}

size_t checkBoundaryShapes() {
  size_t cases = 0;
  // Partial token tiles, the staging-capacity crossover, and int-count alignment.
  for (int ranks : {31, 32, 33}) {
    for (int capacity : {133, GpuNetIoStagingSlots - 1, GpuNetIoStagingSlots, GpuNetIoStagingSlots + 1}) {
      for (int topk : {1, 8, 9}) {
        checkLayout(ranks, capacity, topk, 4096);
        ++cases;
      }
    }
  }
  return cases;
}

}  // namespace

int main() {
  try {
    const size_t shapes = checkOriginalShapes();
    const size_t boundaries = checkBoundaryShapes();
    std::cout << "PASS expanded alias, registered regions, and unchanged legacy allocation: " << shapes << " shapes + "
              << boundaries << " boundary shapes\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL layout invariant: " << error.what() << '\n';
    return 1;
  }
}
