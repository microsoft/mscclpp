// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "config.hpp"

using namespace mscclpp::ep;
using namespace mscclpp::ep::low_latency;

static void require(bool value) {
  if (!value) throw std::runtime_error("layout invariant failed");
}

int main() {
  size_t cases = 0;
  for (int ranks : {1, 2, 8, 32, 64}) {
    for (int capacity : {1, 8, 128, 1024}) {
      for (int topk : {1, 2, 8, 9}) {
        for (int hidden : {4096, 7168, 9216}) {
          const int experts = ranks * 4;
          const PayloadView<Bf16> bf16(hidden, topk);
          const PayloadView<Fp8E4M3, float> fp8(hidden, topk, 128);
          const auto align = [](size_t n) { return configAlign<size_t>(n, BufferAlignmentBytes); };
          const size_t tokenBytes = static_cast<size_t>(ranks) * capacity * hidden * sizeof(Bf16);
          const size_t metadataBytes = align(static_cast<size_t>(ranks + experts) * sizeof(uint64_t));
          const size_t dispatchBytes =
              metadataBytes + static_cast<size_t>(ranks) * capacity * align(std::max(bf16.numBytes_, fp8.numBytes_));
          const size_t tokenOffset = rankMajorTokenOffset(ranks, experts, capacity, topk);
          const size_t staging =
              align(GpuNetIoStagingSlots * align(hidden * sizeof(Bf16) + topk * (sizeof(int) + sizeof(float))));
          const size_t flags = align(static_cast<size_t>(ranks) * GpuNetIoMaxQpsPerPeer * sizeof(uint64_t));
          for (bool rankMajor : {false, true}) {
            const size_t combineBytes =
                rankMajor ? tokenBytes : static_cast<size_t>(experts) * capacity * hidden * sizeof(Bf16);
            const size_t expectedRecv = align(std::max({dispatchBytes, tokenOffset + tokenBytes, combineBytes}));
            const Layout legacy(nullptr, capacity, hidden, ranks, experts, topk, rankMajor);
            require(legacy.recvBufferBytes_ == expectedRecv);
            require(legacy.totalBytes_ == 2 * expectedRecv + staging + 2 * flags + align(tokenBytes));
          }
          const uintptr_t base = 0x100000;
          const Layout expanded(reinterpret_cast<void*>(base), capacity, hidden, ranks, experts, topk, true, true);
          const auto offset = [base](void* ptr) { return reinterpret_cast<uintptr_t>(ptr) - base; };
          const size_t rows = static_cast<size_t>(ranks) * capacity * topk;
          require(offset(expanded.rankMajorTokenBuffer_) + tokenBytes * topk <= expanded.recvBufferBytes_);
          require(offset(expanded.rankMajorExpertOutputBuffer_) == expanded.recvBufferBytes_);
          require(tokenBytes * topk <= expanded.recvBufferBytes_);
          require(offset(expanded.gpuNetIoStagingBuffer_) == 2 * expanded.recvBufferBytes_);
          require(offset(expanded.gpuNetIoStagingBuffer_) + capacity * expanded.gpuNetIoSlotStride_ <=
                  offset(expanded.gpuNetIoFlagsBuffer_));
          require(offset(expanded.gpuNetIoCombineLandingBuffer_) + tokenBytes * topk <=
                  offset(expanded.expandedSendIds_));
          require(offset(expanded.expandedSendIds_) + rows * sizeof(int) <= offset(expanded.expandedSendWeights_));
          require(offset(expanded.expandedSendWeights_) + rows * sizeof(float) <= offset(expanded.expandedSyncFlags_));
          require(offset(expanded.expandedSyncFlags_) + ranks * sizeof(uint64_t) <=
                  offset(expanded.expandedSyncEpoch_));
          require(offset(expanded.expandedSyncEpoch_) + sizeof(uint64_t) <= expanded.totalBytes_);
          require(symmetricBufferSize(capacity, hidden, ranks, experts, topk, true, true) == expanded.totalBytes_);
          ++cases;
        }
      }
    }
  }
  std::cout << "PASS expanded allocation and legacy size invariants: " << cases << " shapes\n";
}