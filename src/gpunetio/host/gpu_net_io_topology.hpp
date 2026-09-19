// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_TOPOLOGY_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_TOPOLOGY_HPP_

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace mscclpp {
namespace detail {
namespace gpunetio {

struct HcaTopology {
  std::string name;
  std::string pciPath;
  int numaNode;
};

inline int pciPathDistance(const std::string& left, const std::string& right) {
  if (left.empty() || right.empty()) return 1 << 10;
  size_t common = 0;
  size_t leftPos = 0;
  size_t rightPos = 0;
  while (leftPos < left.size() && rightPos < right.size()) {
    const size_t leftEnd = left.find('/', leftPos + 1);
    const size_t rightEnd = right.find('/', rightPos + 1);
    const size_t leftLen = (leftEnd == std::string::npos ? left.size() : leftEnd) - leftPos;
    const size_t rightLen = (rightEnd == std::string::npos ? right.size() : rightEnd) - rightPos;
    if (leftLen != rightLen || left.compare(leftPos, leftLen, right, rightPos, rightLen) != 0) break;
    ++common;
    leftPos = leftEnd == std::string::npos ? left.size() : leftEnd;
    rightPos = rightEnd == std::string::npos ? right.size() : rightEnd;
  }
  const int leftDepth = static_cast<int>(std::count(left.begin(), left.end(), '/'));
  const int rightDepth = static_cast<int>(std::count(right.begin(), right.end(), '/'));
  return leftDepth + rightDepth - 2 * static_cast<int>(common);
}

inline int hcaAffinityScore(const std::string& gpuPath, int gpuNumaNode, const HcaTopology& hca) {
  constexpr int NumaMismatchPenalty = 1 << 20;
  int score = pciPathDistance(gpuPath, hca.pciPath);
  if (gpuNumaNode >= 0 && hca.numaNode >= 0 && gpuNumaNode != hca.numaNode) {
    score += NumaMismatchPenalty;
  }
  return score;
}

inline std::vector<std::string> selectClosestHcas(const std::string& gpuPath, int gpuNumaNode,
                                                  const std::vector<HcaTopology>& hcas) {
  int bestAffinity = std::numeric_limits<int>::max();
  std::vector<std::string> names;
  for (const auto& hca : hcas) {
    const int affinity = hcaAffinityScore(gpuPath, gpuNumaNode, hca);
    if (affinity < bestAffinity) {
      bestAffinity = affinity;
      names.clear();
    }
    if (affinity == bestAffinity) names.push_back(hca.name);
  }
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace gpunetio
}  // namespace detail
}  // namespace mscclpp

#endif  // MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_TOPOLOGY_HPP_