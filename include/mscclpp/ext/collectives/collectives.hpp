#ifndef MSCCLPP_EXT_COLLECTIVES_COLLECTIVES_HPP_
#define MSCCLPP_EXT_COLLECTIVES_COLLECTIVES_HPP_

#include <mscclpp/algorithm.hpp>

namespace mscclpp {
namespace collective {
class Collectives {
 public:
  AlgorithmCollection buildDefaultAlgorithms(uintptr_t scratchBuffer, size_t scratchBufferSize, int rank);

 private:
  AlgorithmCollection buildDefaultNativeAlgorithms(uintptr_t scratchBuffer, size_t scratchBufferSize);
  AlgorithmCollection buildDefaultDslAlgorithms(int rank);
};
}  // namespace algorithms
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_COLLECTIVES_COLLECTIVES_HPP_