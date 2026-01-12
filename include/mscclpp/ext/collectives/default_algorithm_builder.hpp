#ifndef MSCCLPP_EXT_COLLECTIVES_DEFAULT_ALGORITHM_BUILDER_HPP_
#define MSCCLPP_EXT_COLLECTIVES_DEFAULT_ALGORITHM_BUILDER_HPP_

#include <mscclpp/algorithm.hpp>

namespace mscclpp {
namespace collective {
class DefaultAlgorithmBuilder {
 public:
  AlgorithmCollection buildDefaultAlgorithms(uintptr_t scratchBuffer, size_t scratchBufferSize, int rank);

 private:
  AlgorithmCollection buildDefaultNativeAlgorithms(uintptr_t scratchBuffer, size_t scratchBufferSize);
  AlgorithmCollection buildDefaultDslAlgorithms(int rank);
};
}  // namespace algorithms
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_COLLECTIVES_DEFAULT_ALGORITHM_BUILDER_HPP_