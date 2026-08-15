// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_INCLUDE_LAUNCH_CUH_
#define MSCCLPP_EP_INCLUDE_LAUNCH_CUH_

#include "exception.cuh"

namespace mscclpp {
namespace ep {

class LaunchConfig {
 public:
  LaunchConfig(int numBlocks, int numThreads, size_t sharedBytes, cudaStream_t stream, bool cooperative = false)
      : config_{dim3(numBlocks), dim3(numThreads), sharedBytes, stream, nullptr, 0} {
    if (cooperative) {
      attribute_.id = cudaLaunchAttributeCooperative;
      attribute_.val.cooperative = 1;
      config_.attrs = &attribute_;
      config_.numAttrs = 1;
    }
  }

  LaunchConfig(const LaunchConfig&) = delete;
  LaunchConfig& operator=(const LaunchConfig&) = delete;
  LaunchConfig(LaunchConfig&&) = delete;
  LaunchConfig& operator=(LaunchConfig&&) = delete;

  const cudaLaunchConfig_t* get() const { return &config_; }

 private:
  cudaLaunchAttribute attribute_{};
  cudaLaunchConfig_t config_;
};

#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))

// Throughput kernels are specialized for the rank counts supported by the runtime and
// control-buffer layout.
#define SWITCH_RANKS(num_ranks, case_macro)           \
  do {                                                \
    switch (num_ranks) {                              \
      case 2:                                         \
        case_macro(2);                                \
      case 4:                                         \
        case_macro(4);                                \
      case 8:                                         \
        case_macro(8);                                \
      case 16:                                        \
        case_macro(16);                               \
      default:                                        \
        EP_HOST_ASSERT(false && "Unsupported ranks"); \
    }                                                 \
  } while (false)

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_INCLUDE_LAUNCH_CUH_
