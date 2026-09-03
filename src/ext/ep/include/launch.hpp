// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_LAUNCH_HPP_
#define MSCCLPP_EP_LAUNCH_HPP_

#include "exception.hpp"

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

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_LAUNCH_HPP_
