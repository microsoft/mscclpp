// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "kernels/api.cuh"

namespace mscclpp {
namespace ep {

class MoERuntime {
 public:
  MoERuntime(mscclpp::Communicator& communicator, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode);
  ~MoERuntime() noexcept(false);

  bool is_available() const;
  bool is_internode_available() const;
  int get_num_rdma_ranks() const;
  int get_rdma_rank() const;
  int get_root_rdma_rank(bool global) const;
  int get_local_device_id() const;
  std::string get_local_ipc_handle() const;

  void dispatch(void* output, float* output_scales, int* output_src_info, int64_t* output_layout, int* output_count,
                const void* input, const int64_t* topk_idx, int num_tokens, int hidden, int num_topk,
                int num_max_dispatch_tokens_per_rank, int num_experts, bool use_fp8,
                low_latency::DispatchLayout dispatch_layout, cudaStream_t stream);

  void combine(void* output, const void* input, const float* input_scales, const int64_t* topk_idx,
               const float* topk_weights, const int* src_info, const int64_t* layout_range, int num_tokens, int hidden,
               int num_topk, int num_max_dispatch_tokens_per_rank, int num_experts, bool requires_dequantization,
               cudaStream_t stream);

 private:
  int low_latency_buffer_idx_ = 0;
  int rank_;
  int rdma_rank_;
  int nvl_rank_;
  int num_ranks_;
  int num_rdma_ranks_;
  int num_nvl_ranks_;
  int device_id_;
  int64_t num_nvl_bytes_;
  int64_t num_rdma_bytes_;
  bool low_latency_mode_;
  bool available_ = false;
  int num_proxy_services_ = 1;
  int ll_ranks_per_ipc_domain_ = 0;
  bool ll_ipc_ready_ = false;

  void* rdma_buffer_ptr_ = nullptr;
  void* workspace_ = nullptr;
  cudaStream_t comm_stream_ = nullptr;

  mscclpp::Communicator* communicator_ = nullptr;
  std::vector<std::shared_ptr<mscclpp::ProxyService>> proxy_services_;
  std::vector<mscclpp::PortChannel> port_channels_;
  std::vector<mscclpp::MemoryChannel> memory_channels_;
  std::shared_ptr<mscclpp::PortChannelDeviceHandle> port_channel_handles_device_ptr_;

  std::vector<void*> peer_rdma_bases_;
  void** peer_rdma_bases_gpu_ = nullptr;
  std::vector<mscclpp::MemoryChannel> ll_memory_channels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> ll_memory_channel_handles_device_ptr_;

  void move_fifo_slots(int num_slots = 1);
  void setup();
};

}  // namespace ep
}  // namespace mscclpp
