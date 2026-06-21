// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/types.h>

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <tuple>
#include <vector>

#include "config.hpp"
#include "event.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME mscclpp_ep_cpp
#endif

namespace mscclpp {
namespace ep {

struct Buffer {
  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8 || NUM_MAX_NVL_PEERS == 4,
                   "The number of maximum NVLink peers must be 4 or 8");

 private:
  // Low-latency mode buffer
  int low_latency_buffer_idx = 0;
  bool low_latency_mode = false;

  // NVLink Buffer
  int64_t num_nvl_bytes;
  void* buffer_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  void** buffer_ptrs_gpu = nullptr;
  // Increment 3: byte offset of the peer-mapped recv-output pool within the NVL
  // allocation (buffer_ptrs[*] + recv_pool_off_). -1 until set in the ctor.
  int64_t recv_pool_off_ = -1;
  // Increment 4: VMM-allocated (cuMem FABRIC/POSIX-FD via gpuCallocPhysical)
  // recv-output pool. recv_pool_local_ptr_ is this rank's local pool; the peer
  // bases (imported via registerMemory/recvMemory) live in recv_pool_ptrs_ /
  // recv_pool_ptrs_gpu. These are TMA-eligible peer VAs (unlike cudaIpc maps).
  // Non-null recv_pool_local_ptr_ selects the increment-4 VMM direct-write path.
  void* recv_pool_local_ptr_ = nullptr;
  std::vector<void*> recv_pool_ptrs_;
  void** recv_pool_ptrs_gpu = nullptr;
  // Keep imported peer RegisteredMemory alive so the cuMem mapping persists for
  // the Buffer's lifetime (recv_pool_ptrs_[*] alias their .data()).
  std::vector<mscclpp::RegisteredMemory> recv_pool_remote_mems_;
  // Increment 5 (inc5 flat-domain dispatch): domain-wide recv-pool bases indexed
  // by GLOBAL rank (all num_ranks across the NVLink domain), so the RDMA sender
  // can write each token directly into the destination GPU's recv pool. Populated
  // only when MSCCLPP_EP_DIRECT is set. recv_pool_global_ptrs_[rank]==local pool.
  std::vector<void*> recv_pool_global_ptrs_;
  void** recv_pool_global_ptrs_gpu = nullptr;
  std::vector<mscclpp::RegisteredMemory> recv_pool_global_remote_mems_;
  // Increment 5 combine-direct (Stage 1): per-(source token, dst global rank)
  // recv-pool slot index written by the dispatch sender (= ep_my_idx). Lets the
  // combine path gather each token's contributions straight from the peer pools.
  // Allocated [kEpRecvPoolMaxTokens * num_ranks] ints under MSCCLPP_EP_DIRECT.
  int* ep_combine_recv_idx_gpu = nullptr;

  // NVSHMEM Buffer
  int64_t num_rdma_bytes;
  void* rdma_buffer_ptr = nullptr;

  // Device info and communication
  int device_id;
  int rank, rdma_rank, nvl_rank;
  int num_ranks, num_rdma_ranks, num_nvl_ranks;
  cudaIpcMemHandle_t ipc_handles[NUM_MAX_NVL_PEERS];

  // Stream for communication
  at::cuda::CUDAStream comm_stream;

  // After IPC/NVSHMEM synchronization, this flag will be true
  bool available = false;

  // Task fifo
  int head = 0;
  int* task_fifo_ptrs[NUM_MAX_NVL_PEERS] = {nullptr};
  int** task_fifo_ptrs_gpu = nullptr;

  // Workspace
  void* workspace = nullptr;

  // Host-side MoE info
  volatile int* moe_recv_counter = nullptr;
  int* moe_recv_counter_mapped = nullptr;

  // Host-side expert-level MoE info
  volatile int* moe_recv_expert_counter = nullptr;
  int* moe_recv_expert_counter_mapped = nullptr;

  // Host-side RDMA-level MoE info
  volatile int* moe_recv_rdma_counter = nullptr;
  int* moe_recv_rdma_counter_mapped = nullptr;

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  // One ProxyService spawns a single proxy thread that drains every PortChannel
  // FIFO it owns. With LL combine pushing thousands of triggers per iter, the
  // single thread becomes the wall-clock bottleneck on cross-node runs. We
  // shard channels across `proxy_services` so each gets its own thread/FIFO,
  // increasing host-side dispatch parallelism (no kernel changes required).
  // Count is resolved at construction (env `MSCCLPP_EP_NUM_PROXIES` or
  // arch-aware default).
  int num_proxy_services = 1;
  std::vector<std::shared_ptr<mscclpp::ProxyService>> proxy_services;
  std::shared_ptr<mscclpp::Communicator> communicator;
  std::vector<mscclpp::PortChannel> port_channels;
  std::vector<mscclpp::MemoryChannel> memory_channels;
  std::shared_ptr<mscclpp::PortChannelDeviceHandle> port_channel_handles_device_ptr;
  std::shared_ptr<mscclpp::MemoryChannelDeviceHandle> memory_channel_handles_device_ptr;

  // LL fast path: peer-mapped RDMA buffer pointers.
  // ``peer_rdma_bases[r]`` aliases rank ``r``'s ``rdma_buffer_ptr`` through
  // mscclpp's CudaIpc transport. Intranode peers use POSIX-FD CUDA IPC;
  // cross-node peers use cuMem fabric handles routed through nvidia-imex
  // over the NVL72 NVSwitch fabric (Proposal A — replaces RDMA atomicAdd
  // with NVLink atomics, since Azure CX-7 RoCE has IBV_ATOMIC_NONE).
  // Populated in ``sync()`` when ``low_latency_mode``; empty otherwise.
  std::vector<void*> peer_rdma_bases;
  void** peer_rdma_bases_gpu = nullptr;
  // Base MemoryChannels over CUDA IPC used only for the LL barrier ring.
  std::vector<mscclpp::MemoryChannel> ll_memory_channels;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> ll_memory_channel_handles_device_ptr;
  int ll_ranks_per_ipc_domain = 0;
  bool ll_ipc_ready = false;

  // NVLS multicast for HT internode (Wide Proposal B2).
  //
  // When `mscclpp::isNvlsSupported()` is true and `num_rdma_ranks > 1`,
  // we set up a multicast-bound buffer carrying:
  //   - tail counters[num_channels][num_rdma_ranks][num_rdma_ranks] uint64_t
  //   - head counters[num_channels][num_rdma_ranks][num_rdma_ranks] uint64_t
  //   - notify_dispatch barrier epoch[num_rdma_ranks] uint64_t
  //   - notify_dispatch small-data slots[num_rdma_ranks][kSummaryBytes]
  //
  // Cross-node atomic adds use `multimem.red.add.u64` PTX which travels
  // over the NVL72 fabric instead of broken IB atomics on Azure CX-7 RoCE.
  // The kernels select between this NVLS path and the legacy PortChannel
  // path at runtime based on `nvls_ht_enabled`.
  //
  // Falls back gracefully on platforms without NVLS multicast support
  // (e.g. H100+IB, A100+IB clusters): `nvls_ht_enabled` stays `false`,
  // all NVLS pointers stay `nullptr`, and the original PortChannel
  // signal/wait + atomicAdd path remains active.
  bool nvls_ht_enabled = false;
  std::shared_ptr<mscclpp::NvlsConnection> nvls_ht_conn;
  // SwitchChannel keeps the multicast pointer alive (its destructor
  // unbinds the multicast); device pointers below are extracted from it.
  std::shared_ptr<mscclpp::SwitchChannel> nvls_ht_sc;
  // Underlying GpuBuffer (multicast-eligible physical alloc); kept alive
  // for the lifetime of the multicast binding.
  std::shared_ptr<mscclpp::GpuBuffer<uint8_t>> nvls_ht_buffer;
  // mc_ptr: multicast-side device pointer (writes hit all peers via switch).
  // dev_ptr: local-side device pointer (reads see local copy of the same
  // physical memory).
  void* nvls_ht_mc_ptr = nullptr;
  void* nvls_ht_dev_ptr = nullptr;
  // Sub-region byte offsets within the multicast buffer (set in sync()).
  size_t nvls_ht_off_tail = 0;
  size_t nvls_ht_off_head = 0;
  size_t nvls_ht_off_barrier = 0;
  size_t nvls_ht_off_data = 0;
  size_t nvls_ht_total_bytes = 0;
  // Per-call epoch counter for NVLS barrier slots. Incremented on the host
  // before each kernel launch that uses an NVLS barrier; the kernel spins
  // until the barrier slot reaches `epoch * num_ranks`.
  uint64_t nvls_ht_epoch = 0;
  // Independent epoch for cached_notify barrier slots (offsets +24 / +32),
  // since those slots are only touched when the cached path is taken — using
  // the shared `nvls_ht_epoch` would over-count the expected value relative
  // to the number of times those particular slots have actually been bumped.
  uint64_t nvls_ht_cached_epoch = 0;
  // Worst-case shape parameters used to size the buffer:
  //   stride_per_channel = num_rdma_ranks * num_rdma_ranks (counter slots)
  // We allocate for `kNvlsMaxChannels` so any `num_sms` config fits.
  static constexpr int kNvlsMaxChannels = 64;     // num_sms / 2 upper bound
  static constexpr int kNvlsPerPeerBytes = 1024;  // small-data per (sender, receiver) pair
  // Number of distinct barrier slots in the barrier sub-region (each u64).
  static constexpr int kNvlsBarrierSlots = 8;

 private:
  void move_fifo_slots(int num_slots = 1);

 public:
  Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode);

  ~Buffer() noexcept(false);

  bool is_available() const;

  bool is_internode_available() const;

  int get_num_rdma_ranks() const;

  int get_rdma_rank() const;

  int get_root_rdma_rank(bool global) const;

  int get_local_device_id() const;

  pybind11::bytearray get_local_ipc_handle() const;

  pybind11::bytearray get_local_nvshmem_unique_id() const;

  torch::Tensor get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const;

  mscclpp::UniqueId create_unique_id() const;

  void connect(mscclpp::UniqueId root_id);

  void sync(const std::vector<int>& device_ids,
            const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
            const std::optional<pybind11::bytearray>& root_unique_id_opt);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
  get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event,
                      bool async, bool allocate_on_comm_stream);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
             std::optional<EventHandle>>
  intranode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                     const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                     const std::optional<torch::Tensor>& num_tokens_per_rank, const torch::Tensor& is_token_in_rank,
                     const std::optional<torch::Tensor>& num_tokens_per_expert, int cached_num_recv_tokens,
                     const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
                     const std::optional<torch::Tensor>& cached_channel_prefix_matrix, int expert_alignment,
                     const Config& config, std::optional<EventHandle>& previous_event, bool async,
                     bool allocate_on_comm_stream);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> intranode_combine(
      const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights, const torch::Tensor& src_idx,
      const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix,
      const torch::Tensor& send_head, const Config& config, std::optional<EventHandle>& previous_event, bool async,
      bool allocate_on_comm_stream);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor,
             std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
             std::optional<torch::Tensor>, std::optional<EventHandle>>
  internode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                     const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                     const std::optional<torch::Tensor>& num_tokens_per_rank,
                     const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                     const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                     int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                     const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
                     const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                     const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
                     const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum, int expert_alignment,
                     const Config& config, std::optional<EventHandle>& previous_event, bool async,
                     bool allocate_on_comm_stream);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> internode_combine(
      const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights, const torch::Tensor& src_meta,
      const torch::Tensor& is_combined_token_in_rank, const torch::Tensor& rdma_channel_prefix_matrix,
      const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
      const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head, const Config& config,
      std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream);

  void clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);

  std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor,
             std::optional<EventHandle>, std::optional<std::function<void()>>>
  low_latency_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx, int num_max_dispatch_tokens_per_rank,
                       int num_experts, bool use_fp8, bool async, bool return_recv_hook,
                       const std::optional<torch::Tensor>& out_packed_recv_x = std::nullopt,
                       const std::optional<torch::Tensor>& out_packed_recv_x_scales = std::nullopt,
                       const std::optional<torch::Tensor>& out_packed_recv_src_info = std::nullopt,
                       const std::optional<torch::Tensor>& out_packed_recv_layout_range = std::nullopt,
                       const std::optional<torch::Tensor>& out_packed_recv_count = std::nullopt);

  std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> low_latency_combine(
      const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales, const torch::Tensor& topk_idx,
      const torch::Tensor& topk_weights, const torch::Tensor& src_info, const torch::Tensor& layout_range,
      int num_max_dispatch_tokens_per_rank, int num_experts, bool zero_copy, bool async, bool return_recv_hook,
      const std::optional<torch::Tensor>& out = std::nullopt);

  torch::Tensor get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts);
};

}  // namespace ep
}  // namespace mscclpp
