// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <cuda.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <future>
#include <map>
#include <string>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

namespace mscclpp {
namespace ep {

namespace {

using EPProxyService = mscclpp::ProxyService;

int local_world_size() {
  int local_world_size = NUM_MAX_NVL_PEERS;
  if (const char* env = std::getenv("MSCCLPP_EP_LOCAL_WORLD_SIZE")) {
    int v = std::atoi(env);
    if (v > 0 && v <= NUM_MAX_NVL_PEERS) local_world_size = v;
  }
  return local_world_size;
}

int resolve_num_proxy_services(int num_ranks, int local_world_size) {
  if (const char* env = std::getenv("MSCCLPP_EP_NUM_PROXIES")) {
    int v = std::atoi(env);
    return v > 0 ? v : 1;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 8;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 8;
  int lws = local_world_size > 0 ? local_world_size : 1;
  int num_rdma_ranks_local = std::max(1, num_ranks / lws);
  if (prop.major >= 10) return num_rdma_ranks_local > 1 ? lws : 1;
  return 8;
}

bool resolve_fabric_ipc_supported() {
  if (const char* env = std::getenv("MSCCLPP_EP_FABRIC_IPC")) {
    std::string v(env);
    for (auto& c : v) c = std::tolower(static_cast<unsigned char>(c));
    if (v == "0" || v == "off" || v == "false" || v == "no") return false;
    if (v == "1" || v == "on" || v == "true" || v == "yes" || v == "force") return true;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  CUdevice cu_dev;
  if (cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) return false;
  int supported = 0;
  if (cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cu_dev) != CUDA_SUCCESS) {
    return false;
  }
  if (!supported) return false;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
  return prop.major >= 10;
}

}  // namespace

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, int64_t num_nvl_bytes, int64_t num_rdma_bytes,
                       bool low_latency_mode)
    : rank_(communicator.bootstrap()->getRank()),
      num_ranks_(communicator.bootstrap()->getNranks()),
      num_nvl_bytes_(num_nvl_bytes),
      num_rdma_bytes_(num_rdma_bytes),
      low_latency_mode_(low_latency_mode),
      communicator_(&communicator) {
  EP_HOST_ASSERT(low_latency_mode_);
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(num_nvl_bytes_ == 0);
  EP_HOST_ASSERT(num_rdma_bytes_ % NUM_BUFFER_ALIGNMENT_BYTES == 0);
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < num_ranks_);

  CUDA_CHECK(cudaGetDevice(&device_id_));
  int lws = local_world_size();
  rdma_rank_ = rank_ / lws;
  nvl_rank_ = rank_ % lws;
  num_rdma_ranks_ = std::max(1, num_ranks_ / lws);
  num_nvl_ranks_ = std::min(num_ranks_, lws);

  num_proxy_services_ = resolve_num_proxy_services(num_ranks_, lws);
  proxy_services_.reserve(num_proxy_services_);
  for (int i = 0; i < num_proxy_services_; ++i) proxy_services_.emplace_back(std::make_shared<EPProxyService>());

  CUDA_CHECK(cudaStreamCreateWithFlags(&comm_stream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&workspace_, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemsetAsync(workspace_, 0, NUM_WORKSPACE_BYTES, comm_stream_));
  for (auto& ps : proxy_services_) ps->startProxy();
}

MoERuntime::~MoERuntime() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  if (peer_rdma_bases_gpu_ != nullptr) CUDA_CHECK(cudaFree(peer_rdma_bases_gpu_));
  for (auto& ps : proxy_services_) ps->stopProxy();
  if (workspace_ != nullptr) CUDA_CHECK(cudaFree(workspace_));
  if (rdma_buffer_ptr_ != nullptr) {
    // gpuCallocPhysical allocations are intentionally process-lifetime in the
    // existing EP runtime. cudaFree is safe for the fallback cudaMalloc path.
    // Ignore failures from fabric allocations during process teardown.
    cudaFree(rdma_buffer_ptr_);
  }
  if (comm_stream_ != nullptr) CUDA_CHECK(cudaStreamDestroy(comm_stream_));
}

bool MoERuntime::is_available() const { return available_; }
bool MoERuntime::is_internode_available() const { return is_available() && num_ranks_ > NUM_MAX_NVL_PEERS; }
int MoERuntime::get_num_rdma_ranks() const { return num_rdma_ranks_; }
int MoERuntime::get_rdma_rank() const { return rdma_rank_; }
int MoERuntime::get_root_rdma_rank(bool global) const { return global ? nvl_rank_ : 0; }
int MoERuntime::get_local_device_id() const { return device_id_; }
std::string MoERuntime::get_local_ipc_handle() const { return {}; }

void MoERuntime::sync(const std::vector<int>& device_ids,
                      const std::vector<std::optional<std::string>>& all_gathered_handles,
                      const std::optional<std::string>& root_unique_id_opt) {
  (void)device_ids;
  (void)all_gathered_handles;
  (void)root_unique_id_opt;
  EP_HOST_ASSERT(!available_);
  EP_HOST_ASSERT(communicator_ != nullptr);

  const std::vector<mscclpp::Transport> ib_transports = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5, mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  const auto ipc_transport = mscclpp::Transport::CudaIpc;
  const auto ib_transport = ib_transports[device_id_];
  const mscclpp::TransportFlags all_transport = ipc_transport | ib_transport;

  const bool fabric_ipc_supported = resolve_fabric_ipc_supported();
  const bool use_fabric_ipc_alloc = mscclpp::isNvlsSupported() && fabric_ipc_supported;
  if (use_fabric_ipc_alloc) {
    rdma_buffer_ptr_ = mscclpp::detail::gpuCallocPhysical(num_rdma_bytes_);
  } else {
    CUDA_CHECK(cudaMalloc(&rdma_buffer_ptr_, num_rdma_bytes_));
  }
  CUDA_CHECK(cudaMemset(rdma_buffer_ptr_, 0, num_rdma_bytes_));
  communicator_->bootstrap()->barrier();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::map<int, mscclpp::MemoryId> memory_ids;
  auto add_memory_to_all = [&](mscclpp::RegisteredMemory mem) -> mscclpp::MemoryId {
    mscclpp::MemoryId id = static_cast<mscclpp::MemoryId>(-1);
    for (auto& ps : proxy_services_) {
      auto cur = ps->addMemory(mem);
      if (id == static_cast<mscclpp::MemoryId>(-1)) id = cur;
      EP_HOST_ASSERT(cur == id);
    }
    return id;
  };

  auto local_rdma_buffer_mem = communicator_->registerMemory(rdma_buffer_ptr_, num_rdma_bytes_, all_transport);
  memory_ids[rank_] = add_memory_to_all(local_rdma_buffer_mem);

  constexpr int kRdmaTag = 1;
  for (int r = 0; r < num_ranks_; ++r) {
    if (r != rank_) communicator_->sendMemory(local_rdma_buffer_mem, r, kRdmaTag);
  }
  for (int r = 0; r < num_ranks_; ++r) {
    if (r == rank_) continue;
    auto mem = communicator_->recvMemory(r, kRdmaTag).get();
    memory_ids[r] = add_memory_to_all(std::move(mem));
  }

  std::unordered_map<int, std::vector<mscclpp::Connection>> connections;
  const mscclpp::EndpointConfig ipc_cfg(ipc_transport);
  const mscclpp::EndpointConfig ib_cfg(ib_transport);
  connections[rank_].emplace_back(communicator_->connect(ipc_cfg, rank_, kRdmaTag).get());

  constexpr int kNumIbConnectionsPerRank = 12;
  for (int r = 0; r < num_ranks_; ++r) {
    if (r == rank_) continue;
    std::vector<std::shared_future<mscclpp::Connection>> futures;
    futures.reserve(kNumIbConnectionsPerRank);
    for (int i = 0; i < kNumIbConnectionsPerRank; ++i)
      futures.emplace_back(communicator_->connect(ib_cfg, r, kRdmaTag));
    for (auto& f : futures) connections[r].emplace_back(f.get());
  }

  std::unordered_map<int, std::vector<std::pair<int, mscclpp::SemaphoreId>>> sema_ids;
  constexpr int kNumSemaphoresPerRank = 16;
  for (int i = 0; i < kNumSemaphoresPerRank; ++i) {
    for (int r = 0; r < num_ranks_; ++r) {
      auto& conns = connections[r];
      auto& conn = conns[i % conns.size()];
      int proxy_idx = (i * num_ranks_ + r) % num_proxy_services_;
      auto sema_id = proxy_services_[proxy_idx]->buildAndAddSemaphore(*communicator_, conn);
      sema_ids[r].emplace_back(proxy_idx, sema_id);
    }
  }

  std::vector<mscclpp::PortChannelDeviceHandle> port_channel_handles;
  for (int i = 0; i < kNumSemaphoresPerRank; ++i) {
    for (int r = 0; r < num_ranks_; ++r) {
      auto [proxy_idx, sema_id] = sema_ids[r][i % sema_ids[r].size()];
      auto port_channel = proxy_services_[proxy_idx]->portChannel(sema_id, memory_ids[r], memory_ids[rank_]);
      port_channels_.emplace_back(std::move(port_channel));
      port_channel_handles.emplace_back(port_channels_.rbegin()->deviceHandle());
    }
  }
  port_channel_handles_device_ptr_ =
      mscclpp::detail::gpuCallocShared<mscclpp::PortChannelDeviceHandle>(port_channel_handles.size());
  mscclpp::gpuMemcpy<mscclpp::PortChannelDeviceHandle>(port_channel_handles_device_ptr_.get(),
                                                       port_channel_handles.data(), port_channel_handles.size(),
                                                       cudaMemcpyHostToDevice);

  const int ipc_domain_size = use_fabric_ipc_alloc ? num_ranks_ : num_nvl_ranks_;
  auto is_ipc_peer = [&](int peer) {
    return peer != rank_ && ipc_domain_size > 1 && rank_ / ipc_domain_size == peer / ipc_domain_size;
  };
  const bool want_peer_ipc = use_fabric_ipc_alloc || (low_latency_mode_ && ipc_domain_size > 1);
  if (want_peer_ipc) {
    constexpr int kLlIpcTag = 2;
    auto rdma_mem_ipc = communicator_->registerMemory(rdma_buffer_ptr_, num_rdma_bytes_, ipc_transport);
    std::vector<std::shared_future<mscclpp::RegisteredMemory>> remote_futures(num_ranks_);
    for (int r = 0; r < num_ranks_; ++r) {
      if (r == rank_ || !is_ipc_peer(r)) continue;
      communicator_->sendMemory(rdma_mem_ipc, r, kLlIpcTag);
      remote_futures[r] = communicator_->recvMemory(r, kLlIpcTag);
    }

    std::vector<mscclpp::Connection> ll_ipc_conns(num_ranks_);
    std::vector<std::shared_future<mscclpp::Connection>> conn_futures(num_ranks_);
    for (int r = 0; r < num_ranks_; ++r) {
      if (r == rank_ || !is_ipc_peer(r)) continue;
      conn_futures[r] = communicator_->connect(ipc_cfg, r, kLlIpcTag);
    }
    for (int r = 0; r < num_ranks_; ++r) {
      if (r == rank_ || !is_ipc_peer(r)) continue;
      ll_ipc_conns[r] = conn_futures[r].get();
    }

    peer_rdma_bases_.assign(num_ranks_, nullptr);
    peer_rdma_bases_[rank_] = rdma_buffer_ptr_;
    std::vector<mscclpp::RegisteredMemory> remote_mems(num_ranks_);
    for (int r = 0; r < num_ranks_; ++r) {
      if (r == rank_ || !is_ipc_peer(r)) continue;
      remote_mems[r] = remote_futures[r].get();
      peer_rdma_bases_[r] = remote_mems[r].data();
    }
    CUDA_CHECK(cudaMalloc(&peer_rdma_bases_gpu_, sizeof(void*) * num_ranks_));
    CUDA_CHECK(
        cudaMemcpy(peer_rdma_bases_gpu_, peer_rdma_bases_.data(), sizeof(void*) * num_ranks_, cudaMemcpyHostToDevice));

    std::vector<mscclpp::BaseMemoryChannelDeviceHandle> ll_handles(num_ranks_);
    for (int r = 0; r < num_ranks_; ++r) {
      if (r == rank_ || !is_ipc_peer(r)) continue;
      auto sema = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator_, ll_ipc_conns[r]);
      ll_memory_channels_.emplace_back(sema, remote_mems[r], rdma_mem_ipc);
      ll_handles[r] = ll_memory_channels_.rbegin()->deviceHandle();
    }
    ll_memory_channel_handles_device_ptr_ =
        mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(num_ranks_);
    mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(ll_memory_channel_handles_device_ptr_.get(),
                                                               ll_handles.data(), num_ranks_, cudaMemcpyHostToDevice);
    ll_ranks_per_ipc_domain_ = ipc_domain_size;
    ll_ipc_ready_ = ipc_domain_size >= num_ranks_;
  }

  available_ = true;
}

void MoERuntime::dispatch(void* output, float* output_scales, int* output_src_info, int64_t* output_layout,
                          int* output_count, const void* input, const int64_t* topk_idx, int num_tokens, int hidden,
                          int num_topk, int num_max_dispatch_tokens_per_rank, int num_experts, bool use_fp8,
                          cudaStream_t stream) {
  EP_HOST_ASSERT(low_latency_mode_);
  EP_HOST_ASSERT(hidden % sizeof(int4) == 0 && hidden % 128 == 0);
  EP_HOST_ASSERT(num_tokens <= num_max_dispatch_tokens_per_rank);
  EP_HOST_ASSERT(num_experts % num_ranks_ == 0);

  LowLatencyLayout layout(rdma_buffer_ptr_, num_max_dispatch_tokens_per_rank, hidden, num_ranks_, num_experts);
  EP_HOST_ASSERT(layout.total_bytes <= static_cast<size_t>(num_rdma_bytes_));
  auto buffer = layout.buffers[low_latency_buffer_idx_];
  auto next_buffer = layout.buffers[low_latency_buffer_idx_ ^= 1];
  auto next_clean_meta = next_buffer.clean_meta();

  low_latency::DispatchConfig config{.numTokens_ = num_tokens,
                                     .hidden_ = hidden,
                                     .numTopk_ = num_topk,
                                     .numExperts_ = num_experts,
                                     .numMaxTokensPerRank_ = num_max_dispatch_tokens_per_rank,
                                     .inputDType_ = low_latency::DType::BF16,
                                     .outputDType_ = use_fp8 ? low_latency::DType::F8E4M3 : low_latency::DType::BF16};
  low_latency::BufferSet current_buf{.sendDataBuffer_ = buffer.dispatch_rdma_send_buffer,
                                     .sendCountBuffer_ = nullptr,
                                     .recvDataBuffer_ = buffer.dispatch_rdma_recv_data_buffer,
                                     .recvCountBuffer_ = buffer.dispatch_rdma_recv_count_buffer,
                                     .cleanupRegion_ = nullptr,
                                     .cleanupSize_ = 0};
  low_latency::BufferSet next_buf{.sendDataBuffer_ = nullptr,
                                  .sendCountBuffer_ = nullptr,
                                  .recvDataBuffer_ = nullptr,
                                  .recvCountBuffer_ = nullptr,
                                  .cleanupRegion_ = next_clean_meta.first,
                                  .cleanupSize_ = next_clean_meta.second};
  low_latency::TransportContext transport{
      .rdmaBufferBase_ = rdma_buffer_ptr_,
      .portChannels_ = port_channel_handles_device_ptr_.get(),
      .memoryChannels_ = ll_memory_channel_handles_device_ptr_ ? ll_memory_channel_handles_device_ptr_.get() : nullptr,
      .peerBases_ = peer_rdma_bases_gpu_,
      .ipcReady_ = ll_ipc_ready_,
      .rank_ = rank_,
      .numRanks_ = num_ranks_,
      .ranksPerIpcDomain_ = ll_ranks_per_ipc_domain_};
  low_latency::dispatch(output, output_scales, output_src_info, output_layout, output_count, input, topk_idx, config,
                        current_buf, next_buf, transport, workspace_, stream, low_latency::SEND_AND_RECV);
}

void MoERuntime::combine(void* output, const void* input, const float* input_scales, const int64_t* topk_idx,
                         const float* topk_weights, const int* src_info, const int64_t* layout_range, int num_tokens,
                         int hidden, int num_topk, int num_max_dispatch_tokens_per_rank, int num_experts,
                         bool input_is_fp8, cudaStream_t stream) {
  EP_HOST_ASSERT(low_latency_mode_);
  EP_HOST_ASSERT(hidden % sizeof(int4) == 0 && hidden % 128 == 0);
  EP_HOST_ASSERT(num_experts % num_ranks_ == 0);

  LowLatencyLayout layout(rdma_buffer_ptr_, num_max_dispatch_tokens_per_rank, hidden, num_ranks_, num_experts);
  EP_HOST_ASSERT(layout.total_bytes <= static_cast<size_t>(num_rdma_bytes_));
  auto buffer = layout.buffers[low_latency_buffer_idx_];
  auto next_buffer = layout.buffers[low_latency_buffer_idx_ ^= 1];
  auto next_clean_meta = next_buffer.clean_meta();

  low_latency::CombineConfig config{.numCombinedTokens_ = num_tokens,
                                    .hidden_ = hidden,
                                    .numTopk_ = num_topk,
                                    .numExperts_ = num_experts,
                                    .numMaxTokensPerRank_ = num_max_dispatch_tokens_per_rank,
                                    .inputDType_ = input_is_fp8 ? low_latency::DType::F8E4M3 : low_latency::DType::BF16,
                                    .outputDType_ = low_latency::DType::BF16,
                                    .zeroCopy_ = false};
  low_latency::BufferSet current_buf{.sendDataBuffer_ = buffer.combine_rdma_send_buffer,
                                     .sendCountBuffer_ = nullptr,
                                     .recvDataBuffer_ = buffer.combine_rdma_recv_data_buffer,
                                     .recvCountBuffer_ = buffer.combine_rdma_recv_flag_buffer,
                                     .cleanupRegion_ = nullptr,
                                     .cleanupSize_ = 0};
  low_latency::BufferSet next_buf{.sendDataBuffer_ = nullptr,
                                  .sendCountBuffer_ = nullptr,
                                  .recvDataBuffer_ = nullptr,
                                  .recvCountBuffer_ = nullptr,
                                  .cleanupRegion_ = next_clean_meta.first,
                                  .cleanupSize_ = next_clean_meta.second};
  low_latency::TransportContext transport{
      .rdmaBufferBase_ = rdma_buffer_ptr_,
      .portChannels_ = port_channel_handles_device_ptr_.get(),
      .memoryChannels_ = ll_memory_channel_handles_device_ptr_ ? ll_memory_channel_handles_device_ptr_.get() : nullptr,
      .peerBases_ = peer_rdma_bases_gpu_,
      .ipcReady_ = ll_ipc_ready_,
      .rank_ = rank_,
      .numRanks_ = num_ranks_,
      .ranksPerIpcDomain_ = ll_ranks_per_ipc_domain_};
  low_latency::combine(output, input, input_scales, topk_idx, topk_weights, src_info, layout_range, config, current_buf,
                       next_buf, transport, workspace_, stream, low_latency::SEND_AND_RECV);
}

void MoERuntime::move_fifo_slots(int) {}

}  // namespace ep
}  // namespace mscclpp
