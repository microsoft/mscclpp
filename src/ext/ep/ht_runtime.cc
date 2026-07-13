// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// Torch-free high-throughput MoE dispatch/combine runtime. This is a
// mechanical de-torch of the former `src/ext/ep/ht/buffer.cc`: tensor params
// became raw device pointers + explicit size ints, output tensors became
// caller-provided pointers, `at::cuda::CUDAStream` became `cudaStream_t`, and
// the torch validation asserts / EventHandle async machinery were dropped. The
// CUDA kernels (`intranode::` / `internode::`), env knobs, `#ifdef
// EP_DISPATCH_NCCLEP` blocks, recv-pool / VMM / IPC logic and diagnostics are
// preserved verbatim.

#include "ht_runtime.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "api.cuh"
#include "constants.cuh"

namespace mscclpp {
namespace ep {

// Upstream MSCCL++ now exposes `Connection::atomicAdd` and
// `PortChannelDeviceHandle::atomicAdd` natively (see commit "atomic add"
// on branch chhwang/new-atomic-add, merged into this tree). The stock
// `ProxyService` recognises `ChannelTrigger.type == 0` as an atomic-add
// request, so no subclass or private-member access is required anymore.
using EPProxyService = mscclpp::ProxyService;

// Number of host-side proxy services (== proxy threads) the runtime creates.
// PortChannels are sharded across these so per-thread FIFO contention drops.
// A single proxy thread caps cross-node LL combine at ~2.8 ms/iter on H100+IB
// platforms; 8 threads bring it down to ~470 us (NIC-bound). On NVSwitch-only
// platforms (e.g. GB200 NVL72) host proxies don't post IB WRs, so 1 is plenty.
//
// Resolution order:
//   1. `MSCCLPP_EP_NUM_PROXIES` env var (clamped to >= 1) if set.
//   2. Auto-detect from current device's compute capability:
//        - Hopper (sm_90, H100/H200) and earlier: 8
//        - Blackwell (sm_100+, B100/B200/GB200): 1
//   3. Fallback: 8.
//
// Empirical sweep on 2x8 H100 + IB (16 ranks, t=128, h=7168, topk=8):
//   N=1:  D 1013 us / C 2801 us
//   N=2:  D  611 us / C  843 us
//   N=4:  D  479 us / C  568 us
//   N=8:  D  445 us / C  469 us  <-- knee
//   N=12: collapses (CPU oversubscription with 8 GPUs/node).
static int resolve_num_proxy_services(int num_ranks, int local_world_size) {
  if (const char* env = std::getenv("MSCCLPP_EP_NUM_PROXIES")) {
    int v = std::atoi(env);
    return v > 0 ? v : 1;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 8;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 8;
  // sm_100+ = Blackwell (GB200 etc.). NVSwitch fabric makes one proxy
  // sufficient intranode, but cross-node RDMA still funnels through that
  // single host thread. When num_rdma_ranks > 1, shard the proxy across
  // local GPUs so each GPU has its own RDMA-driving thread. (P2)
  int lws = local_world_size > 0 ? local_world_size : 1;
  int num_rdma_ranks_local = std::max(1, num_ranks / lws);
  if (prop.major >= 10) return num_rdma_ranks_local > 1 ? lws : 1;
  return 8;
}

// Cross-host cuMem fabric IPC capability.
//
// `isNvlsSupported()` returns true for any device with NVLink multicast,
// including H100. But NVLS by itself only works inside one host's NVLink
// island; cross-host sharing of cuMem allocations / NVLS multicast handles
// requires the device to be on an actual cross-host NVLink fabric (GB200
// NVL72 with nvidia-imex on Azure today). H100 reports
// `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED == 1` too but lacks
// the NVSwitch fabric to actually share fabric handles across hosts; the
// cross-host import then falls back to POSIX-FD, whose handle exchange
// routes through a unix-domain socket on the master host -- which
// worker-node ranks cannot reach (`connect() failed for unix socket to
// /tmp/mscclpp_bootstrap_*.sock`). That is the exact failure signature
// commit 3ab2e43b ("NVLS HT B2 phases 1-3") introduced on H100 / Azure
// CX-7 RoCE.
//
// To stay safe by default we require both:
//   - device attr `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED`
//   - compute capability >= sm_100 (Blackwell+).
//
// Resolution order:
//   1. `MSCCLPP_EP_FABRIC_IPC` env var (`0`/`off`/`false`/`no` => off,
//      `1`/`on`/`true`/`yes`/`force` => on, anything else => auto). When
//      set, env value takes precedence over the device check.
//   2. Auto-detect via the two checks above.
static bool resolve_fabric_ipc_supported() {
  if (const char* env = std::getenv("MSCCLPP_EP_FABRIC_IPC")) {
    std::string v(env);
    for (auto& c : v) c = std::tolower(static_cast<unsigned char>(c));
    if (v == "0" || v == "off" || v == "false" || v == "no") return false;
    if (v == "1" || v == "on" || v == "true" || v == "yes" || v == "force") return true;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  CUdevice cuDev;
  if (cuDeviceGet(&cuDev, dev) != CUDA_SUCCESS) return false;
  int supported = 0;
  if (cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cuDev) != CUDA_SUCCESS) {
    return false;
  }
  if (!supported) return false;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
  // Blackwell+ (sm_100, GB200 NVL72) is the only deployed cross-host
  // NVLink fabric today. H100 (sm_90) advertises fabric-handle support
  // but lacks the nvidia-imex / NVSwitch fabric to actually share them
  // across hosts.
  return prop.major >= 10;
}

MoEHighThroughputRuntime::MoEHighThroughputRuntime(int rank, int numRanks, int64_t numNvlBytes, int64_t numRdmaBytes)
    : num_nvl_bytes(numNvlBytes),
      num_rdma_bytes(numRdmaBytes),
      rank(rank),
      num_ranks(numRanks),
      bootstrap(std::make_shared<mscclpp::TcpBootstrap>(rank, numRanks)) {
  // Communication stream + reusable cross-stream ordering event. Created first
  // because the IPC/workspace setup below issues `cudaMemsetAsync` on it. This
  // replaces the torch `at::cuda::getStreamFromPool(true)` member initializer.
  CUDA_CHECK(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreateWithFlags(&comm_event, cudaEventDisableTiming));

  // Resolve local_world_size early so the Blackwell proxy-sharding (P2) can use it.
  int local_world_size_for_proxy = NUM_MAX_NVL_PEERS;
  if (const char* env = std::getenv("MSCCLPP_EP_LOCAL_WORLD_SIZE")) {
    int v = std::atoi(env);
    if (v > 0 && v <= NUM_MAX_NVL_PEERS) local_world_size_for_proxy = v;
  }
  num_proxy_services = resolve_num_proxy_services(num_ranks, local_world_size_for_proxy);
  proxy_services.reserve(num_proxy_services);
  for (int i = 0; i < num_proxy_services; ++i) {
    proxy_services.emplace_back(std::make_shared<EPProxyService>());
  }
  if (rank == 0) {
    printf("[mscclpp_ep] num_proxy_services=%d (set MSCCLPP_EP_NUM_PROXIES to override)\n", num_proxy_services);
    fflush(stdout);
  }
  // Task fifo memory
  int64_t fifo_bytes = sizeof(int) * NUM_MAX_FIFO_SLOTS;
  int64_t buffer_ptr_bytes = sizeof(void*) * NUM_MAX_NVL_PEERS;
  int64_t task_ptr_bytes = sizeof(int*) * NUM_MAX_NVL_PEERS;

  // Common checks
  EP_HOST_ASSERT(num_nvl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                 (num_nvl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
  EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and
                 (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
  EP_HOST_ASSERT(0 <= rank and rank < num_ranks and
                 (num_ranks <= NUM_MAX_NVL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
  EP_HOST_ASSERT(num_ranks < NUM_MAX_NVL_PEERS or num_ranks % NUM_MAX_NVL_PEERS == 0);
  if (num_rdma_bytes > 0) EP_HOST_ASSERT(num_ranks > NUM_MAX_NVL_PEERS or low_latency_mode);

  // Get ranks
  CUDA_CHECK(cudaGetDevice(&device_id));
  // Allow overriding the local-world-size (number of GPUs per node) via the
  // env var MSCCLPP_EP_LOCAL_WORLD_SIZE. By default the partitioning is
  // pinned to NUM_MAX_NVL_PEERS=8, which mis-classifies all ranks as
  // intra-node on hosts with fewer than 8 GPUs (e.g. GB200x4) and breaks
  // cross-node LL via spurious cudaIpcOpenMemHandle on remote IPC handles.
  int local_world_size = NUM_MAX_NVL_PEERS;
  if (const char* env = std::getenv("MSCCLPP_EP_LOCAL_WORLD_SIZE")) {
    int v = std::atoi(env);
    if (v > 0 && v <= NUM_MAX_NVL_PEERS) local_world_size = v;
  }
  rdma_rank = rank / local_world_size, nvl_rank = rank % local_world_size;
  num_rdma_ranks = std::max(1, num_ranks / local_world_size), num_nvl_ranks = std::min(num_ranks, local_world_size);

  // Get device info
  cudaDeviceProp device_prop = {};
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));

  if (num_nvl_bytes > 0) {
    // Local IPC: alloc local memory and set local IPC handle
    int64_t ep_nvl_alloc = num_nvl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes;
#ifdef EP_DISPATCH_NCCLEP
    // Increment 4: allocate the peer-mapped recv-output pool as VMM memory
    // (cuMem FABRIC/POSIX-FD via gpuCallocPhysical) so peers import TMA-eligible
    // VAs. Falls back to the increment-3 inline carve (cudaIpc, no TMA) on
    // platforms without fabric-IPC support.
    static const bool ep_fabric_pool_ok = mscclpp::isNvlsSupported() && resolve_fabric_ipc_supported();
    if (ep_fabric_pool_ok) {
      const size_t ep_pool_bytes = Config::recv_pool_bytes_static(num_ranks);
      recv_pool_local_ptr_ = mscclpp::detail::gpuCallocPhysical(ep_pool_bytes);
      CUDA_CHECK(cudaMemset(recv_pool_local_ptr_, 0, ep_pool_bytes));
    } else {
      // inc3 fallback: append the pool AFTER the fifo/task regions of the NVL
      // alloc (reached by peers via buffer_ptrs[peer] + recv_pool_off_), NOT in
      // num_nvl_bytes so it does not count against the INT_MAX registered cap.
      recv_pool_off_ = ((ep_nvl_alloc + 127) / 128) * 128;
      ep_nvl_alloc = recv_pool_off_ + static_cast<int64_t>(Config::recv_pool_bytes_static(num_ranks));
    }
#endif
    CUDA_CHECK(cudaMalloc(&buffer_ptrs[nvl_rank], ep_nvl_alloc));
    CUDA_CHECK(cudaIpcGetMemHandle(&ipc_handles[nvl_rank], buffer_ptrs[nvl_rank]));
    buffer_ptrs_gpu =
        reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes + fifo_bytes);

    // Set task fifo
    EP_HOST_ASSERT(NUM_MAX_FIFO_SLOTS % num_nvl_ranks == 0);
    task_fifo_ptrs[nvl_rank] =
        reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes);
    task_fifo_ptrs_gpu = reinterpret_cast<int**>(reinterpret_cast<uint8_t*>(buffer_ptrs[nvl_rank]) + num_nvl_bytes +
                                                 fifo_bytes + buffer_ptr_bytes);

    // No need to synchronize, will do a full device sync during `sync`
    CUDA_CHECK(cudaMemsetAsync(task_fifo_ptrs[nvl_rank], 0, fifo_bytes, comm_stream));
  }

  // Create 32 MiB workspace
  CUDA_CHECK(cudaMalloc(&workspace, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

  // MoE counter
  CUDA_CHECK(cudaMallocHost(&moe_recv_counter, sizeof(int64_t), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
  *moe_recv_counter = -1;

  // MoE expert-level counter
  CUDA_CHECK(cudaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
  for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i) moe_recv_expert_counter[i] = -1;

  // MoE RDMA-level counter
  if (num_rdma_ranks > 0) {
    CUDA_CHECK(cudaMallocHost(&moe_recv_rdma_counter, sizeof(int), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
    *moe_recv_rdma_counter = -1;
  }

  for (auto& ps : proxy_services) ps->startProxy();
}

MoEHighThroughputRuntime::~MoEHighThroughputRuntime() noexcept(false) {
  // Synchronize
  CUDA_CHECK(cudaDeviceSynchronize());

  if (num_nvl_bytes > 0) {
    // Barrier
    intranode::barrier(task_fifo_ptrs_gpu, head, nvl_rank, num_nvl_ranks, comm_stream);
    move_fifo_slots();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Close remote IPC
    if (isAvailable()) {
      for (int i = 0; i < num_nvl_ranks; ++i)
        if (i != nvl_rank) CUDA_CHECK(cudaIpcCloseMemHandle(buffer_ptrs[i]));
    }

    // Free local buffer and error flag
    CUDA_CHECK(cudaFree(buffer_ptrs[nvl_rank]));
  }
#ifdef EP_DISPATCH_NCCLEP
  // Increment 4: release the device-side VMM recv-pool base array. The imported
  // peer mappings are owned by recv_pool_remote_mems_ (auto-released); the local
  // VMM pool follows the rdma_buffer_ptr convention (reclaimed at process exit).
  if (recv_pool_ptrs_gpu != nullptr) {
    CUDA_CHECK(cudaFree(recv_pool_ptrs_gpu));
    recv_pool_ptrs_gpu = nullptr;
  }
  if (recv_pool_global_ptrs_gpu != nullptr) {
    CUDA_CHECK(cudaFree(recv_pool_global_ptrs_gpu));
    recv_pool_global_ptrs_gpu = nullptr;
  }
  if (ep_combine_recv_idx_gpu != nullptr) {
    CUDA_CHECK(cudaFree(ep_combine_recv_idx_gpu));
    ep_combine_recv_idx_gpu = nullptr;
  }
#endif

  // Free NVSHMEM
  if (num_rdma_bytes > 0) {
    // NVSHMEM support is not yet ported; if we got here with
    // num_rdma_bytes > 0 the construction or sync call would already have
    // failed, so there is nothing to tear down.
  }

  // LL fast-path teardown. RegisteredMemory shared_ptrs in ll_memory_channels
  // own the peer IPC mappings; we just release the device-side base array.
  if (ll_ipc_ready) {
    if (peer_rdma_bases_gpu != nullptr) {
      CUDA_CHECK(cudaFree(peer_rdma_bases_gpu));
      peer_rdma_bases_gpu = nullptr;
    }
  }

  for (auto& ps : proxy_services) ps->stopProxy();

  // Free cuBLAS handle, workspace and MoE counter
  CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_counter)));

  // Free chunked mode staffs
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(moe_recv_expert_counter)));

  // Destroy the communication stream + ordering event.
  if (comm_event) CUDA_CHECK(cudaEventDestroy(comm_event));
  if (comm_stream) CUDA_CHECK(cudaStreamDestroy(comm_stream));
}

void MoEHighThroughputRuntime::move_fifo_slots(int num_slots) {
  head = (head + num_ranks * num_slots) % NUM_MAX_FIFO_SLOTS;
}

void MoEHighThroughputRuntime::stream_wait(cudaStream_t dst, cudaStream_t src) {
  // Make `dst` wait for everything currently enqueued on `src`. The reusable
  // event is recorded on `src` then waited on `dst`; record-then-wait is
  // host-sequential within a call so a single event is safe to reuse.
  CUDA_CHECK(cudaEventRecord(comm_event, src));
  CUDA_CHECK(cudaStreamWaitEvent(dst, comm_event, 0));
}

bool MoEHighThroughputRuntime::isAvailable() const { return available; }

bool MoEHighThroughputRuntime::isInternodeAvailable() const { return isAvailable() and num_ranks > NUM_MAX_NVL_PEERS; }

int MoEHighThroughputRuntime::getNumRdmaRanks() const { return num_rdma_ranks; }

int MoEHighThroughputRuntime::getRdmaRank() const { return rdma_rank; }

int MoEHighThroughputRuntime::getRootRdmaRank(bool global) const { return global ? nvl_rank : 0; }

int MoEHighThroughputRuntime::getLocalDeviceId() const { return device_id; }

std::string MoEHighThroughputRuntime::getLocalIpcHandle() const {
  return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

std::string MoEHighThroughputRuntime::getLocalNvshmemUniqueId() const {
  // The MSCCL++ EP port replaces NVSHMEM with PortChannel/MemoryChannel,
  // so there is no NVSHMEM unique id to expose. Kept for ABI parity with
  // DeepEP's Python frontend; callers should use the MSCCL++ bootstrap.
  throw std::runtime_error(
      "mscclpp::ep::MoEHighThroughputRuntime::getLocalNvshmemUniqueId: not applicable (NVSHMEM is not used in "
      "mscclpp_ep)");
}

mscclpp::UniqueId MoEHighThroughputRuntime::createUniqueId() const { return bootstrap->createUniqueId(); }

void MoEHighThroughputRuntime::connect(mscclpp::UniqueId rootId) {
  bootstrap->initialize(rootId);
  communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
}

void MoEHighThroughputRuntime::sync(const std::vector<int>& deviceIds,
                                    const std::vector<std::optional<std::string>>& allGatheredHandles,
                                    const std::optional<std::string>& rootUniqueIdOpt) {
  EP_HOST_ASSERT(not isAvailable());

  const std::vector<mscclpp::Transport> ib_transports = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5, mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  const auto ipc_transport = mscclpp::Transport::CudaIpc;
  const auto ib_transport = ib_transports[device_id];
  const mscclpp::TransportFlags all_transport = ipc_transport | ib_transport;

  // Sync IPC handles
  if (num_nvl_bytes > 0) {
    EP_HOST_ASSERT(num_ranks == deviceIds.size());
    EP_HOST_ASSERT(deviceIds.size() == allGatheredHandles.size());
    for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
      EP_HOST_ASSERT(allGatheredHandles[offset + i].has_value());
      const auto& handle_str = allGatheredHandles[offset + i].value();
      EP_HOST_ASSERT(handle_str.size() == CUDA_IPC_HANDLE_SIZE);
      if (offset + i != rank) {
        std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE);
        CUDA_CHECK(cudaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i], cudaIpcMemLazyEnablePeerAccess));
        task_fifo_ptrs[i] = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buffer_ptrs[i]) + num_nvl_bytes);
      } else {
        EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved, handle_str.c_str(), CUDA_IPC_HANDLE_SIZE) == 0);
      }
    }

    // Copy all buffer and task pointers to GPU
    CUDA_CHECK(cudaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(task_fifo_ptrs_gpu, task_fifo_ptrs, sizeof(int*) * NUM_MAX_NVL_PEERS, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // create connections
    std::vector<mscclpp::Connection> connections;
    {
      std::vector<std::shared_future<mscclpp::Connection>> connection_futures;
      mscclpp::EndpointConfig local_config(ipc_transport);
      for (int i = 0; i < num_nvl_ranks; ++i) {
        auto r = i + rdma_rank * num_nvl_ranks;
        connection_futures.emplace_back(communicator->connect(local_config, r, 0));
      }
      for (auto& future : connection_futures) {
        connections.emplace_back(future.get());
      }
    }

    auto buffer_mem = communicator->registerMemory(buffer_ptrs[nvl_rank], num_nvl_bytes, ipc_transport);

    std::vector<std::shared_future<mscclpp::RegisteredMemory>> remote_mem_futures(num_nvl_ranks);
    for (int i = 0; i < num_nvl_ranks; ++i) {
      if (i == nvl_rank) continue;
      auto r = i + rdma_rank * num_nvl_ranks;
      communicator->sendMemory(buffer_mem, r, 0);
      remote_mem_futures[i] = communicator->recvMemory(r, 0);
    }
    for (int i = 0; i < num_nvl_ranks; ++i) {
      if (i == nvl_rank) continue;
      auto sema = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator, connections[i]);
      memory_channels.emplace_back(sema, remote_mem_futures[i].get(), buffer_mem);
    }
    std::vector<mscclpp::MemoryChannelDeviceHandle> memory_channel_handles(num_nvl_ranks);
    for (int i = 0; i < num_nvl_ranks; ++i) {
      if (i == nvl_rank) continue;
      memory_channel_handles[i] = memory_channels.rbegin()->deviceHandle();
    }

#ifdef EP_DISPATCH_NCCLEP
    // Increment 4: exchange the VMM recv-output pool base pointers across the
    // local NVL peers. registerMemory(ipc) on a gpuCallocPhysical buffer carries
    // FABRIC/POSIX-FD handles, so the imported `.data()` peer VA is dereferenceable
    // (and TMA-eligible) — unlike the cudaIpc-mapped buffer_ptrs. Imported
    // RegisteredMemory is retained in recv_pool_remote_mems_ to keep the mapping.
    if (recv_pool_local_ptr_ != nullptr) {
      constexpr int kRecvPoolTag = 7;
      const size_t ep_pool_bytes = Config::recv_pool_bytes_static(num_ranks);
      auto pool_mem = communicator->registerMemory(recv_pool_local_ptr_, ep_pool_bytes, ipc_transport);
      std::vector<std::shared_future<mscclpp::RegisteredMemory>> pool_futs(num_nvl_ranks);
      for (int i = 0; i < num_nvl_ranks; ++i) {
        if (i == nvl_rank) continue;
        auto r = i + rdma_rank * num_nvl_ranks;
        communicator->sendMemory(pool_mem, r, kRecvPoolTag);
        pool_futs[i] = communicator->recvMemory(r, kRecvPoolTag);
      }
      recv_pool_ptrs_.assign(NUM_MAX_NVL_PEERS, nullptr);
      recv_pool_ptrs_[nvl_rank] = recv_pool_local_ptr_;
      recv_pool_remote_mems_.resize(num_nvl_ranks);
      for (int i = 0; i < num_nvl_ranks; ++i) {
        if (i == nvl_rank) continue;
        recv_pool_remote_mems_[i] = pool_futs[i].get();
        recv_pool_ptrs_[i] = recv_pool_remote_mems_[i].data();
      }
      CUDA_CHECK(cudaMalloc(&recv_pool_ptrs_gpu, sizeof(void*) * NUM_MAX_NVL_PEERS));
      CUDA_CHECK(cudaMemcpy(recv_pool_ptrs_gpu, recv_pool_ptrs_.data(), sizeof(void*) * NUM_MAX_NVL_PEERS,
                            cudaMemcpyHostToDevice));
      // Intranode combine-direct: allocate the per-(token, dst rank) gather map so the
      // INTRA_DIRECT dispatch sender can record each token's recv-pool slot and the TMA
      // direct-gather combine can read it back. Idempotent with the internode (inc5)
      // allocation below (guarded by == nullptr); sized for the worst-case source tokens.
      if (ep_combine_recv_idx_gpu == nullptr)
        CUDA_CHECK(cudaMalloc(&ep_combine_recv_idx_gpu,
                              sizeof(int) * static_cast<size_t>(Config::kEpRecvPoolMaxTokens) * num_ranks));

      if (rank == 0) {
        printf("[mscclpp_ep] inc4 VMM recv-pool peer bases (rank 0):");
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) printf(" [%d]=%p", i, recv_pool_ptrs_[i]);
        printf("\n");
        fflush(stdout);
      }
    }
#endif

    memory_channel_handles_device_ptr =
        mscclpp::detail::gpuCallocShared<mscclpp::MemoryChannelDeviceHandle>(num_nvl_ranks);
    mscclpp::gpuMemcpy<mscclpp::MemoryChannelDeviceHandle>(
        memory_channel_handles_device_ptr.get(), memory_channel_handles.data(), num_nvl_ranks, cudaMemcpyHostToDevice);
  }

  // RDMA buffer setup (replaces DeepEP's NVSHMEM symmetric-heap allocation).
  //
  // Unlike DeepEP which used `nvshmem_align` to place the RDMA buffer on the
  // symmetric heap, all our internode communication goes through MSCCL++
  // `PortChannel` (proxy-based RDMA), so a plain `cudaMalloc` + IB memory
  // registration is sufficient. The bootstrap barrier replaces
  // `nvshmem_barrier_all`.
  if (num_rdma_bytes > 0) {
    EP_HOST_ASSERT(communicator != nullptr);
    EP_HOST_ASSERT(bootstrap != nullptr);

    // Allocate the RDMA buffer.
    //
    // For low-latency mode on platforms that support NVLink-SHARP / NVLS
    // (GB200 NVL72 with nvidia-imex configured), use mscclpp's
    // `gpuCallocPhysical` (cuMemCreate + cuMemMap with POSIX_FD|FABRIC
    // handle types) so the buffer is eligible for cuMem fabric IPC — the
    // LL fast path then maps the buffer across the NVL72 fabric via
    // nvidia-imex and performs atomicAdd over NVLink rather than RDMA
    // (which has IBV_ATOMIC_NONE on Azure CX-7 RoCE).
    //
    // Fallback: on platforms without NVLS / multicast support (e.g.
    // H100 + IB, A100 + IB), `gpuCallocPhysical` would either fail or
    // produce non-fabric-IPC memory; fall back to plain `cudaMalloc` and
    // let the LL path use the existing PortChannel proxy mechanism over
    // IB. For HT internode on Azure GB200 (Phase 4), we also use
    // fabric-IPC so cross-node `handle.put(data)` can be replaced by
    // direct kernel-side writes via NVL72 fabric pointers — bypassing
    // the broken Azure CX-7 RoCE RDMA WRITE path.
    //
    // NVLS-supported but FABRIC-unsupported deployments (e.g. H100 on
    // Azure NDv5) must not take this path for the HT cross-host case:
    // their `gpuCallocPhysical` result is a POSIX-FD-only handle whose
    // cross-host import falls back to a master-local unix socket which
    // worker ranks cannot reach.
    static const bool fabric_ipc_supported = resolve_fabric_ipc_supported();
    const bool use_fabric_ipc_alloc =
        mscclpp::isNvlsSupported() && fabric_ipc_supported && (low_latency_mode || num_rdma_ranks > 1);
    if (use_fabric_ipc_alloc) {
      rdma_buffer_ptr = mscclpp::detail::gpuCallocPhysical(num_rdma_bytes);
      CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));
    } else {
      CUDA_CHECK(cudaMalloc(&rdma_buffer_ptr, num_rdma_bytes));
      CUDA_CHECK(cudaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));
    }
    if (rank == 0) {
      printf("[mscclpp_ep] rdma_buffer allocator: %s (low_latency=%d, nvls=%d, fabric_ipc=%d)\n",
             use_fabric_ipc_alloc ? "gpuCallocPhysical (fabric-IPC)" : "cudaMalloc", (int)low_latency_mode,
             (int)mscclpp::isNvlsSupported(), (int)fabric_ipc_supported);
      fflush(stdout);
    }
    bootstrap->barrier();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Rank -> RDMA buffer IDs. MemoryIds are local to each ProxyService;
    // we register every memory in every proxy in the same global order so
    // a single int identifies the memory across all of them.
    std::map<int, mscclpp::MemoryId> memory_ids;

    auto add_memory_to_all = [&](mscclpp::RegisteredMemory mem) -> mscclpp::MemoryId {
      mscclpp::MemoryId id = static_cast<mscclpp::MemoryId>(-1);
      for (auto& ps : proxy_services) {
        auto cur = ps->addMemory(mem);
        if (id == static_cast<mscclpp::MemoryId>(-1)) id = cur;
        EP_HOST_ASSERT(cur == id && "MemoryIds drifted across proxy services");
      }
      return id;
    };

    // Register local memory
    auto local_rdma_buffer_mem = communicator->registerMemory(rdma_buffer_ptr, num_rdma_bytes, all_transport);
    memory_ids[rank] = add_memory_to_all(local_rdma_buffer_mem);

    // Send local memory to other ranks.
    //
    // NOTE: DeepEP filters this to same-GPU-ID peers in low_latency_mode
    // because LL there uses NVSHMEM, not port channels. This port drives
    // LL kernels through PortChannel, so every peer must have a real
    // memory/connection/semaphore/port channel entry. Treat LL and HT
    // sync identically: always connect all peers.
    //
    // Caveat: for a pure intra-node LL launch (``num_nvl_bytes == 0`` with
    // every peer on the same host) the resulting port channels go through
    // the CPU proxy over IB loopback between different HCAs, which on
    // this platform does not deliver atomics reliably and currently
    // deadlocks LL dispatch. See `src/ext/ep/README.md` for the full
    // discussion. Cross-node LL (DeepEP's recommended 1-GPU-per-node
    // topology) is unaffected.
    // Use tag=1 to disambiguate from the NVL phase's tag=0 traffic with same-node peers.
    constexpr int kRdmaTag = 1;
    for (int r = 0; r < num_ranks; ++r) {
      if (r == rank) continue;
      communicator->sendMemory(local_rdma_buffer_mem, r, kRdmaTag);
    }

    // Receive remote memory from other ranks.
    for (int r = 0; r < num_ranks; ++r) {
      if (r == rank) continue;
      auto f = communicator->recvMemory(r, kRdmaTag);
      auto mem = f.get();
      memory_ids[r] = add_memory_to_all(std::move(mem));
    }

    // Rank -> vector of connections
    std::unordered_map<int, std::vector<mscclpp::Connection>> connections;
    const mscclpp::EndpointConfig ipc_cfg(ipc_transport);
    const mscclpp::EndpointConfig ib_cfg(ib_transport);

    // Self connection for local memory (CUDA IPC).
    connections[rank].emplace_back(communicator->connect(ipc_cfg, rank, kRdmaTag).get());

    // Remote IB connections (multi-QP per peer).
    const int num_ib_connections_per_rank = 12;  // #QPs per rank (mirrors DeepEP).
    for (int r = 0; r < num_ranks; ++r) {
      if (r == rank) continue;
      std::vector<std::shared_future<mscclpp::Connection>> futures;
      futures.reserve(num_ib_connections_per_rank);
      for (int i = 0; i < num_ib_connections_per_rank; ++i) {
        futures.emplace_back(communicator->connect(ib_cfg, r, kRdmaTag));
      }
      for (auto& f : futures) connections[r].emplace_back(f.get());
    }

    // Rank -> vector of (proxy_idx, semaphore_id_within_proxy). Iterate
    // peers in sorted rank order so semaphore pairings between nodes line
    // up deterministically. Channels — and therefore their backing
    // semaphores — are sharded across `proxy_services`: channel at flat
    // index `i*num_ranks + r` lives on proxy `(i*num_ranks + r) %
    // num_proxy_services`. SemaphoreIds are local to each proxy, so we
    // record (proxy_idx, sid) pairs.
    std::unordered_map<int, std::vector<std::pair<int, mscclpp::SemaphoreId>>> sema_ids;
    const int num_semaphores_per_rank = 16;
    for (int i = 0; i < num_semaphores_per_rank; ++i) {
      for (int r = 0; r < num_ranks; ++r) {
        auto conn_it = connections.find(r);
        EP_HOST_ASSERT(conn_it != connections.end());
        auto& conns = conn_it->second;
        auto& conn = conns[i % conns.size()];
        int proxy_idx = (i * num_ranks + r) % num_proxy_services;
        auto sema_id = proxy_services[proxy_idx]->buildAndAddSemaphore(*communicator, conn);
        sema_ids[r].emplace_back(proxy_idx, sema_id);
      }
    }

    // Create port channels + device handles.
    //
    // The kernels index `port_channel_handles[channel_id * num_ranks + peer_rank]`
    // where peer_rank is a GLOBAL rank in [0..num_ranks). So the outer stride must
    // be num_ranks with peers in ascending rank order. Iterating `memory_ids` (an
    // `unordered_map`) yields hash order and would misroute signals, deadlocking.
    // Each channel inherits the proxy of the semaphore it was built on, so the
    // resulting `PortChannelDeviceHandle` routes its FIFO pushes to the correct
    // proxy thread.
    const int num_port_channels_per_rank = num_semaphores_per_rank;
    std::vector<mscclpp::PortChannelDeviceHandle> port_channel_handles;
    for (int i = 0; i < num_port_channels_per_rank; ++i) {
      for (int r = 0; r < num_ranks; ++r) {
        auto mem_it = memory_ids.find(r);
        EP_HOST_ASSERT(mem_it != memory_ids.end());
        auto memory_id = mem_it->second;
        auto [proxy_idx, sema_id] = sema_ids[r][i % sema_ids[r].size()];
        auto port_channel = proxy_services[proxy_idx]->portChannel(sema_id, memory_id, memory_ids[rank]);
        port_channels.emplace_back(std::move(port_channel));
        port_channel_handles.emplace_back(port_channels.rbegin()->deviceHandle());
      }
    }

    port_channel_handles_device_ptr =
        mscclpp::detail::gpuCallocShared<mscclpp::PortChannelDeviceHandle>(port_channel_handles.size());
    mscclpp::gpuMemcpy<mscclpp::PortChannelDeviceHandle>(port_channel_handles_device_ptr.get(),
                                                         port_channel_handles.data(), port_channel_handles.size(),
                                                         cudaMemcpyHostToDevice);

    // ------------------------------------------------------------------
    // HT internode NVLS multicast setup (Wide Proposal B2).
    //
    // On platforms with NVLink-SHARP / multicast support (GB200 NVL72
    // with nvidia-imex), allocate a multicast-bound buffer used by the HT
    // dispatch / combine / notify_dispatch kernels for:
    //   - tail / head atomic counters (replaces the 4 atomicAdd sites)
    //   - notify_dispatch barrier epoch (replaces port_channel.signal/wait)
    //   - notify_dispatch small-data delivery (replaces putWithSignal)
    //
    // All cross-node atomic adds become `multimem.red.add.u64` PTX which
    // travels over the NVL72 fabric — bypassing IB entirely. This is the
    // same fabric path that LL Proposal A validated cross-node.
    //
    // Fallback (existing IB platforms): when `isNvlsSupported()` is false
    // or there is only one RDMA rank (intranode-only), `nvls_ht_enabled`
    // stays `false` and kernels select the legacy PortChannel path.
    //
    // Additionally: NVLS alone is insufficient for cross-host. H100 has
    // NVLS within a node but no cross-host NVSwitch fabric, so
    // `connectNvlsCollective` either fails or builds a per-node-only
    // multicast object, and the POSIX-FD fallback for handle exchange
    // routes through a master-local unix socket that worker ranks cannot
    // reach. Gate this path on `fabric_ipc_supported` so non-Blackwell
    // deployments cleanly use the legacy PortChannel path.
    //
    // Skipped for `low_latency_mode` since LL has its own (working)
    // fabric-IPC path via Proposal A and does not use the HT counter
    // protocol.
    // ------------------------------------------------------------------
    nvls_ht_enabled = false;
    if (!low_latency_mode && num_rdma_ranks > 1 && mscclpp::isNvlsSupported() && fabric_ipc_supported) {
      // Worst-case sizing — chosen so the same multicast buffer fits any
      // (num_sms, num_rdma_ranks) configuration the kernels may launch with.
      const size_t kCounterBytesPerChannel =
          static_cast<size_t>(NUM_MAX_RDMA_PEERS) * NUM_MAX_RDMA_PEERS * sizeof(uint64_t);
      const size_t tail_bytes = static_cast<size_t>(kNvlsMaxChannels) * kCounterBytesPerChannel;
      const size_t head_bytes = tail_bytes;
      const size_t barrier_bytes = static_cast<size_t>(kNvlsBarrierSlots) * sizeof(uint64_t);
      // Data region: per-sender slot of [num_rdma_ranks_max × per_peer_bytes],
      // one slot per global rank (worst-case num_ranks = NUM_MAX_RDMA_PEERS *
      // NUM_MAX_NVL_PEERS). Each rank writes its own slot via `multimem.st`;
      // every receiver then reads the sub-position destined to it.
      const size_t kPerSenderSlotBytes = static_cast<size_t>(NUM_MAX_RDMA_PEERS) * kNvlsPerPeerBytes;
      const size_t kMaxRanks = static_cast<size_t>(NUM_MAX_RDMA_PEERS) * NUM_MAX_NVL_PEERS;
      const size_t data_bytes = kMaxRanks * kPerSenderSlotBytes;

      // 256 B alignment for each sub-region to keep `multimem` ops well-aligned.
      auto align256 = [](size_t x) { return (x + 255) & ~size_t(255); };
      nvls_ht_off_tail = 0;
      nvls_ht_off_head = align256(nvls_ht_off_tail + tail_bytes);
      nvls_ht_off_barrier = align256(nvls_ht_off_head + head_bytes);
      nvls_ht_off_data = align256(nvls_ht_off_barrier + barrier_bytes);
      nvls_ht_total_bytes = align256(nvls_ht_off_data + data_bytes);

      // GpuBuffer auto-uses gpuCallocPhysicalShared (cuMem fabric handle)
      // when isNvlsSupported() — required for multicast bind.
      nvls_ht_buffer = std::make_shared<mscclpp::GpuBuffer<uint8_t>>(nvls_ht_total_bytes);
      CUDA_CHECK(cudaMemset(nvls_ht_buffer->data(), 0, nvls_ht_buffer->bytes()));

      std::vector<int> all_ranks;
      all_ranks.reserve(num_ranks);
      for (int r = 0; r < num_ranks; ++r) all_ranks.push_back(r);

      // Collective: every rank must call. If it fails (e.g. IMEX
      // misconfigured, peers in different fabrics), the exception
      // propagates — there is no clean fallback mid-collective. The
      // `isNvlsSupported()` gate above is the production guard.
      nvls_ht_conn = mscclpp::connectNvlsCollective(communicator, all_ranks, nvls_ht_buffer->bytes());
      auto sw = nvls_ht_conn->bindAllocatedMemory(reinterpret_cast<CUdeviceptr>(nvls_ht_buffer->data()),
                                                  nvls_ht_buffer->bytes());
      nvls_ht_sc = std::make_shared<mscclpp::SwitchChannel>(std::move(sw));
      auto h = nvls_ht_sc->deviceHandle();
      nvls_ht_mc_ptr = h.mcPtr;
      nvls_ht_dev_ptr = h.devicePtr;
      nvls_ht_enabled = (nvls_ht_mc_ptr != nullptr) && (nvls_ht_dev_ptr != nullptr);

      // DIAG: print mcPtr/devicePtr/buf-VA per rank to verify whether
      // connectNvlsCollective produced a multicast that actually spans
      // both nodes (suspected: per-node only on Azure GB200).
      printf("[mscclpp_ep] NVLS HT diag rank=%d mcPtr=%p devicePtr=%p bufVA=%p bytes=%zu\n", rank,
             (void*)nvls_ht_mc_ptr, (void*)nvls_ht_dev_ptr, (void*)nvls_ht_buffer->data(),
             (size_t)nvls_ht_buffer->bytes());
      fflush(stdout);

      bootstrap->barrier();

      if (rank == 0) {
        printf(
            "[mscclpp_ep] NVLS HT multicast: enabled=%d total=%zu KB "
            "(tail@%zu head@%zu barrier@%zu data@%zu)\n",
            (int)nvls_ht_enabled, nvls_ht_total_bytes / 1024, nvls_ht_off_tail, nvls_ht_off_head, nvls_ht_off_barrier,
            nvls_ht_off_data);
        fflush(stdout);
      }
    } else if (rank == 0) {
      printf(
          "[mscclpp_ep] NVLS HT multicast: disabled (low_latency=%d, num_rdma_ranks=%d, "
          "nvls_supported=%d, fabric_ipc=%d)\n",
          (int)low_latency_mode, num_rdma_ranks, (int)mscclpp::isNvlsSupported(), (int)fabric_ipc_supported);
      fflush(stdout);
    }

    // ------------------------------------------------------------------
    // Intra-node LL fast path setup.
    //
    // When all ranks sit on the same host (num_rdma_ranks == 1), LL dispatch
    // and combine still go through `PortChannel` above — which internally
    // uses the proxy service over IB loopback between different HCAs on
    // this platform. That path is correct but slow (caps at ~170 GB/s vs.
    // NVLink's multi-TB/s). We additionally set up CUDA-IPC peer pointers
    // to each peer's `rdma_buffer_ptr` plus a set of per-peer MemoryChannels
    // for a barrier ring. The LL kernels select this path at launch time.
    // Cross-node LL uses cuMem fabric IPC (Proposal A): peers map
    // `rdma_buffer_ptr` through the NVL72 NVSwitch fabric via nvidia-imex,
    // and the LL kernels do direct `st.global` + atomicAdd through those
    // peer pointers. This bypasses the RDMA path entirely (Azure CX-7 RoCE
    // has IBV_ATOMIC_NONE which makes the proxy-emulated atomicAdd hang).
    // Requires nvidia-imex active on every rank's host with a shared
    // `nodes_config.cfg` covering all node IPs.
    // ------------------------------------------------------------------
    // ------------------------------------------------------------------
    // Single-node LL uses regular CUDA IPC peer pointers. Cross-node LL/HT on
    // GB200 uses fabric-IPC peer pointers, letting kernels write tokens directly
    // into each peer's `rdma_buffer_ptr` via the NVL72 fabric VA.
    // ------------------------------------------------------------------
    const int ipc_domain_size = use_fabric_ipc_alloc ? num_ranks : num_nvl_ranks;
    auto is_ipc_peer = [&](int peer) {
      return peer != rank && ipc_domain_size > 1 && rank / ipc_domain_size == peer / ipc_domain_size;
    };
    const bool want_peer_ipc = use_fabric_ipc_alloc || (low_latency_mode && ipc_domain_size > 1);
    if (want_peer_ipc) {
      // Reuse the local RDMA registration's CudaIpc transport entry. The
      // existing `local_rdma_buffer_mem` was registered with `all_transport`
      // (= ipc | ib), so its CudaIpc TransportInfo is already populated
      // with the FABRIC handle (when supported by the underlying physical
      // allocation). We need a separate registration only because the
      // remote-side `recvMemory` below is tagged independently.
      constexpr int kLlIpcTag = 2;
      auto rdma_mem_ipc = communicator->registerMemory(rdma_buffer_ptr, num_rdma_bytes, ipc_transport);
      std::vector<std::shared_future<mscclpp::RegisteredMemory>> remote_futures(num_ranks);
      for (int r = 0; r < num_ranks; ++r) {
        if (r == rank || !is_ipc_peer(r)) continue;
        communicator->sendMemory(rdma_mem_ipc, r, kLlIpcTag);
        remote_futures[r] = communicator->recvMemory(r, kLlIpcTag);
      }
      std::vector<mscclpp::Connection> ll_ipc_conns(num_ranks);
      {
        std::vector<std::shared_future<mscclpp::Connection>> conn_futures(num_ranks);
        mscclpp::EndpointConfig cfg(ipc_transport);
        for (int r = 0; r < num_ranks; ++r) {
          if (r == rank || !is_ipc_peer(r)) continue;
          conn_futures[r] = communicator->connect(cfg, r, kLlIpcTag);
        }
        for (int r = 0; r < num_ranks; ++r) {
          if (r == rank || !is_ipc_peer(r)) continue;
          ll_ipc_conns[r] = conn_futures[r].get();
        }
      }

      // Resolve peer base pointers from the (now mapped) remote
      // RegisteredMemory. `data()` returns the locally-mapped peer pointer;
      // for fabric handles this address lives in the cuMem fabric VA range,
      // while single-node LL gets a regular CUDA IPC mapping.
      peer_rdma_bases.assign(num_ranks, nullptr);
      peer_rdma_bases[rank] = rdma_buffer_ptr;
      std::vector<mscclpp::RegisteredMemory> remote_mems(num_ranks);
      for (int r = 0; r < num_ranks; ++r) {
        if (r == rank || !is_ipc_peer(r)) continue;
        remote_mems[r] = remote_futures[r].get();
        peer_rdma_bases[r] = remote_mems[r].data();
      }
      CUDA_CHECK(cudaMalloc(&peer_rdma_bases_gpu, sizeof(void*) * num_ranks));
      CUDA_CHECK(
          cudaMemcpy(peer_rdma_bases_gpu, peer_rdma_bases.data(), sizeof(void*) * num_ranks, cudaMemcpyHostToDevice));
      if (rank == 0) {
        printf("[mscclpp_ep] Phase 4 fabric-IPC peer bases (rank 0):\n");
        for (int r = 0; r < num_ranks; ++r) {
          printf("  peer_rdma_bases[%d] = %p\n", r, peer_rdma_bases[r]);
        }
        fflush(stdout);
      }

#ifdef EP_DISPATCH_NCCLEP
      {
        // Increment 5 (inc5 flat-domain dispatch): when MSCCLPP_EP_DIRECT is set,
        // exchange the cuMem-fabric recv-output pool base to ALL ranks (indexed by
        // global rank), mirroring the peer_rdma_bases exchange. Lets the RDMA sender
        // write each token directly into the destination GPU's recv pool over
        // fabric-VA, removing the rail-aligned rdma_channel bounce + forwarder
        // transpose. Same cuMem FABRIC handle as inc4a's pool. Env-gated so the
        // inc4a baseline (MSCCLPP_EP_DIRECT unset) is byte-for-byte unchanged.
        const char* e_direct = std::getenv("MSCCLPP_EP_DIRECT");
        const bool ep_direct = (e_direct != nullptr && std::atoi(e_direct) != 0);
        if (ep_direct && recv_pool_local_ptr_ != nullptr) {
          constexpr int kRecvPoolGlobalTag = 9;
          const size_t ep_pool_bytes_g = Config::recv_pool_bytes_static(num_ranks);
          auto pool_mem_g = communicator->registerMemory(recv_pool_local_ptr_, ep_pool_bytes_g, all_transport);
          for (int r = 0; r < num_ranks; ++r) {
            if (r == rank) continue;
            communicator->sendMemory(pool_mem_g, r, kRecvPoolGlobalTag);
          }
          std::vector<std::shared_future<mscclpp::RegisteredMemory>> gpool_futs(num_ranks);
          for (int r = 0; r < num_ranks; ++r) {
            if (r == rank) continue;
            gpool_futs[r] = communicator->recvMemory(r, kRecvPoolGlobalTag);
          }
          recv_pool_global_ptrs_.assign(num_ranks, nullptr);
          recv_pool_global_ptrs_[rank] = recv_pool_local_ptr_;
          recv_pool_global_remote_mems_.resize(num_ranks);
          for (int r = 0; r < num_ranks; ++r) {
            if (r == rank) continue;
            recv_pool_global_remote_mems_[r] = gpool_futs[r].get();
            recv_pool_global_ptrs_[r] = recv_pool_global_remote_mems_[r].data();
          }
          CUDA_CHECK(cudaMalloc(&recv_pool_global_ptrs_gpu, sizeof(void*) * num_ranks));
          CUDA_CHECK(cudaMemcpy(recv_pool_global_ptrs_gpu, recv_pool_global_ptrs_.data(), sizeof(void*) * num_ranks,
                                cudaMemcpyHostToDevice));
          if (rank == 0) {
            printf("[mscclpp_ep] inc5 domain-wide recv-pool bases (rank 0):");
            for (int r = 0; r < num_ranks; ++r) printf(" [%d]=%p", r, recv_pool_global_ptrs_[r]);
            printf("\n");
            fflush(stdout);
          }
          // inc5 combine-direct (Stage 1): allocate the per-(token, dst global
          // rank) gather map once, sized for the worst-case source token count.
          // The dispatch sender fills it; combine gathers from it.
          if (ep_combine_recv_idx_gpu == nullptr)
            CUDA_CHECK(cudaMalloc(&ep_combine_recv_idx_gpu,
                                  sizeof(int) * static_cast<size_t>(Config::kEpRecvPoolMaxTokens) * num_ranks));
        }
      }
#endif

      if (low_latency_mode) {
        // LL barrier ring needs MemoryChannels.
        std::vector<mscclpp::BaseMemoryChannelDeviceHandle> ll_handles(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
          if (r == rank || !is_ipc_peer(r)) continue;
          auto sema = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator, ll_ipc_conns[r]);
          ll_memory_channels.emplace_back(sema, remote_mems[r], rdma_mem_ipc);
          ll_handles[r] = ll_memory_channels.rbegin()->deviceHandle();
        }
        ll_memory_channel_handles_device_ptr =
            mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(num_ranks);
        mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(
            ll_memory_channel_handles_device_ptr.get(), ll_handles.data(), num_ranks, cudaMemcpyHostToDevice);

        ll_ranks_per_ipc_domain = ipc_domain_size;
        ll_ipc_ready = ipc_domain_size >= num_ranks;
      }
    }
  }

  // Ready to use
  available = true;
}

void MoEHighThroughputRuntime::getDispatchLayout(int* numTokensPerRank, int* numTokensPerRdmaRank,
                                                 int* numTokensPerExpert, bool* isTokenInRank, const int64_t* topkIdx,
                                                 int numTokens, int numTopk, int numExperts, cudaStream_t stream) {
  EP_HOST_ASSERT(numExperts > 0);

  // Make comm_stream wait for the caller's stream (replaces the torch
  // stream_wait dance), then launch the layout kernel on comm_stream.
  stream_wait(comm_stream, stream);

  internode::get_dispatch_layout(topkIdx, numTokensPerRank, numTokensPerRdmaRank, numTokensPerExpert, isTokenInRank,
                                 numTokens, numTopk, num_ranks, numExperts, comm_stream);

  // Make the caller's stream wait for comm_stream so the outputs are visible.
  stream_wait(stream, comm_stream);
}

void MoEHighThroughputRuntime::computeIntranodeChannels(int xElementSize, const Config& config, int& dispatchNumSms,
                                                        bool& allSender, int& numChannels) const {
  // MSCCLPP_EP_DISPATCH_NSM (intranode): set the dispatch block count independently of the
  // combine grid. The dispatch is channel-partitioned (notify + senders share num_channels),
  // so DISPATCH_NSM drives this whole dispatch+layout pipeline consistently. Constrained to
  // [2, config.num_sms] (the NVL buffers are sized for config.num_sms/2 channels), even.
  int dispatch_num_sms = config.num_sms;
  {
    const char* e_dnsm = std::getenv("MSCCLPP_EP_DISPATCH_NSM");
    if (e_dnsm != nullptr) {
      int n = std::atoi(e_dnsm);
      n &= ~1;  // round down to even (two blocks per channel)
      if (n >= 2) dispatch_num_sms = std::min(n, config.num_sms);
    }
  }
  // MSCCLPP_EP_INTRA_ALLSENDER (INTRA_DIRECT only): make EVERY block a sender (one channel
  // per block) instead of the 50/50 sender/receiver split, so all dispatch_num_sms blocks
  // move hidden directly to the dest pools (matching NCCL-EP's all-sender block count).
  // Metadata goes to the pool META region + a drain pass. The all-sender layout is only
  // consumable by the TMA combine (the ring fallback expects num_sms/2 channels), so this
  // defaults ON under INTRA_DIRECT + TMA combine and auto-disables if the ring combine is
  // forced (MSCCLPP_EP_COMBINE_TMA=0). Explicit MSCCLPP_EP_INTRA_ALLSENDER=0/1 overrides.
  static const bool ep_intra_allsender_env = [] {
    const char* d = std::getenv("MSCCLPP_EP_INTRA_DIRECT");
    if (not(d != nullptr and std::atoi(d) != 0)) return false;  // requires INTRA_DIRECT
    const char* ct = std::getenv("MSCCLPP_EP_COMBINE_TMA");
    if (ct != nullptr and std::atoi(ct) == 0) return false;  // ring combine can't consume all-sender layout
    const char* e = std::getenv("MSCCLPP_EP_INTRA_ALLSENDER");
    return e == nullptr or std::atoi(e) != 0;  // default ON, disable only with explicit =0
  }();
  // The all-sender path requires the BF16 direct pool (xElementSize == 2 mirrors the original
  // `x.scalar_type() == torch::kBFloat16` gate).
  const bool all_sender = ep_intra_allsender_env and recv_pool_local_ptr_ != nullptr and
                          recv_pool_ptrs_gpu != nullptr and xElementSize == 2;
  // All-sender: one channel per block (num_channels = grid). Otherwise: two blocks per channel.
  dispatchNumSms = dispatch_num_sms;
  allSender = all_sender;
  numChannels = all_sender ? dispatch_num_sms : dispatch_num_sms / 2;
}

int MoEHighThroughputRuntime::getIntranodeDispatchNumChannels(int xElementSize, const Config& config) const {
  int dispatch_num_sms = 0, num_channels = 0;
  bool all_sender = false;
  computeIntranodeChannels(xElementSize, config, dispatch_num_sms, all_sender, num_channels);
  return num_channels;
}

void* MoEHighThroughputRuntime::resolveIntranodeRecvXBuffer(int numRecvTokens, int hidden, int xElementSize,
                                                            const Config& config) const {
  // Sender-direct (MSCCLPP_EP_INTRA_DIRECT): make recv_x a zero-copy view of this rank's
  // peer-mapped recv pool so the sender writes hidden straight to its final slot,
  // eliminating the 2-hop ring + receiver hidden drain. Mirrors the ep_intra_direct gate
  // in intranodeDispatch exactly; returns nullptr when the caller should allocate recvX.
#ifdef EP_DISPATCH_NCCLEP
  const size_t ep_intra_pool_header_bytes = config.get_recv_pool_header_bytes(num_ranks);
  const char* e_intra_direct = std::getenv("MSCCLPP_EP_INTRA_DIRECT");
  const bool ep_intra_direct = e_intra_direct != nullptr && std::atoi(e_intra_direct) != 0 &&
                               recv_pool_local_ptr_ != nullptr && recv_pool_ptrs_gpu != nullptr &&
                               numRecvTokens <= Config::kEpRecvPoolMaxTokens &&
                               static_cast<int64_t>(numRecvTokens) * hidden * static_cast<int64_t>(xElementSize) <=
                                   static_cast<int64_t>(Config::recv_pool_bytes_static(num_ranks)) -
                                       static_cast<int64_t>(ep_intra_pool_header_bytes);
  if (ep_intra_direct) {
    return static_cast<uint8_t*>(recv_pool_local_ptr_) + ep_intra_pool_header_bytes;
  }
  return nullptr;
#else
  (void)numRecvTokens;
  (void)hidden;
  (void)xElementSize;
  (void)config;
  return nullptr;
#endif
}

int MoEHighThroughputRuntime::intranodeNotifyDispatch(int* rankPrefixMatrix, int* channelPrefixMatrix,
                                                      int* numRecvTokensPerExpert, const int* numTokensPerRank,
                                                      const int* numTokensPerExpert, const bool* isTokenInRank,
                                                      int numTokens, int numExperts, int xElementSize,
                                                      int expertAlignment, const Config& config, cudaStream_t stream) {
  // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  int dispatch_num_sms = 0, num_channels = 0;
  bool all_sender = false;
  computeIntranodeChannels(xElementSize, config, dispatch_num_sms, all_sender, num_channels);

  const int num_local_experts = numExperts / num_ranks;

  // Make comm_stream wait for the caller's stream before launching.
  stream_wait(comm_stream, stream);

  // Barrier or send sizes
  // To clean: channel start/end offset, head and tail
  int num_memset_int = num_channels * num_ranks * 4;

  // Send sizes
  // Meta information:
  //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
  //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
  // NOTES: no more token dropping in this version
  *moe_recv_counter = -1;
  for (int i = 0; i < num_local_experts; ++i) moe_recv_expert_counter[i] = -1;
  EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
  intranode::notify_dispatch(numTokensPerRank, moe_recv_counter_mapped, num_ranks, numTokensPerExpert,
                             moe_recv_expert_counter_mapped, numExperts, numTokens, isTokenInRank, channelPrefixMatrix,
                             rankPrefixMatrix, num_memset_int, expertAlignment, buffer_ptrs_gpu, task_fifo_ptrs_gpu,
                             head, rank, comm_stream, num_channels);
  move_fifo_slots(3);

  // Synchronize total received tokens and tokens per expert
  int num_recv_tokens = -1;
  auto start_time = std::chrono::high_resolution_clock::now();
  while (true) {
    // Read total count
    num_recv_tokens = static_cast<int>(*moe_recv_counter);

    // Read per-expert count
    bool ready = (num_recv_tokens >= 0);
    for (int i = 0; i < num_local_experts and ready; ++i) ready &= moe_recv_expert_counter[i] >= 0;

    if (ready) break;

    // Timeout check
    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time)
            .count() > NUM_CPU_TIMEOUT_SECS)
      throw std::runtime_error("DeepEP error: CPU recv timeout");
  }
  for (int i = 0; i < num_local_experts; ++i) numRecvTokensPerExpert[i] = moe_recv_expert_counter[i];

  // Make the caller's stream wait for comm_stream so the prefix matrices are visible.
  stream_wait(stream, comm_stream);
  return num_recv_tokens;
}

void MoEHighThroughputRuntime::intranodeDispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx,
                                                 float* recvTopkWeights, int* recvSrcIdx, int* sendHead,
                                                 int* recvChannelPrefixMatrix, const void* x, const float* xScales,
                                                 const int64_t* topkIdx, const float* topkWeights,
                                                 const bool* isTokenInRank, const int* rankPrefixMatrix,
                                                 const int* channelPrefixMatrix, int numTokens, int hidden, int numTopk,
                                                 int numScales, int numExperts, int xElementSize, int numRecvTokens,
                                                 bool cachedMode, const Config& config, cudaStream_t stream) {
  // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  int dispatch_num_sms = 0, num_channels = 0;
  bool all_sender = false;
  computeIntranodeChannels(xElementSize, config, dispatch_num_sms, all_sender, num_channels);

  // num_experts is 0 in cached mode (matches the original tail, where it was
  // derived as `cached_mode ? 0 : num_tokens_per_expert->size(0)`).
  const int num_experts = cachedMode ? 0 : numExperts;

  // Input optional pointers (nullptr == not present).
  const int64_t* topk_idx_ptr = topkIdx;
  const float* topk_weights_ptr = topkWeights;
  const float* x_scales_ptr = xScales;
  int num_topk = numTopk;
  int num_scales = numScales;

  // Make comm_stream wait for the caller's stream before launching.
  stream_wait(comm_stream, stream);

  // Barrier (cached mode only). The non-cached caller already ran
  // intranodeNotifyDispatch (which issued notify_dispatch + the barrier).
  int num_memset_int = num_channels * num_ranks * 4;
  if (cachedMode) {
    // Copy rank prefix matrix and clean flags
    intranode::cached_notify_dispatch(rankPrefixMatrix, num_memset_int, buffer_ptrs_gpu, task_fifo_ptrs_gpu, head, rank,
                                      num_ranks, comm_stream);
    move_fifo_slots(2);
  }

  // Sender-direct (MSCCLPP_EP_INTRA_DIRECT): make recv_x a zero-copy view of this rank's
  // peer-mapped recv pool so the sender writes hidden straight to its final slot,
  // eliminating the 2-hop ring + receiver hidden drain. The caller resolves recvX (= the
  // pool view) via resolveIntranodeRecvXBuffer when this gate is active.
  void** ep_intra_recv_pool_ptrs = nullptr;
#ifdef EP_DISPATCH_NCCLEP
  const size_t ep_intra_pool_header_bytes = config.get_recv_pool_header_bytes(num_ranks);
  const char* e_intra_direct = std::getenv("MSCCLPP_EP_INTRA_DIRECT");
  const bool ep_intra_direct = e_intra_direct != nullptr && std::atoi(e_intra_direct) != 0 &&
                               recv_pool_local_ptr_ != nullptr && recv_pool_ptrs_gpu != nullptr &&
                               numRecvTokens <= Config::kEpRecvPoolMaxTokens &&
                               static_cast<int64_t>(numRecvTokens) * hidden * static_cast<int64_t>(xElementSize) <=
                                   static_cast<int64_t>(Config::recv_pool_bytes_static(num_ranks)) -
                                       static_cast<int64_t>(ep_intra_pool_header_bytes);
#else
  const size_t ep_intra_pool_header_bytes = 0;
  const bool ep_intra_direct = false;
#endif
  if (ep_intra_direct) {
    ep_intra_recv_pool_ptrs = recv_pool_ptrs_gpu;
    // recvX already points at recv_pool_local_ptr_ + header (caller-resolved).
  }

  // Caller-provided recv output pointers (no torch allocation here).
  void* recv_x = recvX;
  int64_t* recv_topk_idx_ptr = recvTopkIdx;
  float* recv_topk_weights_ptr = recvTopkWeights;
  float* recv_x_scales_ptr = recvXScales;

  // Dispatch
  const int dispatch_hidden_int4 = static_cast<int>(hidden * xElementSize / sizeof(int4));
  if (all_sender) {
    // All-sender direct path: every block sends hidden straight to the dest pools; the
    // hidden ring + receiver are unused. Requires the direct pool (assert it resolved).
    EP_HOST_ASSERT(ep_intra_direct and ep_intra_recv_pool_ptrs != nullptr);
    // The TMA combine is token-parallel and ignores channel_prefix_matrix; zero the recv
    // copy so any (non-TMA) consumer sees a defined tensor.
    CUDA_CHECK(cudaMemsetAsync(recvChannelPrefixMatrix, 0, static_cast<size_t>(num_ranks) * num_channels * sizeof(int),
                               comm_stream));
    const int64_t meta_base = static_cast<int64_t>(Config::get_recv_pool_meta_base(num_ranks));
    const int64_t meta_slot_bytes = Config::kEpRecvPoolMetaBytes;
    intranode::dispatch_allsender(sendHead, x, topk_idx_ptr, topk_weights_ptr, x_scales_ptr, isTokenInRank,
                                  channelPrefixMatrix, numTokens, dispatch_hidden_int4, num_topk, num_experts,
                                  num_scales, buffer_ptrs_gpu, rank, num_ranks, comm_stream, num_channels,
                                  recv_pool_ptrs_gpu, static_cast<int64_t>(ep_intra_pool_header_bytes), meta_base,
                                  meta_slot_bytes, ep_combine_recv_idx_gpu);
    // Unpack the per-token metadata the senders packed into our pool META region.
    intranode::intranode_meta_drain(recv_pool_local_ptr_, meta_base, numRecvTokens, recvSrcIdx, recv_topk_idx_ptr,
                                    recv_topk_weights_ptr, recv_x_scales_ptr, num_topk, num_scales, meta_slot_bytes,
                                    comm_stream);
  } else {
    EP_HOST_ASSERT(
        num_ranks * num_ranks * sizeof(int) +             // Size prefix matrix
            num_channels * num_ranks * sizeof(int) +      // Channel start offset
            num_channels * num_ranks * sizeof(int) +      // Channel end offset
            num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden * xElementSize +  // Data buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(int) +  // Source index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk *
                sizeof(int64_t) +  // Top-k index buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk *
                sizeof(float) +  // Top-k weight buffer
            num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) *
                num_scales  // FP8 scale buffer
        <= num_nvl_bytes);
    intranode::dispatch(recv_x, recv_x_scales_ptr, recvSrcIdx, recv_topk_idx_ptr, recv_topk_weights_ptr,
                        recvChannelPrefixMatrix, sendHead, x, x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        isTokenInRank, channelPrefixMatrix, numTokens, dispatch_hidden_int4, num_topk, num_experts,
                        num_scales, buffer_ptrs_gpu, rank, num_ranks, comm_stream, dispatch_num_sms,
                        config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens,
                        ep_intra_recv_pool_ptrs, static_cast<int64_t>(ep_intra_pool_header_bytes),
                        ep_intra_direct ? ep_combine_recv_idx_gpu : nullptr);
  }

  // Make the caller's stream wait for comm_stream so the recv outputs are visible.
  stream_wait(stream, comm_stream);
}

void MoEHighThroughputRuntime::intranodeCombine(void* combinedX, float* combinedTopkWeights, const void* x,
                                                const float* topkWeights, const int* srcIdx,
                                                const int* rankPrefixMatrix, const int* channelPrefixMatrix,
                                                const int* sendHead, int numTokens, int numRecvTokens, int hidden,
                                                int numTopk, int xElementSize, int ringNumChannels,
                                                const Config& config, cudaStream_t stream) {
  // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  int num_channels = config.num_sms / 2;

  // The 2-hop ring fallback consumes channel_prefix_matrix; the TMA direct-gather combine
  // ignores it. The all-sender dispatch produces a wider matrix (one column per block), and
  // DISPATCH_NSM may produce fewer, so the column count is validated only inside the ring
  // fallback (where it must fit the ring buffers); the TMA path tolerates any column count.
  const int ring_num_channels = ringNumChannels;
  EP_HOST_ASSERT((hidden * xElementSize) % sizeof(int4) == 0);

  // Make comm_stream wait for the caller's stream before launching.
  stream_wait(comm_stream, stream);

  const float* topk_weights_ptr = topkWeights;
  float* recv_topk_weights_ptr = combinedTopkWeights;
  int num_topk = numTopk;

  // Combine output. Default: TMA-staged direct-gather (MSCCLPP_EP_COMBINE_TMA != 0) reads
  // each token's contributions straight from the peer recv pools through a cp.async.bulk SMEM
  // pipeline. The grid is token-parallel, so MSCCLPP_EP_COMBINE_NSM sets the block count
  // independently of the dispatch channel count. Falls back to the 2-hop ring combine when the
  // direct-gather inputs are unavailable (no INTRA_DIRECT recv pools) or the escape hatch is set.
  void* recv_x = combinedX;
  bool used_tma_combine = false;
  {
    static const bool ep_combine_tma_disabled = [] {
      const char* e = std::getenv("MSCCLPP_EP_COMBINE_TMA");
      return e != nullptr and std::atoi(e) == 0;
    }();
    // HT combine operates on BF16 (xElementSize == 2 mirrors the original
    // `x.scalar_type() == torch::kBFloat16` gate); the TMA combine requires it.
    const bool tma_inputs_ready =
        recv_pool_ptrs_gpu != nullptr and ep_combine_recv_idx_gpu != nullptr and xElementSize == 2;
    if (not ep_combine_tma_disabled and tma_inputs_ready) {
      int combine_sms = config.num_sms;
      const char* e_cnsm = std::getenv("MSCCLPP_EP_COMBINE_NSM");
      if (e_cnsm != nullptr and std::atoi(e_cnsm) >= 1) combine_sms = std::atoi(e_cnsm);
      const int64_t intra_pool_header_bytes = static_cast<int64_t>(config.get_recv_pool_header_bytes(num_ranks));
      used_tma_combine = intranode::combine_tma(
          CUDA_R_16BF, recv_x, recv_topk_weights_ptr, const_cast<int*>(sendHead), numRecvTokens, hidden, num_topk,
          num_ranks, recv_pool_ptrs_gpu, ep_combine_recv_idx_gpu, intra_pool_header_bytes, combine_sms, comm_stream);
    }
  }

  if (not used_tma_combine) {
    // 2-hop ring fallback. Uses the channel count dispatch actually produced (ring_num_channels).
    // The ring buffers are sized for config.num_sms/2 channels, so the all-sender dispatch
    // (which produces more columns and writes hidden directly, not into the ring) is
    // incompatible with this fallback -- it requires the TMA combine.
    EP_HOST_ASSERT(ring_num_channels <= num_channels);
    const int ring_num_sms = ring_num_channels * 2;
    EP_HOST_ASSERT(ring_num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu, const_cast<int*>(sendHead), ring_num_channels, numRecvTokens,
                                     ring_num_channels * num_ranks * 2, task_fifo_ptrs_gpu, head, rank, num_ranks,
                                     comm_stream);

    // NOTES: this function uses two FIFO slots (barrier before and after)
    move_fifo_slots(2);

    EP_HOST_ASSERT(ring_num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                       ring_num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden *
                           xElementSize +  // Data buffer
                       ring_num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                           sizeof(int) +  // Source index buffer
                       ring_num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk *
                           sizeof(float)  // Top-k weight buffer
                   <= num_nvl_bytes);
    intranode::combine(CUDA_R_16BF, recv_x, recv_topk_weights_ptr, x, topk_weights_ptr, srcIdx, rankPrefixMatrix,
                       channelPrefixMatrix, const_cast<int*>(sendHead), numTokens, numRecvTokens, hidden, num_topk,
                       buffer_ptrs_gpu, rank, num_ranks, comm_stream, ring_num_sms,
                       config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens);
  }

  // Make the caller's stream wait for comm_stream so combinedX is visible.
  stream_wait(stream, comm_stream);
}

// -----------------------------------------------------------------------------
// Internode (NVLink + RDMA) high-throughput path. Ported from DeepEP
// `csrc/deep_ep.cpp`; the kernels it drives are in
// `src/ext/ep/kernels/internode.cu`. Validated end-to-end on 2 x H100 x 8
// via `test/python/ep/test_internode_multirank.py`. De-torched the same
// way as the intranode path: tensor params became raw pointers + size ints,
// output tensors became caller pointers, the EventHandle / async / record_stream
// machinery became comm_stream stream_wait brackets, and the original single
// `internode_dispatch` was split into internodeNotifyDispatch (the non-cached
// notify phase) + internodeDispatch (the dispatch-kernel tail), mirroring the
// intranode split (see SIGNATURES.md for the notify->dispatch boundary). The
// kernels, env knobs, `#ifdef EP_DISPATCH_NCCLEP` blocks, recv-pool / VMM / NVLS
// logic and diagnostics are preserved verbatim.
// -----------------------------------------------------------------------------

void* MoEHighThroughputRuntime::resolveInternodeRecvXBuffer(int numRecvTokens, int hidden, int xElementSize,
                                                            const Config& config) const {
  // Increment 4 (VMM pool): in the non-cached internode dispatch, recv_x is a zero-copy view of
  // this rank's peer-mapped VMM recv-output pool (recv_pool_local_ptr_ + header) so the cross-GPU
  // forwarder writes hidden straight to the destination's recv_x via TMA-eligible peer VAs and the
  // direct-gather combine reads it back. Mirrors the `ep_use_direct` gate in internodeDispatch
  // exactly (minus the cached-mode term, since the helper is only consulted on the non-cached
  // forward path); returns nullptr when the caller should allocate recvX itself.
#ifdef EP_DISPATCH_NCCLEP
  const size_t ep_pool_header_bytes = config.get_recv_pool_header_bytes(num_ranks);
  const bool ep_use_direct = num_nvl_bytes > 0 and recv_pool_local_ptr_ != nullptr and recv_pool_ptrs_gpu != nullptr and
                             numRecvTokens <= Config::kEpRecvPoolMaxTokens;
  (void)hidden;
  (void)xElementSize;
  if (ep_use_direct) {
    return static_cast<uint8_t*>(recv_pool_local_ptr_) + ep_pool_header_bytes;
  }
  return nullptr;
#else
  (void)numRecvTokens;
  (void)hidden;
  (void)xElementSize;
  (void)config;
  return nullptr;
#endif
}

int MoEHighThroughputRuntime::getInternodeDispatchNumChannels(const Config& config) const {
#ifdef EP_DISPATCH_NCCLEP
  return ep_flat_dispatch_channels(config.num_sms);
#else
  return config.num_sms / 2;
#endif
}

int MoEHighThroughputRuntime::getSourceMetaBytes() const { return internode::get_source_meta_bytes(); }

int MoEHighThroughputRuntime::getNumMaxNvlPeers() const { return NUM_MAX_NVL_PEERS; }

int MoEHighThroughputRuntime::internodeNotifyDispatch(
    int* rdmaChannelPrefixMatrix, int* recvRdmaRankPrefixSum, int* gblChannelPrefixMatrix, int* recvGblRankPrefixSum,
    int* numRecvTokensPerExpert, int* numRdmaRecvTokens, const int* numTokensPerRank, const int* numTokensPerRdmaRank,
    const int* numTokensPerExpert, const bool* isTokenInRank, int numTokens, int numExperts, int hidden, int numScales,
    int numTopk, int xElementSize, int expertAlignment, const Config& config, cudaStream_t stream) {
  // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
  int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  EP_HOST_ASSERT(0 < num_rdma_ranks and num_rdma_ranks <= NUM_MAX_RDMA_PEERS);

  // inc6/inc7 (kEpFlat): MSCCLPP_EP_DISPATCH_NSM sets the flat dispatch block count
  // (== num_channels, since the flat all-sender path launches one sender block per
  // channel) independently of config.num_sms. num_channels is the token-partitioning
  // granularity shared by notify_dispatch, the prefix-matrix allocations, and the
  // dispatch grid, so resolving it here keeps the whole dispatch pipeline self-consistent
  // (the NSM sweep already varied num_channels 8..76 with byte-correct recv). Clamped to
  // [1, config.num_sms] (the flat path has no forwarder, so it can use the FULL SM budget:
  // num_sms=16 + MSCCLPP_EP_DISPATCH_NSM=16 -> 16 blocks, matching NCCL-EP); the matching
  // RDMA/NVL buffer is sized for the same channel count via ep_buffer_channels(). Unset ->
  // num_sms/2 (byte-identical default). Flat-only; the 2-hop path keeps num_sms/2. The notify
  // phase is always non-cached, so the original `if (not cached_mode)` guard is unconditional here.
#ifdef EP_DISPATCH_NCCLEP
  num_channels = ep_flat_dispatch_channels(config.num_sms);
#endif

  const int num_local_experts = numExperts / num_ranks;
  const int hidden_int4 = static_cast<int>(hidden * xElementSize / sizeof(int4));

  // Make comm_stream wait for the caller's stream before launching.
  stream_wait(comm_stream, stream);

  // Send sizes
  *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
  for (int i = 0; i < num_local_experts; ++i) moe_recv_expert_counter[i] = -1;
  // NVLS Phase 2: bump the per-call epoch counter so the kernel's
  // barrier spin uses a fresh expected value (epoch * num_ranks).
  if (nvls_ht_enabled) ++nvls_ht_epoch;
  internode::notify_dispatch(
      numTokensPerRank, moe_recv_counter_mapped, num_ranks, numTokensPerRdmaRank, moe_recv_rdma_counter_mapped,
      numTokensPerExpert, moe_recv_expert_counter_mapped, numExperts, isTokenInRank, numTokens, num_channels,
      hidden_int4, numScales, numTopk, expertAlignment, rdmaChannelPrefixMatrix, recvRdmaRankPrefixSum,
      gblChannelPrefixMatrix, recvGblRankPrefixSum, rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
      buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens, task_fifo_ptrs_gpu, head, rank, comm_stream,
      config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks), num_nvl_bytes, low_latency_mode,
      port_channel_handles_device_ptr.get(), memory_channel_handles_device_ptr.get(),
      nvls_ht_enabled ? nvls_ht_mc_ptr : nullptr, nvls_ht_enabled ? nvls_ht_dev_ptr : nullptr, nvls_ht_off_barrier,
      nvls_ht_off_data, nvls_ht_epoch, kNvlsPerPeerBytes);
  move_fifo_slots(3);

  // Synchronize total received tokens and tokens per expert
  int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
  auto start_time = std::chrono::high_resolution_clock::now();
  while (true) {
    num_recv_tokens = static_cast<int>(*moe_recv_counter);
    num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

    bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
    for (int i = 0; i < num_local_experts and ready; ++i) ready &= moe_recv_expert_counter[i] >= 0;

    if (ready) break;

    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time)
            .count() > NUM_CPU_TIMEOUT_SECS) {
      printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens,
             num_rdma_recv_tokens);
      for (int i = 0; i < num_local_experts; ++i)
        printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
      throw std::runtime_error("mscclpp::ep error: timeout (internode_dispatch CPU)");
    }
  }
  for (int i = 0; i < num_local_experts; ++i) numRecvTokensPerExpert[i] = moe_recv_expert_counter[i];
  *numRdmaRecvTokens = num_rdma_recv_tokens;

  // Make the caller's stream wait for comm_stream so the prefix matrices are visible.
  stream_wait(stream, comm_stream);
  return num_recv_tokens;
}

void MoEHighThroughputRuntime::internodeDispatch(
    void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights, void* recvSrcMeta,
    int* recvRdmaChannelPrefixMatrix, int* recvGblChannelPrefixMatrix, int* sendRdmaHead, int* sendNvlHead,
    const void* x, const float* xScales, const int64_t* topkIdx, const float* topkWeights, const bool* isTokenInRank,
    const int* rdmaChannelPrefixMatrix, const int* recvRdmaRankPrefixSum, const int* gblChannelPrefixMatrix,
    const int* recvGblRankPrefixSum, int numTokens, int hidden, int numTopk, int numScales, int numExperts,
    int xElementSize, int numRecvTokens, int numRdmaRecvTokens, bool cachedMode, const Config& config,
    cudaStream_t stream) {
  int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  EP_HOST_ASSERT(0 < num_rdma_ranks and num_rdma_ranks <= NUM_MAX_RDMA_PEERS);

  // inc6/inc7 (kEpFlat): MSCCLPP_EP_DISPATCH_NSM sets the flat dispatch block count
  // (== num_channels, since the flat all-sender path launches one sender block per
  // channel) independently of config.num_sms. num_channels is the token-partitioning
  // granularity shared by notify_dispatch, the prefix-matrix allocations, and the
  // dispatch grid, so resolving it here keeps the whole dispatch pipeline self-consistent
  // (the NSM sweep already varied num_channels 8..76 with byte-correct recv). Clamped to
  // [1, config.num_sms] (the flat path has no forwarder, so it can use the FULL SM budget:
  // num_sms=16 + MSCCLPP_EP_DISPATCH_NSM=16 -> 16 blocks, matching NCCL-EP); the matching
  // RDMA/NVL buffer is sized for the same channel count via ep_buffer_channels(). Unset ->
  // num_sms/2 (byte-identical default). Flat-only; the 2-hop path keeps num_sms/2.
#ifdef EP_DISPATCH_NCCLEP
  if (not cachedMode) num_channels = ep_flat_dispatch_channels(config.num_sms);
#endif

  const int hidden_int4 = static_cast<int>(hidden * xElementSize / sizeof(int4));
  // num_experts is 0 in cached mode (matches the original tail, where it was
  // derived as `cached_mode ? 0 : num_tokens_per_expert->size(0)`).
  const int num_experts = cachedMode ? 0 : numExperts;

  // Input optional pointers (nullptr == not present).
  const int64_t* topk_idx_ptr = topkIdx;
  const float* topk_weights_ptr = topkWeights;
  const float* x_scales_ptr = xScales;
  int num_topk = numTopk;
  int num_scales = numScales;

  // Make comm_stream wait for the caller's stream before launching.
  stream_wait(comm_stream, stream);

  // Barrier or send sizes. The non-cached caller already ran internodeNotifyDispatch
  // (which issued notify_dispatch + the host busy-wait); cached mode just needs the barrier.
  if (cachedMode) {
    // Just a barrier and clean flags
    if (nvls_ht_enabled) ++nvls_ht_cached_epoch;
    internode::cached_notify(hidden_int4, num_scales, num_topk, num_topk, num_ranks, num_channels, 0, nullptr, nullptr,
                             nullptr, nullptr, rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens, task_fifo_ptrs_gpu, head, rank,
                             comm_stream, config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_nvl_bytes, true, low_latency_mode, port_channel_handles_device_ptr.get(),
                             memory_channel_handles_device_ptr.get(), nvls_ht_enabled ? nvls_ht_mc_ptr : nullptr,
                             nvls_ht_enabled ? nvls_ht_dev_ptr : nullptr, nvls_ht_off_barrier, nvls_ht_cached_epoch);
    move_fifo_slots(2);
  }

  // Allocate new tensors
  void** ep_recv_pool_ptrs = nullptr;         // non-null selects the increment-4 VMM direct-write path
  void** ep_recv_pool_global_ptrs = nullptr;  // inc5: domain-wide pool bases (sender direct-write)
#ifdef EP_DISPATCH_NCCLEP
  // Increment 4 (VMM pool): when num_recv_tokens fits the fixed pool, back recv_x
  // by the local VMM recv-output pool so the cross-GPU forwarder can write hidden
  // straight to the destination's recv_x via TMA-eligible peer VAs. Publish our
  // recv_gbl_rank_prefix_sum into the peer-readable pool header. recv_x (= recvX) is the
  // caller-resolved pool view (recv_pool_local_ptr_ + header) from resolveInternodeRecvXBuffer.
  const bool ep_use_direct = (not cachedMode) and num_nvl_bytes > 0 and recv_pool_local_ptr_ != nullptr and
                             recv_pool_ptrs_gpu != nullptr and numRecvTokens <= Config::kEpRecvPoolMaxTokens;
  if (ep_use_direct) {
    ep_recv_pool_ptrs = recv_pool_ptrs_gpu;
    ep_recv_pool_global_ptrs = recv_pool_global_ptrs_gpu;  // inc5: null unless MSCCLPP_EP_DIRECT exchanged
    void* pool_base = recv_pool_local_ptr_;
    CUDA_CHECK(cudaMemcpyAsync(pool_base, recvGblRankPrefixSum, static_cast<size_t>(num_ranks) * sizeof(int),
                               cudaMemcpyDeviceToDevice, comm_stream));
  }
#endif

  // Caller-provided recv output pointers (no torch allocation here). The recv-side metadata
  // outputs (recvSrcMeta / recv*ChannelPrefixMatrix / send*Head) are passed straight into the
  // kernel with the original `cached_mode ? nullptr : ...` gating below.
  void* recv_x = recvX;
  int64_t* recv_topk_idx_ptr = recvTopkIdx;
  float* recv_topk_weights_ptr = recvTopkWeights;
  float* recv_x_scales_ptr = recvXScales;

  // Launch data dispatch
  // Phase 3: pass NVLS counter region pointers (head/tail × mc/dev). When
  // `nvls_ht_enabled` is false, all four are nullptr and the kernel falls
  // back to the legacy PortChannel/atomicAdd path.
  void* nvls_head_mc =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_mc_ptr) + nvls_ht_off_head) : nullptr;
  void* nvls_head_dev =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_dev_ptr) + nvls_ht_off_head) : nullptr;
  void* nvls_tail_mc =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_mc_ptr) + nvls_ht_off_tail) : nullptr;
  void* nvls_tail_dev =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_dev_ptr) + nvls_ht_off_tail) : nullptr;
  internode::dispatch(recv_x, recv_x_scales_ptr, recv_topk_idx_ptr, recv_topk_weights_ptr,
                      cachedMode ? nullptr : recvSrcMeta, x, x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                      cachedMode ? nullptr : sendRdmaHead, cachedMode ? nullptr : sendNvlHead,
                      cachedMode ? nullptr : recvRdmaChannelPrefixMatrix,
                      cachedMode ? nullptr : recvGblChannelPrefixMatrix, rdmaChannelPrefixMatrix, recvRdmaRankPrefixSum,
                      gblChannelPrefixMatrix, recvGblRankPrefixSum, numTokens, hidden_int4, num_scales, num_topk,
                      num_experts, isTokenInRank, rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens,
                      config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens,
                      config.num_max_nvl_chunked_recv_tokens, rank, num_ranks, cachedMode, comm_stream, num_channels,
                      low_latency_mode, port_channel_handles_device_ptr.get(), memory_channel_handles_device_ptr.get(),
                      nvls_head_mc, nvls_head_dev, nvls_tail_mc, nvls_tail_dev, peer_rdma_bases_gpu, ep_recv_pool_ptrs,
                      ep_recv_pool_global_ptrs, ep_recv_pool_global_ptrs ? ep_combine_recv_idx_gpu : nullptr);

#ifdef EP_DISPATCH_NCCLEP
  // Increment 6 (kEpFlat): when MSCCLPP_EP_FLAT is set, the sender wrote each token's
  // metadata straight into the destination pool's META region. Drain it into the
  // recv_* output tensors on comm_stream right after dispatch. With the receiver role
  // still present (approach 1), the dispatch kernel does not exit until all senders'
  // pool writes are visible, so this same-stream drain sees complete metadata (the
  // same guarantee inc5's recv_x relies on). Overwrites the receiver's metadata output
  // with identical values; this validates the flat producer before the receiver is
  // removed in the all-sender stage.
  {
    const char* e_direct = std::getenv("MSCCLPP_EP_DIRECT");
    const char* e_flat = std::getenv("MSCCLPP_EP_FLAT");
    const bool ep_flat = (e_flat && std::atoi(e_flat) != 0) && (e_direct && std::atoi(e_direct) != 0);
    if (ep_flat and ep_use_direct and not cachedMode and topk_idx_ptr != nullptr) {
      const int64_t ep_meta_base = static_cast<int64_t>(Config::get_recv_pool_meta_base(num_ranks));
      internode::flat_meta_drain(recv_pool_local_ptr_, ep_meta_base, numRecvTokens, recvSrcMeta, recv_x_scales_ptr,
                                 recv_topk_idx_ptr, recv_topk_weights_ptr, num_scales, num_topk, num_experts, num_ranks,
                                 rank, Config::kEpRecvPoolMetaBytes, comm_stream);
    }
  }
#endif

  // Make the caller's stream wait for comm_stream so the recv outputs are visible.
  stream_wait(stream, comm_stream);
}

void MoEHighThroughputRuntime::internodeCombine(void* combinedX, float* combinedTopkWeights, const void* x,
                                                const float* topkWeights, const void* srcMeta,
                                                const bool* isCombinedTokenInRank, const int* rdmaChannelPrefixMatrix,
                                                const int* rdmaRankPrefixSum, const int* gblChannelPrefixMatrix,
                                                const int* combinedRdmaHead, const int* combinedNvlHead, int numTokens,
                                                int numCombinedTokens, int hidden, int numTopk, int xElementSize,
                                                const Config& config, cudaStream_t stream) {
  const int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);

  // inc6 (kEpFlat): detect the flat direct-gather combine up front. Under flat the
  // gather path ignores the per-channel prefix matrices entirely (they are vestigial
  // and, with MSCCLPP_EP_DISPATCH_NSM, may be sized for a different channel count), so
  // their num_channels shape checks below are relaxed. The flag is reused later to skip
  // cached_notify and to honor MSCCLPP_EP_COMBINE_NSM.
  bool ep_flat_combine = false;
#ifdef EP_DISPATCH_NCCLEP
  {
    const char* e_direct = std::getenv("MSCCLPP_EP_DIRECT");
    const char* e_flat = std::getenv("MSCCLPP_EP_FLAT");
    const bool ep_flat = (e_flat && std::atoi(e_flat) != 0) && (e_direct && std::atoi(e_direct) != 0);
    ep_flat_combine = ep_flat and recv_pool_global_ptrs_gpu != nullptr and ep_combine_recv_idx_gpu != nullptr;
  }
#endif

  const int hidden_int4 = static_cast<int>(hidden * xElementSize / sizeof(int4));
  EP_HOST_ASSERT((hidden * xElementSize) % sizeof(int4) == 0);

  // Make comm_stream wait for the caller's stream before launching.
  stream_wait(comm_stream, stream);

  const float* topk_weights_ptr = topkWeights;
  float* combined_topk_weights_ptr = combinedTopkWeights;
  int num_topk = numTopk;

  EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
  EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

  // inc6 (kEpFlat): under the flat all-sender path the combine kernel early-returns
  // to the pure direct-gather branch (recv_pool_global_ptrs + ep_combine_recv_idx),
  // so it never touches the rdma/nvl ring or the head breadcrumbs. cached_notify's
  // ring cleanup + head-prep is therefore unnecessary AND unsafe: its sm_id>=3 branch
  // indexes rdma_channel_prefix_matrix (the forwarder-produced recv_rdma_channel_prefix_matrix)
  // to derive token ranges, then writes combined_nvl_head over that range. Under
  // all-sender flat the forwarder is removed so those tensors are never written ->
  // garbage ranges -> out-of-bounds writes (illegal memory access). Skip the whole
  // cached_notify + fifo-advance step under flat; the direct gather needs none of it.
  // (ep_flat_combine was computed once near the top of this function.)
  if (not ep_flat_combine) {
    internode::cached_notify(
        hidden_int4, 0, 0, num_topk, num_ranks, num_channels, numCombinedTokens, const_cast<int*>(combinedRdmaHead),
        rdmaChannelPrefixMatrix, rdmaRankPrefixSum, const_cast<int*>(combinedNvlHead), rdma_buffer_ptr,
        config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens,
        task_fifo_ptrs_gpu, head, rank, comm_stream,
        config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks), num_nvl_bytes, false, low_latency_mode,
        port_channel_handles_device_ptr.get(), memory_channel_handles_device_ptr.get(),
        nvls_ht_enabled ? nvls_ht_mc_ptr : nullptr, nvls_ht_enabled ? nvls_ht_dev_ptr : nullptr, nvls_ht_off_barrier,
        (nvls_ht_enabled ? ++nvls_ht_cached_epoch : 0));
    move_fifo_slots(2);
  }

  // inc6 (kEpFlat): the flat combine is a pure direct-gather whose grid is
  // independent of the dispatch channel partitioning -- it strides every warp over
  // num_combined_tokens and never indexes the per-channel prefix matrices. The
  // combine SM sweep showed it saturates the NVLink ceiling at ~76 blocks, so the
  // grid can be capped independently of config.num_sms via MSCCLPP_EP_COMBINE_NSM
  // (combine block count; default keeps the dispatch-tied num_channels*2). Only
  // applied under flat; the 2-hop combine MUST keep num_channels matching the
  // prefix-matrix column count, so its grid is left untouched. Capped at
  // num_channels (=config.num_sms/2) so the combine grid never exceeds the SM budget.
  int combine_grid_channels = num_channels;
#ifdef EP_DISPATCH_NCCLEP
  if (ep_flat_combine) {
    const char* e_cnsm = std::getenv("MSCCLPP_EP_COMBINE_NSM");
    if (e_cnsm != nullptr) {
      int cnsm_blocks = std::atoi(e_cnsm);
      if (cnsm_blocks >= 2) {
        int cnsm_channels = cnsm_blocks / 2;
        combine_grid_channels = (cnsm_channels < num_channels) ? cnsm_channels : num_channels;
      }
    }
  }
#endif

  void* combined_x = combinedX;
  // Phase 3: NVLS counter region pointers for combine kernel.
  void* combine_nvls_head_mc =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_mc_ptr) + nvls_ht_off_head) : nullptr;
  void* combine_nvls_head_dev =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_dev_ptr) + nvls_ht_off_head) : nullptr;
  void* combine_nvls_tail_mc =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_mc_ptr) + nvls_ht_off_tail) : nullptr;
  void* combine_nvls_tail_dev =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_dev_ptr) + nvls_ht_off_tail) : nullptr;
  internode::combine(CUDA_R_16BF, combined_x, combined_topk_weights_ptr, isCombinedTokenInRank, x, topk_weights_ptr,
                     combinedRdmaHead, combinedNvlHead, srcMeta, rdmaChannelPrefixMatrix, rdmaRankPrefixSum,
                     gblChannelPrefixMatrix, numTokens, numCombinedTokens, hidden, num_topk, rdma_buffer_ptr,
                     config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
                     config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens, rank, num_ranks,
                     comm_stream, combine_grid_channels, low_latency_mode, port_channel_handles_device_ptr.get(),
                     memory_channel_handles_device_ptr.get(), combine_nvls_head_mc, combine_nvls_head_dev,
                     combine_nvls_tail_mc, combine_nvls_tail_dev, peer_rdma_bases_gpu, recv_pool_global_ptrs_gpu,
                     ep_combine_recv_idx_gpu);

  // Make the caller's stream wait for comm_stream so combinedX is visible.
  stream_wait(stream, comm_stream);
}

}  // namespace ep
}  // namespace mscclpp
