#include "buffer.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <cuda_runtime.h>
#include <pybind11/functional.h>
#include <torch/python.h>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mscclpp/gpu_utils.hpp>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"

namespace mscclpp {
namespace ep {

// Upstream MSCCL++ now exposes `Connection::atomicAdd` and
// `PortChannelDeviceHandle::atomicAdd` natively (see commit "atomic add"
// on branch chhwang/new-atomic-add, merged into this tree). The stock
// `ProxyService` recognises `ChannelTrigger.type == 0` as an atomic-add
// request, so no subclass or private-member access is required anymore.
using EPProxyService = mscclpp::ProxyService;

// Number of host-side proxy services (== proxy threads) the Buffer creates.
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

Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode)
    : rank(rank),
      num_ranks(num_ranks),
      num_nvl_bytes(num_nvl_bytes),
      num_rdma_bytes(num_rdma_bytes),
      low_latency_mode(low_latency_mode),
      comm_stream(at::cuda::getStreamFromPool(true)),
      bootstrap(std::make_shared<mscclpp::TcpBootstrap>(rank, num_ranks)) {
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

Buffer::~Buffer() noexcept(false) {
  // Synchronize
  CUDA_CHECK(cudaDeviceSynchronize());

  if (num_nvl_bytes > 0) {
    // Barrier
    intranode::barrier(task_fifo_ptrs_gpu, head, nvl_rank, num_nvl_ranks, comm_stream);
    move_fifo_slots();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Close remote IPC
    if (is_available()) {
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
}

void Buffer::move_fifo_slots(int num_slots) { head = (head + num_ranks * num_slots) % NUM_MAX_FIFO_SLOTS; }

bool Buffer::is_available() const { return available; }

bool Buffer::is_internode_available() const { return is_available() and num_ranks > NUM_MAX_NVL_PEERS; }

int Buffer::get_num_rdma_ranks() const { return num_rdma_ranks; }

int Buffer::get_rdma_rank() const { return rdma_rank; }

int Buffer::get_root_rdma_rank(bool global) const { return global ? nvl_rank : 0; }

int Buffer::get_local_device_id() const { return device_id; }

pybind11::bytearray Buffer::get_local_ipc_handle() const {
  return {ipc_handles[nvl_rank].reserved, CUDA_IPC_HANDLE_SIZE};
}

pybind11::bytearray Buffer::get_local_nvshmem_unique_id() const {
  // The MSCCL++ EP port replaces NVSHMEM with PortChannel/MemoryChannel,
  // so there is no NVSHMEM unique id to expose. Kept for ABI parity with
  // DeepEP's Python frontend; callers should use the MSCCL++ bootstrap.
  throw std::runtime_error(
      "mscclpp::ep::Buffer::get_local_nvshmem_unique_id: not applicable (NVSHMEM is not used in mscclpp_ep)");
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset,
                                              bool use_rdma_buffer) const {
  torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
  auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
  auto base_ptr = reinterpret_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[nvl_rank]) + offset;
  auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_nvl_bytes;
  return torch::from_blob(base_ptr, num_bytes / element_bytes,
                          torch::TensorOptions().dtype(casted_dtype).device(at::kCUDA));
}

mscclpp::UniqueId Buffer::create_unique_id() const { return bootstrap->createUniqueId(); }

void Buffer::connect(mscclpp::UniqueId root_id) {
  bootstrap->initialize(root_id);
  communicator = std::make_shared<mscclpp::Communicator>(bootstrap);
}

void Buffer::sync(const std::vector<int>& device_ids,
                  const std::vector<std::optional<pybind11::bytearray>>& all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
  EP_HOST_ASSERT(not is_available());

  const std::vector<mscclpp::Transport> ib_transports = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5, mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  const auto ipc_transport = mscclpp::Transport::CudaIpc;
  const auto ib_transport = ib_transports[device_id];
  const mscclpp::TransportFlags all_transport = ipc_transport | ib_transport;

  // Sync IPC handles
  if (num_nvl_bytes > 0) {
    EP_HOST_ASSERT(num_ranks == device_ids.size());
    EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
    for (int i = 0, offset = rdma_rank * num_nvl_ranks; i < num_nvl_ranks; ++i) {
      EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
      auto handle_str = std::string(all_gathered_handles[offset + i].value());
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
    // Phase 4: HT internode mode also benefits from fabric-IPC peer
    // pointers. On Azure CX-7 the IB RDMA WRITE that PortChannel uses
    // for the dispatch/combine token data payload hangs cross-node
    // (same root cause as signal/wait), so we set up the same per-peer
    // mapped pointers as LL and the kernels write tokens directly into
    // the peer's `rdma_buffer_ptr` via the NVL72 fabric VA.
    // ------------------------------------------------------------------
    const bool want_fabric_ipc = use_fabric_ipc_alloc;
    if (want_fabric_ipc) {
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
        if (r == rank) continue;
        communicator->sendMemory(rdma_mem_ipc, r, kLlIpcTag);
        remote_futures[r] = communicator->recvMemory(r, kLlIpcTag);
      }
      std::vector<mscclpp::Connection> ll_ipc_conns(num_ranks);
      {
        std::vector<std::shared_future<mscclpp::Connection>> conn_futures(num_ranks);
        mscclpp::EndpointConfig cfg(ipc_transport);
        for (int r = 0; r < num_ranks; ++r) {
          if (r == rank) continue;
          conn_futures[r] = communicator->connect(cfg, r, kLlIpcTag);
        }
        for (int r = 0; r < num_ranks; ++r) {
          if (r == rank) continue;
          ll_ipc_conns[r] = conn_futures[r].get();
        }
      }

      // Resolve peer base pointers from the (now mapped) remote
      // RegisteredMemory. `data()` returns the locally-mapped peer pointer;
      // for fabric handles this address lives in the cuMem fabric VA range
      // and is dereferenceable from the GPU.
      peer_rdma_bases.assign(num_ranks, nullptr);
      peer_rdma_bases[rank] = rdma_buffer_ptr;
      std::vector<mscclpp::RegisteredMemory> remote_mems(num_ranks);
      for (int r = 0; r < num_ranks; ++r) {
        if (r == rank) continue;
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

      if (low_latency_mode) {
        // LL barrier ring needs MemoryChannels.
        std::vector<mscclpp::MemoryChannelDeviceHandle> ll_handles(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
          if (r == rank) continue;
          auto sema = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator, ll_ipc_conns[r]);
          ll_memory_channels.emplace_back(sema, remote_mems[r], rdma_mem_ipc);
          ll_handles[r] = ll_memory_channels.rbegin()->deviceHandle();
        }
        ll_memory_channel_handles_device_ptr =
            mscclpp::detail::gpuCallocShared<mscclpp::MemoryChannelDeviceHandle>(num_ranks);
        mscclpp::gpuMemcpy<mscclpp::MemoryChannelDeviceHandle>(ll_memory_channel_handles_device_ptr.get(),
                                                               ll_handles.data(), num_ranks, cudaMemcpyHostToDevice);

        ll_ipc_ready = true;
      }
    }
  }

  // Ready to use
  available = true;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts, std::optional<EventHandle>& previous_event,
                            bool async, bool allocate_on_comm_stream) {
  EP_HOST_ASSERT(topk_idx.dim() == 2);
  EP_HOST_ASSERT(topk_idx.is_contiguous());
  EP_HOST_ASSERT(num_experts > 0);

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() and async);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
  auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
  auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
  auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kCUDA));
  auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kCUDA));
  if (is_internode_available())
    num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

  internode::get_dispatch_layout(
      topk_idx.data_ptr<int64_t>(), num_tokens_per_rank.data_ptr<int>(),
      num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
      num_tokens_per_expert.data_ptr<int>(), is_token_in_rank.data_ptr<bool>(), num_tokens, num_topk, num_ranks,
      num_experts, comm_stream);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {num_tokens_per_rdma_rank}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream) to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

  return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
           std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           std::optional<EventHandle>>
Buffer::intranode_dispatch(
    const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales, const std::optional<torch::Tensor>& topk_idx,
    const std::optional<torch::Tensor>& topk_weights, const std::optional<torch::Tensor>& num_tokens_per_rank,
    const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
    int cached_num_recv_tokens, const std::optional<torch::Tensor>& cached_rank_prefix_matrix,
    const std::optional<torch::Tensor>& cached_channel_prefix_matrix, int expert_alignment, const Config& config,
    std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
  bool cached_mode = cached_rank_prefix_matrix.has_value();

  // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  int num_channels = config.num_sms / 2;
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
    EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());
  }

  // Type checks
  EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
  } else {
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
  }

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
  EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
  EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
  EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
    EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
    EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
    EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and
                   cached_channel_prefix_matrix->size(1) == num_channels);
  } else {
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
  }

  auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
  auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
       num_local_experts = num_experts / num_ranks;

  // Top-k checks
  int num_topk = 0;
  int64_t* topk_idx_ptr = nullptr;
  float* topk_weights_ptr = nullptr;
  EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
  if (topk_idx.has_value()) {
    num_topk = static_cast<int>(topk_idx->size(1));
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
    EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
    EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
    EP_HOST_ASSERT(num_topk == topk_weights->size(1));
    EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
    topk_idx_ptr = topk_idx->data_ptr<int64_t>();
    topk_weights_ptr = topk_weights->data_ptr<float>();
  }

  // FP8 scales checks
  float* x_scales_ptr = nullptr;
  int num_scales = 0;
  if (x_scales.has_value()) {
    EP_HOST_ASSERT(x.element_size() == 1);
    EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(x_scales->dim() > 0 and x_scales->dim() < 3 and x_scales->is_contiguous());
    EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
    num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
    x_scales_ptr = x_scales->data_ptr<float>();
  }

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() and async);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Create handles (only return for non-cached mode)
  int num_recv_tokens = -1;
  auto rank_prefix_matrix = torch::Tensor();
  auto channel_prefix_matrix = torch::Tensor();
  std::vector<int> num_recv_tokens_per_expert_list;

  // Barrier or send sizes
  // To clean: channel start/end offset, head and tail
  int num_memset_int = num_channels * num_ranks * 4;
  if (cached_mode) {
    num_recv_tokens = cached_num_recv_tokens;
    rank_prefix_matrix = cached_rank_prefix_matrix.value();
    channel_prefix_matrix = cached_channel_prefix_matrix.value();

    // Copy rank prefix matrix and clean flags
    intranode::cached_notify_dispatch(rank_prefix_matrix.data_ptr<int>(), num_memset_int, buffer_ptrs_gpu,
                                      task_fifo_ptrs_gpu, head, rank, num_ranks, comm_stream);
    move_fifo_slots(2);
  } else {
    rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));

    // Send sizes
    // Meta information:
    //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
    //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
    // NOTES: no more token dropping in this version
    *moe_recv_counter = -1;
    for (int i = 0; i < num_local_experts; ++i) moe_recv_expert_counter[i] = -1;
    EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_nvl_bytes);
    intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
                               num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
                               num_tokens, is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
                               rank_prefix_matrix.data_ptr<int>(), num_memset_int, expert_alignment, buffer_ptrs_gpu,
                               task_fifo_ptrs_gpu, head, rank, comm_stream, num_channels);
    move_fifo_slots(3);

    // Synchronize total received tokens and tokens per expert
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
    num_recv_tokens_per_expert_list =
        std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
  }

  // Allocate new tensors
  auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
  auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kCUDA));
  auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
       recv_x_scales = std::optional<torch::Tensor>();
  auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
  auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

  // Assign pointers
  int64_t* recv_topk_idx_ptr = nullptr;
  float* recv_topk_weights_ptr = nullptr;
  float* recv_x_scales_ptr = nullptr;
  if (topk_idx.has_value()) {
    recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
    recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
    recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
    recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
  }
  if (x_scales.has_value()) {
    recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                         : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
    recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
  }

  // Dispatch
  EP_HOST_ASSERT(num_ranks * num_ranks * sizeof(int) +             // Size prefix matrix
                     num_channels * num_ranks * sizeof(int) +      // Channel start offset
                     num_channels * num_ranks * sizeof(int) +      // Channel end offset
                     num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                     num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden *
                         recv_x.element_size() +  // Data buffer
                     num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                         sizeof(int) +  // Source index buffer
                     num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk *
                         sizeof(int64_t) +  // Top-k index buffer
                     num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk *
                         sizeof(float) +  // Top-k weight buffer
                     num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * sizeof(float) *
                         num_scales  // FP8 scale buffer
                 <= num_nvl_bytes);
  intranode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(), recv_topk_idx_ptr,
                      recv_topk_weights_ptr, recv_channel_prefix_matrix.data_ptr<int>(), send_head.data_ptr<int>(),
                      x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr, is_token_in_rank.data_ptr<bool>(),
                      channel_prefix_matrix.data_ptr<int>(), num_tokens,
                      static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)), num_topk, num_experts,
                      num_scales, buffer_ptrs_gpu, rank, num_ranks, comm_stream, config.num_sms,
                      config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, recv_x, recv_src_idx,
                    recv_channel_prefix_matrix, send_head}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to :
         {x_scales, topk_idx, topk_weights, num_tokens_per_rank, num_tokens_per_expert, cached_channel_prefix_matrix,
          cached_rank_prefix_matrix, recv_topk_idx, recv_topk_weights, recv_x_scales}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream) to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

  // Return values
  return {recv_x,
          recv_x_scales,
          recv_topk_idx,
          recv_topk_weights,
          num_recv_tokens_per_expert_list,
          rank_prefix_matrix,
          channel_prefix_matrix,
          recv_channel_prefix_matrix,
          recv_src_idx,
          send_head,
          event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::intranode_combine(
    const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights, const torch::Tensor& src_idx,
    const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix, const torch::Tensor& send_head,
    const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
  EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
  EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
  EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
  EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and
                 rank_prefix_matrix.scalar_type() == torch::kInt32);
  EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and
                 channel_prefix_matrix.scalar_type() == torch::kInt32);

  // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  int num_channels = config.num_sms / 2;

  auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
  auto num_recv_tokens = static_cast<int>(send_head.size(0));
  EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
  EP_HOST_ASSERT(send_head.size(1) == num_ranks);
  EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
  EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
  EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

  // Allocate all tensors on comm stream if set
  // NOTES: do not allocate tensors upfront!
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() and async);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  int num_topk = 0;
  auto recv_topk_weights = std::optional<torch::Tensor>();
  float* topk_weights_ptr = nullptr;
  float* recv_topk_weights_ptr = nullptr;
  if (topk_weights.has_value()) {
    EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
    EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
    num_topk = static_cast<int>(topk_weights->size(1));
    topk_weights_ptr = topk_weights->data_ptr<float>();
    recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
    recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
  }

  // Launch barrier and reset queue head and tail
  EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_nvl_bytes);
  intranode::cached_notify_combine(buffer_ptrs_gpu, send_head.data_ptr<int>(), num_channels, num_recv_tokens,
                                   num_channels * num_ranks * 2, task_fifo_ptrs_gpu, head, rank, num_ranks,
                                   comm_stream);

  // NOTES: this function uses two FIFO slots (barrier before and after)
  move_fifo_slots(2);

  // Combine data
  auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
  EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +  // Queue head and tail
                     num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * hidden *
                         x.element_size() +  // Data buffer
                     num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens *
                         sizeof(int) +  // Source index buffer
                     num_channels * num_ranks * config.num_max_nvl_chunked_recv_tokens * num_topk *
                         sizeof(float)  // Top-k weight buffer
                 <= num_nvl_bytes);
  intranode::combine(at::cuda::ScalarTypeToCudaDataType(x.scalar_type()), recv_x.data_ptr(), recv_topk_weights_ptr,
                     x.data_ptr(), topk_weights_ptr, src_idx.data_ptr<int>(), rank_prefix_matrix.data_ptr<int>(),
                     channel_prefix_matrix.data_ptr<int>(), send_head.data_ptr<int>(), num_tokens, num_recv_tokens,
                     hidden, num_topk, buffer_ptrs_gpu, rank, num_ranks, comm_stream, config.num_sms,
                     config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {topk_weights, recv_topk_weights}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream) to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  // Switch back compute stream
  if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

  return {recv_x, recv_topk_weights, event};
}

// -----------------------------------------------------------------------------
// Internode (NVLink + RDMA) high-throughput path. Ported from DeepEP
// `csrc/deep_ep.cpp`; the kernels it drives are in
// `src/ext/ep/kernels/internode.cu`. Validated end-to-end on 2 x H100 x 8
// via `test/python/ext/ep/test_internode_multirank.py`. The low-latency
// (pure RDMA) paths further below are a structural port and have not
// been validated on real hardware.
// -----------------------------------------------------------------------------

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
           std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor,
           std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>,
           std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::internode_dispatch(
    const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales, const std::optional<torch::Tensor>& topk_idx,
    const std::optional<torch::Tensor>& topk_weights, const std::optional<torch::Tensor>& num_tokens_per_rank,
    const std::optional<torch::Tensor>& num_tokens_per_rdma_rank, const torch::Tensor& is_token_in_rank,
    const std::optional<torch::Tensor>& num_tokens_per_expert, int cached_num_recv_tokens,
    int cached_num_rdma_recv_tokens, const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix,
    const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
    const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix,
    const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum, int expert_alignment, const Config& config,
    std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
  // In dispatch, CPU will busy-wait until GPU receive tensor size metadata from other ranks, which can be quite long.
  pybind11::gil_scoped_release release;

  const int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);
  EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

  bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
    EP_HOST_ASSERT(num_tokens_per_expert.has_value());
  }

  // Type checks
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
  }

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
  EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
  if (cached_mode) {
    EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and
                   cached_rdma_channel_prefix_matrix->is_contiguous());
    EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and
                   cached_rdma_channel_prefix_matrix->size(1) == num_channels);
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
    EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
    EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and
                   cached_gbl_channel_prefix_matrix->size(1) == num_channels);
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
    EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
  } else {
    EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
    EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
    EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
  }

  auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
       hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
  auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)),
       num_local_experts = num_experts / num_ranks;

  // Top-k checks
  int num_topk = 0;
  int64_t* topk_idx_ptr = nullptr;
  float* topk_weights_ptr = nullptr;
  EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
  if (topk_idx.has_value()) {
    num_topk = static_cast<int>(topk_idx->size(1));
    EP_HOST_ASSERT(num_experts > 0);
    EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
    EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
    EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
    EP_HOST_ASSERT(num_topk == topk_weights->size(1));
    EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
    topk_idx_ptr = topk_idx->data_ptr<int64_t>();
    topk_weights_ptr = topk_weights->data_ptr<float>();
  }

  // FP8 scales checks
  float* x_scales_ptr = nullptr;
  int num_scales = 0;
  if (x_scales.has_value()) {
    EP_HOST_ASSERT(x.element_size() == 1);
    EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(x_scales->dim() > 0 and x_scales->dim() < 3 and x_scales->is_contiguous());
    EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
    num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
    x_scales_ptr = x_scales->data_ptr<float>();
  }

  // Allocate all tensors on comm stream if set
  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() and async);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  // Wait previous tasks to be finished
  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  // Create handles (only return for non-cached mode)
  int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
  auto rdma_channel_prefix_matrix = torch::Tensor();
  auto recv_rdma_rank_prefix_sum = torch::Tensor();
  auto gbl_channel_prefix_matrix = torch::Tensor();
  auto recv_gbl_rank_prefix_sum = torch::Tensor();
  std::vector<int> num_recv_tokens_per_expert_list;

  // Barrier or send sizes
  if (cached_mode) {
    num_recv_tokens = cached_num_recv_tokens;
    num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
    rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
    recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
    gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
    recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

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
  } else {
    rdma_channel_prefix_matrix =
        torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kCUDA));

    // Send sizes
    *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
    for (int i = 0; i < num_local_experts; ++i) moe_recv_expert_counter[i] = -1;
    // NVLS Phase 2: bump the per-call epoch counter so the kernel's
    // barrier spin uses a fresh expected value (epoch * num_ranks).
    if (nvls_ht_enabled) ++nvls_ht_epoch;
    internode::notify_dispatch(
        num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
        num_tokens_per_rdma_rank->data_ptr<int>(), moe_recv_rdma_counter_mapped, num_tokens_per_expert->data_ptr<int>(),
        moe_recv_expert_counter_mapped, num_experts, is_token_in_rank.data_ptr<bool>(), num_tokens, num_channels,
        hidden_int4, num_scales, num_topk, expert_alignment, rdma_channel_prefix_matrix.data_ptr<int>(),
        recv_rdma_rank_prefix_sum.data_ptr<int>(), gbl_channel_prefix_matrix.data_ptr<int>(),
        recv_gbl_rank_prefix_sum.data_ptr<int>(), rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
        buffer_ptrs_gpu, config.num_max_nvl_chunked_recv_tokens, task_fifo_ptrs_gpu, head, rank, comm_stream,
        config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks), num_nvl_bytes, low_latency_mode,
        port_channel_handles_device_ptr.get(), memory_channel_handles_device_ptr.get(),
        nvls_ht_enabled ? nvls_ht_mc_ptr : nullptr, nvls_ht_enabled ? nvls_ht_dev_ptr : nullptr, nvls_ht_off_barrier,
        nvls_ht_off_data, nvls_ht_epoch, kNvlsPerPeerBytes);
    move_fifo_slots(3);

    // Synchronize total received tokens and tokens per expert
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
    num_recv_tokens_per_expert_list =
        std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
  }

  // Allocate new tensors
  void** ep_recv_pool_ptrs = nullptr;  // non-null selects the increment-4 VMM direct-write path
#ifdef EP_DISPATCH_NCCLEP
  // Increment 4 (VMM pool): when num_recv_tokens fits the fixed pool, back recv_x
  // by the local VMM recv-output pool so the cross-GPU forwarder can write hidden
  // straight to the destination's recv_x via TMA-eligible peer VAs. Publish our
  // recv_gbl_rank_prefix_sum into the peer-readable pool header.
  const size_t ep_pool_header_bytes = config.get_recv_pool_header_bytes(num_ranks);
  const bool ep_use_direct = (not cached_mode) and num_nvl_bytes > 0 and recv_pool_local_ptr_ != nullptr and
                             recv_pool_ptrs_gpu != nullptr and num_recv_tokens <= Config::kEpRecvPoolMaxTokens;
  torch::Tensor recv_x;
  if (ep_use_direct) {
    ep_recv_pool_ptrs = recv_pool_ptrs_gpu;
    void* pool_base = recv_pool_local_ptr_;
    CUDA_CHECK(cudaMemcpyAsync(pool_base, recv_gbl_rank_prefix_sum.data_ptr<int>(),
                               static_cast<size_t>(num_ranks) * sizeof(int), cudaMemcpyDeviceToDevice, comm_stream));
    void* recv_x_ptr = static_cast<uint8_t*>(pool_base) + ep_pool_header_bytes;
    recv_x = torch::from_blob(recv_x_ptr, {num_recv_tokens, hidden}, x.options());
  } else {
    recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
  }
#else
  auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
#endif
  auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(),
       recv_x_scales = std::optional<torch::Tensor>();
  auto recv_src_meta = std::optional<torch::Tensor>();
  auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
  auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
  auto send_rdma_head = std::optional<torch::Tensor>();
  auto send_nvl_head = std::optional<torch::Tensor>();
  if (not cached_mode) {
    recv_src_meta =
        torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kCUDA));
    recv_rdma_channel_prefix_matrix =
        torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kCUDA));
    send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kCUDA));
    send_nvl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_NVL_PEERS}, dtype(torch::kInt32).device(torch::kCUDA));
  }

  int64_t* recv_topk_idx_ptr = nullptr;
  float* recv_topk_weights_ptr = nullptr;
  float* recv_x_scales_ptr = nullptr;
  if (topk_idx.has_value()) {
    recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
    recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
    recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
    recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
  }
  if (x_scales.has_value()) {
    recv_x_scales = x_scales->dim() == 1 ? torch::empty({num_recv_tokens}, x_scales->options())
                                         : torch::empty({num_recv_tokens, num_scales}, x_scales->options());
    recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
  }

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
  internode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_topk_idx_ptr, recv_topk_weights_ptr,
                      cached_mode ? nullptr : recv_src_meta->data_ptr(), x.data_ptr(), x_scales_ptr, topk_idx_ptr,
                      topk_weights_ptr, cached_mode ? nullptr : send_rdma_head->data_ptr<int>(),
                      cached_mode ? nullptr : send_nvl_head->data_ptr<int>(),
                      cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                      cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                      rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                      gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(), num_tokens,
                      hidden_int4, num_scales, num_topk, num_experts, is_token_in_rank.data_ptr<bool>(),
                      rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                      buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens, config.num_max_nvl_chunked_recv_tokens,
                      rank, num_ranks, cached_mode, comm_stream, num_channels, low_latency_mode,
                      port_channel_handles_device_ptr.get(), memory_channel_handles_device_ptr.get(), nvls_head_mc,
                      nvls_head_dev, nvls_tail_mc, nvls_tail_dev, peer_rdma_bases_gpu, ep_recv_pool_ptrs);

  // Wait streams
  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x, is_token_in_rank, recv_x, rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {x_scales, topk_idx, topk_weights, num_tokens_per_rank, num_tokens_per_rdma_rank,
                     num_tokens_per_expert, cached_rdma_channel_prefix_matrix, cached_recv_rdma_rank_prefix_sum,
                     cached_gbl_channel_prefix_matrix, cached_recv_gbl_rank_prefix_sum, recv_topk_idx,
                     recv_topk_weights, recv_x_scales, recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix,
                     send_rdma_head, send_nvl_head, recv_src_meta}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream) to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

  return {recv_x,
          recv_x_scales,
          recv_topk_idx,
          recv_topk_weights,
          num_recv_tokens_per_expert_list,
          rdma_channel_prefix_matrix,
          gbl_channel_prefix_matrix,
          recv_rdma_channel_prefix_matrix,
          recv_rdma_rank_prefix_sum,
          recv_gbl_channel_prefix_matrix,
          recv_gbl_rank_prefix_sum,
          recv_src_meta,
          send_rdma_head,
          send_nvl_head,
          event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> Buffer::internode_combine(
    const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights, const torch::Tensor& src_meta,
    const torch::Tensor& is_combined_token_in_rank, const torch::Tensor& rdma_channel_prefix_matrix,
    const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
    const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_nvl_head, const Config& config,
    std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
  const int num_channels = config.num_sms / 2;
  EP_HOST_ASSERT(config.num_sms % 2 == 0);

  // Shape and contiguous checks
  EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
  EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
  EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and
                 is_combined_token_in_rank.scalar_type() == torch::kBool);
  EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and
                 rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
  EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and
                 rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
  EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and
                 gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
  EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and
                 combined_rdma_head.scalar_type() == torch::kInt32);
  EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.is_contiguous() and
                 combined_nvl_head.scalar_type() == torch::kInt32);

  auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)),
       hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
  auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
  EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
  EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
  EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
  EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and
                 rdma_channel_prefix_matrix.size(1) == num_channels);
  EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
  EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
  EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and
                 combined_rdma_head.size(1) == num_rdma_ranks);
  EP_HOST_ASSERT(combined_nvl_head.dim() == 2 and combined_nvl_head.size(1) == NUM_MAX_NVL_PEERS);

  auto compute_stream = at::cuda::getCurrentCUDAStream();
  if (allocate_on_comm_stream) {
    EP_HOST_ASSERT(previous_event.has_value() and async);
    at::cuda::setCurrentCUDAStream(comm_stream);
  }

  if (previous_event.has_value()) {
    stream_wait(comm_stream, previous_event.value());
  } else {
    stream_wait(comm_stream, compute_stream);
  }

  int num_topk = 0;
  auto combined_topk_weights = std::optional<torch::Tensor>();
  float* topk_weights_ptr = nullptr;
  float* combined_topk_weights_ptr = nullptr;
  if (topk_weights.has_value()) {
    EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
    EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
    EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
    num_topk = static_cast<int>(topk_weights->size(1));
    topk_weights_ptr = topk_weights->data_ptr<float>();
    combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
    combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
  }

  EP_HOST_ASSERT(config.num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
  EP_HOST_ASSERT(config.num_max_nvl_chunked_send_tokens <= config.num_max_nvl_chunked_recv_tokens / num_rdma_ranks);

  internode::cached_notify(
      hidden_int4, 0, 0, num_topk, num_ranks, num_channels, num_combined_tokens, combined_rdma_head.data_ptr<int>(),
      rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(),
      combined_nvl_head.data_ptr<int>(), rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu,
      config.num_max_nvl_chunked_recv_tokens, task_fifo_ptrs_gpu, head, rank, comm_stream,
      config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks), num_nvl_bytes, false, low_latency_mode,
      port_channel_handles_device_ptr.get(), memory_channel_handles_device_ptr.get(),
      nvls_ht_enabled ? nvls_ht_mc_ptr : nullptr, nvls_ht_enabled ? nvls_ht_dev_ptr : nullptr, nvls_ht_off_barrier,
      (nvls_ht_enabled ? ++nvls_ht_cached_epoch : 0));
  move_fifo_slots(2);

  auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
  // Phase 3: NVLS counter region pointers for combine kernel.
  void* combine_nvls_head_mc =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_mc_ptr) + nvls_ht_off_head) : nullptr;
  void* combine_nvls_head_dev =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_dev_ptr) + nvls_ht_off_head) : nullptr;
  void* combine_nvls_tail_mc =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_mc_ptr) + nvls_ht_off_tail) : nullptr;
  void* combine_nvls_tail_dev =
      nvls_ht_enabled ? static_cast<void*>(static_cast<char*>(nvls_ht_dev_ptr) + nvls_ht_off_tail) : nullptr;
  internode::combine(
      at::cuda::ScalarTypeToCudaDataType(x.scalar_type()), combined_x.data_ptr(), combined_topk_weights_ptr,
      is_combined_token_in_rank.data_ptr<bool>(), x.data_ptr(), topk_weights_ptr, combined_rdma_head.data_ptr<int>(),
      combined_nvl_head.data_ptr<int>(), src_meta.data_ptr(), rdma_channel_prefix_matrix.data_ptr<int>(),
      rdma_rank_prefix_sum.data_ptr<int>(), gbl_channel_prefix_matrix.data_ptr<int>(), num_tokens, num_combined_tokens,
      hidden, num_topk, rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens,
      config.num_max_rdma_chunked_recv_tokens, buffer_ptrs_gpu, config.num_max_nvl_chunked_send_tokens,
      config.num_max_nvl_chunked_recv_tokens, rank, num_ranks, comm_stream, num_channels, low_latency_mode,
      port_channel_handles_device_ptr.get(), memory_channel_handles_device_ptr.get(), combine_nvls_head_mc,
      combine_nvls_head_dev, combine_nvls_tail_mc, combine_nvls_tail_dev, peer_rdma_bases_gpu);

  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(comm_stream);
    for (auto& t : {x, src_meta, is_combined_token_in_rank, rdma_channel_prefix_matrix, rdma_rank_prefix_sum,
                    gbl_channel_prefix_matrix, combined_x, combined_rdma_head, combined_nvl_head}) {
      t.record_stream(comm_stream);
      if (allocate_on_comm_stream) t.record_stream(compute_stream);
    }
    for (auto& to : {topk_weights, combined_topk_weights}) {
      to.has_value() ? to->record_stream(comm_stream) : void();
      if (allocate_on_comm_stream) to.has_value() ? to->record_stream(compute_stream) : void();
    }
  } else {
    stream_wait(compute_stream, comm_stream);
  }

  if (allocate_on_comm_stream) at::cuda::setCurrentCUDAStream(compute_stream);

  return {combined_x, combined_topk_weights, event};
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
  EP_HOST_ASSERT(low_latency_mode);

  auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
  auto clean_meta_0 = layout.buffers[0].clean_meta();
  auto clean_meta_1 = layout.buffers[1].clean_meta();

  auto check_boundary = [=](void* ptr, size_t num_bytes) {
    auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
    EP_HOST_ASSERT(0 <= offset and offset + static_cast<int64_t>(num_bytes) <= num_rdma_bytes);
  };
  check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
  check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

  internode_ll::clean_low_latency_buffer(
      clean_meta_0.first, clean_meta_0.second, clean_meta_1.first, clean_meta_1.second, rank, num_ranks,
      port_channel_handles_device_ptr.get(),
      ll_memory_channel_handles_device_ptr ? ll_memory_channel_handles_device_ptr.get() : nullptr, ll_ipc_ready,
      at::cuda::getCurrentCUDAStream());
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor,
           std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                             int num_max_dispatch_tokens_per_rank, int num_experts, bool use_fp8, bool async,
                             bool return_recv_hook, const std::optional<torch::Tensor>& out_packed_recv_x,
                             const std::optional<torch::Tensor>& out_packed_recv_x_scales,
                             const std::optional<torch::Tensor>& out_packed_recv_src_info,
                             const std::optional<torch::Tensor>& out_packed_recv_layout_range,
                             const std::optional<torch::Tensor>& out_packed_recv_count) {
  EP_HOST_ASSERT(low_latency_mode);

  EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
  EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
  EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
  EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
  EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
  EP_HOST_ASSERT(num_experts % num_ranks == 0);

  auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
  auto num_scales = hidden / 128, num_topk = static_cast<int>(topk_idx.size(1));
  int num_local_experts = num_experts / num_ranks;

  LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
  EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
  auto buffer = layout.buffers[low_latency_buffer_idx];
  auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

  auto compute_stream = at::cuda::getCurrentCUDAStream();
  auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
  EP_HOST_ASSERT(not(async and return_recv_hook));
  if (not return_recv_hook) stream_wait(launch_stream, compute_stream);

  // Reusable output tensors. The largest (`packed_recv_x` ~58 MB at 7K hidden)
  // is what motivates the reuse path: a fresh torch::empty per call adds
  // measurable host overhead (~10us cumulative for the 4 allocations) which
  // shows up against NCCL-EP's preallocated bench at small payloads.
  const auto recv_x_dtype = use_fp8 ? torch::kFloat8_e4m3fn : torch::kBFloat16;
  torch::Tensor packed_recv_x;
  if (out_packed_recv_x.has_value()) {
    EP_HOST_ASSERT(out_packed_recv_x->dim() == 3 and out_packed_recv_x->is_contiguous());
    EP_HOST_ASSERT(out_packed_recv_x->size(0) == num_local_experts);
    EP_HOST_ASSERT(out_packed_recv_x->size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(out_packed_recv_x->size(2) == hidden);
    EP_HOST_ASSERT(out_packed_recv_x->scalar_type() == recv_x_dtype);
    packed_recv_x = out_packed_recv_x.value();
  } else {
    packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                                 x.options().dtype(recv_x_dtype));
  }
  torch::Tensor packed_recv_src_info;
  if (out_packed_recv_src_info.has_value()) {
    EP_HOST_ASSERT(out_packed_recv_src_info->dim() == 2 and out_packed_recv_src_info->is_contiguous());
    EP_HOST_ASSERT(out_packed_recv_src_info->size(0) == num_local_experts);
    EP_HOST_ASSERT(out_packed_recv_src_info->size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(out_packed_recv_src_info->scalar_type() == torch::kInt32);
    packed_recv_src_info = out_packed_recv_src_info.value();
  } else {
    packed_recv_src_info = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank},
                                        torch::dtype(torch::kInt32).device(torch::kCUDA));
  }
  torch::Tensor packed_recv_layout_range;
  if (out_packed_recv_layout_range.has_value()) {
    EP_HOST_ASSERT(out_packed_recv_layout_range->dim() == 2 and out_packed_recv_layout_range->is_contiguous());
    EP_HOST_ASSERT(out_packed_recv_layout_range->size(0) == num_local_experts);
    EP_HOST_ASSERT(out_packed_recv_layout_range->size(1) == num_ranks);
    EP_HOST_ASSERT(out_packed_recv_layout_range->scalar_type() == torch::kInt64);
    packed_recv_layout_range = out_packed_recv_layout_range.value();
  } else {
    packed_recv_layout_range =
        torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  }
  torch::Tensor packed_recv_count;
  if (out_packed_recv_count.has_value()) {
    EP_HOST_ASSERT(out_packed_recv_count->dim() == 1 and out_packed_recv_count->is_contiguous());
    EP_HOST_ASSERT(out_packed_recv_count->size(0) == num_local_experts);
    EP_HOST_ASSERT(out_packed_recv_count->scalar_type() == torch::kInt32);
    packed_recv_count = out_packed_recv_count.value();
  } else {
    packed_recv_count = torch::empty({num_local_experts}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  }

  auto packed_recv_x_scales = std::optional<torch::Tensor>();
  float* packed_recv_x_scales_ptr = nullptr;
  if (use_fp8) {
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and
                   "TMA requires the number of tokens to be multiple of 4");
    if (out_packed_recv_x_scales.has_value()) {
      // Caller-provided scales tensor must already be in the kernel's
      // expected (transposed) layout: shape [num_local_experts,
      // num_ranks*max_tokens, num_scales], strides such that
      // size(1)=num_ranks*max_tokens with the actual storage
      // [num_local_experts, num_scales, num_ranks*max_tokens] (i.e.
      // produced by `torch.empty(...).transpose(1, 2)`).
      EP_HOST_ASSERT(out_packed_recv_x_scales->dim() == 3);
      EP_HOST_ASSERT(out_packed_recv_x_scales->size(0) == num_local_experts);
      EP_HOST_ASSERT(out_packed_recv_x_scales->size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
      EP_HOST_ASSERT(out_packed_recv_x_scales->size(2) == num_scales);
      EP_HOST_ASSERT(out_packed_recv_x_scales->scalar_type() == torch::kFloat32);
      packed_recv_x_scales = out_packed_recv_x_scales.value();
    } else {
      packed_recv_x_scales = torch::empty({num_local_experts, num_scales, num_ranks * num_max_dispatch_tokens_per_rank},
                                          torch::dtype(torch::kFloat32).device(torch::kCUDA));
      packed_recv_x_scales = torch::transpose(packed_recv_x_scales.value(), 1, 2);
    }
    packed_recv_x_scales_ptr = packed_recv_x_scales->data_ptr<float>();
  }

  auto next_clean_meta = next_buffer.clean_meta();
  auto port_handles = port_channel_handles_device_ptr.get();
  auto mem_handles = ll_memory_channel_handles_device_ptr ? ll_memory_channel_handles_device_ptr.get() : nullptr;
  auto peer_bases = peer_rdma_bases_gpu;
  const bool use_ipc = ll_ipc_ready;
  auto rdma_base = rdma_buffer_ptr;
  auto launcher = [=](int phases) {
    internode_ll::dispatch(packed_recv_x.data_ptr(), packed_recv_x_scales_ptr, packed_recv_src_info.data_ptr<int>(),
                           packed_recv_layout_range.data_ptr<int64_t>(), packed_recv_count.data_ptr<int>(),
                           buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
                           buffer.dispatch_rdma_send_buffer, x.data_ptr(), topk_idx.data_ptr<int64_t>(),
                           next_clean_meta.first, next_clean_meta.second, num_tokens, hidden,
                           num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank, num_ranks, use_fp8, workspace,
                           launch_stream, phases, rdma_base, port_handles, peer_bases, mem_handles, use_ipc);
  };
  launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(launch_stream);
  } else if (not return_recv_hook) {
    stream_wait(compute_stream, launch_stream);
  }

  std::optional<std::function<void()>> recv_hook = std::nullopt;
  if (return_recv_hook) recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

  return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event,
          recv_hook};
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>> Buffer::low_latency_combine(
    const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
    const torch::Tensor& src_info, const torch::Tensor& layout_range, int num_max_dispatch_tokens_per_rank,
    int num_experts, bool zero_copy, bool async, bool return_recv_hook, const std::optional<torch::Tensor>& out) {
  EP_HOST_ASSERT(low_latency_mode);

  EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
  EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
  EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
  EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
  EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
  EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
  EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
  EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
  EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
  EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
  EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
  EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
  EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
  EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
  EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);
  auto hidden = static_cast<int>(x.size(2));
  auto num_topk = static_cast<int>(topk_weights.size(1));
  auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

  LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
  EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
  auto buffer = layout.buffers[low_latency_buffer_idx];
  auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

  auto compute_stream = at::cuda::getCurrentCUDAStream();
  auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
  EP_HOST_ASSERT(not(async and return_recv_hook));
  if (not return_recv_hook) stream_wait(launch_stream, compute_stream);

  torch::Tensor combined_x;
  if (out.has_value()) {
    EP_HOST_ASSERT(out->dim() == 2 and out->is_contiguous());
    EP_HOST_ASSERT(out->size(0) == num_combined_tokens and out->size(1) == hidden);
    EP_HOST_ASSERT(out->scalar_type() == x.scalar_type());
    combined_x = out.value();
  } else {
    combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
  }

  auto next_clean_meta = next_buffer.clean_meta();
  auto port_handles = port_channel_handles_device_ptr.get();
  auto mem_handles = ll_memory_channel_handles_device_ptr ? ll_memory_channel_handles_device_ptr.get() : nullptr;
  auto peer_bases = peer_rdma_bases_gpu;
  const bool use_ipc = ll_ipc_ready;
  auto rdma_base = rdma_buffer_ptr;
  auto launcher = [=](int phases) {
    internode_ll::combine(
        combined_x.data_ptr(), buffer.combine_rdma_recv_data_buffer, buffer.combine_rdma_recv_flag_buffer,
        buffer.combine_rdma_send_buffer, x.data_ptr(), topk_idx.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
        src_info.data_ptr<int>(), layout_range.data_ptr<int64_t>(), next_clean_meta.first, next_clean_meta.second,
        num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank, num_topk, num_experts, rank, num_ranks,
        workspace, launch_stream, phases, zero_copy, rdma_base, port_handles, peer_bases, mem_handles, use_ipc);
  };
  launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

  std::optional<EventHandle> event;
  if (async) {
    event = EventHandle(launch_stream);
  } else if (not return_recv_hook) {
    stream_wait(compute_stream, launch_stream);
  }

  std::optional<std::function<void()>> recv_hook = std::nullopt;
  if (return_recv_hook) recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

  return {combined_x, event, recv_hook};
}

torch::Tensor Buffer::get_next_low_latency_combine_buffer(int num_max_dispatch_tokens_per_rank, int hidden,
                                                          int num_experts) {
  LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
  auto buffer = layout.buffers[low_latency_buffer_idx];
  auto dtype = torch::kBFloat16;
  auto num_msg_elems = static_cast<int>(buffer.num_bytes_per_combine_msg / elementSize(torch::kBFloat16));

  EP_HOST_ASSERT(buffer.num_bytes_per_combine_msg % elementSize(torch::kBFloat16) == 0);
  return torch::from_blob(buffer.combine_rdma_send_buffer_data_start,
                          {num_experts / num_ranks, num_ranks * num_max_dispatch_tokens_per_rank, hidden},
                          {num_ranks * num_max_dispatch_tokens_per_rank * num_msg_elems, num_msg_elems, 1},
                          torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
}

}  // namespace ep
}  // namespace mscclpp
