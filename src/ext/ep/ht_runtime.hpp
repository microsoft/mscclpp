// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// High-throughput (HT) MoE dispatch/combine runtime. This is a torch-free,
// raw-pointer port of the DeepEP-style `Buffer` (formerly `src/ext/ep/ht/
// buffer.{cc,hpp}`), following the same boundary convention as the
// low-latency `MoERuntime` (`src/ext/ep/moe_runtime.{cc,hpp}`): tensor params
// become raw device pointers + explicit size ints, output tensors become
// caller-provided pointers, and torch CUDA streams become `cudaStream_t`. The
// CUDA kernels it drives (`intranode::` / `internode::`) are unchanged.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <optional>
#include <string>
#include <vector>

#include "ht/config.hpp"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

namespace mscclpp {
namespace ep {

// High-throughput expert-parallel runtime. Sibling to `MoERuntime` (low
// latency); both expose a raw-pointer, torch-free boundary so the extension
// module never links libtorch.
//
// Dynamic recv sizing is handled with an explicit two-phase API: the caller
// first runs `intranodeNotifyDispatch` (returns `numRecvTokens`), allocates the
// recv output buffers, then runs `intranodeDispatch`. The cached fast path
// skips the notify phase (`cachedMode == true`).
struct MoEHighThroughputRuntime {
  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8 || NUM_MAX_NVL_PEERS == 4,
                   "The number of maximum NVLink peers must be 4 or 8");

 private:
  // Low-latency mode buffer
  int low_latency_buffer_idx = 0;
  // Retained as a compile-time-false member so the original construction /
  // sync / dispatch bodies (which branch on `low_latency_mode`) port verbatim.
  // The HT runtime is always high-throughput; the low-latency path is served
  // by `MoERuntime`.
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
  // the runtime's lifetime (recv_pool_ptrs_[*] alias their .data()).
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

  // Stream for communication. Replaces the torch `at::cuda::CUDAStream`
  // member; created in the ctor (`cudaStreamCreateWithFlags`) and destroyed in
  // the dtor. The kernels still launch on this stream exactly as before.
  cudaStream_t comm_stream = nullptr;
  // Reusable event for cross-stream ordering between the caller-provided stream
  // and `comm_stream` (replaces the torch EventHandle / stream_wait dance).
  cudaEvent_t comm_event = nullptr;

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

  // Cross-stream ordering helper: makes `dst` wait for all work currently
  // enqueued on `src`, using the reusable `comm_event`. Replaces the torch
  // `stream_wait` / EventHandle machinery.
  void stream_wait(cudaStream_t dst, cudaStream_t src);

  // Resolve the intranode dispatch channel layout from the config + env knobs
  // (MSCCLPP_EP_DISPATCH_NSM / MSCCLPP_EP_INTRA_DIRECT / MSCCLPP_EP_COMBINE_TMA /
  // MSCCLPP_EP_INTRA_ALLSENDER) and the input dtype. Shared by
  // intranodeNotifyDispatch, intranodeDispatch and getIntranodeDispatchNumChannels
  // so all three resolve identical channel counts. `xElementSize == 2` (BF16)
  // is required for the all-sender path.
  void computeIntranodeChannels(int xElementSize, const Config& config, int& dispatchNumSms, bool& allSender,
                                int& numChannels) const;

 public:
  MoEHighThroughputRuntime(int rank, int numRanks, int64_t numNvlBytes, int64_t numRdmaBytes);

  ~MoEHighThroughputRuntime() noexcept(false);

  bool isAvailable() const;

  bool isInternodeAvailable() const;

  int getNumRdmaRanks() const;

  int getRdmaRank() const;

  int getRootRdmaRank(bool global) const;

  int getLocalDeviceId() const;

  std::string getLocalIpcHandle() const;

  std::string getLocalNvshmemUniqueId() const;

  mscclpp::UniqueId createUniqueId() const;

  void connect(mscclpp::UniqueId rootId);

  void sync(const std::vector<int>& deviceIds, const std::vector<std::optional<std::string>>& allGatheredHandles,
            const std::optional<std::string>& rootUniqueIdOpt);

  // Compute layout metadata for `topkIdx`. Outputs (caller pointers):
  //   numTokensPerRank      [numRanks]                (int)
  //   numTokensPerRdmaRank  [numRdmaRanks]            (int, may be nullptr when
  //                                                    !isInternodeAvailable())
  //   numTokensPerExpert    [numExperts]              (int)
  //   isTokenInRank         [numTokens * numRanks]    (bool)
  void getDispatchLayout(int* numTokensPerRank, int* numTokensPerRdmaRank, int* numTokensPerExpert, bool* isTokenInRank,
                         const int64_t* topkIdx, int numTokens, int numTopk, int numExperts, cudaStream_t stream);

  // Number of channels the intranode dispatch resolves for this config + dtype.
  // The caller uses it to size `channelPrefixMatrix` / `recvChannelPrefixMatrix`
  // ([numRanks * numChannels]) and to pass `ringNumChannels` to `intranodeCombine`.
  // Mirrors the env-driven channel-count logic in `intranodeDispatch` exactly.
  int getIntranodeDispatchNumChannels(int xElementSize, const Config& config) const;

  // Resolve the recv-x output pointer for `intranodeDispatch`. When the
  // zero-copy direct / all-sender path is active for this shape, returns the
  // internal recv-pool view (recv_pool_local_ptr_ + header); the caller must
  // pass this same pointer back as `recvX` (so expert outputs land in the pool
  // for the TMA combine to gather). Returns nullptr when the caller should
  // allocate `recvX` itself. Mirrors the `ep_intra_direct` gate in
  // `intranodeDispatch` exactly so both agree.
  void* resolveIntranodeRecvXBuffer(int numRecvTokens, int hidden, int xElementSize, const Config& config) const;

  // Resolve the recv-x output pointer for `internodeDispatch`. When the
  // non-cached internode direct path is active for this shape, returns the
  // internal recv-pool view (recv_pool_local_ptr_ + header); the caller must
  // pass this same pointer back as `recvX` (so the cross-GPU forwarder writes
  // hidden straight into the pool and the direct-gather combine reads it back).
  // Returns nullptr when the caller should allocate `recvX` itself. Mirrors the
  // `ep_use_direct` gate in `internodeDispatch` exactly (minus the cached-mode
  // term, since the helper is only consulted on the non-cached forward path).
  void* resolveInternodeRecvXBuffer(int numRecvTokens, int hidden, int xElementSize, const Config& config) const;

  // Number of channels the non-cached internode dispatch resolves for this
  // config (mirrors `internodeDispatch`'s `ep_flat_dispatch_channels(num_sms)`
  // under EP_DISPATCH_NCCLEP, else `num_sms/2`). The caller uses it to size the
  // rdma/gbl channel-prefix matrices ([numRdmaRanks|numRanks * numChannels]).
  int getInternodeDispatchNumChannels(const Config& config) const;

  // Per-token source-meta row width in bytes (recvSrcMeta is
  // [numRecvTokens * getSourceMetaBytes()]). Exposes
  // `internode::get_source_meta_bytes()` so the caller can size recvSrcMeta.
  int getSourceMetaBytes() const;

  // Compile-time NUM_MAX_NVL_PEERS (4 on GB200, 8 on HGX). The caller uses it to
  // size sendNvlHead ([numRdmaRecvTokens * NUM_MAX_NVL_PEERS]) and to validate
  // combinedNvlHead's second dimension.
  int getNumMaxNvlPeers() const;

  // Phase A (non-cached): send sizes / notify. Writes rankPrefixMatrix
  // [numRanks*numRanks], channelPrefixMatrix [numRanks*numChannels] and the host
  // array numRecvTokensPerExpert [numLocalExperts]; returns numRecvTokens.
  // `xElementSize` (2 == BF16) is required to resolve the all-sender channel
  // count identically to `intranodeDispatch`.
  int intranodeNotifyDispatch(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert,
                              const int* numTokensPerRank, const int* numTokensPerExpert, const bool* isTokenInRank,
                              int numTokens, int numExperts, int xElementSize, int expertAlignment,
                              const Config& config, cudaStream_t stream);

  // Phase B (shared/cached tail): run the dispatch kernel. When `cachedMode` is
  // true the caller supplies rankPrefixMatrix / channelPrefixMatrix / numRecvTokens
  // from a prior dispatch and this method runs the cached barrier first; when
  // false it assumes `intranodeNotifyDispatch` already ran. Writes the recv
  // output pointers (recvXScales / recvTopkIdx / recvTopkWeights may be nullptr).
  void intranodeDispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights, int* recvSrcIdx,
                         int* sendHead, int* recvChannelPrefixMatrix, const void* x, const float* xScales,
                         const int64_t* topkIdx, const float* topkWeights, const bool* isTokenInRank,
                         const int* rankPrefixMatrix, const int* channelPrefixMatrix, int numTokens, int hidden,
                         int numTopk, int numScales, int numExperts, int xElementSize, int numRecvTokens,
                         bool cachedMode, const Config& config, cudaStream_t stream);

  // Combine (scatter-reduce). Output combinedX [numRecvTokens * hidden] where
  // numRecvTokens == number of ORIGINAL tokens (== sendHead row count, known to
  // the caller); combinedTopkWeights [numRecvTokens * numTopk] may be nullptr.
  // `numTokens` is the dispatched token count (x row count); `ringNumChannels`
  // is channelPrefixMatrix's column count (== getIntranodeDispatchNumChannels()).
  void intranodeCombine(void* combinedX, float* combinedTopkWeights, const void* x, const float* topkWeights,
                        const int* srcIdx, const int* rankPrefixMatrix, const int* channelPrefixMatrix,
                        const int* sendHead, int numTokens, int numRecvTokens, int hidden, int numTopk,
                        int xElementSize, int ringNumChannels, const Config& config, cudaStream_t stream);

  // ---------------------------------------------------------------------------
  // Internode (NVLink + RDMA) path. Declared torch-free; the bodies are
  // throwing stubs in this drop and will be ported next (the intranode path is
  // validated single-node first). See SIGNATURES.md for the notify->dispatch
  // boundary contract.
  // ---------------------------------------------------------------------------

  // Phase A (non-cached): writes rdmaChannelPrefixMatrix [numRdmaRanks*numChannels],
  // recvRdmaRankPrefixSum [numRdmaRanks], gblChannelPrefixMatrix [numRanks*numChannels],
  // recvGblRankPrefixSum [numRanks], host numRecvTokensPerExpert [numLocalExperts];
  // returns the primary count numRecvTokens and the secondary count via
  // *numRdmaRecvTokens.
  int internodeNotifyDispatch(int* rdmaChannelPrefixMatrix, int* recvRdmaRankPrefixSum, int* gblChannelPrefixMatrix,
                              int* recvGblRankPrefixSum, int* numRecvTokensPerExpert, int* numRdmaRecvTokens,
                              const int* numTokensPerRank, const int* numTokensPerRdmaRank,
                              const int* numTokensPerExpert, const bool* isTokenInRank, int numTokens, int numExperts,
                              int hidden, int numScales, int numTopk, int xElementSize, int expertAlignment,
                              const Config& config, cudaStream_t stream);

  // Phase B (shared/cached tail): runs the internode dispatch kernel. Writes the
  // recv outputs (recvXScales / recvTopkIdx / recvTopkWeights / recvSrcMeta /
  // recvRdmaChannelPrefixMatrix / recvGblChannelPrefixMatrix / sendRdmaHead /
  // sendNvlHead may be nullptr in cached mode).
  void internodeDispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights,
                         void* recvSrcMeta, int* recvRdmaChannelPrefixMatrix, int* recvGblChannelPrefixMatrix,
                         int* sendRdmaHead, int* sendNvlHead, const void* x, const float* xScales,
                         const int64_t* topkIdx, const float* topkWeights, const bool* isTokenInRank,
                         const int* rdmaChannelPrefixMatrix, const int* recvRdmaRankPrefixSum,
                         const int* gblChannelPrefixMatrix, const int* recvGblRankPrefixSum, int numTokens, int hidden,
                         int numTopk, int numScales, int numExperts, int xElementSize, int numRecvTokens,
                         int numRdmaRecvTokens, bool cachedMode, const Config& config, cudaStream_t stream);

  // Combine. Output combinedX [numCombinedTokens * hidden] (numCombinedTokens ==
  // is_combined_token_in_rank row count, known to the caller); combinedTopkWeights
  // [numCombinedTokens * numTopk] may be nullptr.
  void internodeCombine(void* combinedX, float* combinedTopkWeights, const void* x, const float* topkWeights,
                        const void* srcMeta, const bool* isCombinedTokenInRank, const int* rdmaChannelPrefixMatrix,
                        const int* rdmaRankPrefixSum, const int* gblChannelPrefixMatrix, const int* combinedRdmaHead,
                        const int* combinedNvlHead, int numTokens, int numCombinedTokens, int hidden, int numTopk,
                        int xElementSize, const Config& config, cudaStream_t stream);
};

}  // namespace ep
}  // namespace mscclpp
