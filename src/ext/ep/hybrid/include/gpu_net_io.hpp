// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_HYBRID_GPU_NET_IO_HPP_
#define MSCCLPP_EP_HYBRID_GPU_NET_IO_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <string>

namespace mscclpp::ep {

/// Device view of the EP GPUNetIO channels: one PortChannel per (peer, queue), each bound to its own queue pair over
/// an EndpointConfig::Ib::Mode::GpuNetIo connection. Every channel's dst_/src_ are the peer's / local symmetric buffer,
/// so offsets below are symmetric-buffer offsets. Unlike PortChannelDeviceHandle::put(), these helpers do not fence;
/// callers order their source writes (see the EP row-fence modes).
struct EpGpuNetIoDeviceContext {
  PortChannelDeviceHandle* channels;
  int numPeers;
  int numQpsPerPeer;
  int numHcas;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE static void require(bool valid) { GpuNetIoDeviceHandle::require(valid); }

  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle channel(int peer, int queue) const {
    require(channels != nullptr && peer >= 0 && peer < numPeers && queue >= 0 && queue < numQpsPerPeer);
    const auto handle = channels[static_cast<size_t>(peer) * numQpsPerPeer + queue];
    require(handle.backend_ == PortChannelBackend::GpuNetIo && handle.gpuNetIo_.qp != nullptr);
    return handle;
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_qp* qp(const PortChannelDeviceHandle& handle) {
    return static_cast<doca_gpu_dev_verbs_qp*>(handle.gpuNetIo_.qp);
  }

  /// Remote (peer) address of `bytes` at `offset`, bounds-checked against the channel's destination memory.
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_addr remoteAddr(const PortChannelDeviceHandle& handle,
                                                                  uint64_t offset, uint64_t bytes) {
    const auto memory = handle.gpuNetIo_.memory(handle.dst_, false, offset, bytes);
    return {memory.addr + offset, memory.key};
  }

  /// Local address of `bytes` at `offset`, bounds-checked against the channel's source memory.
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_addr localAddr(const PortChannelDeviceHandle& handle, uint64_t offset,
                                                                 uint64_t bytes) {
    const auto memory = handle.gpuNetIo_.memory(handle.src_, true, offset, bytes);
    return {memory.addr + offset, memory.key};
  }

  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_addr atomicScratch(const PortChannelDeviceHandle& handle) {
    return {handle.gpuNetIo_.atomicScratchAddr, handle.gpuNetIo_.atomicScratchKey};
  }
#endif

  MSCCLPP_DEVICE_INLINE void put(int peer, uint64_t destination, uint64_t source, uint64_t bytes, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        qp(handle), remoteAddr(handle, destination, bytes), localAddr(handle, source, bytes), bytes, &ticket);
#else
    require(false);
#endif
  }

  MSCCLPP_DEVICE_INLINE void atomicAdd(int peer, uint64_t destination, int64_t value, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    const auto remote = remoteAddr(handle, destination, sizeof(uint64_t));
    require(remote.addr % sizeof(uint64_t) == 0);
    const auto scratch = atomicScratch(handle);
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        qp(handle), remote, scratch, 0, remote, scratch, static_cast<uint64_t>(value), &ticket);
#else
    require(false);
#endif
  }

  /// RDMA write followed by an ordered remote atomic add of `signalValue` at symmetric offset `signalOffset`.
  MSCCLPP_DEVICE_INLINE void putWithSignal(int peer, uint64_t destination, uint64_t source, uint64_t bytes,
                                           uint64_t signalOffset, uint64_t signalValue, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    const auto signal = remoteAddr(handle, signalOffset, sizeof(uint64_t));
    require(signal.addr % sizeof(uint64_t) == 0);
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        qp(handle), remoteAddr(handle, destination, bytes), localAddr(handle, source, bytes), bytes, signal,
        atomicScratch(handle), signalValue, &ticket);
#else
    require(false);
#endif
  }

  MSCCLPP_DEVICE_INLINE void flush(int peer, int queue = 0) { channel(peer, queue).flush(-1); }

  /// Poll for completion of every WQE posted so far on (peer, queue) for at most `spins` iterations.
  /// @return 0 when drained, EBUSY if still pending, or the completion error status.
  MSCCLPP_DEVICE_INLINE int tryFlush(int peer, uint64_t spins, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto* q = qp(channel(peer, queue));
    const uint64_t ticket =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->sq_rsvd_index);
    if (ticket == 0) return 0;
    for (uint64_t spin = 0; spin < spins; ++spin) {
      const int status =
          doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, ticket - 1);
      if (status != EBUSY) return status;
    }
    return EBUSY;
#else
    require(false);
    return 0;
#endif
  }

  /// Send-queue reservation index of (peer, queue): every WQE posted so far lies below it.
  MSCCLPP_DEVICE_INLINE uint64_t reservedTicket(int peer, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    return doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        &qp(channel(peer, queue))->sq_rsvd_index);
#else
    require(false);
    return 0;
#endif
  }

  /// Wait until every WQE below `ticket` (from reservedTicket) on (peer, queue) has completed.
  MSCCLPP_DEVICE_INLINE void waitTicket(int peer, uint64_t ticket, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (ticket == 0) return;
    auto* q = qp(channel(peer, queue));
    int status;
    do {
      status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, ticket - 1);
    } while (status == EBUSY);
    require(status == 0);
#else
    require(false);
#endif
  }

  MSCCLPP_DEVICE_INLINE void putBatched3(int peer, int queue, uint64_t dst0, uint64_t src0, uint64_t size0,
                                         uint64_t dst1, uint64_t src1, uint64_t size1, uint64_t dst2, uint64_t src2,
                                         uint64_t size2) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    const uint64_t destinations[3] = {dst0, dst1, dst2};
    const uint64_t sources[3] = {src0, src1, src2};
    const uint64_t sizes[3] = {size0, size1, size2};
    doca_gpu_dev_verbs_addr remote[3], local[3];
    for (int index = 0; index < 3; ++index) {
      require(sizes[index] <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
      remote[index] = remoteAddr(handle, destinations[index], sizes[index]);
      local[index] = localAddr(handle, sources[index], sizes[index]);
    }
    auto* q = qp(handle);
    const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        q, 3, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#pragma unroll
    for (int index = 0; index < 3; ++index) {
      auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(q, base + index);
      doca_gpu_dev_verbs_wqe_prepare_write(q, wqe, base + index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, remote[index].addr,
                                           remote[index].key, local[index].addr, local[index].key, sizes[index]);
    }
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, base, base + 2);
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(q, base + 3,
                                                                         DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#else
    require(false);
#endif
  }
#endif
};

class EpGpuNetIoService {
 public:
  EpGpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& devices, int cudaDevice);
  ~EpGpuNetIoService();
  EpGpuNetIoService(const EpGpuNetIoService&) = delete;
  EpGpuNetIoService& operator=(const EpGpuNetIoService&) = delete;
  void setup(void* buffer, size_t bytes);
  EpGpuNetIoDeviceContext* deviceContext() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mscclpp::ep

#endif