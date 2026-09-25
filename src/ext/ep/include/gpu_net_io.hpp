// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_GPU_NET_IO_HPP_
#define MSCCLPP_EP_GPU_NET_IO_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <string>

namespace mscclpp::ep {

struct EpGpuNetIoDeviceContext {
  PortChannelDeviceHandle* channels;
  int numPeers;
  int numQpsPerPeer;
  int numHcas;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle channel(int peer, int queue) const {
    BasePortChannelDeviceHandle::requireGpuNetIo(peer >= 0 && peer < numPeers && queue >= 0 && queue < numQpsPerPeer &&
                                                 channels != nullptr);
    auto handle = channels[static_cast<size_t>(peer) * numQpsPerPeer + queue];
    handle.validateGpuNetIo();
    return handle;
  }

  MSCCLPP_DEVICE_INLINE void put(int peer, uint64_t destination, uint64_t source, uint64_t bytes, int queue = 0) {
    channel(peer, queue).put(destination, source, bytes);
  }

  MSCCLPP_DEVICE_INLINE void atomicAdd(int peer, uint64_t destination, int64_t value, int queue = 0) {
    channel(peer, queue).atomicAdd(destination, value);
  }

  MSCCLPP_DEVICE_INLINE void putWithSignal(int peer, uint64_t destination, uint64_t source, uint64_t bytes,
                                           uint64_t signalOffset, uint64_t signalValue, int queue = 0) {
    auto handle = channel(peer, queue);
    const auto remote = handle.gpuNetIoMemory(handle.dst_, peer, destination, bytes);
    const auto local = handle.gpuNetIoMemory(handle.src_, handle.gpuNetIoLocalRank_, source, bytes);
    auto signal = handle.gpuNetIoMemory(handle.dst_, peer, signalOffset, sizeof(uint64_t));
    signal.base += signalOffset;
    signal.bytes = sizeof(uint64_t);
    BasePortChannelDeviceHandle::requireGpuNetIo(signal.base % sizeof(uint64_t) == 0 && signalValue == 1);
    handle.gpuNetIo_->putRegisteredWithSignal(peer, handle.gpuNetIoQpIndex_, remote, destination, local, source, bytes,
                                              signal);
  }

  MSCCLPP_DEVICE_INLINE void flush(int peer, int queue = 0) { channel(peer, queue).flush(-1); }

  MSCCLPP_DEVICE_INLINE int tryFlush(int peer, uint64_t spins, int queue = 0) {
    const auto handle = channel(peer, queue);
    return handle.gpuNetIo_->tryFlush(peer, spins, handle.gpuNetIoQpIndex_);
  }

  MSCCLPP_DEVICE_INLINE void putBatched3(int peer, int queue, uint64_t dst0, uint64_t src0, uint64_t size0,
                                         uint64_t dst1, uint64_t src1, uint64_t size1, uint64_t dst2, uint64_t src2,
                                         uint64_t size2) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    const uint64_t destinations[3] = {dst0, dst1, dst2};
    const uint64_t sources[3] = {src0, src1, src2};
    const uint64_t sizes[3] = {size0, size1, size2};
    GpuNetIoMemoryDeviceHandle remote[3], local[3];
    for (int index = 0; index < 3; ++index) {
      BasePortChannelDeviceHandle::requireGpuNetIo(sizes[index] <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
      remote[index] = handle.gpuNetIoMemory(handle.dst_, peer, destinations[index], sizes[index]);
      local[index] = handle.gpuNetIoMemory(handle.src_, handle.gpuNetIoLocalRank_, sources[index], sizes[index]);
    }
    auto* qp = detail::gpuNetIoQp(*handle.gpuNetIo_, peer, handle.gpuNetIoQpIndex_);
    const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        qp, 3, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#pragma unroll
    for (int index = 0; index < 3; ++index) {
      auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, base + index);
      doca_gpu_dev_verbs_wqe_prepare_write(
          qp, wqe, base + index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
          remote[index].base + destinations[index], detail::gpuNetIoHtobe32(remote[index].key),
          local[index].base + sources[index], detail::gpuNetIoHtobe32(local[index].key), sizes[index]);
    }
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, base, base + 2);
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(qp, base + 3,
                                                                         DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#else
    BasePortChannelDeviceHandle::requireGpuNetIo(false);
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