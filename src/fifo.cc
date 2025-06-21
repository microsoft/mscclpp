// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/env.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>

#include "api.h"
#include "atomic.hpp"

namespace mscclpp {

struct Fifo::Impl {
  detail::UniqueGpuHostPtr<ProxyTrigger> triggers;
  detail::UniqueGpuPtr<uint64_t> head;
  detail::UniqueGpuPtr<uint64_t> tailReplica;
  const int size;

  // The original tail of this fifo allocated on the host. If a tail replica is used
  // (when `env()->fifoUseTailReplica == true`), it always holds that *tailReplica <= *hostTail.
  std::shared_ptr<uint64_t> hostTail;

  // for transferring fifo tail
  CudaStreamWithFlags stream;

  Impl(int size)
      : triggers(detail::gpuCallocHostUnique<ProxyTrigger>(size)),
        head(detail::gpuCallocUnique<uint64_t>()),
        tailReplica(env()->fifoUseTailReplica ? detail::gpuCallocUnique<uint64_t>() : nullptr),
        size(size),
        hostTail(env()->fifoUseTailReplica ? std::make_shared<uint64_t>(0) : detail::gpuCallocHostShared<uint64_t>()),
        stream(cudaStreamNonBlocking) {}
};

MSCCLPP_API_CPP Fifo::Fifo(int size) {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  int numaNode = getDeviceNumaNode(device);
  if (numaNode >= 0) {
    numaBind(numaNode);
  }
  pimpl_ = std::make_unique<Impl>(size);
}

MSCCLPP_API_CPP Fifo::~Fifo() = default;

MSCCLPP_API_CPP ProxyTrigger Fifo::poll() {
  ProxyTrigger trigger;
  ProxyTrigger* ptr = &pimpl_->triggers.get()[*(pimpl_->hostTail) % pimpl_->size];
  // we are loading fst first. if fst is non-zero then snd is also valid
  trigger.fst = atomicLoad(&(ptr->fst), memoryOrderAcquire);
  trigger.snd = ptr->snd;
  return trigger;
}

MSCCLPP_API_CPP void Fifo::pop() {
  uint64_t curTail = *(pimpl_->hostTail);
  atomicStore(&(pimpl_->triggers.get()[curTail % pimpl_->size].fst), uint64_t{0}, memoryOrderRelease);
  *(pimpl_->hostTail) = curTail + 1;
}

MSCCLPP_API_CPP void Fifo::flushTail([[maybe_unused]] bool sync) {
  if (!env()->fifoUseTailReplica) {
    // Nothing to flush if the tail is not replicated.
    return;
  }
#if defined(MSCCLPP_DEVICE_HIP)
  *(pimpl_->tailReplica.get()) = *(pimpl_->hostTail.get());
#else   // !defined(MSCCLPP_DEVICE_HIP)
  // Flush the tail to device memory. This is either triggered every ProxyFlushPeriod to make sure that the fifo can
  // make progress even if there is no request mscclppSync. However, mscclppSync type is for flush request.
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(pimpl_->tailReplica.get(), pimpl_->hostTail.get(), sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, pimpl_->stream));
  if (sync) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(pimpl_->stream));
  }
#endif  // !defined(MSCCLPP_DEVICE_HIP)
}

MSCCLPP_API_CPP int Fifo::size() const { return pimpl_->size; }

MSCCLPP_API_CPP FifoDeviceHandle Fifo::deviceHandle() const {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl_->triggers.get();
  deviceHandle.head = pimpl_->head.get();
  // tailReplica refers to the original tail if `fifoUseTailReplica == false`.
  deviceHandle.tailReplica = env()->fifoUseTailReplica ? pimpl_->tailReplica.get() : pimpl_->hostTail.get();
  deviceHandle.size = pimpl_->size;
  return deviceHandle;
}

}  // namespace mscclpp
