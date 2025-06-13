// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/env.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "api.h"
#include "atomic.hpp"

namespace mscclpp {

struct Fifo::Impl {
  detail::UniqueGpuHostPtr<ProxyTrigger> triggers;
  detail::UniqueGpuPtr<uint64_t> head;
  detail::UniqueGpuPtr<uint64_t> tailReplica;
  detail::UniqueGpuPtr<uint64_t> tailCache;
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
        tailCache(detail::gpuCallocUnique<uint64_t>()),
        size(size),
        hostTail(env()->fifoUseTailReplica ? std::make_shared<uint64_t>(0) : detail::gpuCallocHostShared<uint64_t>()),
        stream(cudaStreamNonBlocking) {}
};

MSCCLPP_API_CPP Fifo::Fifo(int size) : pimpl(std::make_unique<Impl>(size)) {}
MSCCLPP_API_CPP Fifo::~Fifo() = default;

MSCCLPP_API_CPP ProxyTrigger Fifo::poll() {
  ProxyTrigger trigger;
  ProxyTrigger* ptr = &pimpl->triggers.get()[*(pimpl->hostTail) % pimpl->size];
  // we are loading fst first. if fst is non-zero then snd is also valid
  trigger.fst = atomicLoad(&(ptr->fst), memoryOrderAcquire);
  trigger.snd = ptr->snd;
  return trigger;
}

MSCCLPP_API_CPP void Fifo::pop() {
  uint64_t curTail = *(pimpl->hostTail);
  atomicStore(&(pimpl->triggers.get()[curTail % pimpl->size].fst), uint64_t{0}, memoryOrderRelease);
  *(pimpl->hostTail) = curTail + 1;
}

MSCCLPP_API_CPP void Fifo::flushTail(bool sync) {
  if (!env()->fifoUseTailReplica) {
    // Nothing to flush if the tail is not replicated.
    return;
  }
#if defined(MSCCLPP_DEVICE_HIP)
  *(pimpl->tailReplica.get()) = *(pimpl->hostTail.get());
#else   // !defined(MSCCLPP_DEVICE_HIP)
  // Flush the tail to device memory. This is either triggered every ProxyFlushPeriod to make sure that the fifo can
  // make progress even if there is no request mscclppSync. However, mscclppSync type is for flush request.
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(pimpl->tailReplica.get(), pimpl->hostTail.get(), sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, pimpl->stream));
  if (sync) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(pimpl->stream));
  }
#endif  // !defined(MSCCLPP_DEVICE_HIP)
}

MSCCLPP_API_CPP int Fifo::size() const { return pimpl->size; }

MSCCLPP_API_CPP FifoDeviceHandle Fifo::deviceHandle() const {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl->triggers.get();
  deviceHandle.head = pimpl->head.get();
  deviceHandle.tail = env()->fifoUseTailReplica ? pimpl->tailReplica.get() : pimpl->hostTail.get();
  deviceHandle.tailCache = pimpl->tailCache.get();
  deviceHandle.size = pimpl->size;
  return deviceHandle;
}

}  // namespace mscclpp
