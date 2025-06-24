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
  std::shared_ptr<uint64_t> tailHost;
  std::shared_ptr<uint64_t> tailReplica;
  const int size;

  // for transferring fifo tail
  CudaStreamWithFlags stream;

  Impl(int size)
      : triggers(detail::gpuCallocHostUnique<ProxyTrigger>(size)),
        head(detail::gpuCallocUnique<uint64_t>()),
        tailHost(env()->fifoUseTailReplica ? std::make_shared<uint64_t>(0) : detail::gpuCallocHostShared<uint64_t>()),
        tailReplica(env()->fifoUseTailReplica ? detail::gpuCallocUnique<uint64_t>() : nullptr),
        size(size) {
    if (env()->fifoUseTailReplica) {
      stream.set(cudaStreamNonBlocking);
    }
  }
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
  ProxyTrigger* ptr = &pimpl_->triggers.get()[*(pimpl_->tailHost) % pimpl_->size];
  // we are loading fst first. if fst is non-zero then snd is also valid
  trigger.fst = atomicLoad(&(ptr->fst), memoryOrderAcquire);
  trigger.snd = ptr->snd;
  return trigger;
}

MSCCLPP_API_CPP void Fifo::pop() {
  uint64_t curTail = *(pimpl_->tailHost);
  pimpl_->triggers.get()[curTail % pimpl_->size].fst = 0;
  *(pimpl_->tailHost) = curTail + 1;
}

MSCCLPP_API_CPP void Fifo::flushTail([[maybe_unused]] bool sync) {
  if (!env()->fifoUseTailReplica) {
    // Nothing to flush if the tail is not replicated.
    return;
  }
  // Flush the tail to device memory. This is either triggered every ProxyFlushPeriod to make sure that the fifo can
  // make progress even if there is no request mscclppSync. However, mscclppSync type is for flush request.
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(pimpl_->tailReplica.get(), pimpl_->tailHost.get(), sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, pimpl_->stream));
  if (sync) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(pimpl_->stream));
  }
}

MSCCLPP_API_CPP int Fifo::size() const { return pimpl_->size; }

MSCCLPP_API_CPP FifoDeviceHandle Fifo::deviceHandle() const {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl_->triggers.get();
  deviceHandle.head = pimpl_->head.get();
  // tailReplica refers to the original tail if `fifoUseTailReplica == false`.
  deviceHandle.tailReplica = env()->fifoUseTailReplica ? pimpl_->tailReplica.get() : pimpl_->tailHost.get();
  deviceHandle.size = pimpl_->size;
  return deviceHandle;
}

}  // namespace mscclpp
