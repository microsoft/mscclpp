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
  detail::UniqueGpuHostPtr<uint64_t> tail;
  detail::UniqueGpuPtr<uint64_t> tailCache;
  const int size;

  Impl(int size)
      : triggers(detail::gpuCallocHostUnique<ProxyTrigger>(size)),
        head(detail::gpuCallocUnique<uint64_t>()),
        tail(detail::gpuCallocHostUnique<uint64_t>()),
        tailCache(detail::gpuCallocUnique<uint64_t>()),
        size(size) {}
};

MSCCLPP_API_CPP Fifo::Fifo(int size) : pimpl(std::make_unique<Impl>(size)) {}

MSCCLPP_API_CPP Fifo::~Fifo() = default;

MSCCLPP_API_CPP ProxyTrigger Fifo::poll() {
  ProxyTrigger trigger;
  ProxyTrigger* ptr = &pimpl->triggers.get()[*(pimpl->tail) % pimpl->size];
  // we are loading fst first. if fst is non-zero then snd is also valid
  trigger.fst = atomicLoad(&(ptr->fst), memoryOrderAcquire);
  trigger.snd = ptr->snd;
  return trigger;
}

MSCCLPP_API_CPP void Fifo::pop() {
  uint64_t curTail = *(pimpl->tail);
  atomicStore(&(pimpl->triggers.get()[curTail % pimpl->size].fst), uint64_t{0}, memoryOrderRelease);
  *(pimpl->tail) = curTail + 1;
}

MSCCLPP_API_CPP int Fifo::size() const { return pimpl->size; }

MSCCLPP_API_CPP FifoDeviceHandle Fifo::deviceHandle() const {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl->triggers.get();
  deviceHandle.head = pimpl->head.get();
  deviceHandle.tail = pimpl->tail.get();
  deviceHandle.tailCache = pimpl->tailCache.get();
  deviceHandle.size = pimpl->size;
  return deviceHandle;
}

}  // namespace mscclpp
