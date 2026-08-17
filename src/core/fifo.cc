// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/env.hpp>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>

#include "api.h"
#include "atomic.hpp"
#include "logger.hpp"

namespace mscclpp {

namespace {
// size is validated to be a positive power of two, so this is exact.
uint64_t shiftOf(int size) {
  uint64_t shift = 0;
  while ((1 << shift) < size) shift++;
  return shift;
}
}  // namespace

struct Fifo::Impl {
  detail::UniqueGpuHostPtr<ProxyTrigger> triggers;
  detail::UniqueGpuPtr<uint64_t> head;
  detail::UniqueGpuHostPtr<uint64_t> tail;
  detail::UniqueGpuPtr<uint64_t> tailCache;
  const int size;
  const uint64_t sizeMask;
  const uint64_t sizeShift;

  Impl(int size)
      : triggers(detail::gpuCallocHostUnique<ProxyTrigger>(size)),
        head(detail::gpuCallocUnique<uint64_t>()),
        tail(detail::gpuCallocHostUnique<uint64_t>()),
        tailCache(detail::gpuCallocUnique<uint64_t>()),
        size(size),
        sizeMask(uint64_t(size) - 1),
        sizeShift(shiftOf(size)) {}
};

MSCCLPP_API_CPP Fifo::Fifo(int size) {
  if (size <= 0 || (size & (size - 1)) != 0) {
    THROW(GPU, Error, ErrorCode::InvalidUsage, "FIFO size must be a positive power of two, got ", size);
  }
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  int numaNode = getDeviceNumaNode(device);
  if (numaNode >= 0) {
    numaBind(numaNode);
  }
  pimpl_ = std::make_unique<Impl>(size);
}

MSCCLPP_API_CPP Fifo::~Fifo() = default;

MSCCLPP_API_CPP bool Fifo::poll(ProxyTrigger& trigger) {
  const uint64_t curTail = *(pimpl_->tail);
  ProxyTrigger* ptr = &pimpl_->triggers.get()[curTail & pimpl_->sizeMask];

  // snd is the commit word: the producer writes it last, with release, carrying the parity of the
  // lap this slot is on. A match means the payload written before it is visible too.
  const uint64_t expectedParity = ((curTail >> pimpl_->sizeShift) & 1ULL) ^ 1ULL;
  ProxyTrigger candidate;
  candidate.snd = atomicLoad(&(ptr->snd), memoryOrderAcquire);
  if (candidate.fields.reserved != expectedParity) return false;

  candidate.fields.reserved = 0;
  candidate.fst = atomicLoad(&(ptr->fst), memoryOrderRelaxed);
  trigger = candidate;
  return true;
}

MSCCLPP_API_CPP void Fifo::pop() {
  // The slot is not cleared: the next lap's parity makes what it holds stale.
  atomicStore(pimpl_->tail.get(), *(pimpl_->tail) + 1, memoryOrderRelease);
}

MSCCLPP_API_CPP uint64_t Fifo::tail() const { return *(pimpl_->tail); }

MSCCLPP_API_CPP int Fifo::size() const { return pimpl_->size; }

MSCCLPP_API_CPP FifoDeviceHandle Fifo::deviceHandle() const {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl_->triggers.get();
  deviceHandle.head = pimpl_->head.get();
  deviceHandle.tail = pimpl_->tail.get();
  deviceHandle.tailCache = pimpl_->tailCache.get();
  deviceHandle.size = pimpl_->size;
  deviceHandle.sizeMask = pimpl_->sizeMask;
  deviceHandle.sizeShift = pimpl_->sizeShift;
  return deviceHandle;
}

}  // namespace mscclpp
