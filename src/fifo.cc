// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <emmintrin.h>

#include <mscclpp/cuda_utils.hpp>
#include <mscclpp/fifo.hpp>
#include <stdexcept>

#include "api.h"

namespace mscclpp {

struct HostProxyFifo::Impl {
  UniqueCudaHostPtr<ProxyTrigger[]> triggers;
  UniqueCudaPtr<uint64_t> head;
  UniqueCudaPtr<uint64_t> tailReplica;

  // allocated on the host. Only accessed by the host. This is a copy of the
  // value pointed to by fifoTailDev and the invariant is that
  // *fifoTailDev <= hostTail. Meaning that host's copy of tail is
  // always ahead of the device's copy and host updates the device's copy
  // only when it is needed. Therefore, hostTail is the "true" tail
  // and fifoTailDev is a "stale" tail. See proxy.cc to undertand how
  // these updates are pushed to the device.
  uint64_t hostTail;

  // for transferring fifo tail
  CudaStreamWithFlags stream;

  Impl()
      : triggers(makeUniqueCudaHost<ProxyTrigger[]>(MSCCLPP_PROXY_FIFO_SIZE)),
        head(allocUniqueCuda<uint64_t>()),
        tailReplica(allocUniqueCuda<uint64_t>()),
        hostTail(0),
        stream(cudaStreamNonBlocking) {}
};

MSCCLPP_API_CPP HostProxyFifo::HostProxyFifo() : pimpl(std::make_unique<Impl>()) {}
MSCCLPP_API_CPP HostProxyFifo::~HostProxyFifo() = default;

MSCCLPP_API_CPP void HostProxyFifo::poll(ProxyTrigger* trigger) {
  __m128i xmm0 = _mm_load_si128((__m128i*)&pimpl->triggers.get()[pimpl->hostTail % MSCCLPP_PROXY_FIFO_SIZE]);
  _mm_store_si128((__m128i*)trigger, xmm0);
}

MSCCLPP_API_CPP void HostProxyFifo::pop() {
  *(volatile uint64_t*)(&pimpl->triggers.get()[pimpl->hostTail % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
  (pimpl->hostTail)++;
}

MSCCLPP_API_CPP void HostProxyFifo::flushTail(bool sync) {
  // Flush the tail to device memory. This is either triggered every ProxyFlushPeriod to make sure that the fifo can
  // make progress even if there is no request mscclppSync. However, mscclppSync type is for flush request.
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(pimpl->tailReplica.get(), &pimpl->hostTail, sizeof(uint64_t),
                                    cudaMemcpyHostToDevice, pimpl->stream));
  if (sync) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(pimpl->stream));
  }
}

MSCCLPP_API_CPP DeviceProxyFifo HostProxyFifo::deviceFifo() {
  DeviceProxyFifo deviceFifo;
  deviceFifo.triggers = pimpl->triggers.get();
  deviceFifo.head = pimpl->head.get();
  deviceFifo.tailReplica = pimpl->tailReplica.get();
  return deviceFifo;
}

}  // namespace mscclpp
