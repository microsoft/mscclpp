#include <cuda_runtime.h>
#include <emmintrin.h>

#include <mscclpp/fifo.hpp>
#include <stdexcept>

#include "alloc.h"
#include "api.h"
#include "checks.hpp"

namespace mscclpp {

struct HostProxyFifo::Impl {
  DeviceProxyFifo deviceFifo;

  // allocated on the host. Only accessed by the host. This is a copy of the
  // value pointed to by fifoTailDev and the invariant is that
  // *fifoTailDev <= hostTail. Meaning that host's copy of tail is
  // always ahead of the device's copy and host updates the device's copy
  // only when it is needed. Therefore, hostTail is the "true" tail
  // and fifoTailDev is a "stale" tail. See proxy.cc to undertand how
  // these updates are pushed to the device.
  uint64_t hostTail;

  // for transferring fifo tail
  cudaStream_t stream;
};

MSCCLPP_API_CPP HostProxyFifo::HostProxyFifo() {
  pimpl = std::make_unique<Impl>();
  MSCCLPPTHROW(mscclppCudaCalloc(&pimpl->deviceFifo.head, 1));
  MSCCLPPTHROW(mscclppCudaHostCalloc(&pimpl->deviceFifo.triggers, MSCCLPP_PROXY_FIFO_SIZE));
  MSCCLPPTHROW(mscclppCudaCalloc(&pimpl->deviceFifo.tailReplica, 1));
  CUDATHROW(cudaStreamCreateWithFlags(&pimpl->stream, cudaStreamNonBlocking));
  pimpl->hostTail = 0;
}

MSCCLPP_API_CPP HostProxyFifo::~HostProxyFifo() {
  mscclppCudaFree(pimpl->deviceFifo.head);
  mscclppCudaHostFree(pimpl->deviceFifo.triggers);
  mscclppCudaFree(pimpl->deviceFifo.tailReplica);
  cudaStreamDestroy(pimpl->stream);
}

MSCCLPP_API_CPP void HostProxyFifo::poll(ProxyTrigger* trigger) {
  __m128i xmm0 = _mm_load_si128((__m128i*)&pimpl->deviceFifo.triggers[pimpl->hostTail % MSCCLPP_PROXY_FIFO_SIZE]);
  _mm_store_si128((__m128i*)trigger, xmm0);
}

MSCCLPP_API_CPP void HostProxyFifo::pop() {
  *(volatile uint64_t*)(&pimpl->deviceFifo.triggers[pimpl->hostTail % MSCCLPP_PROXY_FIFO_SIZE]) = 0;
  (pimpl->hostTail)++;
}

MSCCLPP_API_CPP void HostProxyFifo::flushTail(bool sync) {
  // Flush the tail to device memory. This is either triggered every MSCCLPP_PROXY_FIFO_FLUSH_COUNTER to make sure
  // that the fifo can make progress even if there is no request mscclppSync. However, mscclppSync type is for flush
  // request.
  CUDATHROW(cudaMemcpyAsync(pimpl->deviceFifo.tailReplica, &pimpl->hostTail, sizeof(uint64_t), cudaMemcpyHostToDevice,
                            pimpl->stream));
  if (sync) {
    CUDATHROW(cudaStreamSynchronize(pimpl->stream));
  }
}

MSCCLPP_API_CPP DeviceProxyFifo HostProxyFifo::deviceFifo() { return pimpl->deviceFifo; }

}  // namespace mscclpp
