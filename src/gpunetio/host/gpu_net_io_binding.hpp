#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_BINDING_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_BINDING_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>

namespace mscclpp::detail {

struct GpuNetIoMemoryExchange {
  uint64_t base;
  uint64_t bytes;
  uint32_t rkey;
  uint32_t kind;
  int32_t rank;
  int32_t peer;
  int32_t qpIndex;
  uint32_t version;
};

inline GpuNetIoMemoryExchange exchangeGpuNetIoMemory(Bootstrap& bootstrap, GpuNetIoMemoryExchange outgoing, int tag) {
  GpuNetIoMemoryExchange incoming{};
  if (outgoing.rank < outgoing.peer) {
    bootstrap.send(&outgoing, sizeof(outgoing), outgoing.peer, tag);
    bootstrap.recv(&incoming, sizeof(incoming), outgoing.peer, tag);
  } else {
    bootstrap.recv(&incoming, sizeof(incoming), outgoing.peer, tag);
    bootstrap.send(&outgoing, sizeof(outgoing), outgoing.peer, tag);
  }
  if (incoming.version != 1 || incoming.kind != outgoing.kind || incoming.rank != outgoing.peer ||
      incoming.peer != outgoing.rank || incoming.qpIndex != outgoing.qpIndex || incoming.base == 0 ||
      incoming.bytes == 0 || incoming.bytes > UINTPTR_MAX - incoming.base) {
    throw Error("Mismatched GPUNetIO connection or memory exchange", ErrorCode::InvalidUsage);
  }
  return incoming;
}

}  // namespace mscclpp::detail

#endif