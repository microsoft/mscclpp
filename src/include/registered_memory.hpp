#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include "mscclpp.hpp"
#include "mscclpp.h"
#include "ib.h"
#include "communicator.hpp"
#include <cuda_runtime.h>

namespace mscclpp {

struct TransportInfo {
  TransportFlags transport;

  // TODO: rewrite this using std::variant or something
  bool ibLocal;
  union {
    cudaIpcMemHandle_t cudaIpcHandle;
    mscclppIbMr* ibMr;
    mscclppIbMrInfo ibMrInfo;
  };
};

struct RegisteredMemory::Impl {
  void* data;
  size_t size;
  int rank;
  TransportFlags transports;
  std::vector<TransportInfo> transportInfos;

  Impl(void* data, size_t size, int rank, TransportFlags transports, Communicator::Impl& commImpl);
  Impl(const std::vector<char>& data);

  TransportInfo& getTransportInfo(TransportFlags transport) {
    for (auto& entry : transportInfos) {
      if (entry.transport == transport) {
        return entry;
      }
    }
    throw std::runtime_error("Transport data not found");
  }
};

} // namespace mscclpp

#endif // MSCCLPP_REGISTERED_MEMORY_HPP_
