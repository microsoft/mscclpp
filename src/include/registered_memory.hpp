#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include <cuda_runtime.h>

#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>

#include "communicator.hpp"
#include "ib.hpp"

namespace mscclpp {

struct TransportInfo {
  Transport transport;

  // TODO: rewrite this using std::variant or something
  bool ibLocal;
  union {
    struct {
      cudaIpcMemHandle_t cudaIpcBaseHandle;
      size_t cudaIpcOffsetFromBase;
    };
    struct {
      const IbMr* ibMr;
      IbMrInfo ibMrInfo;
    };
  };
};

struct RegisteredMemory::Impl {
  void* data;
  size_t size;
  int rank;
  uint64_t hostHash;
  TransportFlags transports;
  std::vector<TransportInfo> transportInfos;

  Impl(void* data, size_t size, int rank, TransportFlags transports, Communicator::Impl& commImpl);
  Impl(const std::vector<char>& data);

  TransportInfo& getTransportInfo(Transport transport) {
    for (auto& entry : transportInfos) {
      if (entry.transport == transport) {
        return entry;
      }
    }
    throw Error("Transport data not found", ErrorCode::InternalError);
  }
};

}  // namespace mscclpp

#endif  // MSCCLPP_REGISTERED_MEMORY_HPP_
