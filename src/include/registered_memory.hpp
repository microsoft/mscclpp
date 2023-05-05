#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include "communicator.hpp"
#include "ib.hpp"
#include "mscclpp.h"
#include "mscclpp.hpp"
#include <cuda_runtime.h>

namespace mscclpp {

struct TransportInfo
{
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

struct RegisteredMemory::Impl
{
  void* data;
  size_t size;
  int rank;
  uint64_t hostHash;
  TransportFlags transports;
  std::vector<TransportInfo> transportInfos;

  Impl(void* data, size_t size, int rank, TransportFlags transports, Communicator::Impl& commImpl);
  Impl(const std::vector<char>& data);

  TransportInfo& getTransportInfo(Transport transport)
  {
    for (auto& entry : transportInfos) {
      if (entry.transport == transport) {
        return entry;
      }
    }
    throw MscclppError("Transport data not found", mscclppInternalError);
  }
};

} // namespace mscclpp

#endif // MSCCLPP_REGISTERED_MEMORY_HPP_
