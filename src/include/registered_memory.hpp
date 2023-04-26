#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include "mscclpp.hpp"
#include "mscclpp.h"
#include "ib.h"
#include <variant>
#include <cuda_runtime.h>

namespace mscclpp {

struct TransportInfo {
  TransportFlags transport;
  std::variant<std::monostate, cudaIpcMemHandle_t, mscclppIbMr*, mscclppIbMrInfo> data;
};

struct RegisteredMemory::Impl {
  void* data;
  size_t size;
  int rank;
  TransportFlags transports;
  std::vector<TransportInfo> transportInfos;

  Impl(void* data, size_t size, int rank, TransportFlags transports);
  Impl(const std::vector<char>& data);

  template<class T> T& getTransportInfo(TransportFlags transport) {
    for (auto& entry : transportInfos) {
      if (entry.transport == transport) {
        return std::get<T>(entry.data);
      }
    }
    throw std::runtime_error("Transport data not found");
  }
};

} // namespace mscclpp

#endif // MSCCLPP_REGISTERED_MEMORY_HPP_
