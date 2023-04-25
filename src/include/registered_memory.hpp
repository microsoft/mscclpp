#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include "mscclpp.hpp"
#include "ib.h"
#include <variant>
#include <cuda_runtime.h>

namespace mscclpp {

struct IBTransportData {
  mscclppIbMr localIbMr;
  mscclppIbMrInfo remoteIbMrInfo;
};

struct TransportData {
  TransportFlags transport;
  union {
    void* cudaIpcPtr;
    IBTransportData ibData;
  }
};

struct RegisteredMemory::Impl {
  void* data;
  size_t size;
  TransportFlags transports;
  std::vector<TransportData> transportData;

  Impl(void* data, size_t size, TransportFlags transports);

  ~Impl();

  template<typename T> T& getTransportData(TransportFlags transport) {
    for (auto& data : transportData) {
      if (data.transport == transport) {
        return data;
      }
    }
    throw std::runtime_error("Transport data not found");
  }
};

} // namespace mscclpp

#endif // MSCCLPP_REGISTERED_MEMORY_HPP_
