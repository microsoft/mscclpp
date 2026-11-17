// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rdma_provider.hpp"

#include "ib.hpp"
#include "logger.hpp"
#include "mrc.hpp"

namespace mscclpp {

// Classification includes metadata imports even when local IB support is disabled.
TransportFlags getRdmaTransports() { return AllIBTransports | AllMrcTransports; }

bool areRdmaTransportsCompatible(Transport local, Transport remote) {
  return (AllIBTransports.has(local) && AllIBTransports.has(remote)) ||
         (AllMrcTransports.has(local) && AllMrcTransports.has(remote));
}

void checkMrcSupport([[maybe_unused]] TransportFlags transports) {
#if !defined(MSCCLPP_USE_MRC)
  if ((transports & AllMrcTransports).any()) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "Native MRC support is disabled");
  }
#endif
}

std::shared_ptr<RdmaProvider> createRdmaProvider(Transport transport) {
  if (AllIBTransports.has(transport)) {
#if defined(MSCCLPP_USE_IB)
    return std::make_shared<IbProvider>(transport);
#else
    THROW(NET, Error, ErrorCode::InvalidUsage, "IB support is not enabled");
#endif
  }
  if (AllMrcTransports.has(transport)) {
#if defined(MSCCLPP_USE_MRC)
    return std::make_shared<MrcProvider>(transport);
#else
    THROW(NET, Error, ErrorCode::InvalidUsage, "Native MRC support is disabled");
#endif
  }
  THROW(NET, Error, ErrorCode::InvalidUsage, "Unsupported RDMA transport: ", transport);
}

std::shared_ptr<RdmaEndpoint> deserializeRdmaEndpoint(const EndpointConfig& config,
                                                      std::vector<char>::const_iterator& it,
                                                      [[maybe_unused]] std::vector<char>::const_iterator end) {
  if (AllIBTransports.has(config.transport)) {
    if (end - it < static_cast<std::ptrdiff_t>(sizeof(IbQpInfo) + sizeof(bool))) {
      THROW(NET, Error, ErrorCode::InvalidUsage, "Truncated IB endpoint metadata");
    }
    return deserializeIbEndpoint(it);
  }
  if (AllMrcTransports.has(config.transport)) {
#if defined(MSCCLPP_USE_MRC)
    return deserializeMrcEndpoint(config, it, end);
#else
    THROW(NET, Error, ErrorCode::InvalidUsage, "Native MRC support is disabled");
#endif
  }
  THROW(NET, Error, ErrorCode::InvalidUsage, "Unsupported RDMA transport: ", config.transport);
}

std::shared_ptr<RdmaMemory> deserializeRdmaMemory(Transport transport, std::vector<char>::const_iterator& it,
                                                  std::vector<char>::const_iterator end,
                                                  [[maybe_unused]] void* originalData, [[maybe_unused]] size_t size,
                                                  [[maybe_unused]] TransportFlags advertised,
                                                  [[maybe_unused]] TransportFlags& seen) {
  if (AllIBTransports.has(transport)) {
    if (end - it < static_cast<std::ptrdiff_t>(sizeof(IbMrInfo))) {
      THROW(NET, Error, ErrorCode::InvalidUsage, "Truncated IB memory metadata");
    }
    return deserializeIbMemory(it);
  }
  if (AllMrcTransports.has(transport)) {
#if defined(MSCCLPP_USE_MRC)
    return deserializeMrcMemory(transport, it, end, originalData, size, advertised, seen);
#else
    THROW(NET, Error, ErrorCode::InvalidUsage, "Native MRC support is disabled");
#endif
  }
  THROW(NET, Error, ErrorCode::InvalidUsage, "Unsupported RDMA transport: ", transport);
}

void validateRdmaMemoryRecords(TransportFlags advertised, TransportFlags seen) {
  if (seen != (advertised & AllMrcTransports)) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "Missing native MRC memory records");
  }
}

}  // namespace mscclpp
