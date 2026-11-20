// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rdma_provider.hpp"

#include "ib.hpp"
#include "logger.hpp"

namespace mscclpp {

// Classification includes metadata imports even when local IB support is disabled.
TransportFlags getRdmaTransports() { return AllIBTransports; }

bool areRdmaTransportsCompatible(Transport local, Transport remote) {
  return AllIBTransports.has(local) && AllIBTransports.has(remote);
}

std::shared_ptr<RdmaProvider> createRdmaProvider(Transport transport) {
  if (AllIBTransports.has(transport)) {
#if defined(MSCCLPP_USE_IB)
    return std::make_shared<IbProvider>(transport);
#else
    THROW(NET, Error, ErrorCode::InvalidUsage, "IB support is not enabled");
#endif
  }
  THROW(NET, Error, ErrorCode::InvalidUsage, "Unsupported RDMA transport: ", transport);
}

std::shared_ptr<RdmaEndpoint> deserializeRdmaEndpoint(const EndpointConfig& config,
                                                      std::vector<char>::const_iterator& it) {
  if (AllIBTransports.has(config.transport)) {
    return deserializeIbEndpoint(it);
  }
  THROW(NET, Error, ErrorCode::InvalidUsage, "Unsupported RDMA transport: ", config.transport);
}

std::shared_ptr<RdmaMemory> deserializeRdmaMemory(Transport transport, std::vector<char>::const_iterator& it) {
  if (AllIBTransports.has(transport)) {
    return deserializeIbMemory(it);
  }
  THROW(NET, Error, ErrorCode::InvalidUsage, "Unsupported RDMA transport: ", transport);
}

}  // namespace mscclpp
