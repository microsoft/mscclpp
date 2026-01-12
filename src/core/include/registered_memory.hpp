// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu.hpp>

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
    struct {
      union {
        char shareableHandle[64];
        struct {
          // These are only defined for multicast (NVLS) capability
          int rootFd;
          int rootPid;
        };
      };
      size_t offsetFromBase;
    };
  };
};

struct RegisteredMemory::Impl {
  // This is the data pointer returned by RegisteredMemory::data(), which may be different from the original data
  // pointer for deserialized remote memory.
  void* data;
  // This is the original data pointer the RegisteredMemory was created with.
  void* originalDataPtr;
  size_t size;
  // This is the size returned by cuMemGetAddressRange of data
  size_t baseDataSize;
  uint64_t hostHash;
  uint64_t pidHash;
  bool isCuMemMapAlloc;
  TransportFlags transports;
  std::vector<TransportInfo> transportInfos;
  std::shared_ptr<void> peerMemHandle;

  // Only used for IB transport
  std::unordered_map<Transport, std::unique_ptr<const IbMr>> ibMrMap;

  // For sharing memory handle via file descriptor
  int fileDesc = -1;

  Impl(void* data, size_t size, TransportFlags transports, Context::Impl& contextImpl);
  Impl(const std::vector<char>::const_iterator& begin, const std::vector<char>::const_iterator& end);
  /// Constructs a RegisteredMemory::Impl from a vector of data. The constructor should only be used for the remote
  /// memory.
  Impl(const std::vector<char>& data);
  ~Impl();

  const TransportInfo& getTransportInfo(Transport transport) const;
};

}  // namespace mscclpp

#endif  // MSCCLPP_REGISTERED_MEMORY_HPP_
