// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_REGISTERED_MEMORY_HPP_
#define MSCCLPP_REGISTERED_MEMORY_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu.hpp>

#include "communicator.hpp"
#include "gpu_ipc_mem.hpp"
#include "ib.hpp"

namespace mscclpp {

struct TransportInfo {
  Transport transport;

  // TODO: rewrite this using std::variant or something
  bool ibLocal;
  union {
    GpuIpcMemHandle gpuIpcMemHandle;
    struct {
      const IbMr* ibMr;
      IbMrInfo ibMrInfo;
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
  uint64_t hostHash;
  uint64_t pidHash;
  TransportFlags transports;
  std::vector<TransportInfo> transportInfos;

  UniqueGpuIpcMemHandle localGpuIpcHandle;
  std::unique_ptr<GpuIpcMem> remoteGpuIpcMem;

  // Only used for IB transport
  std::unique_ptr<const IbMr> ibMr;

  Impl(void* data, size_t size, TransportFlags transports, Context::Impl& contextImpl);
  Impl(const std::vector<char>::const_iterator& begin, const std::vector<char>::const_iterator& end);
  /// Constructs a RegisteredMemory::Impl from a vector of data. The constructor should only be used for the remote
  /// memory.
  Impl(const std::vector<char>& data);

  const TransportInfo& getTransportInfo(Transport transport) const;
};

}  // namespace mscclpp

#endif  // MSCCLPP_REGISTERED_MEMORY_HPP_
