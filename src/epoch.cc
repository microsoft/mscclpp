#include <mscclpp/epoch.hpp>

#include "api.h"

namespace mscclpp {

MSCCLPP_API_CPP DeviceEpoch::DeviceEpoch(Communicator& communicator, std::shared_ptr<Connection> connection)
    : BaseEpoch(connection, allocUniqueCuda<uint64_t>(), allocUniqueCuda<uint64_t>()) {
  setup(communicator);
}

MSCCLPP_API_CPP DeviceEpoch::DeviceHandle DeviceEpoch::deviceHandle() {
  DeviceEpoch::DeviceHandle device;
  device.inboundEpochId = inboundEpochId_.get();
  device.expectedInboundEpochId = expectedInboundEpochId_.get();
  return device;
}

MSCCLPP_API_CPP HostEpoch::HostEpoch(Communicator& communicator, std::shared_ptr<Connection> connection)
    : BaseEpoch(connection, std::make_unique<uint64_t>(), std::make_unique<uint64_t>()) {
  if (connection->transport() == Transport::CudaIpc) {
    throw Error("HostEpoch cannot be used with CudaIpc transport", ErrorCode::InvalidUsage);
  }
  setup(communicator);
}

MSCCLPP_API_CPP void HostEpoch::wait() {
  (*expectedInboundEpochId_) += 1;
  while (*(volatile uint64_t*)inboundEpochId_.get() < (*expectedInboundEpochId_)) {
  }
}

MSCCLPP_API_CPP DirectEpoch::DirectEpoch(Communicator& communicator, std::shared_ptr<Connection> connection)
    : localInboundEpochId_(allocUniqueCuda<uint64_t>()),
      expectedInboundEpochId_(allocUniqueCuda<uint64_t>()),
      outboundEpochId_(allocUniqueCuda<uint64_t>()) {
  if (connection->transport() != Transport::CudaIpc) {
    throw Error("DirectEpoch can only be used with CudaIpc transport", ErrorCode::InvalidUsage);
  }
  auto localInboundEpochIdsRegMem =
      communicator.registerMemory(localInboundEpochId_.get(), sizeof(uint64_t), connection->transport());

  communicator.sendMemoryOnSetup(localInboundEpochIdsRegMem, connection->remoteRank(), connection->tag());
  remoteInboundEpochIdsRegMem_ = communicator.recvMemoryOnSetup(connection->remoteRank(), connection->tag());
}

MSCCLPP_API_CPP DirectEpoch::DeviceHandle DirectEpoch::deviceHandle() {
  DirectEpoch::DeviceHandle device;
  device.remoteInboundEpochId = reinterpret_cast<uint64_t*>(remoteInboundEpochIdsRegMem_.get().data());
  device.inboundEpochId = localInboundEpochId_.get();
  device.expectedInboundEpochId = expectedInboundEpochId_.get();
  device.outboundEpochId = outboundEpochId_.get();
  return device;
};

}  // namespace mscclpp
