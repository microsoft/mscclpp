#include <mscclpp/epoch.hpp>

#include "api.h"
#include "debug.h"

namespace mscclpp {

static NonblockingFuture<RegisteredMemory> setupInboundEpochId(Communicator& communicator, Connection* connection,
                                                               void* localInboundEpochId) {
  auto localInboundEpochIdsRegMem =
      communicator.registerMemory(localInboundEpochId, sizeof(uint64_t), connection->transport());
  communicator.sendMemoryOnSetup(localInboundEpochIdsRegMem, connection->remoteRank(), connection->tag());
  return communicator.recvMemoryOnSetup(connection->remoteRank(), connection->tag());
}

MSCCLPP_API_CPP DeviceEpoch::DeviceEpoch(Communicator& communicator, std::shared_ptr<Connection> connection)
    : BaseEpoch(allocUniqueCuda<uint64_t>(), allocUniqueCuda<uint64_t>(), std::make_unique<uint64_t>()),
      connection_(connection) {
  remoteInboundEpochIdsRegMem_ = setupInboundEpochId(communicator, connection.get(), localInboundEpochId_.get());
}

MSCCLPP_API_CPP void DeviceEpoch::signal() {
  connection_->updateAndSync(remoteInboundEpochIdsRegMem_.get(), 0, outboundEpochId_.get(), *outboundEpochId_ + 1);
}

MSCCLPP_API_CPP DeviceEpoch::DeviceHandle DeviceEpoch::deviceHandle() {
  DeviceEpoch::DeviceHandle device;
  device.inboundEpochId = localInboundEpochId_.get();
  device.expectedInboundEpochId = expectedInboundEpochId_.get();
  return device;
}

MSCCLPP_API_CPP HostEpoch::HostEpoch(Communicator& communicator, std::shared_ptr<Connection> connection)
    : BaseEpoch(std::make_unique<uint64_t>(), std::make_unique<uint64_t>(), std::make_unique<uint64_t>()),
      connection_(connection) {
  if (connection->transport() == Transport::CudaIpc) {
    throw Error("HostEpoch cannot be used with CudaIpc transport", ErrorCode::InvalidUsage);
  }
  remoteInboundEpochIdsRegMem_ = setupInboundEpochId(communicator, connection.get(), localInboundEpochId_.get());
}

MSCCLPP_API_CPP void HostEpoch::signal() {
  connection_->updateAndSync(remoteInboundEpochIdsRegMem_.get(), 0, outboundEpochId_.get(), *outboundEpochId_ + 1);
}

MSCCLPP_API_CPP void HostEpoch::wait() {
  (*expectedInboundEpochId_) += 1;
  while (*(volatile uint64_t*)localInboundEpochId_.get() < (*expectedInboundEpochId_)) {
  }
}

MSCCLPP_API_CPP SmEpoch::SmEpoch(Communicator& communicator, std::shared_ptr<Connection> connection)
    : BaseEpoch(allocUniqueCuda<uint64_t>(), allocUniqueCuda<uint64_t>(), allocUniqueCuda<uint64_t>()) {
  if (connection->transport() == Transport::CudaIpc) {
    remoteInboundEpochIdsRegMem_ = setupInboundEpochId(communicator, connection.get(), localInboundEpochId_.get());
    INFO(MSCCLPP_INIT, "Creating a direct epoch for CudaIPC transport from %d to %d",
         communicator.bootstrapper()->getRank(), connection->remoteRank());
    isRemoteInboundEpochIdSet_ = true;
  } else if (AllIBTransports.has(connection->transport())) {
    // We don't need to really with any of the IB transports, since the values will be local
    INFO(MSCCLPP_INIT, "Creating a direct epoch for IB transport from %d to %d", communicator.bootstrapper()->getRank(),
         connection->remoteRank());
    isRemoteInboundEpochIdSet_ = false;
  }
}

MSCCLPP_API_CPP SmEpoch::DeviceHandle SmEpoch::deviceHandle() {
  SmEpoch::DeviceHandle device;
  device.remoteInboundEpochId =
      isRemoteInboundEpochIdSet_ ? reinterpret_cast<uint64_t*>(remoteInboundEpochIdsRegMem_.get().data()) : nullptr;
  device.inboundEpochId = localInboundEpochId_.get();
  device.expectedInboundEpochId = expectedInboundEpochId_.get();
  device.outboundEpochId = outboundEpochId_.get();
  return device;
};

}  // namespace mscclpp
