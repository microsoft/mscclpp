#include "epoch.hpp"
#include "checks.hpp"
#include "alloc.h"
#include "api.h"

namespace mscclpp {

BaseEpoch::BaseEpoch(std::shared_ptr<Connection> connection) : connection_(connection){}

void BaseEpoch::setup(Communicator& communicator) {
  localEpochIdsRegMem_ = communicator.registerMemory(epochIds_, sizeof(epochIds_), connection_->transport());
  communicator.sendMemoryOnSetup(localEpochIdsRegMem_, connection_->remoteRank(), connection_->tag());
  remoteEpochIdsRegMem_ = communicator.recvMemoryOnSetup(connection_->remoteRank(), connection_->tag());
}

void BaseEpoch::signal() {
  connection_->write(remoteEpochIdsRegMem_.get(), offsetof(EpochIds, inboundReplica), localEpochIdsRegMem_, offsetof(EpochIds, outbound), sizeof(epochIds_));
}

MSCCLPP_API_CPP DeviceEpoch::DeviceEpoch(Communicator& communicator, std::shared_ptr<Connection> connection) : BaseEpoch(connection) {
  MSCCLPPTHROW(mscclppCudaCalloc(&epochIds_, 1));
  MSCCLPPTHROW(mscclppCudaCalloc(&expectedInboundEpochId_, 1));
  setup(communicator);
}

MSCCLPP_API_CPP DeviceEpoch::~DeviceEpoch() {
  mscclppCudaFree(epochIds_);
  mscclppCudaFree(expectedInboundEpochId_);
}

MSCCLPP_API_CPP void DeviceEpoch::signal() {
  BaseEpoch::signal();
}

MSCCLPP_API_CPP DeviceEpoch::DeviceHandle DeviceEpoch::deviceHandle() {
  DeviceEpoch::DeviceHandle device;
  device.epochIds = epochIds_;
  device.expectedInboundEpochId = expectedInboundEpochId_;
  return device;
}

MSCCLPP_API_CPP HostEpoch::HostEpoch(Communicator& communicator, std::shared_ptr<Connection> connection) : BaseEpoch(connection) {
  if (connection->transport() == Transport::CudaIpc){
    throw std::runtime_error("HostEpoch cannot be used with CudaIpc transport");
  }
  epochIds_ = new EpochIds();
  expectedInboundEpochId_ = new uint64_t();
  setup(communicator);
}

MSCCLPP_API_CPP HostEpoch::~HostEpoch() {
  delete epochIds_;
  delete expectedInboundEpochId_;
}

MSCCLPP_API_CPP void HostEpoch::increamentAndSignal() {
  *(volatile uint64_t*)&(epochIds_->outbound) += 1;
  signal();
}

MSCCLPP_API_CPP void HostEpoch::wait(){
  (*expectedInboundEpochId_) += 1;
  while (*(volatile uint64_t*)&(epochIds_->inboundReplica) < (*expectedInboundEpochId_));
}

} // namespace mscclpp
