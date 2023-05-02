#include "epoch.hpp"
#include "checks.hpp"
#include "alloc.h"

namespace mscclpp {

Epoch::Epoch(Communicator& communicator, std::shared_ptr<Connection> connection) : connection_(connection) {
  MSCCLPPTHROW(mscclppCudaCalloc(&device_.epochIds_, 1));
  MSCCLPPTHROW(mscclppCudaCalloc(&device_.expectedInboundEpochId_, 1));

  localEpochIdsRegMem_ = communicator.registerMemory(device_.epochIds_, sizeof(device_.epochIds_), connection->transport());
  communicator.bootstrapper()->send(localEpochIdsRegMem_.serialize(), connection->remoteRank(), connection->tag());
  std::vector<char> serializedRemoteEpochIds;
  communicator.bootstrapper()->recv(serializedRemoteEpochIds, connection->remoteRank(), connection->tag());
  remoteEpochIdsRegMem_ = RegisteredMemory::deserialize(serializedRemoteEpochIds);
}

Epoch::~Epoch() {
  MSCCLPPTHROW(mscclppCudaFree(&device_.epochIds_));
  MSCCLPPTHROW(mscclppCudaFree(&device_.expectedInboundEpochId_));
}

void Epoch::signal() {
  connection_->write(remoteEpochIdsRegMem_, offsetof(EpochIds, inboundReplica_), localEpochIdsRegMem_, offsetof(EpochIds, outbound_), sizeof(device_.epochIds_));
}

} // namespace mscclpp