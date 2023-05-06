#include "epoch.hpp"
#include "alloc.h"
#include "api.h"
#include "checks.hpp"

namespace mscclpp {

MSCCLPP_API_CPP Epoch::Epoch(Communicator& communicator, std::shared_ptr<Connection> connection)
  : connection_(connection)
{
  MSCCLPPTHROW(mscclppCudaCalloc(&device_.epochIds_, 1));
  MSCCLPPTHROW(mscclppCudaCalloc(&device_.expectedInboundEpochId_, 1));

  localEpochIdsRegMem_ =
    communicator.registerMemory(device_.epochIds_, sizeof(device_.epochIds_), connection->transport());
  communicator.sendMemoryOnSetup(localEpochIdsRegMem_, connection->remoteRank(), connection->tag());
  remoteEpochIdsRegMem_ = communicator.recvMemoryOnSetup(connection->remoteRank(), connection->tag());
}

MSCCLPP_API_CPP Epoch::~Epoch()
{
  mscclppCudaFree(device_.epochIds_);
  mscclppCudaFree(device_.expectedInboundEpochId_);
}

MSCCLPP_API_CPP void Epoch::signal()
{
  connection_->write(remoteEpochIdsRegMem_.get(), offsetof(EpochIds, inboundReplica_), localEpochIdsRegMem_,
                     offsetof(EpochIds, outbound_), sizeof(device_.epochIds_));
}

} // namespace mscclpp
