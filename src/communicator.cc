#include "mscclpp.hpp"
#include "mscclpp.h"

namespace mscclpp {

struct Communicator::impl {
    mscclppComm_t comm;
};

void Communicator::initRank(int nranks, const char* ipPortPair, int rank) {
  
}

void Communicator::initRankFromId(int nranks, UniqueId id, int rank) {
  
}

void Communicator::bootstrapAllGather(void* data, int size) {
  
}

void Communicator::bootstrapBarrier() {
  
}

std::shared_ptr<HostConnection> Communicator::connect(int remoteRank, int tag, void* localBuff, uint64_t buffSize,
                                        TransportType transportType, const char* ibDev = 0) {
  
}

void Communicator::connectionSetup() {
  
}

void Communicator::destroy() {
  
}

int Communicator::rank() {
  
}

int Communicator::size() {
  
}

void Communicator::setBootstrapConnTimeout(unsigned timeout) {
  
}

} // namespace mscclpp