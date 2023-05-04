#ifndef MSCCLPP_EPOCH_HPP_
#define MSCCLPP_EPOCH_HPP_

#include "mscclpp.hpp"

namespace mscclpp {

struct alignas(16) EpochIds
{
  uint64_t outbound_;
  uint64_t inboundReplica_;
};

struct DeviceEpoch
{
#ifdef __CUDACC__
  __forceinline__ __device__ void wait()
  {
    (*expectedInboundEpochId_) += 1;
    while (*(volatile uint64_t*)&(epochIds_->inboundReplica_) < (*expectedInboundEpochId_));
  }

  __forceinline__ __device__ void epochIncrement()
  {
    *(volatile uint64_t*)&(epochIds_->outbound_) += 1;
  }
#endif // __CUDACC__

  EpochIds* epochIds_;
  uint64_t* expectedInboundEpochId_;
};

class Epoch
{
  std::shared_ptr<Connection> connection_;
  DeviceEpoch device_;
  RegisteredMemory localEpochIdsRegMem_;
  NonblockingFuture<RegisteredMemory> remoteEpochIdsRegMem_;

public:
  Epoch(Communicator& communicator, std::shared_ptr<Connection> connection);
  Epoch(const Epoch&) = delete;
  ~Epoch();

  void signal();

  DeviceEpoch deviceEpoch() { return device_; }
};

} // namespace mscclpp

#endif // MSCCLPP_EPOCH_HPP_