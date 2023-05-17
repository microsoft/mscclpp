#ifndef MSCCLPP_EPOCH_HPP_
#define MSCCLPP_EPOCH_HPP_

#include <mscclpp/core.hpp>

namespace mscclpp {

struct alignas(16) EpochIds {
  uint64_t outbound;
  uint64_t inboundReplica;
};

class BaseEpoch {
 private:
  std::shared_ptr<Connection> connection_;
  RegisteredMemory localEpochIdsRegMem_;

 protected:
  EpochIds* epochIds_;
  NonblockingFuture<RegisteredMemory> remoteEpochIdsRegMem_;
  uint64_t* expectedInboundEpochId_;

 public:
  BaseEpoch(std::shared_ptr<Connection> connection);
  void setup(Communicator& communicator);
  BaseEpoch(const BaseEpoch&) = delete;
  void signal();
};

class DeviceEpoch : BaseEpoch {
 public:
  DeviceEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);
  DeviceEpoch(const DeviceEpoch&) = delete;
  ~DeviceEpoch();
  void signal();

  struct DeviceHandle {
#ifdef __CUDACC__
    __forceinline__ __device__ void wait() {
      (*expectedInboundEpochId) += 1;
      while (*(volatile uint64_t*)&(epochIds->inboundReplica) < (*expectedInboundEpochId))
        ;
    }

    __forceinline__ __device__ void epochIncrement() { *(volatile uint64_t*)&(epochIds->outbound) += 1; }

    __forceinline__ __device__ void signalDirect() {
      // This fence ensures that the writes from a preceding putDirect() are visible on the peer GPU before the
      // incremented epoch id is visible.
      __threadfence_system();
      epochIncrement();
      *(volatile uint64_t*)&(remoteEpochIds->inboundReplica) = epochIds->outbound;
    }
#endif  // __CUDACC__

    EpochIds* epochIds;
    EpochIds* remoteEpochIds;
    uint64_t* expectedInboundEpochId;
  };

  DeviceHandle deviceHandle();
};

class HostEpoch : BaseEpoch {
 public:
  HostEpoch(Communicator& communicator, std::shared_ptr<Connection> connection);
  HostEpoch(const HostEpoch&) = delete;
  ~HostEpoch();

  void increamentAndSignal();
  void wait();
};

}  // namespace mscclpp

#endif  // MSCCLPP_EPOCH_HPP_
