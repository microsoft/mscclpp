#ifndef MSCCLPP_EPOCH_HPP_
#define MSCCLPP_EPOCH_HPP_

#include "mscclpp.hpp"

namespace mscclpp {

struct alignas(16) SignalEpochId {
  // every signal(), increaments this and either:
  // 1) proxy thread pushes it to the remote peer's localSignalEpochId->proxy
  // 2) gpu thread directly writes it to remoteSignalEpochId->device
  uint64_t device;
  // signal() function triggers the cpu proxy thread to write to it
  uint64_t proxy;
};

struct DeviceEpoch {
#ifdef __CUDACC__
  __forceinline__ __device__ void wait()
  {
    (*waitEpochId) += 1;
    while (*(volatile uint64_t*)&(localSignalEpochId->proxy) < (*waitEpochId))
      ;
  }

  __forceinline__ __device__ void epochIncrement()
  {
    *(volatile uint64_t*)&(localSignalEpochId->device) += 1;
  }
#endif // __CUDACC__

  SignalEpochId* localSignalEpochId;
  SignalEpochId* remoteSignalEpochId;
  uint64_t* waitEpochId;
};


class Epoch {
  struct Impl;
  std::unique_ptr<Impl> pimpl;
public:
  Epoch();
  ~Epoch();

  void signal();

  DeviceEpoch& getDeviceEpoch();
};

} // namespace mscclpp

#endif // MSCCLPP_EPOCH_HPP_