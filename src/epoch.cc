#include "epoch.hpp"
#include "checks.hpp"

namespace mscclpp {

struct Epoch::Impl {
  DeviceEpoch deviceEpoch;

  Impl() {
    MSCCLPPTHROW(mscclppCudaCalloc(&deviceEpoch.localSignalEpochId, 1));
    MSCCLPPTHROW(mscclppCudaCalloc(&deviceEpoch.waitEpochId, 1));
  }

  ~Impl() {
    MSCCLPPTHROW(mscclppCudaFree(deviceEpoch.localSignalEpochId));
    MSCCLPPTHROW(mscclppCudaFree(deviceEpoch.waitEpochId));
  }
};

Epoch::Epoch() : pimpl(std::make_unique<Impl>()) {}

} // namespace mscclpp