#include <mscclpp/channel.hpp>

#include "api.h"
#include "checks_internal.hpp"
#include "debug.h"
#include "numa.hpp"

namespace mscclpp {
namespace channel {

MSCCLPP_API_CPP DeviceChannelService::DeviceChannelService(Communicator& communicator)
    : communicator_(communicator),
      proxy_([&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); }) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  deviceNumaNode = getDeviceNumaNode(cudaDevice);
}

MSCCLPP_API_CPP void DeviceChannelService::bindThread() {
  if (deviceNumaNode >= 0) {
    numaBind(deviceNumaNode);
    INFO(MSCCLPP_INIT, "NUMA node of DeviceChannelService proxy thread is set to %d", deviceNumaNode);
  }
}

}  // namespace channel
}  // namespace mscclpp
