#include "channel.hpp"
#include "api.h"
#include "checks.hpp"
#include "debug.h"
#include "utils.h"

namespace mscclpp {
namespace channel {

MSCCLPP_API_CPP DeviceChannelService::DeviceChannelService(Communicator& communicator)
  : communicator_(communicator),
    proxy_([&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); })
{
  int cudaDevice;
  CUDATHROW(cudaGetDevice(&cudaDevice));
  MSCCLPPTHROW(getDeviceNumaNode(cudaDevice, &deviceNumaNode));
}

MSCCLPP_API_CPP void DeviceChannelService::bindThread()
{
  if (deviceNumaNode >= 0) {
    MSCCLPPTHROW(numaBind(deviceNumaNode));
    INFO(MSCCLPP_INIT, "NUMA node of DeviceChannelService proxy thread is set to %d", deviceNumaNode);
  }
}

} // namespace channel
} // namespace mscclpp
