#include <mscclpp/proxy_channel_device.hpp>

extern "C" __global__ void __launch_bounds__(1024, 1)
    nw_in_wait_kernel(mscclpp::SimpleProxyChannelDeviceHandle* proxyChannel, int num_handles) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  printf("globalIndex %d\n", globalIndex);
  if (globalIndex == 0) {
    for (int i=0; i<num_handles; i++) {
        printf("wait %d\n", i);
        proxyChannel[i].wait();
    }
  }
}