#include <mscclpp/proxy_channel_device.hpp>

extern "C" __global__ void __launch_bounds__(1024, 1)
    nw_out_put_kernel(mscclpp::SimpleProxyChannelDeviceHandle* proxyChannel, int num_handles) {
    // nw_out_put_kernel(mscclpp::SimpleProxyChannelDeviceHandle* proxyChannel, int offset, int dataSize, int num_handles, int rank) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  printf("globalIndex %d\n", globalIndex);
  if (globalIndex == 0) {
    for (int i=0; i<num_handles; i++) {
      printf("put %d\n", i);
      // if (rank == 0) {
        proxyChannel[i].putWithSignal(0, 8);
        proxyChannel[i].flush();
    }
  }
}