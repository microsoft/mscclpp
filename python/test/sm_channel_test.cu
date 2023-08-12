#include <mscclpp/sm_channel_device.hpp>

#include "mscclpp_common.h"

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL sm_channel
#define N_SHARDS 2
#define TD int
#define USE_PACKET false
#endif

// END_DEFINES //

// be careful about using channels[my_rank] as it is inavlie and it is there just for simplicity of indexing
extern "C" __global__ void __launch_bounds__(1024, N_SHARDS)
    KERNEL(Plist<mscclpp::SmChannelDeviceHandle, N_SHARDS> channels, int my_rank, int nranks, int num_elements) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  uint64_t size_per_rank = (num_elements * sizeof(TD)) / nranks;
  uint64_t my_offset = size_per_rank * my_rank;
  uint64_t my_nghr_offset = size_per_rank * bid;
  int flag = 123;
  if (bid < nranks && bid != my_rank) {
    if (USE_PACKET) {
      channels[bid].putPackets(2 * my_offset, my_offset, size_per_rank, tid, blockDim.x, flag);
      channels[bid].getPackets(my_nghr_offset, 2 * my_nghr_offset, size_per_rank, tid, blockDim.x, flag);
    } else {
      channels[bid].put(my_offset, my_offset, size_per_rank, tid, blockDim.x);
      __syncthreads();
      if (!USE_PACKET && tid == 0) {
        channels[bid].signal();
        channels[bid].wait();
      }
    }
  }
}
