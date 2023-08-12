#include <mscclpp/packet.hpp>
#include <mscclpp/proxy_channel_device.hpp>

#include "mscclpp_common.h"

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL simple_proxy_channel
#define N_SHARDS 8
#define TD int
#define USE_PACKET false
#endif

// END_DEFINES //

// be careful about using channels[my_rank] as it is inavlie and it is there just for simplicity of indexing
extern "C" __global__ void __launch_bounds__(1024, 1)
    KERNEL(Plist<mscclpp::SimpleProxyChannelDeviceHandle, N_SHARDS> channels, int my_rank, int nranks, TD* data,
           TD* scratch, int num_elements) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  uint64_t size_per_rank = (num_elements * sizeof(TD)) / nranks;
  uint64_t my_offset = size_per_rank * my_rank;
  int nthreads_per_rank = nthreads / nranks;
  int my_nghr = tid / nthreads_per_rank;
  uint64_t my_nghr_offset = size_per_rank * my_nghr;
  __syncthreads();
  int flag = 123;
  if (USE_PACKET) {
    mscclpp::putPackets(scratch, 2 * my_offset, data, my_offset, size_per_rank, tid, nthreads, flag);
    __syncthreads();
    if (tid < nranks && tid != my_rank) {
      channels[tid].put(2 * my_offset, 2 * my_offset, 2 * size_per_rank);
    }
    if (my_nghr != my_rank && my_nghr < nranks)
      mscclpp::getPackets(data, my_nghr_offset, scratch, 2 * my_nghr_offset, size_per_rank, tid % nthreads_per_rank,
                          nthreads_per_rank, flag);
  } else {
    if (tid < nranks && tid != my_rank) {
      channels[tid].putWithSignal(my_offset, my_offset, size_per_rank);
      channels[tid].wait();
    }
  }
}
