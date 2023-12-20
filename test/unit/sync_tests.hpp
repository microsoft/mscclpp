/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include <gtest/gtest.h>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_utils.hpp>

using TYPE = mscclpp::DeviceSyncer;

MSCCLPP_DEVICE_INLINE void
timestamp(uint64_t& clk) {
    asm volatile("s_memrealtime %0\n"
                 "s_waitcnt lgkmcnt(0)\n"
                    : "=s" (clk));
}

__global__ void synchronize(TYPE *syncer, uint64_t *timer) {
    uint64_t start {0};
    uint64_t end {0};
    timestamp(start);
    syncer->sync(gridDim.x);
    timestamp(end);
    if (threadIdx.x == 0) {
      timer[blockIdx.x] = end - start;
    }
}

class DeviceSyncerTestFixture : public ::testing::Test {
 public:
  DeviceSyncerTestFixture() {
    MSCCLPP_CUDATHROW(cudaMalloc((void**)&syncer_d, sizeof(TYPE)));
    syncer_h = (TYPE*)malloc(sizeof(TYPE));
    MSCCLPP_CUDATHROW(cudaMalloc((void**)&timer_d, sizeof(uint64_t) * MAX_BLOCKS));
    timer_h = (uint64_t*)malloc(sizeof(uint64_t) * MAX_BLOCKS);
  }

  ~DeviceSyncerTestFixture() {
    if (timer_h) { free(timer_h); }
    if (timer_d) { (void)cudaFree(timer_d); }
    if (syncer_h) { free(syncer_h); }
    if (syncer_d) { (void)cudaFree(syncer_d); }
  }

  void execute(uint32_t x_block_dim, uint32_t x_grid_dim) {
    assert(x_grid_dim <= MAX_BLOCKS);

    const dim3 blocksize(x_block_dim, 1, 1);
    const dim3 gridsize(x_grid_dim, 1, 1);

    cudaStream_t stream;
    MSCCLPP_CUDATHROW(cudaStreamCreate(&stream));

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    MSCCLPP_CUDATHROW(cudaEventCreate(&start_event));
    MSCCLPP_CUDATHROW(cudaEventCreate(&stop_event));

    memset(syncer_h, 0, sizeof(TYPE));
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(syncer_d, syncer_h, sizeof(TYPE), cudaMemcpyHostToDevice, stream));
    MSCCLPP_CUDATHROW(cudaEventRecord(start_event, stream));
    synchronize<<<gridsize, blocksize, 0, stream>>>(syncer_d, timer_d);
    MSCCLPP_CUDATHROW(cudaEventRecord(stop_event, stream));
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(timer_h, timer_d, sizeof(uint64_t) * MAX_BLOCKS, cudaMemcpyDeviceToHost, stream));
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

    float event_time;
    MSCCLPP_CUDATHROW(cudaEventElapsedTime(&event_time, start_event, stop_event));

    MSCCLPP_CUDATHROW(cudaEventDestroy(stop_event));
    MSCCLPP_CUDATHROW(cudaEventDestroy(start_event));
    MSCCLPP_CUDATHROW(cudaStreamDestroy(stream));

    printf("event time: %f ms\n", event_time);
    timer_us(x_grid_dim);
  }

 protected:
  uint64_t memrealtime_freq_mhz() {
    cudaDeviceProp deviceProp{};
    MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&deviceProp, 0));
#if defined(__HIP_PLATFORM_AMD__) && (__HIP_PLATFORM_AMD__ == 1)
    switch (deviceProp.gcnArch) {
      case 900: return 27;
      case 906: return 25;
      case 908: return 25;
      case 910: return 25;
      default:
        assert(false && "clock data unavailable");
        return 0;
    }
#endif
  }

  double gpu_cycles_to_us(uint64_t cycles) {
    double div {(double)cycles / memrealtime_freq_mhz()};
    return div;
  }

  void timer_us(uint32_t num_blocks) {
    for (uint32_t i{0}; i < num_blocks; i++) {
      printf("block %d : latency %f us\n", i, gpu_cycles_to_us(timer_h[i]));
    }
  }

  const uint32_t MAX_BLOCKS{1024};
  TYPE *syncer_h{nullptr};
  TYPE *syncer_d{nullptr};
  uint64_t *timer_h{nullptr};
  uint64_t *timer_d{nullptr};
};
