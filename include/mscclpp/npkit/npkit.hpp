// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NPKIT_H_
#define NPKIT_H_

#include <string>
#include <thread>
#include <vector>

#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/npkit/npkit_event.hpp>
#include <mscclpp/npkit/npkit_struct.hpp>

#if defined(__HIP_PLATFORM_AMD__)
#define NPKIT_GET_GPU_TIMESTAMP wall_clock64
#else
#define NPKIT_GET_GPU_TIMESTAMP clock64
#endif

#define NPKIT_SHM_NUM_EVENTS 64

class NpKit {
 public:
  static const uint64_t kNumGpuEventBuffers = 1024;

  static const uint64_t kNumCpuEventBuffers = 64;

  static void Init(int rank);

  static void Dump(const std::string& dump_dir);

  static void Shutdown();

  static NpKitEventCollectContext* GetGpuEventCollectContexts();

#if defined(MSCCLPP_DEVICE_COMPILE)
  static inline __device__ void CollectGpuEventShm(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
                                                   NpKitEvent* event_buffer, uint64_t* event_buffer_head) {
    if (*event_buffer_head < NPKIT_SHM_NUM_EVENTS) {
      if (threadIdx.x == 0) {
        NpKitEvent& event = event_buffer[*event_buffer_head];
        event.fields.type = type;
        event.fields.size = size;
        event.fields.rsvd = rsvd;
        event.fields.timestamp = timestamp;
      }
      (*event_buffer_head)++;
    }
  }
#endif

  static void CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp, int channel_id);

  static uint64_t* GetCpuTimestamp();

 private:
  static void CpuTimestampUpdateThread();

  // 64K * 1024 * 16B = 1GB per GPU
  static const uint64_t kMaxNumGpuEventsPerBuffer = 1ULL << 16;

  // 64K * 2 (send/recv) * (1024/64) = 2M, 2M * 64 * 16B = 2GB per CPU
  static const uint64_t kMaxNumCpuEventsPerBuffer = 1ULL << 21;

  static std::vector<mscclpp::UniqueCudaPtr<NpKitEvent>> gpu_event_buffers_;
  static std::vector<std::unique_ptr<NpKitEvent[]>> cpu_event_buffers_;

  static mscclpp::UniqueCudaPtr<NpKitEventCollectContext> gpu_collect_contexts_;
  static std::unique_ptr<NpKitEventCollectContext[]> cpu_collect_contexts_;

  static uint64_t rank_;

  static mscclpp::UniqueCudaHostPtr<uint64_t> cpu_timestamp_;
  static std::unique_ptr<std::thread> cpu_timestamp_update_thread_;
  static volatile bool cpu_timestamp_update_thread_should_stop_;
};

#endif
