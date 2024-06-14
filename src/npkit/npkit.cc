// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <unistd.h>

#include <chrono>
#include <fstream>
#include <mscclpp/gpu.hpp>
#include <mscclpp/npkit/npkit.hpp>

#include "debug.h"

uint64_t NpKit::rank_ = 0;

std::vector<mscclpp::UniqueCudaPtr<NpKitEvent>> NpKit::gpu_event_buffers_;
std::vector<std::unique_ptr<NpKitEvent[]>> NpKit::cpu_event_buffers_;

mscclpp::UniqueCudaPtr<NpKitEventCollectContext> NpKit::gpu_collect_contexts_;
std::unique_ptr<NpKitEventCollectContext[]> NpKit::cpu_collect_contexts_;

mscclpp::UniqueCudaHostPtr<uint64_t> NpKit::cpu_timestamp_;
std::unique_ptr<std::thread> NpKit::cpu_timestamp_update_thread_;
volatile bool NpKit::cpu_timestamp_update_thread_should_stop_ = false;

void NpKit::CpuTimestampUpdateThread() {
  uint64_t init_system_clock = std::chrono::system_clock::now().time_since_epoch().count();
  uint64_t init_steady_clock = std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t curr_steady_clock = 0;
  volatile uint64_t* volatile_cpu_timestamp_ = cpu_timestamp_.get();
  while (!cpu_timestamp_update_thread_should_stop_) {
    curr_steady_clock = std::chrono::steady_clock::now().time_since_epoch().count();
    *volatile_cpu_timestamp_ = init_system_clock + (curr_steady_clock - init_steady_clock);
  }
}

void NpKit::Init(int rank) {
#if defined(ENABLE_NPKIT)
  uint64_t i = 0;
  NpKitEventCollectContext ctx;
  ctx.event_buffer_head = 0;
  rank_ = rank;

  // Init event data structures
  gpu_collect_contexts_ = mscclpp::allocUniqueCuda<NpKitEventCollectContext>(NpKit::kNumGpuEventBuffers);
  for (i = 0; i < NpKit::kNumGpuEventBuffers; i++) {
    gpu_event_buffers_.emplace_back(mscclpp::allocUniqueCuda<NpKitEvent>(kMaxNumGpuEventsPerBuffer));
    ctx.event_buffer = gpu_event_buffers_[i].get();
    mscclpp::memcpyCuda(gpu_collect_contexts_.get() + i, &ctx, 1);
  }

  cpu_collect_contexts_ = std::make_unique<NpKitEventCollectContext[]>(NpKit::kNumCpuEventBuffers);
  for (i = 0; i < NpKit::kNumCpuEventBuffers; i++) {
    cpu_event_buffers_.emplace_back(std::make_unique<NpKitEvent[]>(kMaxNumCpuEventsPerBuffer));
    ctx.event_buffer = cpu_event_buffers_[i].get();
    cpu_collect_contexts_[i] = ctx;
  }

  // Init timestamp
  cpu_timestamp_ = mscclpp::makeUniqueCudaHost<uint64_t>();
  volatile uint64_t* volatile_cpu_timestamp = cpu_timestamp_.get();
  *volatile_cpu_timestamp = std::chrono::system_clock::now().time_since_epoch().count();
  cpu_timestamp_update_thread_should_stop_ = false;
  cpu_timestamp_update_thread_ = std::make_unique<std::thread>(CpuTimestampUpdateThread);
#else
  WARN("NpKit::Init(%d) : MSCCLPP library was not built with NPKit enabled.", rank);
#endif
}

#if defined(ENABLE_NPKIT)
static int GetGpuClockRateInKhz() {
  int dev_id;
#if defined(__HIP_PLATFORM_AMD__)
  cudaDeviceProp dev_prop;
  char gcn_arch[256];
  MSCCLPP_CUDATHROW(cudaGetDevice(&dev_id));
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&dev_prop, dev_id));
  char* gcnArchNameToken = strtok(dev_prop.gcnArchName, ":");
  strcpy(gcn_arch, gcnArchNameToken);
  if (strncmp("gfx94", gcn_arch, 5) == 0)
    return 100000;
  else
    return 25000;
#else
  cudaDeviceProp dev_prop;
  MSCCLPP_CUDATHROW(cudaGetDevice(&dev_id));
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&dev_prop, dev_id));
  return dev_prop.clockRate;
#endif
}
#endif

void NpKit::Dump(const std::string& dump_dir) {
#if defined(ENABLE_NPKIT)
  uint64_t i = 0;
  std::string dump_file_path;

  // Dump CPU events
  for (i = 0; i < NpKit::kNumCpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/cpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_channel_";
    dump_file_path += std::to_string(i);
    auto cpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    cpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[i].get()),
                         cpu_collect_contexts_[i].event_buffer_head * sizeof(NpKitEvent));
    cpu_trace_file.close();
  }

  // Dump CPU clock info
  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_num_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_num_str = std::to_string(std::chrono::steady_clock::duration::period::num);
  auto clock_period_num_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_num_file.write(clock_period_num_str.c_str(), clock_period_num_str.length());
  clock_period_num_file.close();

  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_den_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_den_str = std::to_string(std::chrono::steady_clock::duration::period::den);
  auto clock_period_den_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_den_file.write(clock_period_den_str.c_str(), clock_period_den_str.length());
  clock_period_den_file.close();

  // Dump GPU events, reuse CPU struct
  for (i = 0; i < NpKit::kNumGpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/gpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_buf_";
    dump_file_path += std::to_string(i);
    mscclpp::memcpyCuda(cpu_event_buffers_[0].get(), gpu_event_buffers_[i].get(), kMaxNumGpuEventsPerBuffer);
    mscclpp::memcpyCuda(cpu_collect_contexts_.get(), gpu_collect_contexts_.get() + i, 1);
    auto gpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    gpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[0].get()),
                         cpu_collect_contexts_[0].event_buffer_head * sizeof(NpKitEvent));
    gpu_trace_file.close();
  }

  // Dump GPU clockRate
  dump_file_path = dump_dir;
  dump_file_path += "/gpu_clock_rate_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_rate_str = std::to_string(GetGpuClockRateInKhz());
  auto gpu_clock_rate_file = std::fstream(dump_file_path, std::ios::out);
  gpu_clock_rate_file.write(clock_rate_str.c_str(), clock_rate_str.length());
  gpu_clock_rate_file.close();
#else
  WARN("NpKit::Dump(%s) : MSCCLPP library was not built with NPKit enabled.", dump_dir.c_str());
#endif
}

void NpKit::Shutdown() {
#if defined(ENABLE_NPKIT)
  // Stop CPU timestamp updating thread
  cpu_timestamp_update_thread_should_stop_ = true;
  cpu_timestamp_update_thread_->join();

  // Free CPU event data structures
  cpu_event_buffers_.clear();
  cpu_collect_contexts_.reset();

  // Free GPU event data structures
  gpu_event_buffers_.clear();
  gpu_collect_contexts_.reset();

  // Free timestamp
  cpu_timestamp_update_thread_.reset();
  cpu_timestamp_.reset();
#endif
}

NpKitEventCollectContext* NpKit::GetGpuEventCollectContexts() { return gpu_collect_contexts_.get(); }

void NpKit::CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp, int channel_id) {
  uint64_t event_buffer_head = cpu_collect_contexts_[channel_id].event_buffer_head;
  if (event_buffer_head < kMaxNumCpuEventsPerBuffer) {
    NpKitEvent& event = cpu_collect_contexts_[channel_id].event_buffer[event_buffer_head];
    event.fields.type = type;
    event.fields.size = size;
    event.fields.rsvd = rsvd;
    event.fields.timestamp = timestamp;
    cpu_collect_contexts_[channel_id].event_buffer_head++;
  }
}

uint64_t* NpKit::GetCpuTimestamp() { return cpu_timestamp_.get(); }
