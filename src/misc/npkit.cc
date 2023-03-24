#include <chrono>
#include <fstream>
#include <unistd.h>

#include "alloc.h"
#include "npkit/npkit.h"

uint64_t NpKit::rank_ = 0;

NpKitEvent** NpKit::gpu_event_buffers_ = nullptr;
NpKitEvent** NpKit::cpu_event_buffers_ = nullptr;

NpKitEventCollectContext* NpKit::gpu_collect_contexts_ = nullptr;
NpKitEventCollectContext* NpKit::cpu_collect_contexts_ = nullptr;
uint64_t* NpKit::cpu_timestamp_ = nullptr;

std::thread* NpKit::cpu_timestamp_update_thread_ = nullptr;
volatile bool NpKit::cpu_timestamp_update_thread_should_stop_ = false;

void NpKit::CpuTimestampUpdateThread() {
  uint64_t init_system_clock = std::chrono::system_clock::now().time_since_epoch().count();
  uint64_t init_steady_clock = std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t curr_steady_clock = 0;
  volatile uint64_t* volatile_cpu_timestamp_ = cpu_timestamp_;
  while (!cpu_timestamp_update_thread_should_stop_) {
    curr_steady_clock = std::chrono::steady_clock::now().time_since_epoch().count();
    *volatile_cpu_timestamp_ = init_system_clock + (curr_steady_clock - init_steady_clock);
  }
}

mscclppResult_t NpKit::Init(int rank) {
  uint64_t i = 0;
  NpKitEventCollectContext ctx;
  ctx.event_buffer_head = 0;
  rank_ = rank;

  // Init event data structures
  MSCCLPPCHECK(mscclppCalloc(&gpu_event_buffers_, kNumGpuEventBuffers));
  MSCCLPPCHECK(mscclppCudaCalloc(&gpu_collect_contexts_, kNumGpuEventBuffers));
  for (i = 0; i < kNumGpuEventBuffers; i++) {
    MSCCLPPCHECK(mscclppCudaCalloc(gpu_event_buffers_ + i, kMaxNumGpuEventsPerBuffer));
    ctx.event_buffer = gpu_event_buffers_[i];
    MSCCLPPCHECK(mscclppCudaMemcpy(gpu_collect_contexts_ + i, &ctx, 1));
  }

  MSCCLPPCHECK(mscclppCalloc(&cpu_event_buffers_, kNumCpuEventBuffers));
  MSCCLPPCHECK(mscclppCalloc(&cpu_collect_contexts_, kNumCpuEventBuffers));
  for (i = 0; i < kNumCpuEventBuffers; i++) {
    MSCCLPPCHECK(mscclppCalloc(cpu_event_buffers_ + i, kMaxNumCpuEventsPerBuffer));
    ctx.event_buffer = cpu_event_buffers_[i];
    cpu_collect_contexts_[i] = ctx;
  }

  // Init timestamp
  MSCCLPPCHECK(mscclppCudaHostCalloc(&cpu_timestamp_, 1));
  volatile uint64_t* volatile_cpu_timestamp = cpu_timestamp_;
  *volatile_cpu_timestamp = std::chrono::system_clock::now().time_since_epoch().count();
  cpu_timestamp_update_thread_should_stop_ = false;
  cpu_timestamp_update_thread_ = new std::thread(CpuTimestampUpdateThread);

  return mscclppSuccess;
}

mscclppResult_t NpKit::Dump(const std::string& dump_dir) {
  uint64_t i = 0;
  std::string dump_file_path;

  // Dump CPU events
  for (i = 0; i < kNumCpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/cpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_channel_";
    dump_file_path += std::to_string(i);
    auto cpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    cpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[i]),
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
  for (i = 0; i < kNumGpuEventBuffers; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/gpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_buf_";
    dump_file_path += std::to_string(i);
    MSCCLPPCHECK(mscclppCudaMemcpy(cpu_event_buffers_[0], gpu_event_buffers_[i], kMaxNumGpuEventsPerBuffer));
    MSCCLPPCHECK(mscclppCudaMemcpy(cpu_collect_contexts_, gpu_collect_contexts_ + i, 1));
    auto gpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    gpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[0]),
        cpu_collect_contexts_[0].event_buffer_head * sizeof(NpKitEvent));
    gpu_trace_file.close();
  }

  // Dump GPU clockRate
  dump_file_path = dump_dir;
  dump_file_path += "/gpu_clock_rate_rank_";
  dump_file_path += std::to_string(rank_);
  cudaDeviceProp dev_prop;
  int dev;
  CUDACHECK(cudaGetDevice(&dev));
  CUDACHECK(cudaGetDeviceProperties(&dev_prop, dev));
  std::string clock_rate_str = std::to_string(dev_prop.clockRate);
  auto gpu_clock_rate_file = std::fstream(dump_file_path, std::ios::out);
  gpu_clock_rate_file.write(clock_rate_str.c_str(), clock_rate_str.length());
  gpu_clock_rate_file.close();

  return mscclppSuccess;
}

mscclppResult_t NpKit::Shutdown() {
  uint64_t i = 0;

  // Stop CPU timestamp updating thread
  cpu_timestamp_update_thread_should_stop_ = true;
  cpu_timestamp_update_thread_->join();

  // Free CPU event data structures
  for (i = 0; i < kNumCpuEventBuffers; i++) {
    free(cpu_event_buffers_[i]);
  }
  free(cpu_event_buffers_);
  free(cpu_collect_contexts_);

  // Free GPU event data structures
  for (i = 0; i < kNumGpuEventBuffers; i++) {
    CUDACHECK(cudaFree(gpu_event_buffers_[i]));
  }
  free(gpu_event_buffers_);
  CUDACHECK(cudaFree(gpu_collect_contexts_));

  // Free timestamp
  MSCCLPPCHECK(mscclppCudaHostFree(cpu_timestamp_));

  return mscclppSuccess;
}

NpKitEventCollectContext* NpKit::GetGpuEventCollectContexts() {
  return gpu_collect_contexts_;
}

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

uint64_t* NpKit::GetCpuTimestamp() {
  return cpu_timestamp_;
}