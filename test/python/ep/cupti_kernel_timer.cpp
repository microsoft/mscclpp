// In-process CUPTI kernel timer for the mscclpp LL benchmark.
// Faithful port of NCCL-EP ep_bench.cu's KernelTimer: uses the CUPTI Activity
// API to record per-kernel GPU execution windows (end-start), keyed by kernel
// name, and exposes an extern "C" ABI so it can be driven from Python via ctypes
// (no cuda-python CUPTI bindings exist in this env, but libcupti.so is loadable).
//
// This is near-zero host perturbation (out-of-band buffer callbacks), unlike
// torch.profiler's in-process tracing which serialized the LL dispatch recv-spin
// and inflated one rank's device time into the millisecond range. It matches
// ep_bench's methodology exactly: start() after warmup, stop() after the timed
// loop, get_avg_us("dispatch"/"combine") buckets by mangled-name substring.
//
// COOPERATIVE-LAUNCH NOTE (GB200 / CUDA 13): the mscclpp LL dispatch/combine
// kernels are launched with cudaLaunchCooperativeKernel. Those are NOT reported
// by CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL on this driver, but they ARE reported
// by CUPTI_ACTIVITY_KIND_KERNEL (the serialized-kernel activity), which is what
// we subscribe to below. KIND_KERNEL only serializes *inter*-kernel concurrency;
// in this dispatch->sync->combine->sync paired loop the kernels already run one
// at a time, so the measured per-kernel GPU duration is unaffected. The activity
// record carries the RAW MANGLED name (e.g. ...low_latency8dispatch...), so the
// caller matches the substring "dispatch"/"combine" (present in the mangled form)
// rather than the demangled "low_latency::dispatch".
//
// Build (host-only C++, links libcupti):
//   g++ -O2 -fPIC -shared cupti_kernel_timer.cpp -o libcupti_kernel_timer.so \
//       -I<cuda>/targets/sbsa-linux/include -L<cuda>/targets/sbsa-linux/lib -lcupti
#include <cupti.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace {

struct KernelStat {
  uint64_t total_ns = 0;
  uint64_t count = 0;
};

std::map<std::string, KernelStat> g_stats;
std::mutex g_mutex;

constexpr size_t kBufSize = 8 * 1024 * 1024;  // 8 MB, matches ep_bench

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  // 8-byte aligned; aligned_alloc requires size to be a multiple of alignment.
  *buffer = static_cast<uint8_t*>(aligned_alloc(8, kBufSize));
  *size = kBufSize;
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
  CUpti_Activity* record = nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
    if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      // CUpti_ActivityKernel10 is the record layout for CUDA 13 CUPTI. start/end
      // (GPU HW timestamps, ns) and name have been stable across versions.
      auto* k = reinterpret_cast<CUpti_ActivityKernel10*>(record);
      if (k->name) {
        auto& s = g_stats[k->name];
        s.total_ns += (k->end - k->start);
        s.count += 1;
      }
    }
  }
  free(buffer);
}

}  // namespace

extern "C" {

// Clear stats, register the buffer callbacks, and enable concurrent-kernel
// activity recording. Call AFTER warmup (like ep_bench's KernelTimer::start()).
int kt_start() {
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stats.clear();
  }
  CUptiResult r = cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted);
  if (r != CUPTI_SUCCESS) return static_cast<int>(r);
  r = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
  return static_cast<int>(r);
}

// Flush pending buffers and disable recording. Returns CUPTI result code (0=ok).
int kt_stop() {
  cuptiActivityFlushAll(0);
  CUptiResult r = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
  return static_cast<int>(r);
}

// Average GPU execution time (microseconds) over every recorded kernel whose
// mangled name contains `substr`. Returns 0 if none matched.
double kt_get_avg_us(const char* substr) {
  std::lock_guard<std::mutex> lock(g_mutex);
  uint64_t total_ns = 0, count = 0;
  for (const auto& kv : g_stats) {
    if (kv.first.find(substr) != std::string::npos) {
      total_ns += kv.second.total_ns;
      count += kv.second.count;
    }
  }
  return count ? static_cast<double>(total_ns) / static_cast<double>(count) / 1000.0 : 0.0;
}

// Number of recorded kernel instances whose name contains `substr`.
long kt_get_count(const char* substr) {
  std::lock_guard<std::mutex> lock(g_mutex);
  uint64_t count = 0;
  for (const auto& kv : g_stats) {
    if (kv.first.find(substr) != std::string::npos) count += kv.second.count;
  }
  return static_cast<long>(count);
}

}  // extern "C"
