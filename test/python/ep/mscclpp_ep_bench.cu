// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// mscclpp_ep_bench: a pure-C++/MPI low-latency EP benchmark that calls
// mscclpp::ep::MoERuntime::dispatch / ::combine directly (no Python), so mscclpp
// EP can be compared with NVIDIA NCCL-EP's ep_bench on an equal footing --
// C++ host launch, and CUPTI kernel timing via CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
// (the same activity kind ep_bench uses; unlike the torch/Kineto in-process path
// it does report mscclpp's cooperative-launch LL kernels).
//
// It mirrors ep_bench's LL measurement methodology and emits the identical
// "=== Summary (Low Latency, across N ranks) ===" block so the unified driver
// (run_ep_bench.py) parses it with no changes.
//
// Scope: low-latency (LL), BF16, EXPERT_MAJOR layout. Single- or multi-node
// (the bootstrap uses an MPI_Bcast of a TcpBootstrap UniqueId).

#include <mpi.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <mscclpp/core.hpp>

#include "config.hpp"        // mscclpp::ep::getLowLatencyRdmaSizeHint
#include "kernels/api.cuh"   // mscclpp::ep::MoEMode, DispatchLayout
#include "moe_runtime.hpp"   // mscclpp::ep::MoERuntime

#define CUDA_CHECK(x)                                                                     \
  do {                                                                                    \
    cudaError_t _e = (x);                                                                 \
    if (_e != cudaSuccess) {                                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
    }                                                                                     \
  } while (0)

#define CUPTI_CHECK(x)                                                                    \
  do {                                                                                    \
    CUptiResult _e = (x);                                                                 \
    if (_e != CUPTI_SUCCESS) {                                                            \
      const char* _s = nullptr;                                                           \
      cuptiGetResultString(_e, &_s);                                                      \
      fprintf(stderr, "CUPTI error %s at %s:%d\n", _s ? _s : "?", __FILE__, __LINE__);    \
    }                                                                                     \
  } while (0)

// ---------------------------------------------------------------------------
// KernelTimer: per-kernel GPU timing via the CUPTI Activity API, a faithful
// analog of ep_bench's KernelTimer. Uses CONCURRENT_KERNEL (ep_bench's kind).
// Records are bucketed by mangled-name substring ("dispatch"/"combine").
// ---------------------------------------------------------------------------
namespace {

struct KernStat {
  uint64_t total_ns = 0;
  uint64_t count = 0;
};
std::map<std::string, KernStat> g_kernel_stats;
int g_activity_kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;

void CUPTIAPI bufferRequested(uint8_t** buffer, size_t* size, size_t* maxNumRecords) {
  constexpr size_t kBufSize = 8 * 1024 * 1024;
  *buffer = static_cast<uint8_t*>(aligned_alloc(8, kBufSize));
  *size = kBufSize;
  *maxNumRecords = 0;
}

void CUPTIAPI bufferCompleted(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t validSize) {
  CUpti_Activity* record = nullptr;
  while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ||
        record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = reinterpret_cast<CUpti_ActivityKernel10*>(record);
      if (k->name) {
        auto& e = g_kernel_stats[k->name];
        e.total_ns += (k->end - k->start);
        e.count += 1;
      }
    }
  }
  free(buffer);
}

class KernelTimer {
 public:
  KernelTimer() {
    if (const char* env = std::getenv("MSCCLPP_EP_BENCH_KERNEL_KIND")) {
      if (std::string(env) == "kernel") g_activity_kind = CUPTI_ACTIVITY_KIND_KERNEL;
    }
  }
  int start() {
    g_kernel_stats.clear();
    CUPTI_CHECK(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
    return cuptiActivityEnable(static_cast<CUpti_ActivityKind>(g_activity_kind));
  }
  void stop() {
    CUPTI_CHECK(cuptiActivityFlushAll(1));
    CUPTI_CHECK(cuptiActivityDisable(static_cast<CUpti_ActivityKind>(g_activity_kind)));
  }
  // Mean GPU time (us) over all kernels whose (mangled) name contains substr.
  double get_avg_us(const char* substr) const {
    uint64_t total_ns = 0, count = 0;
    for (const auto& kv : g_kernel_stats) {
      if (kv.first.find(substr) != std::string::npos) {
        total_ns += kv.second.total_ns;
        count += kv.second.count;
      }
    }
    return count ? (static_cast<double>(total_ns) / count) / 1e3 : 0.0;
  }
  uint64_t get_count(const char* substr) const {
    uint64_t count = 0;
    for (const auto& kv : g_kernel_stats)
      if (kv.first.find(substr) != std::string::npos) count += kv.second.count;
    return count;
  }
};

struct Args {
  int num_tokens = 128;
  int hidden = 7168;
  int num_topk = 8;
  int num_experts = 256;
  int num_warmup = 10;
  int num_iters = 50;
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> int { return (i + 1 < argc) ? std::atoi(argv[++i]) : 0; };
    if (s == "-a" || s == "--algorithm") { ++i; /* ll only */ }
    else if (s == "-t" || s == "--num-tokens") a.num_tokens = next();
    else if (s == "-d" || s == "--hidden") a.hidden = next();
    else if (s == "-k" || s == "--num-topk") a.num_topk = next();
    else if (s == "-e" || s == "--num-experts") a.num_experts = next();
    else if (s == "-w" || s == "--num-warmup") a.num_warmup = next();
    else if (s == "-i" || s == "--num-iters") a.num_iters = next();
  }
  return a;
}

struct Stat {
  double avg, mn, mx;
};
Stat stats(const std::vector<double>& v) {
  double s = 0, mn = 1e30, mx = -1e30;
  for (double x : v) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
  return {v.empty() ? 0.0 : s / v.size(), mn, mx};
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, nRanks = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  int localRank = 0;
  if (const char* env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK")) localRank = std::atoi(env);
  CUDA_CHECK(cudaSetDevice(localRank));

  Args args = parse_args(argc, argv);
  const int T = args.num_tokens, H = args.hidden, K = args.num_topk, E = args.num_experts;
  const int W = nRanks, warmup = args.num_warmup, iters = args.num_iters;
  if (E % W != 0) {
    if (rank == 0) fprintf(stderr, "num_experts (%d) must be divisible by world_size (%d)\n", E, W);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const int Elocal = E / W;

  // --- Bootstrap mscclpp::Communicator (TcpBootstrap + MPI_Bcast of UniqueId). ---
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nRanks);
  mscclpp::UniqueId uid;
  if (rank == 0) uid = bootstrap->createUniqueId();
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(uid);
  mscclpp::Communicator comm(bootstrap);

  const int64_t numRdmaBytes =
      static_cast<int64_t>(mscclpp::ep::getLowLatencyRdmaSizeHint(T, H, W, E));
  mscclpp::ep::MoERuntime rt(comm, /*numNvlBytes=*/0, numRdmaBytes, mscclpp::ep::MoEMode::LOW_LATENCY);
  if (!rt.isAvailable()) {
    if (rank == 0) fprintf(stderr, "MoERuntime not available\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (rank == 0) {
    printf("[cfg] algorithm=LOW_LATENCY num_ranks=%d tokens/rank=%d hidden=%d num_experts=%d "
           "top_k=%d warmup=%d iters=%d num_rdma_bytes=%lld is_internode=%d\n",
           W, T, H, E, K, warmup, iters, (long long)numRdmaBytes, (int)rt.isInternodeAvailable());
    fflush(stdout);
  }

  // --- Device buffers (hoisted out of the timed loop). ---
  const size_t slots = (size_t)W * T;  // recv slots per local expert
  __nv_bfloat16 *d_x = nullptr, *d_out = nullptr, *d_recv = nullptr;
  int64_t *d_topk = nullptr, *d_layout = nullptr;
  float* d_weights = nullptr;
  int *d_srcinfo = nullptr, *d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, (size_t)T * H * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_out, (size_t)T * H * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_recv, (size_t)Elocal * slots * H * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_topk, (size_t)T * K * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_weights, (size_t)T * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_srcinfo, (size_t)Elocal * slots * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_layout, (size_t)Elocal * W * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_count, (size_t)Elocal * sizeof(int)));

  // Inputs (content is immaterial to timing; give every token K distinct experts).
  CUDA_CHECK(cudaMemset(d_x, 0, (size_t)T * H * sizeof(__nv_bfloat16)));
  std::vector<int64_t> h_topk((size_t)T * K);
  std::vector<float> h_weights((size_t)T * K, 1.0f);
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < K; ++j) h_topk[(size_t)t * K + j] = ((int64_t)t * K + j) % E;
  CUDA_CHECK(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

  const long long num_valid_selections = (long long)T * K;
  const double disp_bytes = (double)num_valid_selections * H * 2.0;  // BF16
  const double comb_bytes = disp_bytes;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto dispatch = [&]() {
    rt.dispatch(d_recv, /*outputScales=*/nullptr, d_srcinfo, d_layout, d_count, d_x, d_topk, T, H, K,
                /*numMaxDispatchTokensPerRank=*/T, E, /*requiresQuantization=*/false,
                mscclpp::ep::DispatchLayout::EXPERT_MAJOR, stream);
  };
  auto combine = [&]() {
    rt.combine(d_out, d_recv, /*inputScales=*/nullptr, d_topk, d_weights, d_srcinfo, d_layout, T, H, K,
               /*numMaxDispatchTokensPerRank=*/T, E, /*requiresDequantization=*/false, stream);
  };

  // --- Warmup (paired), then per-iter timed (paired), matching ep_bench. ---
  for (int w = 0; w < warmup; ++w) {
    dispatch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    combine();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
  }

  KernelTimer ktimer;
  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);
  int kt_rc = ktimer.start();

  std::vector<cudaEvent_t> ds(iters), de(iters), cs(iters), ce(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventCreate(&ds[i]));
    CUDA_CHECK(cudaEventCreate(&de[i]));
    CUDA_CHECK(cudaEventCreate(&cs[i]));
    CUDA_CHECK(cudaEventCreate(&ce[i]));
  }
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(ds[i], stream));
    dispatch();
    CUDA_CHECK(cudaEventRecord(de[i], stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(cs[i], stream));
    combine();
    CUDA_CHECK(cudaEventRecord(ce[i], stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(MPI_COMM_WORLD);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  if (kt_rc == CUPTI_SUCCESS) ktimer.stop();

  // --- Collect per-iter host times (ms->us), trim first (warmup outlier). ---
  std::vector<double> disp_us, comb_us, tot_us;
  for (int i = 0; i < iters; ++i) {
    float d_ms = 0, c_ms = 0, t_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&d_ms, ds[i], de[i]));
    CUDA_CHECK(cudaEventElapsedTime(&c_ms, cs[i], ce[i]));
    CUDA_CHECK(cudaEventElapsedTime(&t_ms, ds[i], ce[i]));
    if (i == 0 && iters > 1) continue;
    disp_us.push_back(d_ms * 1e3);
    comb_us.push_back(c_ms * 1e3);
    tot_us.push_back(t_ms * 1e3);
  }
  Stat d = stats(disp_us), c = stats(comb_us), tt = stats(tot_us);

  // --- Cross-rank reduction (MPI), mirroring ep_bench / ep_bench_ll. ---
  auto reduce3 = [&](double avg, double mn, double mx, double& g_avg, double& g_min, double& g_max) {
    MPI_Reduce(&avg, &g_avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mn, &g_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mx, &g_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    g_avg /= W;
  };
  double gda, gdmn, gdmx, gca, gcmn, gcmx, gta, gtmn, gtmx;
  reduce3(d.avg, d.mn, d.mx, gda, gdmn, gdmx);
  reduce3(c.avg, c.mn, c.mx, gca, gcmn, gcmx);
  reduce3(tt.avg, tt.mn, tt.mx, gta, gtmn, gtmx);

  // Kernel-only (CUPTI). Per-rank mean, then cross-rank avg/min/max.
  double kd = (kt_rc == CUPTI_SUCCESS) ? ktimer.get_avg_us("dispatch") : 0.0;
  double kc = (kt_rc == CUPTI_SUCCESS) ? ktimer.get_avg_us("combine") : 0.0;
  double gkda, gkdmn, gkdmx, gkca, gkcmn, gkcmx;
  reduce3(kd, kd, kd, gkda, gkdmn, gkdmx);
  reduce3(kc, kc, kc, gkca, gkcmn, gkcmx);
  bool kernel_ok = (kt_rc == CUPTI_SUCCESS) && (kd > 0.0) && (kc > 0.0);

  if (std::getenv("MSCCLPP_EP_KDEBUG") && rank == 0) {
    printf("[kdebug] kt_start rc=%d dispatch=%.1fus x%llu combine=%.1fus x%llu (kind=%s)\n", kt_rc,
           kd, (unsigned long long)ktimer.get_count("dispatch"), kc,
           (unsigned long long)ktimer.get_count("combine"),
           g_activity_kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ? "CONCURRENT_KERNEL" : "KERNEL");
  }

  if (rank == 0) {
    printf("\n=== Summary (Low Latency, across %d ranks) ===\n", W);
    printf("\n--- Host-observed performance ---\n");
    printf("Dispatch (BF16):  avg=%.2f us, min=%.2f us, max=%.2f us\n", gda, gdmn, gdmx);
    printf("                  throughput: avg=%.2f GB/s\n", (disp_bytes / 1e9) / (gda * 1e-6));
    printf("Combine (BF16):   avg=%.2f us, min=%.2f us, max=%.2f us\n", gca, gcmn, gcmx);
    printf("                  throughput: avg=%.2f GB/s\n", (comb_bytes / 1e9) / (gca * 1e-6));
    printf("Total (D+C):      avg=%.2f us, min=%.2f us, max=%.2f us\n", gta, gtmn, gtmx);
    printf("                  throughput: avg=%.2f GB/s\n",
           ((disp_bytes + comb_bytes) / 1e9) / (gta * 1e-6));

    printf("\n--- Kernel-only performance ---\n");
    if (kernel_ok) {
      printf("Dispatch:    avg=%.2f us, min=%.2f us, max=%.2f us\n", gkda, gkdmn, gkdmx);
      printf("                  throughput: avg=%.2f GB/s\n", (disp_bytes / 1e9) / (gkda * 1e-6));
      printf("Combine:     avg=%.2f us, min=%.2f us, max=%.2f us\n", gkca, gkcmn, gkcmx);
      printf("                  throughput: avg=%.2f GB/s\n", (comb_bytes / 1e9) / (gkca * 1e-6));
      printf("Total (D+C): %.2f us (kernel dispatch avg + combine avg)\n", gkda + gkca);
    } else {
      printf("  NOTE: CUPTI kernel timing unavailable (rc=%d) or captured 0 LL kernels.\n", kt_rc);
    }

    printf("\nByte counts: dispatch=%.2f MB (BF16), combine=%.2f MB (BF16), selections=%lld\n",
           disp_bytes / 1e6, comb_bytes / 1e6, num_valid_selections);
    fflush(stdout);
  }

  for (int i = 0; i < iters; ++i) {
    cudaEventDestroy(ds[i]);
    cudaEventDestroy(de[i]);
    cudaEventDestroy(cs[i]);
    cudaEventDestroy(ce[i]);
  }
  cudaStreamDestroy(stream);
  cudaFree(d_x); cudaFree(d_out); cudaFree(d_recv); cudaFree(d_topk);
  cudaFree(d_weights); cudaFree(d_srcinfo); cudaFree(d_layout); cudaFree(d_count);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
