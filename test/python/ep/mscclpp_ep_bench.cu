// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// mscclpp_ep_bench: a pure-C++/MPI low-latency EP benchmark that calls
// mscclpp::ep::MoERuntime::dispatch / ::combine directly (no Python), so mscclpp
// EP can be compared with NVIDIA NCCL-EP's ep_bench on an equal footing --
// C++ host launch, and CUPTI kernel timing via CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
// (the same activity kind ep_bench uses).
//
// It mirrors ep_bench's LL measurement methodology and emits the identical
// "=== Summary (Low Latency, across N ranks) ===" block so the unified driver
// (run_ep_bench.py) parses it with no changes.
//
// Scope: low-latency (LL), BF16, EXPERT_MAJOR layout. Single- or multi-node
// (the bootstrap uses an MPI_Bcast of a TcpBootstrap UniqueId).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "api.cuh"
#include "config.hpp"
#include "moe_runtime.hpp"

#define CUDA_CHECK(x)                                                                          \
  do {                                                                                         \
    cudaError_t _e = (x);                                                                      \
    if (_e != cudaSuccess) {                                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                            \
    }                                                                                          \
  } while (0)

#define CUPTI_CHECK(x)                                                                 \
  do {                                                                                 \
    CUptiResult _e = (x);                                                              \
    if (_e != CUPTI_SUCCESS) {                                                         \
      const char* _s = nullptr;                                                        \
      cuptiGetResultString(_e, &_s);                                                   \
      fprintf(stderr, "CUPTI error %s at %s:%d\n", _s ? _s : "?", __FILE__, __LINE__); \
    }                                                                                  \
  } while (0)

// ---------------------------------------------------------------------------
// KernelTimer: per-kernel GPU timing via the CUPTI Activity API, a faithful
// analog of ep_bench's KernelTimer.
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
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = reinterpret_cast<CUpti_ActivityKernel9*>(record);
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
  int num_blocks = mscclpp::ep::low_latency::MaxDispatchBlocks;
  int seed = 0xB3C4;
  bool kernel_timing = false;
  std::string dispatch_dtype = "bf16";
  std::string combine_mode = "rank_local_reduce";
};

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&]() -> int { return (i + 1 < argc) ? std::atoi(argv[++i]) : 0; };
    if (s == "-a" || s == "--algorithm") {
      ++i; /* ll only */
    } else if (s == "-t" || s == "--num-tokens")
      a.num_tokens = next();
    else if (s == "-d" || s == "--hidden")
      a.hidden = next();
    else if (s == "-k" || s == "--num-topk")
      a.num_topk = next();
    else if (s == "-e" || s == "--num-experts")
      a.num_experts = next();
    else if (s == "-w" || s == "--num-warmup")
      a.num_warmup = next();
    else if (s == "-i" || s == "--num-iters")
      a.num_iters = next();
    else if (s == "--num-blocks")
      a.num_blocks = next();
    else if (s == "--seed")
      a.seed = next();
    else if (s == "--kernel-timing")
      a.kernel_timing = true;
    else if (s == "--dispatch-dtype" && i + 1 < argc)
      a.dispatch_dtype = argv[++i];
    else if (s == "--combine-mode" && i + 1 < argc)
      a.combine_mode = argv[++i];
  }
  return a;
}

struct Stat {
  double avg, mn, mx;
};
Stat stats(const std::vector<double>& v) {
  double s = 0, mn = 1e30, mx = -1e30;
  for (double x : v) {
    s += x;
    mn = std::min(mn, x);
    mx = std::max(mx, x);
  }
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
  if (T <= 0 || E <= 0 || warmup < 0 || iters <= 0) {
    if (rank == 0) fprintf(stderr, "tokens, experts, and iters must be positive; warmup must be non-negative\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (H != 4096 && H != 6656 && H != 7168 && H != 8192 && H != 9216) {
    if (rank == 0) fprintf(stderr, "hidden must be one of 4096, 6656, 7168, 8192, 9216\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (K <= 0 || K > 9) {
    if (rank == 0) fprintf(stderr, "num_topk must be in [1, 9]\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (K > E) {
    if (rank == 0) fprintf(stderr, "num_topk must not exceed num_experts\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (E % W != 0) {
    if (rank == 0) fprintf(stderr, "num_experts (%d) must be divisible by world_size (%d)\n", E, W);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const int Elocal = E / W;
  auto dispatchDataType = mscclpp::ep::low_latency::DispatchDataType::BF16;
  if (args.dispatch_dtype == "fp8_e4m3") {
    dispatchDataType = mscclpp::ep::low_latency::DispatchDataType::FP8_E4M3;
  } else if (args.dispatch_dtype == "mxfp8_e4m3") {
    dispatchDataType = mscclpp::ep::low_latency::DispatchDataType::MXFP8_E4M3;
  }
  const auto combineMode = args.combine_mode == "direct_send"
                               ? mscclpp::ep::low_latency::CombineMode::DIRECT_SEND
                               : mscclpp::ep::low_latency::CombineMode::RANK_LOCAL_REDUCE;
  if (args.dispatch_dtype != "bf16" && args.dispatch_dtype != "fp8_e4m3" && args.dispatch_dtype != "mxfp8_e4m3") {
    if (rank == 0) fprintf(stderr, "unsupported --dispatch-dtype=%s\n", args.dispatch_dtype.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (args.combine_mode != "rank_local_reduce" && args.combine_mode != "direct_send") {
    if (rank == 0) fprintf(stderr, "unsupported --combine-mode=%s\n", args.combine_mode.c_str());
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (args.num_blocks < W + mscclpp::ep::low_latency::DispatchControlBlocks ||
      args.num_blocks > mscclpp::ep::low_latency::MaxDispatchBlocks) {
    if (rank == 0) fprintf(stderr, "--num-blocks must be in [world_size + 2, 130]\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const bool fp8Dispatch = dispatchDataType != mscclpp::ep::low_latency::DispatchDataType::BF16;
  const int scaleBlockSize = dispatchDataType == mscclpp::ep::low_latency::DispatchDataType::MXFP8_E4M3 ? 32 : 128;
  const int scaleElementSize =
      dispatchDataType == mscclpp::ep::low_latency::DispatchDataType::MXFP8_E4M3 ? sizeof(uint8_t) : sizeof(float);
  const char* dispatchLabel = dispatchDataType == mscclpp::ep::low_latency::DispatchDataType::MXFP8_E4M3
                                  ? "MXFP8_E4M3"
                                  : (fp8Dispatch ? "FP8_E4M3" : "BF16");

  // --- Bootstrap mscclpp::Communicator (TcpBootstrap + MPI_Bcast of UniqueId). ---
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, nRanks);
  mscclpp::UniqueId uid;
  if (rank == 0) uid = bootstrap->createUniqueId();
  MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);
  bootstrap->initialize(uid);
  mscclpp::Communicator comm(bootstrap);

  mscclpp::ep::MoERuntime rt(comm, T, H, E, K, false);
  if (!rt.isAvailable()) {
    if (rank == 0) fprintf(stderr, "MoERuntime not available\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (rank == 0) {
    const size_t symmetricBufferBytes = mscclpp::ep::low_latency::symmetricBufferSize(T, H, W, E, K);
    printf(
        "[cfg] algorithm=LOW_LATENCY num_ranks=%d tokens/rank=%d hidden=%d num_experts=%d "
        "top_k=%d warmup=%d iters=%d dispatch_dtype=%s combine_mode=%s symmetric_buffer_bytes=%zu is_internode=%d\n",
        W, T, H, E, K, warmup, iters, args.dispatch_dtype.c_str(), args.combine_mode.c_str(), symmetricBufferBytes,
        static_cast<int>(rt.isInternodeAvailable()));
    fflush(stdout);
  }

  // --- Device buffers (hoisted out of the timed loop). ---
  const size_t slots = (size_t)W * T;  // recv slots per local expert
  using Bf16 = mscclpp::ep::low_latency::Bf16;
  using Fp8E4M3 = mscclpp::ep::low_latency::Fp8E4M3;
  Bf16 *d_x = nullptr, *d_out = nullptr, *d_expert_output = nullptr;
  void* d_recv = nullptr;
  void* d_scales = nullptr;
  int64_t *d_topk = nullptr, *d_layout = nullptr;
  float* d_weights = nullptr;
  int *d_srcinfo = nullptr, *d_count = nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, (size_t)T * H * sizeof(Bf16)));
  CUDA_CHECK(cudaMalloc(&d_out, (size_t)T * H * sizeof(Bf16)));
  const size_t recvBytes = (size_t)Elocal * slots * H * (fp8Dispatch ? sizeof(Fp8E4M3) : sizeof(Bf16));
  CUDA_CHECK(cudaMalloc(&d_recv, recvBytes));
  if (fp8Dispatch) {
    CUDA_CHECK(cudaMalloc(&d_scales, (size_t)Elocal * slots * (H / scaleBlockSize) * scaleElementSize));
    CUDA_CHECK(cudaMalloc(&d_expert_output, (size_t)Elocal * slots * H * sizeof(Bf16)));
    CUDA_CHECK(cudaMemset(d_expert_output, 0, (size_t)Elocal * slots * H * sizeof(Bf16)));
  } else {
    d_expert_output = static_cast<Bf16*>(d_recv);
  }
  CUDA_CHECK(cudaMalloc(&d_topk, (size_t)T * K * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_weights, (size_t)T * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_srcinfo, (size_t)Elocal * slots * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_layout, (size_t)Elocal * W * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_count, (size_t)Elocal * sizeof(int)));

  // Inputs. Token payloads are immaterial to timing, but the top-k routing is
  // generated with the SAME scheme as NCCL-EP's ep_bench (generateRandomTopkIndicesLL):
  // per-token abs(randn)+1 scores, take the top-k experts by score, then mask 10
  // random (token, slot) positions with -1 to simulate dropped tokens.
  CUDA_CHECK(cudaMemset(d_x, 0, (size_t)T * H * sizeof(Bf16)));
  std::vector<int64_t> h_topk((size_t)T * K);
  std::vector<float> h_weights((size_t)T * K, 1.0f);
  {
    std::mt19937 gen(args.seed + rank);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<std::pair<float, int>> scoreIdx(E);
    for (int t = 0; t < T; ++t) {
      for (int e = 0; e < E; ++e) scoreIdx[e] = {std::abs(dist(gen)) + 1.0f, e};
      std::partial_sort(scoreIdx.begin(), scoreIdx.begin() + K, scoreIdx.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
      for (int j = 0; j < K; ++j) h_topk[(size_t)t * K + j] = scoreIdx[j].second;
    }
    // Randomly mask 10 positions with -1 (simulates dropped tokens); mirrors
    // ep_bench. Guarded on T > 0 so the distribution bound is valid.
    if (T > 0) {
      std::uniform_int_distribution<int> tokenDist(0, T - 1);
      std::uniform_int_distribution<int> topkDist(0, K - 1);
      for (int i = 0; i < 10; ++i) {
        int ti = tokenDist(gen), ki = topkDist(gen);
        h_topk[(size_t)ti * K + ki] = -1;
      }
    }
  }
  CUDA_CHECK(cudaMemcpy(d_topk, h_topk.data(), h_topk.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

  // Byte accounting counts only valid selections (topk >= 0), matching ep_bench's
  // calculateLowLatencyBytes after the -1 masking above.
  long long num_valid_selections = 0;
  for (size_t i = 0; i < h_topk.size(); ++i)
    if (h_topk[i] >= 0) ++num_valid_selections;
  const double dispatchBytesPerToken = fp8Dispatch ? H + (H / scaleBlockSize) * scaleElementSize : H * sizeof(Bf16);
  const double disp_bytes = (double)num_valid_selections * dispatchBytesPerToken;
  const double comb_bytes = (double)num_valid_selections * H * sizeof(Bf16);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto dispatch = [&]() {
    rt.dispatch(d_recv, d_scales, d_srcinfo, nullptr, nullptr, d_layout, d_count, d_x, d_topk, d_weights, T, H, K,
                /*maxTokensPerRank=*/T, E, /*invalidTokenExpertId=*/E, mscclpp::ep::DispatchLayout::EXPERT_MAJOR,
                dispatchDataType, args.num_blocks, stream);
  };
  auto combine = [&]() {
    rt.combine(d_out, d_expert_output, d_topk, d_weights, d_srcinfo, d_layout, T, H, K,
               /*maxTokensPerRank=*/T, E, mscclpp::ep::DispatchLayout::EXPERT_MAJOR, dispatchDataType, combineMode,
               args.num_blocks - mscclpp::ep::low_latency::DispatchControlBlocks, stream);
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
  int kt_rc = args.kernel_timing ? ktimer.start() : -1;
  int localTimerStarted = kt_rc == CUPTI_SUCCESS;
  int allTimersStarted = 0;
  MPI_Allreduce(&localTimerStarted, &allTimersStarted, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!allTimersStarted) {
    if (localTimerStarted) ktimer.stop();
    kt_rc = -1;
  }
  MPI_Barrier(MPI_COMM_WORLD);

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
  if (args.kernel_timing && kt_rc == CUPTI_SUCCESS) ktimer.stop();

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
  int localKernelOk = (kt_rc == CUPTI_SUCCESS) && (kd > 0.0) && (kc > 0.0);
  int allKernelsOk = 0;
  MPI_Allreduce(&localKernelOk, &allKernelsOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  double gkda, gkdmn, gkdmx, gkca, gkcmn, gkcmx;
  double kdMinInput = localKernelOk ? kd : 1e30;
  double kcMinInput = localKernelOk ? kc : 1e30;
  MPI_Reduce(&kd, &gkda, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&kdMinInput, &gkdmn, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&kd, &gkdmx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&kc, &gkca, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&kcMinInput, &gkcmn, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&kc, &gkcmx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  gkda /= W;
  gkca /= W;
  bool kernel_ok = allKernelsOk != 0;

  if (std::getenv("MSCCLPP_EP_KDEBUG") && rank == 0) {
    printf("[kdebug] kt_start rc=%d dispatch=%.1fus x%llu combine=%.1fus x%llu (kind=%s)\n", kt_rc, kd,
           (unsigned long long)ktimer.get_count("dispatch"), kc, (unsigned long long)ktimer.get_count("combine"),
           g_activity_kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL ? "CONCURRENT_KERNEL" : "KERNEL");
  }

  if (rank == 0) {
    printf("\n=== Summary (Low Latency, across %d ranks) ===\n", W);
    printf("\n--- Host-observed performance ---\n");
    printf("Dispatch (%s):  avg=%.2f us, min=%.2f us, max=%.2f us\n", dispatchLabel, gda, gdmn, gdmx);
    printf("                  throughput: avg=%.2f GB/s\n", (disp_bytes / 1e9) / (gda * 1e-6));
    printf("Combine (BF16):   avg=%.2f us, min=%.2f us, max=%.2f us\n", gca, gcmn, gcmx);
    printf("                  throughput: avg=%.2f GB/s\n", (comb_bytes / 1e9) / (gca * 1e-6));
    printf("Total (D+C):      avg=%.2f us, min=%.2f us, max=%.2f us\n", gta, gtmn, gtmx);
    printf("                  throughput: avg=%.2f GB/s\n", ((disp_bytes + comb_bytes) / 1e9) / (gta * 1e-6));

    printf("\n--- Kernel-only performance ---\n");
    if (kernel_ok) {
      printf("Dispatch:    min=%.2f us (representative)  [avg=%.2f, max=%.2f us -- rank skew]\n", gkdmn, gkda, gkdmx);
      printf("                  throughput @min: %.2f GB/s\n", (disp_bytes / 1e9) / (gkdmn * 1e-6));
      printf("Combine:     min=%.2f us (representative)  [avg=%.2f, max=%.2f us -- rank skew]\n", gkcmn, gkca, gkcmx);
      printf("                  throughput @min: %.2f GB/s\n", (comb_bytes / 1e9) / (gkcmn * 1e-6));
    } else {
      printf("  NOTE: CUPTI kernel timing unavailable (rc=%d) or captured 0 LL kernels.\n", kt_rc);
    }

    printf("\nByte counts: dispatch=%.2f MB (%s), combine=%.2f MB (BF16), selections=%lld\n", disp_bytes / 1e6,
           dispatchLabel, comb_bytes / 1e6, num_valid_selections);
    fflush(stdout);
  }

  for (int i = 0; i < iters; ++i) {
    cudaEventDestroy(ds[i]);
    cudaEventDestroy(de[i]);
    cudaEventDestroy(cs[i]);
    cudaEventDestroy(ce[i]);
  }
  cudaStreamDestroy(stream);
  cudaFree(d_x);
  cudaFree(d_out);
  cudaFree(d_recv);
  if (fp8Dispatch) cudaFree(d_expert_output);
  cudaFree(d_scales);
  cudaFree(d_topk);
  cudaFree(d_weights);
  cudaFree(d_srcinfo);
  cudaFree(d_layout);
  cudaFree(d_count);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
