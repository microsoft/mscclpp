// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mp_unit_tests.hpp"

namespace {

using Bf16 = typename mscclpp::bf16x2::ElementType;
using Fp8E4M3 = typename mscclpp::f8_e4m3x2::ElementType;

constexpr int NumRanks = 8;
constexpr int NumExperts = 8;
constexpr int NumTopk = 8;
constexpr int CorrectnessTokens = 8;
constexpr int CorrectnessHidden = 4096;
constexpr int PerfTokens = 32;
constexpr int PerfHidden = 7168;
constexpr int NumWarmups = 10;
constexpr int PairsPerGraph = 50;
constexpr int NumGraphReplays = 20;
constexpr int Threads = 256;

class CudaStream {
 public:
  CudaStream() { MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
  ~CudaStream() {
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  operator cudaStream_t() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaGraph {
 public:
  ~CudaGraph() { reset(); }

  void reset() {
    if (exec_ != nullptr) cudaGraphExecDestroy(exec_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    exec_ = nullptr;
    graph_ = nullptr;
  }

  template <typename Operation>
  void capture(cudaStream_t stream, Operation operation) {
    MSCCLPP_CUDATHROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    operation();
    MSCCLPP_CUDATHROW(cudaStreamEndCapture(stream, &graph_));
    MSCCLPP_CUDATHROW(cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0));
  }

  void launch(cudaStream_t stream) const { MSCCLPP_CUDATHROW(cudaGraphLaunch(exec_, stream)); }

 private:
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
};

struct TestBuffers {
  TestBuffers(int numTokens, int hidden)
      : input(static_cast<size_t>(numTokens) * hidden),
        output(static_cast<size_t>(numTokens) * hidden),
        expertOutput(static_cast<size_t>(NumRanks) * numTokens * hidden),
        topkIdx(static_cast<size_t>(numTokens) * NumTopk),
        topkWeights(static_cast<size_t>(numTokens) * NumTopk),
        outputScales(static_cast<size_t>(NumRanks) * numTokens * hidden / 128),
        srcInfo(static_cast<size_t>(NumRanks) * numTokens),
        layoutRange(NumRanks),
        outputCount(NumRanks) {}

  mscclpp::GpuBuffer<Bf16> input;
  mscclpp::GpuBuffer<Bf16> output;
  mscclpp::GpuBuffer<Bf16> expertOutput;
  mscclpp::GpuBuffer<int64_t> topkIdx;
  mscclpp::GpuBuffer<float> topkWeights;
  mscclpp::GpuBuffer<float> outputScales;
  mscclpp::GpuBuffer<int> srcInfo;
  mscclpp::GpuBuffer<int64_t> layoutRange;
  mscclpp::GpuBuffer<int> outputCount;
};

__global__ void initializeInputs(Bf16* input, int64_t* topkIdx, float* topkWeights, int rank, int numTokens,
                                 int hidden) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t inputElements = static_cast<size_t>(numTokens) * hidden;
  for (size_t index = thread; index < inputElements; index += stride) {
    const int token = static_cast<int>(index / hidden);
    input[index] = static_cast<Bf16>(static_cast<float>((rank * numTokens + token) * NumTopk));
  }

  const size_t routingElements = static_cast<size_t>(numTokens) * NumTopk;
  for (size_t index = thread; index < routingElements; index += stride) {
    topkIdx[index] = static_cast<int64_t>(index % NumTopk);
    topkWeights[index] = 1.0f / NumTopk;
  }
}

MSCCLPP_DEVICE_INLINE float fp8ToFloat(Fp8E4M3 value) {
  mscclpp::f8_e4m3x2 packed;
  packed.data[0] = value;
  packed.data[1] = value;
  return mscclpp::to<mscclpp::f32x2>(packed).data[0];
}

__global__ void dequantizeExpertMajor(Bf16* output, const Fp8E4M3* input, const float* scales, int rows, int hidden) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t elements = static_cast<size_t>(rows) * hidden;
  for (size_t index = thread; index < elements; index += stride) {
    const int row = static_cast<int>(index / hidden);
    const int hiddenIndex = static_cast<int>(index % hidden);
    const float scale = scales[static_cast<size_t>(hiddenIndex / 128) * rows + row];
    output[index] = static_cast<Bf16>(fp8ToFloat(input[index]) * scale);
  }
}

__global__ void stageRankMajorExpertOutput(Bf16* output, const Bf16* input, const int* topkIdx,
                                           const float* topkWeights, int rank, int rows, int hidden, bool directSend) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t elements = static_cast<size_t>(rows) * hidden * (directSend ? static_cast<size_t>(NumTopk) : size_t{1});
  for (size_t index = thread; index < elements; index += stride) {
    const int hiddenIndex = static_cast<int>(index % hidden);
    const size_t routeIndex = index / hidden;
    const int topkLane = directSend ? static_cast<int>(routeIndex % NumTopk) : 0;
    const int row = static_cast<int>(directSend ? routeIndex / NumTopk : routeIndex);
    float localWeight = 0.0f;
    if (directSend) {
      if (topkIdx[static_cast<size_t>(row) * NumTopk + topkLane] == rank) {
        localWeight = topkWeights[static_cast<size_t>(row) * NumTopk + topkLane];
      }
    } else {
      for (int lane = 0; lane < NumTopk; ++lane) {
        if (topkIdx[static_cast<size_t>(row) * NumTopk + lane] == rank) {
          localWeight += topkWeights[static_cast<size_t>(row) * NumTopk + lane];
        }
      }
    }
    output[index] = localWeight == 0.0f
                        ? static_cast<Bf16>(0.0f)
                        : static_cast<Bf16>(static_cast<float>(input[static_cast<size_t>(row) * hidden + hiddenIndex]) *
                                            localWeight);
  }
}

int numBlocks(size_t elements) { return static_cast<int>(std::min<size_t>((elements + Threads - 1) / Threads, 4096)); }

void initializeTestBuffers(TestBuffers& buffers, int rank, int numTokens, int hidden, cudaStream_t stream) {
  const size_t elements = std::max(static_cast<size_t>(numTokens) * hidden, static_cast<size_t>(numTokens) * NumTopk);
  initializeInputs<<<numBlocks(elements), Threads, 0, stream>>>(buffers.input.data(), buffers.topkIdx.data(),
                                                                buffers.topkWeights.data(), rank, numTokens, hidden);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.output.data(), 0, buffers.output.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.expertOutput.data(), 0, buffers.expertOutput.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.outputScales.data(), 0, buffers.outputScales.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.srcInfo.data(), 0, buffers.srcInfo.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.layoutRange.data(), 0, buffers.layoutRange.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.outputCount.data(), 0, buffers.outputCount.bytes(), stream));
}

std::string caseName(mscclpp::ep::DispatchLayout layout, mscclpp::ep::CombineMode combineMode,
                     mscclpp::ep::DispatchDataType dataType) {
  const char* layoutName = layout == mscclpp::ep::DispatchLayout::EXPERT_MAJOR ? "expert-major" : "rank-major";
  const char* combineName =
      combineMode == mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE ? "rank-local-reduce" : "direct-send";
  const char* dataTypeName = dataType == mscclpp::ep::DispatchDataType::BF16 ? "bf16" : "fp8-e4m3";
  return std::string(layoutName) + "/" + combineName + "/" + dataTypeName;
}

std::string checkOutput(const Bf16* output, int rank, int numTokens, int hidden, float tolerance,
                        float expectedMultiplier = 1.0f) {
  std::vector<Bf16> hostOutput(static_cast<size_t>(numTokens) * hidden);
  MSCCLPP_CUDATHROW(cudaMemcpy(hostOutput.data(), output, hostOutput.size() * sizeof(Bf16), cudaMemcpyDeviceToHost));
  for (int token = 0; token < numTokens; ++token) {
    const float expected = static_cast<float>((rank * numTokens + token) * NumTopk) * expectedMultiplier;
    for (int hiddenIndex = 0; hiddenIndex < hidden; ++hiddenIndex) {
      const float actual = static_cast<float>(hostOutput[static_cast<size_t>(token) * hidden + hiddenIndex]);
      if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        std::ostringstream error;
        error << "mismatch at token " << token << ", hidden " << hiddenIndex << ": expected " << expected << ", got "
              << actual;
        return error.str();
      }
    }
  }
  return {};
}

std::string checkCounts(const int* outputCount, mscclpp::ep::DispatchLayout layout, int numTokens) {
  const int countSize = layout == mscclpp::ep::DispatchLayout::EXPERT_MAJOR ? 1 : NumRanks;
  std::vector<int> hostCount(countSize);
  MSCCLPP_CUDATHROW(cudaMemcpy(hostCount.data(), outputCount, hostCount.size() * sizeof(int), cudaMemcpyDeviceToHost));
  const int expected = layout == mscclpp::ep::DispatchLayout::EXPERT_MAJOR ? NumRanks * numTokens : numTokens;
  for (int index = 0; index < countSize; ++index) {
    if (hostCount[index] != expected) {
      std::ostringstream error;
      error << "output count " << index << ": expected " << expected << ", got " << hostCount[index];
      return error.str();
    }
  }
  return {};
}

void assertCollectiveSuccess(const std::string& localError, const std::string& label) {
  const int localSuccess = localError.empty() ? 1 : 0;
  int globalSuccess = 0;
  MPI_Allreduce(&localSuccess, &globalSuccess, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (globalSuccess == 0) {
    FAIL() << label << ": " << (localError.empty() ? "failed on another rank" : localError);
  }
}

std::unique_ptr<mscclpp::ep::MoERuntime> createRuntime(mscclpp::Communicator& communicator, int numTokens, int hidden,
                                                       mscclpp::ep::DispatchLayout layout,
                                                       mscclpp::ep::CombineMode combineMode) {
  return std::make_unique<mscclpp::ep::MoERuntime>(communicator, mscclpp::ep::MoEMode::LATENCY, numTokens, hidden,
                                                   NumExperts, NumTopk, layout, combineMode);
}

void runCorrectnessCase(mscclpp::Communicator& communicator, int rank, int dispatchBlocks, int combineBlocks,
                        mscclpp::ep::DispatchLayout layout, mscclpp::ep::CombineMode combineMode,
                        mscclpp::ep::DispatchDataType dataType) {
  const std::string label = caseName(layout, combineMode, dataType);
  auto runtime = createRuntime(communicator, CorrectnessTokens, CorrectnessHidden, layout, combineMode);
  ASSERT_TRUE(runtime->isAvailable());
  runtime->initialize();

  CudaStream stream;
  TestBuffers buffers(CorrectnessTokens, CorrectnessHidden);
  initializeTestBuffers(buffers, rank, CorrectnessTokens, CorrectnessHidden, stream);

  void* dispatchOutput = runtime->dispatchOutputBuffer();
  const bool expertMajor = layout == mscclpp::ep::DispatchLayout::EXPERT_MAJOR;
  const bool fp8 = dataType == mscclpp::ep::DispatchDataType::FP8_E4M3;
  ASSERT_FALSE(!expertMajor && fp8);

  auto* outputTopkIdx = expertMajor ? nullptr : static_cast<int*>(runtime->outputTopkIdsBuffer());
  auto* outputTopkWeights = expertMajor ? nullptr : static_cast<float*>(runtime->outputTopkWeightsBuffer());
  runtime->dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::LatencyDispatchRequest{
      .output = dispatchOutput,
      .outputScales = fp8 ? buffers.outputScales.data() : nullptr,
      .outputSrcInfo = expertMajor ? buffers.srcInfo.data() : nullptr,
      .outputTopkIdx = outputTopkIdx,
      .outputTopkWeights = outputTopkWeights,
      .outputLayoutRange = expertMajor ? buffers.layoutRange.data() : nullptr,
      .outputCount = buffers.outputCount.data(),
      .input = buffers.input.data(),
      .topkIdx = buffers.topkIdx.data(),
      .topkWeights = buffers.topkWeights.data(),
      .numTokens = CorrectnessTokens,
      .hidden = CorrectnessHidden,
      .numTopk = NumTopk,
      .maxTokensPerRank = CorrectnessTokens,
      .numExperts = NumExperts,
      .invalidTokenExpertId = NumExperts,
      .dispatchLayout = layout,
      .dispatchDataType = dataType,
      .numBlocks = dispatchBlocks,
      .stream = stream,
  }});

  const void* expertOutput = dispatchOutput;
  if (fp8) {
    const int rows = NumRanks * CorrectnessTokens;
    const size_t elements = static_cast<size_t>(rows) * CorrectnessHidden;
    dequantizeExpertMajor<<<numBlocks(elements), Threads, 0, stream>>>(
        buffers.expertOutput.data(), static_cast<const Fp8E4M3*>(dispatchOutput), buffers.outputScales.data(), rows,
        CorrectnessHidden);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    expertOutput = buffers.expertOutput.data();
  } else if (!expertMajor) {
    auto* combineInput = static_cast<Bf16*>(runtime->combineInputBuffer());
    const bool directSend = combineMode == mscclpp::ep::CombineMode::DIRECT_SEND;
    const int rows = NumRanks * CorrectnessTokens;
    const size_t elements = static_cast<size_t>(rows) * CorrectnessHidden * (directSend ? NumTopk : 1);
    stageRankMajorExpertOutput<<<numBlocks(elements), Threads, 0, stream>>>(
        combineInput, static_cast<const Bf16*>(dispatchOutput), outputTopkIdx, outputTopkWeights, rank, rows,
        CorrectnessHidden, directSend);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    expertOutput = combineInput;
  }

  runtime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::LatencyCombineRequest{
      .output = buffers.output.data(),
      .input = expertOutput,
      .topkIdx = buffers.topkIdx.data(),
      .topkWeights = expertMajor ? buffers.topkWeights.data() : nullptr,
      .srcInfo = expertMajor ? buffers.srcInfo.data() : nullptr,
      .layoutRange = expertMajor ? buffers.layoutRange.data() : nullptr,
      .numTokens = CorrectnessTokens,
      .hidden = CorrectnessHidden,
      .numTopk = NumTopk,
      .maxTokensPerRank = CorrectnessTokens,
      .numExperts = NumExperts,
      .dispatchLayout = layout,
      .dispatchDataType = dataType,
      .combineMode = combineMode,
      .numBlocks = combineBlocks,
      .stream = stream,
  }});
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  std::string error = checkCounts(buffers.outputCount.data(), layout, CorrectnessTokens);
  if (error.empty()) {
    error = checkOutput(buffers.output.data(), rank, CorrectnessTokens, CorrectnessHidden, fp8 ? 1.0f : 0.0f);
  }
  assertCollectiveSuccess(error, label);

  communicator.bootstrap()->barrier();
  runtime.reset();
  communicator.bootstrap()->barrier();
}

void runGraphPerformance(mscclpp::Communicator& communicator, int rank, int dispatchBlocks, int combineBlocks,
                         mscclpp::ep::DispatchLayout layout, const std::string& perfLabel) {
  auto runtime =
      createRuntime(communicator, PerfTokens, PerfHidden, layout, mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE);
  ASSERT_TRUE(runtime->isAvailable());
  runtime->initialize();

  CudaStream stream;
  TestBuffers buffers(PerfTokens, PerfHidden);
  initializeTestBuffers(buffers, rank, PerfTokens, PerfHidden, stream);
  const bool expertMajor = layout == mscclpp::ep::DispatchLayout::EXPERT_MAJOR;
  void* dispatchOutput = runtime->dispatchOutputBuffer();
  void* combineInput = expertMajor ? dispatchOutput : runtime->combineInputBuffer();
  auto* outputTopkIdx = expertMajor ? nullptr : static_cast<int*>(runtime->outputTopkIdsBuffer());
  auto* outputTopkWeights = expertMajor ? nullptr : static_cast<float*>(runtime->outputTopkWeightsBuffer());

  auto dispatch = [&]() {
    runtime->dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::LatencyDispatchRequest{
        .output = dispatchOutput,
        .outputScales = nullptr,
        .outputSrcInfo = expertMajor ? buffers.srcInfo.data() : nullptr,
        .outputTopkIdx = outputTopkIdx,
        .outputTopkWeights = outputTopkWeights,
        .outputLayoutRange = expertMajor ? buffers.layoutRange.data() : nullptr,
        .outputCount = buffers.outputCount.data(),
        .input = buffers.input.data(),
        .topkIdx = buffers.topkIdx.data(),
        .topkWeights = buffers.topkWeights.data(),
        .numTokens = PerfTokens,
        .hidden = PerfHidden,
        .numTopk = NumTopk,
        .maxTokensPerRank = PerfTokens,
        .numExperts = NumExperts,
        .invalidTokenExpertId = NumExperts,
        .dispatchLayout = layout,
        .dispatchDataType = mscclpp::ep::DispatchDataType::BF16,
        .numBlocks = dispatchBlocks,
        .stream = stream,
    }});
  };
  auto combine = [&]() {
    runtime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::LatencyCombineRequest{
        .output = buffers.output.data(),
        .input = combineInput,
        .topkIdx = buffers.topkIdx.data(),
        .topkWeights = expertMajor ? buffers.topkWeights.data() : nullptr,
        .srcInfo = expertMajor ? buffers.srcInfo.data() : nullptr,
        .layoutRange = expertMajor ? buffers.layoutRange.data() : nullptr,
        .numTokens = PerfTokens,
        .hidden = PerfHidden,
        .numTopk = NumTopk,
        .maxTokensPerRank = PerfTokens,
        .numExperts = NumExperts,
        .dispatchLayout = layout,
        .dispatchDataType = mscclpp::ep::DispatchDataType::BF16,
        .combineMode = mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE,
        .numBlocks = combineBlocks,
        .stream = stream,
    }});
  };
  auto dispatchCombine = [&]() {
    dispatch();
    combine();
  };
  const float expectedMultiplier = expertMajor ? 1.0f : static_cast<float>(NumTopk);

  for (int iteration = 0; iteration < NumWarmups; ++iteration) dispatchCombine();
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  assertCollectiveSuccess(checkOutput(buffers.output.data(), rank, PerfTokens, PerfHidden, 0.0f, expectedMultiplier),
                          perfLabel + " warmup");

  CudaGraph graph;
  graph.capture(stream, [&]() {
    for (int iteration = 0; iteration < PairsPerGraph; ++iteration) dispatchCombine();
  });
  for (int iteration = 0; iteration < NumWarmups; ++iteration) graph.launch(stream);
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  communicator.bootstrap()->barrier();
  cudaEvent_t start;
  cudaEvent_t end;
  MSCCLPP_CUDATHROW(cudaEventCreate(&start));
  MSCCLPP_CUDATHROW(cudaEventCreate(&end));
  MSCCLPP_CUDATHROW(cudaEventRecord(start, stream));
  for (int iteration = 0; iteration < NumGraphReplays; ++iteration) graph.launch(stream);
  MSCCLPP_CUDATHROW(cudaEventRecord(end, stream));
  MSCCLPP_CUDATHROW(cudaEventSynchronize(end));

  float elapsedMs;
  MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedMs, start, end));
  MSCCLPP_CUDATHROW(cudaEventDestroy(start));
  MSCCLPP_CUDATHROW(cudaEventDestroy(end));
  assertCollectiveSuccess(checkOutput(buffers.output.data(), rank, PerfTokens, PerfHidden, 0.0f, expectedMultiplier),
                          perfLabel + " replay");

  const double localMicroseconds = static_cast<double>(elapsedMs) * 1000.0 / (NumGraphReplays * PairsPerGraph);
  double maxMicroseconds = 0.0;
  MPI_Allreduce(&localMicroseconds, &maxMicroseconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    ::mscclpp::test::reportPerfResult(perfLabel, maxMicroseconds, "us/iter");
  }

  communicator.bootstrap()->barrier();
  graph.reset();
  runtime.reset();
  communicator.bootstrap()->barrier();
}

}  // namespace

class MoERuntimeTest : public CommunicatorTestBase {
 protected:
  void SetUp() override {
    if (gEnv->worldSize != NumRanks || gEnv->nRanksPerNode != NumRanks) {
      SKIP_TEST() << "MoE runtime tests require exactly eight GPUs on one node";
    }

    const int localRank = rankToLocalRank(gEnv->rank);
    MSCCLPP_CUDATHROW(cudaSetDevice(localRank));
    int major;
    MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, localRank));
    if (major < 9) {
      SKIP_TEST() << "MoE runtime tests require SM90 or newer";
    }
    int numSms;
    MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, localRank));
    dispatchBlocks_ = std::min(numSms, 130);
    combineBlocks_ = std::min(numSms, 128);
    ASSERT_GE(dispatchBlocks_, NumRanks + 2);
    ASSERT_GT(combineBlocks_, 1);

    setNumRanksToUse(NumRanks);
    CommunicatorTestBase::SetUp();
  }

  int dispatchBlocks_ = 0;
  int combineBlocks_ = 0;
};

TEST(MoERuntimeTest, InitializationAndModeValidation) {
  bool rejectedThroughput = false;
  try {
    auto unsupported = std::make_unique<mscclpp::ep::MoERuntime>(
        *communicator, mscclpp::ep::MoEMode::THROUGHPUT, CorrectnessTokens, CorrectnessHidden, NumExperts, NumTopk);
  } catch (const std::invalid_argument&) {
    rejectedThroughput = true;
  }
  ASSERT_TRUE(rejectedThroughput);

  auto runtime = createRuntime(*communicator, CorrectnessTokens, CorrectnessHidden,
                               mscclpp::ep::DispatchLayout::RANK_MAJOR, mscclpp::ep::CombineMode::DIRECT_SEND);
  ASSERT_TRUE(runtime->mode() == mscclpp::ep::MoEMode::LATENCY);
  ASSERT_TRUE(runtime->isAvailable());
  ASSERT_EQ(runtime->rank(), gEnv->rank);
  ASSERT_EQ(runtime->numRanks(), NumRanks);
  ASSERT_EQ(runtime->numNvlRanks(), NumRanks);
  ASSERT_EQ(runtime->numRanksPerIpcDomain(), NumRanks);
  runtime->initialize();

  ASSERT_NE(runtime->dispatchOutputBuffer(), nullptr);
  ASSERT_NE(runtime->outputTopkIdsBuffer(), nullptr);
  ASSERT_NE(runtime->outputTopkWeightsBuffer(), nullptr);
  ASSERT_NE(runtime->combineInputBuffer(), nullptr);
  ASSERT_NE(runtime->combineInputBuffer(), runtime->dispatchOutputBuffer());

  bool rejectedDispatch = false;
  try {
    runtime->dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::ThroughputDispatchRequest{}});
  } catch (const std::invalid_argument&) {
    rejectedDispatch = true;
  }
  ASSERT_TRUE(rejectedDispatch);

  bool rejectedCombine = false;
  try {
    runtime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::ThroughputCombineRequest{}});
  } catch (const std::invalid_argument&) {
    rejectedCombine = true;
  }
  ASSERT_TRUE(rejectedCombine);

  communicator->bootstrap()->barrier();
  runtime.reset();
  communicator->bootstrap()->barrier();
}

TEST(MoERuntimeTest, DispatchCombineCorrectness) {
  for (const auto combineMode : {mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE, mscclpp::ep::CombineMode::DIRECT_SEND}) {
    runCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                       mscclpp::ep::DispatchLayout::EXPERT_MAJOR, combineMode, mscclpp::ep::DispatchDataType::BF16);
    runCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                       mscclpp::ep::DispatchLayout::EXPERT_MAJOR, combineMode, mscclpp::ep::DispatchDataType::FP8_E4M3);
    runCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                       mscclpp::ep::DispatchLayout::RANK_MAJOR, combineMode, mscclpp::ep::DispatchDataType::BF16);
  }
}

PERF_TEST(MoERuntimeTest, GraphDispatchCombine32Tokens) {
  runGraphPerformance(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                      mscclpp::ep::DispatchLayout::EXPERT_MAJOR, "expert-major 32 tokens/rank graph D+C");
}

PERF_TEST(MoERuntimeTest, GraphRankMajorDispatchCombine32Tokens) {
  runGraphPerformance(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                      mscclpp::ep::DispatchLayout::RANK_MAJOR, "rank-major 32 tokens/rank graph D+C");
}
