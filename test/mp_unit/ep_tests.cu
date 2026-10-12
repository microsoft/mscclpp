// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "exception.hpp"
#include "mp_unit_tests.hpp"

namespace {

using Bf16 = typename mscclpp::bf16x2::ElementType;
using Fp8E4M3 = typename mscclpp::f8_e4m3x2::ElementType;

constexpr int NumRanks = 8;
constexpr int NumExperts = 16 * NumRanks;
constexpr int NumTopk = 8;
constexpr int CorrectnessTokens = 8;
constexpr int CorrectnessHidden = 4096;
constexpr int PerfTokens = 32;
constexpr int PerfHidden = 7168;
constexpr int NumWarmups = 10;
constexpr int PairsPerGraph = 50;
constexpr int NumGraphReplays = 100;
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
  TestBuffers(int numTokens, int hidden, int numExperts = NumExperts)
      : input(static_cast<size_t>(numTokens) * hidden),
        output(static_cast<size_t>(numTokens) * hidden),
        expertOutput(static_cast<size_t>(numExperts) * numTokens * hidden),
        topkIdx(static_cast<size_t>(numTokens) * NumTopk),
        topkWeights(static_cast<size_t>(numTokens) * NumTopk),
        outputScales(static_cast<size_t>(numExperts) * numTokens * hidden / 128),
        srcInfo(static_cast<size_t>(numExperts) * numTokens),
        layoutRange(numExperts),
        outputCount(std::max(NumRanks, numExperts / NumRanks)) {}

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

std::array<int, NumTopk> routedExperts(int rank, int token, int numExperts) {
  std::vector<int> experts(numExperts);
  std::iota(experts.begin(), experts.end(), 0);
  std::seed_seq seed{42, rank, token};
  std::mt19937 random(seed);
  std::array<int, NumTopk> routes;
  std::sample(experts.begin(), experts.end(), routes.begin(), NumTopk, random);
  return routes;
}

MSCCLPP_HOST_DEVICE_INLINE float testTokenValue(int rank, int numTokens, int token, int valueModulo) {
  const int index = rank * numTokens + token;
  return static_cast<float>((valueModulo > 0 ? index % valueModulo : index) * NumTopk);
}

__global__ void initializeInputs(Bf16* input, float* topkWeights, int rank, int numTokens, int hidden,
                                 int valueModulo) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t inputElements = static_cast<size_t>(numTokens) * hidden;
  for (size_t index = thread; index < inputElements; index += stride) {
    const int token = static_cast<int>(index / hidden);
    input[index] = static_cast<Bf16>(testTokenValue(rank, numTokens, token, valueModulo));
  }

  const size_t routingElements = static_cast<size_t>(numTokens) * NumTopk;
  for (size_t index = thread; index < routingElements; index += stride) {
    topkWeights[index] = 1.0f / NumTopk;
  }
}

MSCCLPP_DEVICE_INLINE float fp8ToFloat(Fp8E4M3 value) {
  mscclpp::f8_e4m3x2 packed;
  packed.data[0] = value;
  packed.data[1] = value;
  return mscclpp::to<mscclpp::f32x2>(packed).data[0];
}

__global__ void initializeThroughputFp8(Fp8E4M3* input, float* scales, const Bf16* reference, int numTokens,
                                        int hidden) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t elements = static_cast<size_t>(numTokens) * hidden;
  mscclpp::f32x2 value;
  value.data[0] = 1.0f;
  value.data[1] = 1.0f;
  const auto one = mscclpp::to<mscclpp::f8_e4m3x2>(value).data[0];
  for (size_t index = thread; index < elements; index += stride) input[index] = one;
  const int scalesPerRow = hidden / 128;
  for (size_t index = thread; index < static_cast<size_t>(numTokens) * scalesPerRow; index += stride) {
    scales[index] = static_cast<float>(reference[index * 128]);
  }
}

__global__ void stageThroughputFp8ExpertOutput(Bf16* output, const Fp8E4M3* input, const float* scales,
                                               const float* rowWeights, int rows, int hidden) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t index = thread; index < static_cast<size_t>(rows) * hidden; index += stride) {
    const int row = static_cast<int>(index / hidden);
    const float weight = rowWeights[row];
    output[index] =
        weight == 0.0f
            ? static_cast<Bf16>(0.0f)
            : static_cast<Bf16>(fp8ToFloat(input[index]) *
                                scales[static_cast<size_t>(row) * (hidden / 128) + index % hidden / 128] * weight);
  }
}

__global__ void dequantizeExpertMajor(Bf16* output, const Fp8E4M3* input, const float* scales, int rowsPerExpert,
                                      int numLocalExperts, int hidden) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t elements = static_cast<size_t>(numLocalExperts) * rowsPerExpert * hidden;
  for (size_t index = thread; index < elements; index += stride) {
    const int row = static_cast<int>(index / hidden);
    const int hiddenIndex = static_cast<int>(index % hidden);
    const size_t scaleIndex =
        (static_cast<size_t>(row / rowsPerExpert) * (hidden / 128) + hiddenIndex / 128) * rowsPerExpert +
        row % rowsPerExpert;
    const float scale = scales[scaleIndex];
    output[index] = static_cast<Bf16>(fp8ToFloat(input[index]) * scale);
  }
}

__global__ void stageRankMajorExpertOutput(Bf16* output, const Bf16* input, const int* topkIdx,
                                           const float* topkWeights, int rank, int rows, int hidden, bool directSend,
                                           int numExperts) {
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
      const int expert = topkIdx[static_cast<size_t>(row) * NumTopk + topkLane];
      if (expert >= 0 && expert < numExperts && expert / (numExperts / NumRanks) == rank) {
        localWeight = 1.0f;
      }
    } else {
      for (int lane = 0; lane < NumTopk; ++lane) {
        const int expert = topkIdx[static_cast<size_t>(row) * NumTopk + lane];
        if (expert >= 0 && expert < numExperts && expert / (numExperts / NumRanks) == rank) {
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

__global__ void stageThroughputExpertOutput(Bf16* output, const Bf16* input, const float* rowWeights, int rows,
                                            int hidden) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t elements = static_cast<size_t>(rows) * hidden;
  for (size_t index = thread; index < elements; index += stride) {
    const int row = static_cast<int>(index / hidden);
    const float weight = rowWeights[row];
    output[index] =
        weight == 0.0f ? static_cast<Bf16>(0.0f) : static_cast<Bf16>(static_cast<float>(input[index]) * weight);
  }
}

int numBlocks(size_t elements) { return static_cast<int>(std::min<size_t>((elements + Threads - 1) / Threads, 4096)); }

void initializeTestBuffers(TestBuffers& buffers, int rank, int numTokens, int hidden, cudaStream_t stream,
                           int numExperts = NumExperts, int valueModulo = 0) {
  std::vector<int64_t> topkIdx(static_cast<size_t>(numTokens) * NumTopk);
  for (int token = 0; token < numTokens; ++token) {
    const auto experts = routedExperts(rank, token, numExperts);
    std::copy(experts.begin(), experts.end(), topkIdx.begin() + static_cast<size_t>(token) * NumTopk);
  }
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(buffers.topkIdx.data(), topkIdx.data(), topkIdx.size() * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
  const size_t elements = std::max(static_cast<size_t>(numTokens) * hidden, static_cast<size_t>(numTokens) * NumTopk);
  initializeInputs<<<numBlocks(elements), Threads, 0, stream>>>(buffers.input.data(), buffers.topkWeights.data(), rank,
                                                                numTokens, hidden, valueModulo);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.output.data(), 0, buffers.output.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.expertOutput.data(), 0, buffers.expertOutput.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.outputScales.data(), 0, buffers.outputScales.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.srcInfo.data(), 0, buffers.srcInfo.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.layoutRange.data(), 0, buffers.layoutRange.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.outputCount.data(), 0, buffers.outputCount.bytes(), stream));
  // Finish the host routing copy before its source vector is destroyed.
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}

std::string caseName(mscclpp::ep::DispatchLayout layout, mscclpp::ep::CombineMode combineMode,
                     mscclpp::ep::DispatchDataType dataType) {
  const char* layoutName = layout == mscclpp::ep::DispatchLayout::EXPERT_MAJOR ? "expert-major" : "rank-major";
  const char* combineName =
      combineMode == mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE ? "rank-local-reduce" : "direct-send";
  const char* dataTypeName = dataType == mscclpp::ep::DispatchDataType::BF16 ? "bf16" : "fp8-e4m3";
  return std::string(layoutName) + "/" + combineName + "/" + dataTypeName;
}

std::string checkOutput(const Bf16* output, int rank, int numTokens, int hidden, float tolerance, int valueModulo = 0) {
  std::vector<Bf16> hostOutput(static_cast<size_t>(numTokens) * hidden);
  MSCCLPP_CUDATHROW(cudaMemcpy(hostOutput.data(), output, hostOutput.size() * sizeof(Bf16), cudaMemcpyDeviceToHost));
  for (int token = 0; token < numTokens; ++token) {
    const float expected = testTokenValue(rank, numTokens, token, valueModulo);
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

std::string checkCounts(const int* outputCount, mscclpp::ep::DispatchLayout layout, int numTokens, int rank,
                        int numExperts) {
  const bool expertMajor = layout == mscclpp::ep::DispatchLayout::EXPERT_MAJOR;
  const int numLocalExperts = numExperts / NumRanks;
  const int countSize = expertMajor ? numLocalExperts : NumRanks;
  std::vector<int> hostCount(countSize);
  MSCCLPP_CUDATHROW(cudaMemcpy(hostCount.data(), outputCount, hostCount.size() * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> expected(countSize, 0);
  for (int source = 0; source < NumRanks; ++source) {
    for (int token = 0; token < numTokens; ++token) {
      bool received = false;
      for (int expert : routedExperts(source, token, numExperts)) {
        if (expert / numLocalExperts == rank) {
          if (expertMajor) ++expected[expert % numLocalExperts];
          received = true;
        }
      }
      if (!expertMajor && received) ++expected[source];
    }
  }
  for (int index = 0; index < countSize; ++index) {
    if (hostCount[index] != expected[index]) {
      std::ostringstream error;
      error << "output count " << index << ": expected " << expected[index] << ", got " << hostCount[index];
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

struct ThroughputExpectation {
  std::vector<float> rowWeights;
  std::vector<int> outputCount;
  int numRecvTokens = 0;
  int totalRows = 0;
};

ThroughputExpectation makeThroughputExpectation(mscclpp::ep::DispatchLayout layout, int rank,
                                                const std::array<int, NumRanks>& numTokensPerRank, int maxTokensPerRank,
                                                int numExperts) {
  const int numLocalExperts = numExperts / NumRanks;
  ThroughputExpectation expectation;
  if (layout == mscclpp::ep::DispatchLayout::TOKEN_MAJOR) {
    expectation.outputCount.assign(numLocalExperts, 0);
    expectation.rowWeights.reserve(static_cast<size_t>(NumRanks) * maxTokensPerRank);
  } else {
    expectation.outputCount.assign(NumRanks, 0);
    expectation.rowWeights.assign(static_cast<size_t>(NumRanks) * maxTokensPerRank, 0.0f);
  }

  std::vector<int> rankOffsets(NumRanks, 0);
  for (int source = 0; source < NumRanks; ++source) {
    for (int token = 0; token < numTokensPerRank[source]; ++token) {
      float localWeight = 0.0f;
      for (int expert : routedExperts(source, token, numExperts)) {
        if (expert >= 0 && expert < numExperts && expert / numLocalExperts == rank) {
          localWeight += 1.0f / NumTopk;
          if (layout == mscclpp::ep::DispatchLayout::TOKEN_MAJOR) {
            ++expectation.outputCount[expert % numLocalExperts];
          }
        }
      }
      if (localWeight == 0.0f) continue;
      if (layout == mscclpp::ep::DispatchLayout::TOKEN_MAJOR) {
        expectation.rowWeights.push_back(localWeight);
      } else {
        expectation.rowWeights[static_cast<size_t>(source) * maxTokensPerRank + rankOffsets[source]] = localWeight;
        ++expectation.outputCount[source];
        ++rankOffsets[source];
      }
      ++expectation.numRecvTokens;
    }
  }
  expectation.totalRows = layout == mscclpp::ep::DispatchLayout::TOKEN_MAJOR
                              ? expectation.numRecvTokens
                              : static_cast<int>(expectation.rowWeights.size());
  return expectation;
}

std::string checkThroughputCounts(const int* outputCount, const ThroughputExpectation& expectation) {
  std::vector<int> hostCount(expectation.outputCount.size());
  MSCCLPP_CUDATHROW(cudaMemcpy(hostCount.data(), outputCount, hostCount.size() * sizeof(int), cudaMemcpyDeviceToHost));
  for (size_t index = 0; index < hostCount.size(); ++index) {
    if (hostCount[index] != expectation.outputCount[index]) {
      std::ostringstream error;
      error << "throughput output count " << index << ": expected " << expectation.outputCount[index] << ", got "
            << hostCount[index];
      return error.str();
    }
  }
  return {};
}

std::unique_ptr<mscclpp::ep::MoERuntime> createRuntime(mscclpp::Communicator& communicator, int numTokens, int hidden,
                                                       mscclpp::ep::DispatchLayout layout,
                                                       mscclpp::ep::CombineMode combineMode,
                                                       int numExperts = NumExperts) {
  return std::make_unique<mscclpp::ep::MoERuntime>(communicator, mscclpp::ep::MoEMode::LATENCY, numTokens, hidden,
                                                   numExperts, NumTopk, layout, combineMode);
}

std::unique_ptr<mscclpp::ep::MoERuntime> createThroughputRuntime(mscclpp::Communicator& communicator, int numTokens,
                                                                 int hidden, mscclpp::ep::DispatchLayout layout,
                                                                 int numExperts = NumExperts) {
  return std::make_unique<mscclpp::ep::MoERuntime>(communicator, mscclpp::ep::MoEMode::THROUGHPUT, numTokens, hidden,
                                                   numExperts, NumTopk, layout);
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
  const auto handle = runtime->dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::LatencyDispatchRequest{
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
      .maxTokensPerRank = CorrectnessTokens,
      .invalidTokenExpertId = NumExperts,
      .dispatchDataType = dataType,
      .numBlocks = dispatchBlocks,
      .stream = stream,
  }});

  const void* expertOutput = dispatchOutput;
  if (fp8) {
    const int rows = NumRanks * CorrectnessTokens;
    const int numLocalExperts = NumExperts / NumRanks;
    const size_t elements = static_cast<size_t>(numLocalExperts) * rows * CorrectnessHidden;
    dequantizeExpertMajor<<<numBlocks(elements), Threads, 0, stream>>>(
        buffers.expertOutput.data(), static_cast<const Fp8E4M3*>(dispatchOutput), buffers.outputScales.data(), rows,
        numLocalExperts, CorrectnessHidden);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    expertOutput = buffers.expertOutput.data();
  } else if (!expertMajor) {
    auto* combineInput = static_cast<Bf16*>(runtime->combineInputBuffer());
    const bool directSend = combineMode == mscclpp::ep::CombineMode::DIRECT_SEND;
    const int rows = NumRanks * CorrectnessTokens;
    const size_t elements = static_cast<size_t>(rows) * CorrectnessHidden * (directSend ? NumTopk : 1);
    stageRankMajorExpertOutput<<<numBlocks(elements), Threads, 0, stream>>>(
        combineInput, static_cast<const Bf16*>(dispatchOutput), outputTopkIdx, outputTopkWeights, rank, rows,
        CorrectnessHidden, directSend, NumExperts);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    expertOutput = combineInput;
  }

  runtime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::LatencyCombineRequest{
      .output = buffers.output.data(),
      .input = expertOutput,
      .handle = handle,
      .numBlocks = combineBlocks,
      .stream = stream,
  }});
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  std::string error = checkCounts(buffers.outputCount.data(), layout, CorrectnessTokens, rank, NumExperts);
  if (error.empty()) {
    error = checkOutput(buffers.output.data(), rank, CorrectnessTokens, CorrectnessHidden, fp8 ? 1.0f : 0.0f);
  }
  assertCollectiveSuccess(error, label);

  runtime.reset();
  communicator.bootstrap()->barrier();
}

struct ThroughputTestCase {
  bool usePreparation = false;
  int numTokens = CorrectnessTokens;
  int hidden = CorrectnessHidden;
  int runtimeCapacity = CorrectnessTokens;
  mscclpp::ep::DispatchDataType dataType = mscclpp::ep::DispatchDataType::BF16;
  bool varyTokensByRank = false;
};

void runThroughputCorrectnessCase(mscclpp::Communicator& communicator, int rank, int dispatchBlocks, int combineBlocks,
                                  mscclpp::ep::DispatchLayout layout, ThroughputTestCase testCase = {}) {
  const bool prepared = testCase.usePreparation;
  const int maxTokensPerRank = testCase.numTokens;
  std::array<int, NumRanks> numTokensPerRank;
  for (int source = 0; source < NumRanks; ++source) {
    numTokensPerRank[source] = testCase.varyTokensByRank ? source + 1 : testCase.numTokens;
  }
  const int numTokens = numTokensPerRank[rank];
  const int hidden = testCase.hidden;
  const bool useFp8 = testCase.dataType == mscclpp::ep::DispatchDataType::FP8_E4M3;
  const int numScales = useFp8 ? hidden / 128 : 0;
  // Keep larger synthetic inputs exactly representable after BF16 expert weighting.
  const int valueModulo = numTokens > CorrectnessTokens ? 32 : 0;
  const char* layoutName = layout == mscclpp::ep::DispatchLayout::TOKEN_MAJOR ? "token-major" : "rank-major";
  const std::string label = std::string("throughput/") + layoutName + (useFp8 ? "/fp8-e4m3" : "/bf16");
  auto runtime = createThroughputRuntime(communicator, testCase.runtimeCapacity, hidden, layout);
  ASSERT_TRUE(runtime->isAvailable());
  runtime->initialize();

  CudaStream stream;
  TestBuffers buffers(numTokens, hidden);
  initializeTestBuffers(buffers, rank, numTokens, hidden, stream, NumExperts, valueModulo);
  mscclpp::GpuBuffer<Fp8E4M3> fp8Input(useFp8 ? static_cast<size_t>(numTokens) * hidden : 1);
  mscclpp::GpuBuffer<float> inputScales(useFp8 ? static_cast<size_t>(numTokens) * numScales : 1);
  if (useFp8) {
    initializeThroughputFp8<<<numBlocks(static_cast<size_t>(numTokens) * hidden), Threads, 0, stream>>>(
        fp8Input.data(), inputScales.data(), buffers.input.data(), numTokens, hidden);
    MSCCLPP_CUDATHROW(cudaGetLastError());
  }
  const void* input = useFp8 ? static_cast<const void*>(fp8Input.data()) : buffers.input.data();

  const ThroughputExpectation expectation =
      makeThroughputExpectation(layout, rank, numTokensPerRank, maxTokensPerRank, NumExperts);
  mscclpp::GpuBuffer<float> rowWeights(expectation.totalRows > 0 ? static_cast<size_t>(expectation.totalRows) : 1);
  if (expectation.totalRows > 0) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(rowWeights.data(), expectation.rowWeights.data(),
                                      sizeof(float) * static_cast<size_t>(expectation.totalRows),
                                      cudaMemcpyHostToDevice, stream));
  }

  const size_t elements = static_cast<size_t>(expectation.totalRows) * hidden;
  const size_t metadataElements = static_cast<size_t>(expectation.totalRows) * NumTopk;
  auto* receivedTopkIdx = static_cast<int*>(runtime->outputTopkIdsBuffer());
  auto* receivedTopkWeights = static_cast<float*>(runtime->outputTopkWeightsBuffer());
  auto* receivedScales = static_cast<float*>(runtime->outputScalesBuffer());
  void* dispatchOutput = runtime->dispatchOutputBuffer();
  auto* combineInput = static_cast<Bf16*>(runtime->combineInputBuffer());
  mscclpp::ep::PrepareHandle preparation;
  auto prepare = [&] {
    return runtime->prepare({.topkIdx = buffers.topkIdx.data(),
                             .numTokens = numTokens,
                             .maxTokensPerRank = maxTokensPerRank,
                             .numBlocks = dispatchBlocks,
                             .stream = stream,
                             .outputCount = buffers.outputCount.data()});
  };
  if (prepared) preparation = prepare();

  mscclpp::ep::ThroughputDispatchRequest request{
      .input = input,
      .inputScales = useFp8 ? inputScales.data() : nullptr,
      .topkIdx = buffers.topkIdx.data(),
      .topkWeights = buffers.topkWeights.data(),
      .numTokens = numTokens,
      .maxTokensPerRank = maxTokensPerRank,
      .dispatchDataType = testCase.dataType,
      .numBlocks = dispatchBlocks,
      .stream = stream,
      .prepareHandle = preparation,
  };
  auto operation = [&] {
    request.prepareHandle = preparation;
    const auto handle = runtime->dispatch(mscclpp::ep::DispatchRequest{request});

    if (expectation.totalRows > 0) {
      if (useFp8) {
        stageThroughputFp8ExpertOutput<<<numBlocks(elements), Threads, 0, stream>>>(
            combineInput, static_cast<const Fp8E4M3*>(dispatchOutput), receivedScales, rowWeights.data(),
            expectation.totalRows, hidden);
      } else {
        stageThroughputExpertOutput<<<numBlocks(elements), Threads, 0, stream>>>(
            combineInput, static_cast<const Bf16*>(dispatchOutput), rowWeights.data(), expectation.totalRows, hidden);
      }
      MSCCLPP_CUDATHROW(cudaGetLastError());
    }

    runtime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::ThroughputCombineRequest{
        .output = buffers.output.data(),
        .handle = handle,
        .numBlocks = combineBlocks,
        .stream = stream,
    }});
  };
  operation();
  if (prepared) operation();
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  std::string error = prepared ? checkThroughputCounts(buffers.outputCount.data(), expectation) : "";
  if (error.empty()) {
    error = checkOutput(buffers.output.data(), rank, numTokens, hidden, 0.0f, valueModulo);
  }
  assertCollectiveSuccess(error, label);

  std::vector<int> topkIdx(metadataElements);
  std::vector<float> topkWeights(metadataElements);
  std::vector<float> scales(static_cast<size_t>(expectation.totalRows) * numScales);
  if (metadataElements > 0) {
    MSCCLPP_CUDATHROW(
        cudaMemcpy(topkIdx.data(), receivedTopkIdx, metadataElements * sizeof(int), cudaMemcpyDeviceToHost));
    MSCCLPP_CUDATHROW(
        cudaMemcpy(topkWeights.data(), receivedTopkWeights, metadataElements * sizeof(float), cudaMemcpyDeviceToHost));
  }
  if (!scales.empty()) {
    MSCCLPP_CUDATHROW(cudaMemcpy(scales.data(), receivedScales, scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
  }
  int compactRow = 0;
  constexpr int LocalExperts = NumExperts / NumRanks;
  for (int source = 0; source < NumRanks; ++source) {
    int rankRow = 0;
    for (int token = 0; token < numTokensPerRank[source]; ++token) {
      const auto experts = routedExperts(source, token, NumExperts);
      const bool selected =
          std::any_of(experts.begin(), experts.end(), [&](int expert) { return expert / LocalExperts == rank; });
      if (!selected) continue;
      const int row =
          layout == mscclpp::ep::DispatchLayout::RANK_MAJOR ? source * maxTokensPerRank + rankRow++ : compactRow++;
      for (int topk = 0; topk < NumTopk; ++topk) {
        const int localExpert = experts[topk] / LocalExperts == rank ? experts[topk] % LocalExperts : -1;
        const size_t index = static_cast<size_t>(row) * NumTopk + topk;
        EXPECT_EQ(topkIdx[index], localExpert);
        EXPECT_EQ(topkWeights[index], localExpert >= 0 ? 1.0f / NumTopk : 0.0f);
      }
      for (int scale = 0; scale < numScales; ++scale) {
        EXPECT_EQ(scales[static_cast<size_t>(row) * numScales + scale],
                  testTokenValue(source, numTokensPerRank[source], token, valueModulo));
      }
    }
  }

  if (prepared) {
    MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.topkIdx.data(), 0xff, buffers.topkIdx.bytes(), stream));
    preparation = prepare();
    operation();
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    std::vector<int> counts(expectation.outputCount.size());
    std::vector<Bf16> output(static_cast<size_t>(numTokens) * hidden);
    MSCCLPP_CUDATHROW(
        cudaMemcpy(counts.data(), buffers.outputCount.data(), counts.size() * sizeof(int), cudaMemcpyDeviceToHost));
    MSCCLPP_CUDATHROW(
        cudaMemcpy(output.data(), buffers.output.data(), output.size() * sizeof(Bf16), cudaMemcpyDeviceToHost));
    EXPECT_TRUE(std::all_of(counts.begin(), counts.end(), [](int value) { return value == 0; }));
    EXPECT_TRUE(
        std::all_of(output.begin(), output.end(), [](Bf16 value) { return static_cast<float>(value) == 0.0f; }));
  }

  communicator.bootstrap()->barrier();
  runtime.reset();
  communicator.bootstrap()->barrier();
}

template <typename Operation>
bool rejectsEpRequest(Operation operation) {
  try {
    operation();
  } catch (const EPException&) {
    return true;
  }
  return false;
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
    return runtime->dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::LatencyDispatchRequest{
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
        .maxTokensPerRank = PerfTokens,
        .invalidTokenExpertId = NumExperts,
        .dispatchDataType = mscclpp::ep::DispatchDataType::BF16,
        .numBlocks = dispatchBlocks,
        .stream = stream,
    }});
  };
  auto combine = [&](const mscclpp::ep::DispatchHandle& handle) {
    runtime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::LatencyCombineRequest{
        .output = buffers.output.data(),
        .input = combineInput,
        .handle = handle,
        .numBlocks = combineBlocks,
        .stream = stream,
    }});
  };
  auto dispatchCombine = [&]() {
    const auto handle = dispatch();
    if (!expertMajor) {
      const int rows = NumRanks * PerfTokens;
      const size_t elements = static_cast<size_t>(rows) * PerfHidden;
      stageRankMajorExpertOutput<<<numBlocks(elements), Threads, 0, stream>>>(
          static_cast<Bf16*>(combineInput), static_cast<const Bf16*>(dispatchOutput), outputTopkIdx, outputTopkWeights,
          rank, rows, PerfHidden, false, NumExperts);
      MSCCLPP_CUDATHROW(cudaGetLastError());
    }
    combine(handle);
  };
  for (int iteration = 0; iteration < NumWarmups; ++iteration) dispatchCombine();
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  assertCollectiveSuccess(checkOutput(buffers.output.data(), rank, PerfTokens, PerfHidden, 0.0f),
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
  assertCollectiveSuccess(checkOutput(buffers.output.data(), rank, PerfTokens, PerfHidden, 0.0f),
                          perfLabel + " replay");

  const double localMicroseconds = static_cast<double>(elapsedMs) * 1000.0 / (NumGraphReplays * PairsPerGraph);
  double maxMicroseconds = 0.0;
  MPI_Allreduce(&localMicroseconds, &maxMicroseconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) {
    ::mscclpp::test::reportPerfResult(perfLabel, maxMicroseconds, "us/iter");
  }

  graph.reset();
  runtime.reset();
  communicator.bootstrap()->barrier();
}

void runThroughputPerformance(mscclpp::ep::MoERuntime& runtime, mscclpp::Communicator& communicator, int rank,
                              int numTokens, int dispatchBlocks, int combineBlocks,
                              mscclpp::ep::DispatchDataType dataType, cudaStream_t stream) {
  using namespace mscclpp::ep;
  constexpr int Warmups = 3;
  constexpr int Iterations = 200;
  const bool fp8 = dataType == DispatchDataType::FP8_E4M3;

  const size_t elements = static_cast<size_t>(numTokens) * PerfHidden;
  mscclpp::GpuBuffer<Bf16> input(elements);
  mscclpp::GpuBuffer<Bf16> output(elements);
  mscclpp::GpuBuffer<Fp8E4M3> fp8Input(fp8 ? elements : 1);
  mscclpp::GpuBuffer<float> inputScales(fp8 ? static_cast<size_t>(numTokens) * (PerfHidden / 128) : 1);
  std::vector<int64_t> routes(static_cast<size_t>(numTokens) * NumTopk);
  for (int token = 0; token < numTokens; ++token) {
    const auto experts = routedExperts(rank, token, NumExperts);
    std::copy(experts.begin(), experts.end(), routes.begin() + static_cast<size_t>(token) * NumTopk);
  }
  mscclpp::GpuBuffer<int64_t> topkIdx(routes.size());
  mscclpp::GpuBuffer<float> topkWeights(routes.size());
  mscclpp::gpuMemcpy<int64_t>(topkIdx.data(), routes.data(), routes.size(), cudaMemcpyHostToDevice);
  initializeInputs<<<numBlocks(elements), Threads, 0, stream>>>(input.data(), topkWeights.data(), rank, numTokens,
                                                                PerfHidden, 32);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  if (fp8) {
    initializeThroughputFp8<<<numBlocks(elements), Threads, 0, stream>>>(fp8Input.data(), inputScales.data(),
                                                                         input.data(), numTokens, PerfHidden);
    MSCCLPP_CUDATHROW(cudaGetLastError());
  }
  // Prepare representative BF16 expert results outside timing, including for FP8 dispatch.
  auto* expertInput = static_cast<Bf16*>(runtime.combineInputBuffer());
  for (int peer = 0; peer < NumRanks; ++peer) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(expertInput + static_cast<size_t>(peer) * elements, input.data(),
                                      elements * sizeof(Bf16), cudaMemcpyDeviceToDevice, stream));
  }
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  ThroughputDispatchRequest request{
      .input = fp8 ? static_cast<const void*>(fp8Input.data()) : input.data(),
      .inputScales = fp8 ? inputScales.data() : nullptr,
      .topkIdx = topkIdx.data(),
      .topkWeights = topkWeights.data(),
      .numTokens = numTokens,
      .maxTokensPerRank = numTokens,
      .dispatchDataType = dataType,
      .numBlocks = dispatchBlocks,
      .stream = stream,
      .prepareHandle = {},
  };
  auto operation = [&] {
    const auto handle = runtime.dispatch(DispatchRequest{request});
    runtime.combine(CombineRequest{ThroughputCombineRequest{
        .output = output.data(),
        .handle = handle,
        .numBlocks = combineBlocks,
        .stream = stream,
    }});
  };
  for (const bool cached : {false, true}) {
    if (cached) {
      request.prepareHandle = runtime.prepare({topkIdx.data(), numTokens, numTokens, dispatchBlocks, stream});
    }
    for (int iteration = 0; iteration < Warmups; ++iteration) operation();
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

    communicator.bootstrap()->barrier();

    cudaEvent_t start;
    cudaEvent_t end;
    MSCCLPP_CUDATHROW(cudaEventCreate(&start));
    MSCCLPP_CUDATHROW(cudaEventCreate(&end));
    MSCCLPP_CUDATHROW(cudaEventRecord(start, stream));
    for (int iteration = 0; iteration < Iterations; ++iteration) operation();
    MSCCLPP_CUDATHROW(cudaEventRecord(end, stream));
    MSCCLPP_CUDATHROW(cudaEventSynchronize(end));
    float elapsedMs;
    MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedMs, start, end));
    MSCCLPP_CUDATHROW(cudaEventDestroy(start));
    MSCCLPP_CUDATHROW(cudaEventDestroy(end));

    const double localMicroseconds = static_cast<double>(elapsedMs) * 1000.0 / Iterations;
    double maxMicroseconds = 0.0;
    MPI_Allreduce(&localMicroseconds, &maxMicroseconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const std::string label =
          std::string("throughput-dispatch-combine/eager/token-major/format=") + (fp8 ? "fp8-e4m3" : "bf16") +
          "/tokens=" + std::to_string(numTokens) + "/dispatch_blocks=" + std::to_string(dispatchBlocks) +
          "/combine_blocks=" + std::to_string(combineBlocks) + "/preparation=" + (cached ? "cached" : "automatic");
      ::mscclpp::test::reportPerfResult(label, maxMicroseconds, "us/D+C");
    }
    communicator.bootstrap()->barrier();
  }
}

int mxfp8TestExpert(int source, int token, int lane, int ranks, int topk, int generation) {
  if (token % 7 == 0 || (token % 2 == 0 && lane == topk - 1)) return -1;
  const int destination = (source + token + lane / 2 + generation) % ranks;
  return destination * 32 + (token + generation + (token % 3 == 0 ? 0 : lane % 2)) % 32;
}

Bf16 mxfp8TestValue(int source, int token, int column, int generation) {
  // An exactly representable token marker identifies unordered latency rows.
  if (column < 32) {
    const int value = column < 3 ? (token >> (4 * column)) & 15 : (column == 3 ? generation & 15 : 0);
    return static_cast<Bf16>(column == 31 ? 16.0f : static_cast<float>(value));
  }
  constexpr std::array<uint16_t, 12> Boundaries{0x0000, 0x0001, 0x007f, 0x0080, 0x3f80, 0x3fdf,
                                                0x3fe0, 0x3fe1, 0x43df, 0x43e0, 0x43e1, 0x7f7f};
  const int block = column / 32;
  const int selector = (source * 3 + token + block + generation) % 16;
  const uint16_t maximum = selector < static_cast<int>(Boundaries.size())
                               ? Boundaries[selector]
                               : (token * 131 + block * 17 + source * 11 + generation * 271) % 0x7f80;
  const int decrement = column % 32 == (token + block) % 32 ? 0 : column % 32;
  const uint16_t magnitude = maximum > decrement ? maximum - decrement : 0;
  return std::bit_cast<Bf16>(static_cast<uint16_t>(magnitude | ((column + token) % 2 ? 0x8000 : 0)));
}

uint8_t referenceMxfp8Scale(const std::array<Bf16, 32>& values) {
  double maximum = 0.0;
  for (const auto value : values) maximum = std::max(maximum, std::abs(static_cast<double>(static_cast<float>(value))));
  if (maximum == 0.0) return 127;
  return static_cast<uint8_t>(127 + std::clamp(static_cast<int>(std::ceil(std::log2(maximum / 448.0))), -126, 127));
}

__global__ void stageMxfp8RankMajorOutput(Bf16* output, const void* input, const uint8_t* scales, const int* ids,
                                          const float* weights, const int* counts, int rows, int capacity, int hidden,
                                          int topk, int invalidExpert, bool quantized, bool directSend) {
  const size_t elements = static_cast<size_t>(rows) * hidden * (directSend ? topk : 1);
  for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const size_t route = index / hidden;
    const int row = directSend ? route / topk : route;
    float weight = 0.0f;
    if (row % capacity < counts[row / capacity]) {
      if (directSend) {
        const int expert = ids[static_cast<size_t>(row) * topk + route % topk];
        weight = expert >= 0 && expert < invalidExpert ? 1.0f : 0.0f;
      } else {
        for (int lane = 0; lane < topk; ++lane) {
          const int expert = ids[static_cast<size_t>(row) * topk + lane];
          if (expert >= 0 && expert < invalidExpert) weight += weights[static_cast<size_t>(row) * topk + lane];
        }
      }
    }
    float value = 0.0f;
    if (weight != 0.0f) {
      value = quantized ? fp8ToFloat(static_cast<const Fp8E4M3*>(input)[static_cast<size_t>(row) * hidden]) *
                              ldexpf(1.0f, int(scales[static_cast<size_t>(row) * (hidden / 32)]) - 127)
                        : static_cast<float>(static_cast<const Bf16*>(input)[static_cast<size_t>(row) * hidden]);
      value = (value + 1.0f) * weight;
    }
    output[index] = static_cast<Bf16>(value);
  }
}

void runMxfp8Correctness(mscclpp::Communicator& communicator, mscclpp::ep::CombineMode combineMode, int hidden,
                         int tokens, int topk, bool useGraph) {
  using namespace mscclpp::ep;
  const int rank = communicator.bootstrap()->getRank();
  const int ranks = communicator.bootstrap()->getNranks();
  const int experts = ranks * 32;
  const int capacity = 2 * tokens + 3;
  const int blocks = std::max(ranks + 2, 24);
  MoERuntime runtime(communicator, MoEMode::LATENCY, capacity, hidden, experts, topk, DispatchLayout::RANK_MAJOR,
                     combineMode);
  ASSERT_TRUE(runtime.isAvailable());
  runtime.initialize();
  ASSERT_NE(runtime.dispatchOutputBuffer(), runtime.combineInputBuffer());
  ASSERT_NE(runtime.outputScalesBuffer(), nullptr);
  CudaStream stream;
  mscclpp::GpuBuffer<Bf16> input(static_cast<size_t>(capacity) * hidden);
  mscclpp::GpuBuffer<Bf16> output(input.nelems());
  mscclpp::GpuBuffer<int64_t> routes(static_cast<size_t>(capacity) * topk);
  mscclpp::GpuBuffer<float> weights(routes.nelems());
  mscclpp::GpuBuffer<int> counts(ranks);
  const bool directSend = combineMode == CombineMode::DIRECT_SEND;

  for (int iteration = 0; iteration < 5; ++iteration) {
    const bool quantized = iteration != 0;
    const bool useWeights = iteration % 2 == 0;
    const int activeCapacity = tokens + iteration % 2;
    const int rows = ranks * activeCapacity;
    std::vector<int> sourceCounts(ranks);
    for (int source = 0; source < ranks; ++source) {
      sourceCounts[source] = iteration == 2 ? 0 : (iteration == 3 ? std::max(1, tokens / 2) : tokens);
      if (iteration == 1) sourceCounts[source] = source == 0 && ranks > 1 ? 0 : std::max(0, tokens - source % 3);
    }
    const int numTokens = sourceCounts[rank];
    std::vector<Bf16> hostInput(static_cast<size_t>(numTokens) * hidden);
    std::vector<int64_t> hostRoutes(static_cast<size_t>(numTokens) * topk);
    std::vector<float> hostWeights(hostRoutes.size(), 0.125f);
    for (int token = 0; token < numTokens; ++token) {
      for (int lane = 0; lane < topk; ++lane) {
        hostRoutes[static_cast<size_t>(token) * topk + lane] =
            mxfp8TestExpert(rank, token, lane, ranks, topk, iteration);
      }
    }
    if (numTokens > 0) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(routes.data(), hostRoutes.data(), hostRoutes.size() * sizeof(int64_t),
                                        cudaMemcpyHostToDevice, stream));
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(weights.data(), hostWeights.data(), hostWeights.size() * sizeof(float),
                                        cudaMemcpyHostToDevice, stream));
    }
    const LatencyDispatchRequest request{
        .output = runtime.dispatchOutputBuffer(),
        .outputScales = quantized ? runtime.outputScalesBuffer() : nullptr,
        .outputSrcInfo = nullptr,
        .outputTopkIdx = static_cast<int*>(runtime.outputTopkIdsBuffer()),
        .outputTopkWeights = static_cast<float*>(runtime.outputTopkWeightsBuffer()),
        .outputLayoutRange = nullptr,
        .outputCount = counts.data(),
        .input = numTokens == 0 ? nullptr : input.data(),
        .topkIdx = numTokens == 0 ? nullptr : routes.data(),
        .topkWeights = useWeights ? weights.data() : nullptr,
        .numTokens = numTokens,
        .maxTokensPerRank = activeCapacity,
        .invalidTokenExpertId = experts,
        .dispatchDataType = quantized ? DispatchDataType::MXFP8_E4M3 : DispatchDataType::BF16,
        .numBlocks = blocks,
        .stream = stream,
    };
    auto operation = [&] {
      const auto handle = runtime.dispatch(DispatchRequest{request});
      const size_t elements = static_cast<size_t>(rows) * hidden * (directSend ? topk : 1);
      stageMxfp8RankMajorOutput<<<numBlocks(elements), Threads, 0, stream>>>(
          static_cast<Bf16*>(runtime.combineInputBuffer()), request.output,
          static_cast<const uint8_t*>(runtime.outputScalesBuffer()), request.outputTopkIdx, request.outputTopkWeights,
          counts.data(), rows, activeCapacity, hidden, topk, experts, quantized, directSend);
      MSCCLPP_CUDATHROW(cudaGetLastError());
      runtime.combine(
          CombineRequest{LatencyCombineRequest{output.data(), runtime.combineInputBuffer(), handle, blocks, stream}});
    };
    CudaGraph graph;
    if (useGraph) {
      // Latency's existing host epoch protocol needs distinct epochs within a replay.
      graph.capture(stream, [&] {
        operation();
        operation();
      });
    }
    for (int replay = 0; replay < (useGraph ? 3 : 2); ++replay) {
      const int generation = iteration + replay * 5;
      for (int token = 0; token < numTokens; ++token) {
        for (int column = 0; column < hidden; ++column) {
          hostInput[static_cast<size_t>(token) * hidden + column] = mxfp8TestValue(rank, token, column, generation);
        }
      }
      if (numTokens > 0) {
        MSCCLPP_CUDATHROW(cudaMemcpyAsync(input.data(), hostInput.data(), hostInput.size() * sizeof(Bf16),
                                          cudaMemcpyHostToDevice, stream));
      }
      if (useGraph)
        graph.launch(stream);
      else
        operation();
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

      std::vector<int> receivedCounts(ranks);
      std::vector<int> receivedIds(static_cast<size_t>(rows) * topk);
      std::vector<float> receivedWeights(receivedIds.size());
      std::vector<uint8_t> received(static_cast<size_t>(rows) * hidden * (quantized ? 1 : sizeof(Bf16)));
      std::vector<uint8_t> receivedScales(static_cast<size_t>(rows) * (hidden / 32));
      mscclpp::gpuMemcpy<int>(receivedCounts.data(), counts.data(), ranks, cudaMemcpyDeviceToHost);
      mscclpp::gpuMemcpy<int>(receivedIds.data(), request.outputTopkIdx, receivedIds.size(), cudaMemcpyDeviceToHost);
      mscclpp::gpuMemcpy<float>(receivedWeights.data(), request.outputTopkWeights, receivedWeights.size(),
                                cudaMemcpyDeviceToHost);
      MSCCLPP_CUDATHROW(cudaMemcpy(received.data(), request.output, received.size(), cudaMemcpyDeviceToHost));
      if (quantized) {
        MSCCLPP_CUDATHROW(
            cudaMemcpy(receivedScales.data(), request.outputScales, receivedScales.size(), cudaMemcpyDeviceToHost));
      }
      std::string error;
      auto check = [&](bool condition, std::string_view message) {
        if (!condition && error.empty()) error = message;
      };
      for (int source = 0; source < ranks; ++source) {
        std::vector<bool> expected(sourceCounts[source], false), seen(sourceCounts[source], false);
        for (int token = 0; token < sourceCounts[source]; ++token) {
          for (int lane = 0; lane < topk; ++lane) {
            const int expert = mxfp8TestExpert(source, token, lane, ranks, topk, iteration);
            if (expert >= 0 && expert / 32 == rank) expected[token] = true;
          }
        }
        check(receivedCounts[source] == std::count(expected.begin(), expected.end(), true), "rank count mismatch");
        for (int slot = 0; slot < activeCapacity; ++slot) {
          const size_t row = static_cast<size_t>(source) * activeCapacity + slot;
          if (slot >= receivedCounts[source]) {
            for (int lane = 0; lane < topk; ++lane) {
              check(receivedIds[row * topk + lane] == experts && receivedWeights[row * topk + lane] == 0.0f,
                    "padding metadata mismatch");
            }
            continue;
          }
          auto marker = [&](int column) {
            if (quantized) {
              return static_cast<float>(std::bit_cast<Fp8E4M3>(received[row * hidden + column])) *
                     std::ldexp(1.0f, int(receivedScales[row * (hidden / 32)]) - 127);
            }
            Bf16 value;
            std::memcpy(&value, received.data() + (row * hidden + column) * sizeof(Bf16), sizeof(Bf16));
            return static_cast<float>(value);
          };
          const float tokenValue = marker(0) + 16.0f * marker(1) + 256.0f * marker(2);
          if (!std::isfinite(tokenValue) || tokenValue < 0 || tokenValue >= sourceCounts[source]) {
            check(false, "invalid token marker");
            continue;
          }
          const int token = static_cast<int>(tokenValue);
          check(expected[token] && !seen[token], "missing or duplicated token");
          seen[token] = true;
          check(marker(3) == (generation & 15), "stale input after replay");
          for (int lane = 0; lane < topk; ++lane) {
            const int expert = mxfp8TestExpert(source, token, lane, ranks, topk, iteration);
            const bool local = expert >= 0 && expert / 32 == rank;
            const int expectedId = local ? expert : experts;
            check(receivedIds[row * topk + lane] == expectedId, "routed expert mismatch");
            check(receivedWeights[row * topk + lane] == (local ? (useWeights ? 0.125f : 1.0f) : 0.0f),
                  "routed weight mismatch");
          }
          for (int block = 0; block < hidden / 32; ++block) {
            std::array<Bf16, 32> reference;
            for (int element = 0; element < 32; ++element) {
              reference[element] = mxfp8TestValue(source, token, block * 32 + element, generation);
            }
            const uint8_t scale = referenceMxfp8Scale(reference);
            if (quantized) check(receivedScales[row * (hidden / 32) + block] == scale, "E8M0 scale mismatch");
            for (int element = 0; element < 32; ++element) {
              const size_t index = row * hidden + block * 32 + element;
              if (quantized) {
                const float scaled = static_cast<float>(reference[element]) * std::ldexp(1.0f, 127 - int(scale));
                const uint8_t value = std::bit_cast<uint8_t>(static_cast<Fp8E4M3>(scaled));
                if (received[index] != value) {
                  check(false, "E4M3 byte mismatch at row " + std::to_string(row) + " column " +
                                   std::to_string(block * 32 + element));
                }
              } else {
                check(std::memcmp(received.data() + index * sizeof(Bf16), &reference[element], sizeof(Bf16)) == 0,
                      "BF16 regression");
              }
            }
          }
        }
        check(seen == expected, "received token set mismatch");
      }
      std::vector<Bf16> combined(hostInput.size());
      if (numTokens > 0)
        mscclpp::gpuMemcpy<Bf16>(combined.data(), output.data(), combined.size(), cudaMemcpyDeviceToHost);
      for (int token = 0; token < numTokens; ++token) {
        int validRoutes = 0;
        for (int lane = 0; lane < topk; ++lane) {
          validRoutes += mxfp8TestExpert(rank, token, lane, ranks, topk, iteration) >= 0;
        }
        const float expected = ((token & 15) + 1) * validRoutes * (useWeights ? 0.125f : 1.0f);
        for (int column = 0; column < hidden; ++column) {
          check(static_cast<float>(combined[static_cast<size_t>(token) * hidden + column]) == expected,
                "BF16 combine mismatch");
        }
      }
      assertCollectiveSuccess(
          error, "rank-major MXFP8 hidden=" + std::to_string(hidden) + " iteration=" + std::to_string(iteration));
    }
    communicator.bootstrap()->barrier();
  }
  communicator.bootstrap()->barrier();
}

void runMxfp8Performance(mscclpp::Communicator& communicator, int tokens) {
  using namespace mscclpp::ep;
  constexpr int Hidden = 4096;
  constexpr int Pairs = 16;
  constexpr int Samples = 5;
  constexpr int Replays = 20;
  const int rank = communicator.bootstrap()->getRank();
  const int ranks = communicator.bootstrap()->getNranks();
  const int experts = ranks * 32;
  int device, sms;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  const int blocks = std::min(sms, 130);
  const int combineBlocks = std::min(sms, 128);
  MoERuntime runtime(communicator, MoEMode::LATENCY, tokens, Hidden, experts, NumTopk, DispatchLayout::RANK_MAJOR);
  runtime.initialize();
  CudaStream stream;
  mscclpp::GpuBuffer<Bf16> input(static_cast<size_t>(tokens) * Hidden);
  mscclpp::GpuBuffer<Bf16> output(input.nelems());
  mscclpp::GpuBuffer<int64_t> routes(static_cast<size_t>(tokens) * NumTopk);
  mscclpp::GpuBuffer<float> weights(routes.nelems());
  mscclpp::GpuBuffer<int> counts(ranks);
  std::vector<int64_t> hostRoutes(routes.nelems());
  for (int token = 0; token < tokens; ++token) {
    for (int lane = 0; lane < NumTopk; ++lane) {
      hostRoutes[static_cast<size_t>(token) * NumTopk + lane] = ((rank + token + lane / 2) % ranks) * 32 + lane;
    }
  }
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(routes.data(), hostRoutes.data(), hostRoutes.size() * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
  initializeInputs<<<numBlocks(input.nelems()), Threads, 0, stream>>>(input.data(), weights.data(), rank, tokens,
                                                                      Hidden, 32);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  auto combine = [&](const DispatchHandle& handle) {
    runtime.combine(CombineRequest{
        LatencyCombineRequest{output.data(), runtime.combineInputBuffer(), handle, combineBlocks, stream}});
  };
  std::array<cudaEvent_t, Pairs> starts{}, ends{};
  for (int pair = 0; pair < Pairs; ++pair) {
    MSCCLPP_CUDATHROW(cudaEventCreate(&starts[pair]));
    MSCCLPP_CUDATHROW(cudaEventCreate(&ends[pair]));
  }
  std::array<double, 2> timings{};
  for (int format = 0; format < 2; ++format) {
    const LatencyDispatchRequest request{
        .output = runtime.dispatchOutputBuffer(),
        .outputScales = format == 0 ? nullptr : runtime.outputScalesBuffer(),
        .outputSrcInfo = nullptr,
        .outputTopkIdx = static_cast<int*>(runtime.outputTopkIdsBuffer()),
        .outputTopkWeights = static_cast<float*>(runtime.outputTopkWeightsBuffer()),
        .outputLayoutRange = nullptr,
        .outputCount = counts.data(),
        .input = input.data(),
        .topkIdx = routes.data(),
        .topkWeights = weights.data(),
        .numTokens = tokens,
        .maxTokensPerRank = tokens,
        .invalidTokenExpertId = experts,
        .dispatchDataType = format == 0 ? DispatchDataType::BF16 : DispatchDataType::MXFP8_E4M3,
        .numBlocks = blocks,
        .stream = stream,
    };
    auto dispatch = [&] { return runtime.dispatch(DispatchRequest{request}); };
    for (int warmup = 0; warmup < 3; ++warmup) {
      const auto handle = dispatch();
      combine(handle);
    }
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    CudaGraph graph;
    graph.capture(stream, [&] {
      for (int pair = 0; pair < Pairs; ++pair) {
        MSCCLPP_CUDATHROW(cudaEventRecordWithFlags(starts[pair], stream, cudaEventRecordExternal));
        const auto handle = dispatch();
        MSCCLPP_CUDATHROW(cudaEventRecordWithFlags(ends[pair], stream, cudaEventRecordExternal));
        combine(handle);
      }
    });
    std::vector<double> measurements;
    for (int sample = 0; sample < Samples; ++sample) {
      communicator.bootstrap()->barrier();
      for (int replay = 0; replay < Replays; ++replay) graph.launch(stream);
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      for (int pair = 0; pair < Pairs; ++pair) {
        float milliseconds;
        MSCCLPP_CUDATHROW(cudaEventElapsedTime(&milliseconds, starts[pair], ends[pair]));
        measurements.push_back(milliseconds * 1000.0);
      }
    }
    std::sort(measurements.begin(), measurements.end());
    const double median = measurements[measurements.size() / 2];
    MPI_Allreduce(&median, &timings[format], 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }
  for (int pair = 0; pair < Pairs; ++pair) {
    MSCCLPP_CUDATHROW(cudaEventDestroy(starts[pair]));
    MSCCLPP_CUDATHROW(cudaEventDestroy(ends[pair]));
  }
  if (rank == 0) {
    const std::string label =
        "rank-major-dispatch/latency/tokens=" + std::to_string(tokens) + "/blocks=" + std::to_string(blocks);
    mscclpp::test::reportPerfResult(label + "/bf16", timings[0], "us");
    mscclpp::test::reportPerfResult(label + "/mxfp8", timings[1], "us");
    mscclpp::test::reportPerfResult(label + "/quantization-overhead", (timings[1] / timings[0] - 1.0) * 100.0, "%");
  }
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
  for (const int numTopk : {0, 9}) {
    ASSERT_TRUE(rejectsEpRequest([&] {
      mscclpp::ep::MoERuntime invalid(*communicator, mscclpp::ep::MoEMode::THROUGHPUT, CorrectnessTokens,
                                      CorrectnessHidden, NumExperts, numTopk, mscclpp::ep::DispatchLayout::TOKEN_MAJOR);
    }));
  }
  auto throughputRuntime = std::make_unique<mscclpp::ep::MoERuntime>(*communicator, mscclpp::ep::MoEMode::THROUGHPUT,
                                                                     CorrectnessTokens, CorrectnessHidden, NumExperts,
                                                                     NumTopk, mscclpp::ep::DispatchLayout::TOKEN_MAJOR);
  ASSERT_TRUE(throughputRuntime->mode() == mscclpp::ep::MoEMode::THROUGHPUT);
  ASSERT_TRUE(throughputRuntime->isAvailable());
  throughputRuntime->initialize();
  ASSERT_NE(throughputRuntime->dispatchOutputBuffer(), nullptr);
  ASSERT_NE(throughputRuntime->combineInputBuffer(), nullptr);
  ASSERT_NE(throughputRuntime->outputTopkIdsBuffer(), nullptr);
  ASSERT_NE(throughputRuntime->outputTopkWeightsBuffer(), nullptr);
  ASSERT_NE(throughputRuntime->outputScalesBuffer(), nullptr);
  ASSERT_NE(throughputRuntime->combineInputBuffer(), throughputRuntime->dispatchOutputBuffer());

  auto runtime = createRuntime(*communicator, CorrectnessTokens, CorrectnessHidden,
                               mscclpp::ep::DispatchLayout::RANK_MAJOR, mscclpp::ep::CombineMode::DIRECT_SEND);
  ASSERT_TRUE(runtime->mode() == mscclpp::ep::MoEMode::LATENCY);
  ASSERT_TRUE(runtime->isAvailable());
  ASSERT_EQ(runtime->rank(), gEnv->rank);
  ASSERT_EQ(runtime->numRanks(), NumRanks);
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
  } catch (const EPException&) {
    rejectedDispatch = true;
  }
  ASSERT_TRUE(rejectedDispatch);

  bool rejectedCombine = false;
  try {
    runtime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::ThroughputCombineRequest{}});
  } catch (const EPException&) {
    rejectedCombine = true;
  }
  ASSERT_TRUE(rejectedCombine);

  bool rejectedLatencyDispatch = false;
  try {
    throughputRuntime->dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::LatencyDispatchRequest{}});
  } catch (const EPException&) {
    rejectedLatencyDispatch = true;
  }
  ASSERT_TRUE(rejectedLatencyDispatch);

  bool rejectedLatencyCombine = false;
  try {
    throughputRuntime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::LatencyCombineRequest{}});
  } catch (const EPException&) {
    rejectedLatencyCombine = true;
  }
  ASSERT_TRUE(rejectedLatencyCombine);

  communicator->bootstrap()->barrier();
  throughputRuntime.reset();
  runtime.reset();
  communicator->bootstrap()->barrier();
}

TEST(MoERuntimeTest, DispatchCombineCorrectness) {
  for (const auto combineMode : {mscclpp::ep::CombineMode::RANK_LOCAL_REDUCE, mscclpp::ep::CombineMode::DIRECT_SEND}) {
    runCorrectnessCase(*communicator, gEnv->rank, 0, 0, mscclpp::ep::DispatchLayout::EXPERT_MAJOR, combineMode,
                       mscclpp::ep::DispatchDataType::BF16);
    runCorrectnessCase(*communicator, gEnv->rank, 0, 0, mscclpp::ep::DispatchLayout::EXPERT_MAJOR, combineMode,
                       mscclpp::ep::DispatchDataType::FP8_E4M3);
    runCorrectnessCase(*communicator, gEnv->rank, 0, 0, mscclpp::ep::DispatchLayout::RANK_MAJOR, combineMode,
                       mscclpp::ep::DispatchDataType::BF16);
  }
}

TEST(MoERuntimeTest, ThroughputCorrectness) {
  using namespace mscclpp::ep;
  for (const auto layout : {DispatchLayout::TOKEN_MAJOR, DispatchLayout::RANK_MAJOR}) {
    for (const auto dataType : {DispatchDataType::BF16, DispatchDataType::FP8_E4M3}) {
      runThroughputCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_, layout,
                                   {.runtimeCapacity = 2 * CorrectnessTokens, .dataType = dataType});
      runThroughputCorrectnessCase(
          *communicator, gEnv->rank, dispatchBlocks_, combineBlocks_, layout,
          {.usePreparation = true, .runtimeCapacity = 2 * CorrectnessTokens, .dataType = dataType});
      for (const bool prepared : {false, true}) {
        runThroughputCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_, layout,
                                     {.usePreparation = prepared,
                                      .runtimeCapacity = 2 * CorrectnessTokens,
                                      .dataType = dataType,
                                      .varyTokensByRank = true});
      }
      runThroughputCorrectnessCase(*communicator, gEnv->rank, 3, combineBlocks_, layout,
                                   {.numTokens = 67, .hidden = 9216, .runtimeCapacity = 96, .dataType = dataType});
    }
    for (const int tokens : {1025, 4097}) {
      runThroughputCorrectnessCase(*communicator, gEnv->rank, 24, combineBlocks_, layout,
                                   {.numTokens = tokens, .hidden = 136, .runtimeCapacity = tokens + 31});
    }
  }

  constexpr int Tokens = 5;
  constexpr int Hidden = 136;
  constexpr int ExpertsPerRank = 129;  // Cross the 128-thread count-exchange boundary.
  CudaStream stream;
  std::vector<Bf16> input(Tokens * Hidden);
  for (size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<Bf16>(gEnv->rank * Tokens + index / Hidden + 1);
  }
  mscclpp::GpuBuffer<Bf16> deviceInput(input.size());
  mscclpp::GpuBuffer<Bf16> deviceOutput(input.size());
  mscclpp::gpuMemcpy<Bf16>(deviceInput.data(), input.data(), input.size(), cudaMemcpyHostToDevice);

  for (const auto layout : {DispatchLayout::TOKEN_MAJOR, DispatchLayout::RANK_MAJOR}) {
    for (const int numTopk : {1, 2, 3, 4, 5, 8}) {
      auto runtime = std::make_unique<MoERuntime>(*communicator, MoEMode::THROUGHPUT, Tokens, Hidden,
                                                  ExpertsPerRank * NumRanks, numTopk, layout);
      runtime->initialize();
      std::vector<int64_t> routes(Tokens * numTopk);
      std::array<int, Tokens> numDestinations{};
      // Cover unordered destinations, repeated ranks, mixed invalid entries, and empty routes.
      for (int token = 0; token < Tokens; ++token) {
        std::array<bool, NumRanks> selectedRanks{};
        for (int topk = 0; topk < numTopk; ++topk) {
          int peer = (gEnv->rank + NumRanks - 1 - topk) % NumRanks;
          if (token == 1) peer = (gEnv->rank + 3) % NumRanks;
          if (token == 2) peer = (gEnv->rank + topk % 2) % NumRanks;
          const bool valid = token != 3 && (token != 2 || topk % 3 != 0) && (token != 4 || topk == numTopk - 1);
          const int expert = peer * ExpertsPerRank + ExpertsPerRank - 1 - topk;
          routes[token * numTopk + topk] = valid ? expert : (token == 3 ? int64_t{1} << 32 : -1);
          if (valid && !selectedRanks[peer]) {
            selectedRanks[peer] = true;
            ++numDestinations[token];
          }
        }
      }
      mscclpp::GpuBuffer<int64_t> deviceRoutes(routes.size());
      mscclpp::GpuBuffer<int> deviceCounts(ExpertsPerRank);
      mscclpp::gpuMemcpy<int64_t>(deviceRoutes.data(), routes.data(), routes.size(), cudaMemcpyHostToDevice);
      PrepareHandle preparation;
      if (layout == DispatchLayout::TOKEN_MAJOR) {
        preparation = runtime->prepare({.topkIdx = deviceRoutes.data(),
                                        .numTokens = Tokens,
                                        .maxTokensPerRank = Tokens,
                                        .numBlocks = dispatchBlocks_,
                                        .stream = stream,
                                        .outputCount = deviceCounts.data()});
      }
      const auto handle = runtime->dispatch(DispatchRequest{ThroughputDispatchRequest{
          .input = deviceInput.data(),
          .inputScales = nullptr,
          .topkIdx = deviceRoutes.data(),
          .topkWeights = nullptr,
          .numTokens = Tokens,
          .maxTokensPerRank = Tokens,
          .dispatchDataType = DispatchDataType::BF16,
          .numBlocks = dispatchBlocks_,
          .stream = stream,
          .prepareHandle = preparation,
      }});
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(runtime->combineInputBuffer(), runtime->dispatchOutputBuffer(),
                                        static_cast<size_t>(NumRanks) * Tokens * Hidden * sizeof(Bf16),
                                        cudaMemcpyDeviceToDevice, stream));
      runtime->combine(CombineRequest{ThroughputCombineRequest{
          .output = deviceOutput.data(),
          .handle = handle,
          .numBlocks = numTopk == 5 ? std::min(combineBlocks_, 24) : combineBlocks_,
          .stream = stream,
      }});
      MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
      communicator->bootstrap()->barrier();
      std::vector<Bf16> output(input.size());
      mscclpp::gpuMemcpy<Bf16>(output.data(), deviceOutput.data(), output.size(), cudaMemcpyDeviceToHost);
      if (layout == DispatchLayout::TOKEN_MAJOR) {
        std::array<int, ExpertsPerRank> counts{};
        mscclpp::gpuMemcpy<int>(counts.data(), deviceCounts.data(), counts.size(), cudaMemcpyDeviceToHost);
        EXPECT_EQ(counts.front(), 0);
        // Expert 128 receives tokens 0 and 1, plus token 4 when top-k is one.
        EXPECT_EQ(counts.back(), numTopk == 1 ? 3 : 2);
      }
      for (size_t index = 0; index < output.size(); ++index) {
        EXPECT_EQ(static_cast<float>(output[index]),
                  static_cast<float>(input[index]) * numDestinations[index / Hidden]);
      }
      runtime.reset();
      communicator->bootstrap()->barrier();
    }
  }
}

PERF_TEST(MoERuntimeTest, GraphDispatchCombine32Tokens) {
  runGraphPerformance(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                      mscclpp::ep::DispatchLayout::EXPERT_MAJOR, "expert-major 32 tokens/rank graph D+C");
}

PERF_TEST(MoERuntimeTest, GraphRankMajorDispatchCombine32Tokens) {
  runGraphPerformance(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                      mscclpp::ep::DispatchLayout::RANK_MAJOR, "rank-major 32 tokens/rank graph D+weighted staging+C");
}

PERF_TEST(MoERuntimeTest, ThroughputPerformance) {
  auto runtime = createThroughputRuntime(*communicator, 4096, PerfHidden, mscclpp::ep::DispatchLayout::TOKEN_MAJOR);
  ASSERT_TRUE(runtime->isAvailable());
  runtime->initialize();
  CudaStream stream;
  for (const int tokens : {1024, 2048, 4096}) {
    for (const int blocks : {24, dispatchBlocks_}) {
      const int combineBlocks = blocks == 24 ? std::min(32, combineBlocks_) : combineBlocks_;
      for (const auto dataType : {mscclpp::ep::DispatchDataType::BF16, mscclpp::ep::DispatchDataType::FP8_E4M3}) {
        runThroughputPerformance(*runtime, *communicator, gEnv->rank, tokens, blocks, combineBlocks, dataType, stream);
      }
    }
  }
  runtime.reset();
  communicator->bootstrap()->barrier();
}

class Mxfp8RuntimeTest : public CommunicatorTestBase {
 protected:
  void SetUp() override {
    if (gEnv->worldSize > 64) {
      SKIP_TEST() << "MXFP8 runtime tests require 1-64 GPUs in one IPC domain";
    }
    const int device = rankToLocalRank(gEnv->rank);
    MSCCLPP_CUDATHROW(cudaSetDevice(device));
    int major;
    MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    if (major < 9) SKIP_TEST() << "MXFP8 dispatch tests require SM90 or newer";
    setNumRanksToUse(gEnv->worldSize);
    CommunicatorTestBase::SetUp();
    if (communicator->bootstrap()->getNranksPerIpcDomain() != gEnv->worldSize) {
      SKIP_TEST() << "MXFP8 runtime tests require all GPUs in one IPC domain";
    }
  }
};

TEST(Mxfp8RuntimeTest, LatencyCorrectness) {
  using namespace mscclpp::ep;
  for (const auto combine : {CombineMode::RANK_LOCAL_REDUCE, CombineMode::DIRECT_SEND}) {
    runMxfp8Correctness(*communicator, combine, 1024, 9, 8, false);
    runMxfp8Correctness(*communicator, combine, 4096, 17, 8, false);
    runMxfp8Correctness(*communicator, combine, 4352, 5, 3, false);
  }
  runMxfp8Correctness(*communicator, CombineMode::RANK_LOCAL_REDUCE, 9216, 53, 8, false);
}

TEST(Mxfp8RuntimeTest, GraphReplay) {
  using namespace mscclpp::ep;
  runMxfp8Correctness(*communicator, CombineMode::RANK_LOCAL_REDUCE, 4096, 7, 8, true);
}

TEST(Mxfp8RuntimeTest, RequestValidation) {
  using namespace mscclpp::ep;
  for (const auto mode : {MoEMode::LATENCY, MoEMode::THROUGHPUT}) {
    MoERuntime runtime(*communicator, mode, 1, 4096, gEnv->worldSize * 32, 8,
                       mode == MoEMode::LATENCY ? DispatchLayout::EXPERT_MAJOR : DispatchLayout::TOKEN_MAJOR);
    runtime.initialize();
    if (mode == MoEMode::LATENCY) {
      LatencyDispatchRequest request{};
      request.maxTokensPerRank = 1;
      request.dispatchDataType = DispatchDataType::MXFP8_E4M3;
      ASSERT_TRUE(rejectsEpRequest([&] { runtime.dispatch(DispatchRequest{request}); }));
    } else {
      ThroughputDispatchRequest request{};
      request.maxTokensPerRank = 1;
      request.numBlocks = std::max(gEnv->worldSize + 2, 24);
      request.dispatchDataType = DispatchDataType::MXFP8_E4M3;
      ASSERT_TRUE(rejectsEpRequest([&] { runtime.dispatch(DispatchRequest{request}); }));
    }
    communicator->bootstrap()->barrier();
  }
  {
    MoERuntime runtime(*communicator, MoEMode::LATENCY, 1, 4096, gEnv->worldSize * 32, 8, DispatchLayout::RANK_MAJOR);
    runtime.initialize();
    LatencyDispatchRequest request{};
    request.output = runtime.dispatchOutputBuffer();
    request.outputTopkIdx = static_cast<int*>(runtime.outputTopkIdsBuffer());
    request.outputTopkWeights = static_cast<float*>(runtime.outputTopkWeightsBuffer());
    request.maxTokensPerRank = 1;
    request.dispatchDataType = DispatchDataType::MXFP8_E4M3;
    ASSERT_TRUE(rejectsEpRequest([&] { runtime.dispatch(DispatchRequest{request}); }));
    communicator->bootstrap()->barrier();
  }
  MoERuntime runtime(*communicator, MoEMode::THROUGHPUT, 1, 4096, gEnv->worldSize * 32, 8, DispatchLayout::RANK_MAJOR);
  runtime.initialize();
  ThroughputDispatchRequest request{};
  request.maxTokensPerRank = 1;
  request.numBlocks = std::max(gEnv->worldSize + 2, 24);
  request.dispatchDataType = DispatchDataType::MXFP8_E4M3;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime.dispatch(DispatchRequest{request}); }));
  communicator->bootstrap()->barrier();
}

TEST(Mxfp8RuntimeTest, ThroughputPrequantizedInput) {
  using namespace mscclpp::ep;
  constexpr int Tokens = 3;
  constexpr int Capacity = 5;
  constexpr int Hidden = 256;
  constexpr int Blocks = 24;
  const int rank = gEnv->rank;
  const int ranks = gEnv->worldSize;
  const int source = (rank + ranks - 1) % ranks;
  CudaStream stream;
  mscclpp::GpuBuffer<Bf16> reference(Tokens * Hidden), output(Tokens * Hidden);
  mscclpp::GpuBuffer<Fp8E4M3> input(Tokens * Hidden);
  mscclpp::GpuBuffer<float> scales(Tokens * (Hidden / 128)), weights(Tokens * NumTopk);
  mscclpp::GpuBuffer<int64_t> routes(Tokens * NumTopk);
  std::vector<int64_t> hostRoutes(routes.nelems());
  for (size_t index = 0; index < hostRoutes.size(); ++index) {
    hostRoutes[index] = ((rank + 1) % ranks) * 32 + index % NumTopk;
  }
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(routes.data(), hostRoutes.data(), hostRoutes.size() * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
  initializeInputs<<<numBlocks(reference.nelems()), Threads, 0, stream>>>(reference.data(), weights.data(), rank,
                                                                          Tokens, Hidden, 0);
  initializeThroughputFp8<<<numBlocks(input.nelems()), Threads, 0, stream>>>(input.data(), scales.data(),
                                                                             reference.data(), Tokens, Hidden);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  for (const auto layout : {DispatchLayout::TOKEN_MAJOR, DispatchLayout::RANK_MAJOR}) {
    MoERuntime runtime(*communicator, MoEMode::THROUGHPUT, Capacity, Hidden, ranks * 32, NumTopk, layout);
    runtime.initialize();
    const int rows = layout == DispatchLayout::RANK_MAJOR ? ranks * Capacity : Tokens;
    const int firstRow = layout == DispatchLayout::RANK_MAJOR ? source * Capacity : 0;
    std::vector<float> hostRowWeights(rows, 0.0f);
    std::fill_n(hostRowWeights.begin() + firstRow, Tokens, 1.0f);
    mscclpp::GpuBuffer<float> rowWeights(rows);
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(rowWeights.data(), hostRowWeights.data(), rows * sizeof(float),
                                      cudaMemcpyHostToDevice, stream));
    ThroughputDispatchRequest request{
        .input = input.data(),
        .inputScales = scales.data(),
        .topkIdx = routes.data(),
        .topkWeights = weights.data(),
        .numTokens = Tokens,
        .maxTokensPerRank = Capacity,
        .dispatchDataType = DispatchDataType::FP8_E4M3,
        .numBlocks = Blocks,
        .stream = stream,
        .prepareHandle = runtime.prepare({routes.data(), Tokens, Capacity, Blocks, stream}),
    };
    auto unsupported = request;
    unsupported.dispatchDataType = DispatchDataType::MXFP8_E4M3;
    ASSERT_TRUE(rejectsEpRequest([&] { runtime.dispatch(DispatchRequest{unsupported}); }));
    auto operation = [&] {
      const auto handle = runtime.dispatch(DispatchRequest{request});
      stageThroughputFp8ExpertOutput<<<numBlocks(static_cast<size_t>(rows) * Hidden), Threads, 0, stream>>>(
          static_cast<Bf16*>(runtime.combineInputBuffer()), static_cast<const Fp8E4M3*>(runtime.dispatchOutputBuffer()),
          static_cast<const float*>(runtime.outputScalesBuffer()), rowWeights.data(), rows, Hidden);
      MSCCLPP_CUDATHROW(cudaGetLastError());
      runtime.combine(CombineRequest{ThroughputCombineRequest{output.data(), handle, Blocks, stream}});
    };
    operation();
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    CudaGraph graph;
    graph.capture(stream, operation);
    for (int replay = 0; replay < 3; ++replay) graph.launch(stream);
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

    std::string error = checkOutput(output.data(), rank, Tokens, Hidden, 0.0f);
    std::vector<uint8_t> received(Tokens * Hidden);
    MSCCLPP_CUDATHROW(cudaMemcpy(received.data(),
                                 static_cast<uint8_t*>(runtime.dispatchOutputBuffer()) + firstRow * Hidden,
                                 received.size(), cudaMemcpyDeviceToHost));
    const uint8_t expected = std::bit_cast<uint8_t>(static_cast<Fp8E4M3>(1.0f));
    if (error.empty() &&
        !std::all_of(received.begin(), received.end(), [=](uint8_t value) { return value == expected; })) {
      error = "Throughput changed prequantized FP8 bytes";
    }
    std::vector<float> receivedScales(Tokens * (Hidden / 128));
    mscclpp::gpuMemcpy<float>(receivedScales.data(),
                              static_cast<float*>(runtime.outputScalesBuffer()) + firstRow * (Hidden / 128),
                              receivedScales.size(), cudaMemcpyDeviceToHost);
    for (size_t index = 0; index < receivedScales.size() && error.empty(); ++index) {
      if (receivedScales[index] != testTokenValue(source, Tokens, index / (Hidden / 128), 0)) {
        error = "Throughput changed prequantized FP8 scales";
      }
    }
    assertCollectiveSuccess(error, "throughput prequantized FP8 regression");
    communicator->bootstrap()->barrier();
  }
}

PERF_TEST(Mxfp8RuntimeTest, DispatchPerformance) {
  for (const int tokens : {32, 128, 512, 1024, 4096}) runMxfp8Performance(*communicator, tokens);
}
