// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "config.hpp"
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

__global__ void initializeInputs(Bf16* input, float* topkWeights, int rank, int numTokens, int hidden) {
  const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t inputElements = static_cast<size_t>(numTokens) * hidden;
  for (size_t index = thread; index < inputElements; index += stride) {
    const int token = static_cast<int>(index / hidden);
    input[index] = static_cast<Bf16>(static_cast<float>((rank * numTokens + token) * NumTopk));
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
        localWeight = topkWeights[static_cast<size_t>(row) * NumTopk + topkLane];
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
                           int numExperts = NumExperts) {
  std::vector<int64_t> topkIdx(static_cast<size_t>(numTokens) * NumTopk);
  for (int token = 0; token < numTokens; ++token) {
    const auto experts = routedExperts(rank, token, numExperts);
    std::copy(experts.begin(), experts.end(), topkIdx.begin() + static_cast<size_t>(token) * NumTopk);
  }
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(buffers.topkIdx.data(), topkIdx.data(), topkIdx.size() * sizeof(int64_t),
                                    cudaMemcpyHostToDevice, stream));
  const size_t elements = std::max(static_cast<size_t>(numTokens) * hidden, static_cast<size_t>(numTokens) * NumTopk);
  initializeInputs<<<numBlocks(elements), Threads, 0, stream>>>(buffers.input.data(), buffers.topkWeights.data(), rank,
                                                                numTokens, hidden);
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

std::string checkOutput(const Bf16* output, int rank, int numTokens, int hidden, float tolerance) {
  std::vector<Bf16> hostOutput(static_cast<size_t>(numTokens) * hidden);
  MSCCLPP_CUDATHROW(cudaMemcpy(hostOutput.data(), output, hostOutput.size() * sizeof(Bf16), cudaMemcpyDeviceToHost));
  for (int token = 0; token < numTokens; ++token) {
    const float expected = static_cast<float>((rank * numTokens + token) * NumTopk);
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

ThroughputExpectation makeThroughputExpectation(mscclpp::ep::DispatchLayout layout, int rank, int numTokens,
                                                int numExperts) {
  const int numLocalExperts = numExperts / NumRanks;
  ThroughputExpectation expectation;
  if (layout == mscclpp::ep::DispatchLayout::TOKEN_MAJOR) {
    expectation.outputCount.assign(numLocalExperts, 0);
    expectation.rowWeights.reserve(static_cast<size_t>(NumRanks) * numTokens);
  } else {
    expectation.outputCount.assign(NumRanks, 0);
    expectation.rowWeights.assign(static_cast<size_t>(NumRanks) * numTokens, 0.0f);
  }

  std::vector<int> rankOffsets(NumRanks, 0);
  for (int source = 0; source < NumRanks; ++source) {
    for (int token = 0; token < numTokens; ++token) {
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
        expectation.rowWeights[static_cast<size_t>(source) * numTokens + rankOffsets[source]] = localWeight;
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

std::string checkThroughputCounts(const int* outputCount, mscclpp::ep::DispatchLayout layout, int rank, int numTokens,
                                  int numExperts) {
  const ThroughputExpectation expectation = makeThroughputExpectation(layout, rank, numTokens, numExperts);
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

  communicator.bootstrap()->barrier();
  runtime.reset();
  communicator.bootstrap()->barrier();
}

enum class ThroughputPreparation { Automatic, Explicit, Captured };

void runThroughputCorrectnessCase(mscclpp::Communicator& communicator, int rank, int dispatchBlocks, int combineBlocks,
                                  mscclpp::ep::DispatchLayout layout,
                                  ThroughputPreparation preparationMode = ThroughputPreparation::Automatic) {
  const char* layoutName = layout == mscclpp::ep::DispatchLayout::TOKEN_MAJOR ? "token-major" : "rank-major";
  const std::string label = std::string("throughput/") + layoutName + "/bf16";
  auto runtime = createThroughputRuntime(communicator, CorrectnessTokens, CorrectnessHidden, layout);
  ASSERT_TRUE(runtime->isAvailable());
  runtime->initialize();

  CudaStream stream;
  TestBuffers buffers(CorrectnessTokens, CorrectnessHidden);
  initializeTestBuffers(buffers, rank, CorrectnessTokens, CorrectnessHidden, stream);

  const ThroughputExpectation expectation = makeThroughputExpectation(layout, rank, CorrectnessTokens, NumExperts);
  mscclpp::GpuBuffer<float> rowWeights(expectation.totalRows > 0 ? static_cast<size_t>(expectation.totalRows) : 1);
  if (expectation.totalRows > 0) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(rowWeights.data(), expectation.rowWeights.data(),
                                      sizeof(float) * static_cast<size_t>(expectation.totalRows),
                                      cudaMemcpyHostToDevice, stream));
  }

  const bool prepared = preparationMode != ThroughputPreparation::Automatic;
  const size_t elements = static_cast<size_t>(expectation.totalRows) * CorrectnessHidden;
  mscclpp::GpuBuffer<Bf16> dispatchStorage(prepared ? elements + 1 : 1);
  mscclpp::GpuBuffer<Bf16> combineStorage(prepared ? elements + 1 : 1);
  const size_t metadataElements = static_cast<size_t>(expectation.totalRows) * NumTopk;
  mscclpp::GpuBuffer<int> receivedTopkIdx(std::max<size_t>(metadataElements, 1));
  mscclpp::GpuBuffer<float> receivedTopkWeights(std::max<size_t>(metadataElements, 1));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(receivedTopkIdx.data(), 0xff, receivedTopkIdx.bytes(), stream));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(receivedTopkWeights.data(), 0, receivedTopkWeights.bytes(), stream));
  // Exercise aligned copies with explicit preparation and unaligned copies with captured preparation.
  const int storageOffset = preparationMode == ThroughputPreparation::Captured ? 1 : 0;
  void* dispatchOutput = prepared ? dispatchStorage.data() + storageOffset : runtime->dispatchOutputBuffer();
  auto* combineInput =
      prepared ? combineStorage.data() + storageOffset : static_cast<Bf16*>(runtime->combineInputBuffer());
  CudaStream prepareStream;
  mscclpp::ep::PrepareHandle preparation;
  auto prepare = [&](cudaStream_t prepareOn) {
    return runtime->prepare({buffers.topkIdx.data(), CorrectnessTokens, CorrectnessTokens, dispatchBlocks, prepareOn});
  };
  if (preparationMode == ThroughputPreparation::Explicit) preparation = prepare(prepareStream);

  auto operation = [&] {
    if (preparationMode == ThroughputPreparation::Captured) preparation = prepare(stream);
    const auto handle = runtime->dispatch(mscclpp::ep::DispatchRequest{mscclpp::ep::ThroughputDispatchRequest{
        .output = dispatchOutput,
        .outputScales = nullptr,
        .outputTopkIdx = receivedTopkIdx.data(),
        .outputTopkWeights = receivedTopkWeights.data(),
        .outputCount = buffers.outputCount.data(),
        .input = buffers.input.data(),
        .inputScales = nullptr,
        .topkIdx = buffers.topkIdx.data(),
        .topkWeights = buffers.topkWeights.data(),
        .numTokens = CorrectnessTokens,
        .maxTokensPerRank = CorrectnessTokens,
        .dispatchDataType = mscclpp::ep::DispatchDataType::BF16,
        .numBlocks = dispatchBlocks,
        .stream = stream,
        .prepareHandle = preparation,
    }});

    if (expectation.totalRows > 0) {
      stageThroughputExpertOutput<<<numBlocks(elements), Threads, 0, stream>>>(
          combineInput, static_cast<const Bf16*>(dispatchOutput), rowWeights.data(), expectation.totalRows,
          CorrectnessHidden);
      MSCCLPP_CUDATHROW(cudaGetLastError());
    }

    runtime->combine(mscclpp::ep::CombineRequest{mscclpp::ep::ThroughputCombineRequest{
        .output = buffers.output.data(),
        .outputTopkWeights = nullptr,
        .input = combineInput,
        .handle = handle,
        .numBlocks = combineBlocks,
        .stream = stream,
    }});
  };
  operation();
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  CudaGraph graph;
  if (prepared) {
    graph.capture(stream, operation);
    for (int replay = 0; replay < 3; ++replay) graph.launch(stream);
  }
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  if (prepared) {
    cudaPointerAttributes attributes{};
    MSCCLPP_CUDATHROW(cudaPointerGetAttributes(&attributes, preparation.numRecvTokensDevice()));
    ASSERT_TRUE(attributes.type == cudaMemoryTypeDevice);
    MSCCLPP_CUDATHROW(cudaPointerGetAttributes(&attributes, preparation.outputCountsDevice()));
    ASSERT_TRUE(attributes.type == cudaMemoryTypeDevice);
    ASSERT_EQ(preparation.numOutputCounts(), static_cast<int>(expectation.outputCount.size()));
    // Only the test reads back the GPU-resident preparation results.
    int received = -1;
    std::vector<int> counts(preparation.numOutputCounts());
    MSCCLPP_CUDATHROW(cudaMemcpy(&received, preparation.numRecvTokensDevice(), sizeof(int), cudaMemcpyDeviceToHost));
    MSCCLPP_CUDATHROW(cudaMemcpy(counts.data(), preparation.outputCountsDevice(), counts.size() * sizeof(int),
                                 cudaMemcpyDeviceToHost));
    ASSERT_EQ(received, expectation.numRecvTokens);
    ASSERT_TRUE(counts == expectation.outputCount);
  }

  std::string error = checkThroughputCounts(buffers.outputCount.data(), layout, rank, CorrectnessTokens, NumExperts);
  if (error.empty()) {
    error = checkOutput(buffers.output.data(), rank, CorrectnessTokens, CorrectnessHidden, 0.0f);
  }
  assertCollectiveSuccess(error, label);

  std::vector<int> topkIdx(metadataElements);
  std::vector<float> topkWeights(metadataElements);
  if (metadataElements > 0) {
    MSCCLPP_CUDATHROW(
        cudaMemcpy(topkIdx.data(), receivedTopkIdx.data(), metadataElements * sizeof(int), cudaMemcpyDeviceToHost));
    MSCCLPP_CUDATHROW(cudaMemcpy(topkWeights.data(), receivedTopkWeights.data(), metadataElements * sizeof(float),
                                 cudaMemcpyDeviceToHost));
  }
  int compactRow = 0;
  constexpr int LocalExperts = NumExperts / NumRanks;
  for (int source = 0; source < NumRanks; ++source) {
    int rankRow = 0;
    for (int token = 0; token < CorrectnessTokens; ++token) {
      const auto experts = routedExperts(source, token, NumExperts);
      const bool selected =
          std::any_of(experts.begin(), experts.end(), [&](int expert) { return expert / LocalExperts == rank; });
      if (!selected) continue;
      const int row =
          layout == mscclpp::ep::DispatchLayout::RANK_MAJOR ? source * CorrectnessTokens + rankRow++ : compactRow++;
      for (int topk = 0; topk < NumTopk; ++topk) {
        const int localExpert = experts[topk] / LocalExperts == rank ? experts[topk] % LocalExperts : -1;
        const size_t index = static_cast<size_t>(row) * NumTopk + topk;
        ASSERT_EQ(topkIdx[index], localExpert);
        ASSERT_EQ(topkWeights[index], localExpert >= 0 ? 1.0f / NumTopk : 0.0f);
      }
    }
  }

  if (preparationMode == ThroughputPreparation::Captured) {
    MSCCLPP_CUDATHROW(cudaMemsetAsync(buffers.topkIdx.data(), 0xff, buffers.topkIdx.bytes(), stream));
    graph.launch(stream);
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
    int received = -1;
    std::vector<Bf16> output(static_cast<size_t>(CorrectnessTokens) * CorrectnessHidden);
    MSCCLPP_CUDATHROW(cudaMemcpy(&received, preparation.numRecvTokensDevice(), sizeof(int), cudaMemcpyDeviceToHost));
    MSCCLPP_CUDATHROW(
        cudaMemcpy(output.data(), buffers.output.data(), output.size() * sizeof(Bf16), cudaMemcpyDeviceToHost));
    ASSERT_EQ(received, 0);
    ASSERT_TRUE(
        std::all_of(output.begin(), output.end(), [](Bf16 value) { return static_cast<float>(value) == 0.0f; }));
  }

  communicator.bootstrap()->barrier();
  graph.reset();
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

TEST(MoERuntimeTest, ThroughputStorageLayout) {
  constexpr size_t HiddenBytes = size_t{65536} * 16384;
  constexpr size_t MetadataBytes = size_t{65536} * 768;
  constexpr std::array<std::pair<int, size_t>, 4> cases{{{2, 8320}, {4, 16512}, {8, 33024}, {16, 66560}}};
  ASSERT_EQ(mscclpp::ep::ThroughputPayloadView::numBytes(), HiddenBytes + MetadataBytes);
  ASSERT_EQ(mscclpp::ep::ThroughputPayloadView::MetadataOffset, HiddenBytes);
  ASSERT_EQ(mscclpp::ep::ThroughputPayloadView::MetadataSlotBytes, size_t{768});
  ASSERT_EQ(mscclpp::ep::dispatchElementsPerScale(mscclpp::ep::DispatchDataType::FP8_E4M3), 128);
  ASSERT_EQ(mscclpp::ep::dispatchElementsPerScale(mscclpp::ep::DispatchDataType::BF16), 0);

  for (const auto& [numRanks, controlBytes] : cases) {
    const mscclpp::ep::ThroughputStorageLayout layout(nullptr, numRanks);
    ASSERT_EQ(layout.totalBytes_, controlBytes + HiddenBytes + MetadataBytes);
    ASSERT_EQ(layout.recvBuffer_, nullptr);

    std::vector<uint8_t> symmetricBuffer(controlBytes + 1);
    const mscclpp::ep::ThroughputStorageLayout boundLayout(symmetricBuffer.data(), numRanks);
    ASSERT_EQ(boundLayout.recvBuffer_, static_cast<void*>(symmetricBuffer.data() + controlBytes));
  }
}

TEST(MoERuntimeTest, InitializationAndModeValidation) {
  auto throughputRuntime = std::make_unique<mscclpp::ep::MoERuntime>(*communicator, mscclpp::ep::MoEMode::THROUGHPUT,
                                                                     CorrectnessTokens, CorrectnessHidden, NumExperts,
                                                                     NumTopk, mscclpp::ep::DispatchLayout::TOKEN_MAJOR);
  ASSERT_TRUE(throughputRuntime->mode() == mscclpp::ep::MoEMode::THROUGHPUT);
  ASSERT_TRUE(throughputRuntime->isAvailable());
  throughputRuntime->initialize();
  ASSERT_NE(throughputRuntime->dispatchOutputBuffer(), nullptr);
  ASSERT_NE(throughputRuntime->combineInputBuffer(), nullptr);
  ASSERT_EQ(throughputRuntime->combineInputBuffer(), throughputRuntime->dispatchOutputBuffer());
  const mscclpp::ep::ThroughputPayloadView payload(NumTopk);
  const void* receiveBuffer = throughputRuntime->dispatchOutputBuffer();
  for (const int row : {0, mscclpp::ep::ThroughputPayloadView::MaxTokens - 1}) {
    const uintptr_t metadata = reinterpret_cast<uintptr_t>(receiveBuffer) +
                               mscclpp::ep::ThroughputPayloadView::MetadataOffset +
                               static_cast<size_t>(row) * mscclpp::ep::ThroughputPayloadView::MetadataSlotBytes;
    ASSERT_EQ(reinterpret_cast<uintptr_t>(payload.topKIndices(receiveBuffer, row)), metadata);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(payload.topKValues(receiveBuffer, row)), metadata + NumTopk * sizeof(int));
    ASSERT_EQ(reinterpret_cast<uintptr_t>(payload.scaleFactors(receiveBuffer, row)),
              metadata + NumTopk * (sizeof(int) + sizeof(float)));
  }

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

TEST(MoERuntimeTest, ThroughputDispatchCombineCorrectness) {
  runThroughputCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                               mscclpp::ep::DispatchLayout::TOKEN_MAJOR);
  runThroughputCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                               mscclpp::ep::DispatchLayout::RANK_MAJOR);
}

TEST(MoERuntimeTest, ThroughputPreparedDispatchCombineCorrectness) {
  for (const auto layout : {mscclpp::ep::DispatchLayout::TOKEN_MAJOR, mscclpp::ep::DispatchLayout::RANK_MAJOR}) {
    runThroughputCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_, layout,
                                 ThroughputPreparation::Explicit);
  }
}

TEST(MoERuntimeTest, ThroughputCapturedPreparation) {
  for (const auto layout : {mscclpp::ep::DispatchLayout::TOKEN_MAJOR, mscclpp::ep::DispatchLayout::RANK_MAJOR}) {
    runThroughputCorrectnessCase(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_, layout,
                                 ThroughputPreparation::Captured);
  }
}

TEST(MoERuntimeTest, ThroughputPreparationValidation) {
  using namespace mscclpp::ep;
  PrepareHandle empty;
  ASSERT_TRUE(rejectsEpRequest([&] { empty.numRecvTokensDevice(); }));
  ASSERT_TRUE(rejectsEpRequest([&] { empty.outputCountsDevice(); }));
  ASSERT_TRUE(rejectsEpRequest([&] { empty.numOutputCounts(); }));

  auto runtime =
      createThroughputRuntime(*communicator, CorrectnessTokens, CorrectnessHidden, DispatchLayout::TOKEN_MAJOR);
  runtime->initialize();
  CudaStream stream;
  TestBuffers buffers(CorrectnessTokens, CorrectnessHidden);
  initializeTestBuffers(buffers, gEnv->rank, CorrectnessTokens, CorrectnessHidden, stream);
  const PrepareRequest prepareRequest{buffers.topkIdx.data(), CorrectnessTokens, CorrectnessTokens, dispatchBlocks_,
                                      stream};
  auto preparation = runtime->prepare(prepareRequest);
  ThroughputDispatchRequest request{
      .output = runtime->dispatchOutputBuffer(),
      .outputScales = nullptr,
      .outputTopkIdx = nullptr,
      .outputTopkWeights = nullptr,
      .outputCount = buffers.outputCount.data(),
      .input = buffers.input.data(),
      .inputScales = nullptr,
      .topkIdx = buffers.topkIdx.data(),
      .topkWeights = buffers.topkWeights.data(),
      .numTokens = CorrectnessTokens,
      .maxTokensPerRank = CorrectnessTokens,
      .dispatchDataType = DispatchDataType::BF16,
      .numBlocks = dispatchBlocks_,
      .stream = stream,
      .prepareHandle = preparation,
  };

  auto invalidPrepare = prepareRequest;
  invalidPrepare.numTokens = -1;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->prepare(invalidPrepare); }));
  invalidPrepare = prepareRequest;
  invalidPrepare.topkIdx = nullptr;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->prepare(invalidPrepare); }));
  invalidPrepare = prepareRequest;
  invalidPrepare.maxTokensPerRank = CorrectnessTokens + 1;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->prepare(invalidPrepare); }));
  invalidPrepare = prepareRequest;
  invalidPrepare.numBlocks = 0;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->prepare(invalidPrepare); }));

  auto invalid = request;
  invalid.topkIdx = buffers.topkIdx.data() + NumTopk;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->dispatch(DispatchRequest{invalid}); }));
  invalid = request;
  invalid.numTokens -= 1;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->dispatch(DispatchRequest{invalid}); }));
  invalid = request;
  invalid.maxTokensPerRank += 1;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->dispatch(DispatchRequest{invalid}); }));
  invalid = request;
  invalid.numBlocks -= 1;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->dispatch(DispatchRequest{invalid}); }));

  const auto dispatchHandle = runtime->dispatch(DispatchRequest{request});
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  auto replacement = runtime->prepare(prepareRequest);
  ASSERT_TRUE(rejectsEpRequest([&] { preparation.outputCountsDevice(); }));
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->dispatch(DispatchRequest{request}); }));
  ASSERT_TRUE(rejectsEpRequest([&] {
    runtime->combine(CombineRequest{ThroughputCombineRequest{
        .output = buffers.output.data(),
        .outputTopkWeights = nullptr,
        .input = runtime->combineInputBuffer(),
        .handle = dispatchHandle,
        .numBlocks = combineBlocks_,
        .stream = stream,
    }});
  }));

  auto other =
      createThroughputRuntime(*communicator, CorrectnessTokens, CorrectnessHidden, DispatchLayout::TOKEN_MAJOR);
  other->initialize();
  auto otherPreparation = other->prepare(prepareRequest);
  invalid = request;
  invalid.prepareHandle = otherPreparation;
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->dispatch(DispatchRequest{invalid}); }));
  other.reset();
  ASSERT_TRUE(rejectsEpRequest([&] { otherPreparation.numRecvTokensDevice(); }));
  ASSERT_TRUE(rejectsEpRequest([&] { runtime->dispatch(DispatchRequest{invalid}); }));

  auto latency = createRuntime(*communicator, CorrectnessTokens, CorrectnessHidden, DispatchLayout::RANK_MAJOR,
                               CombineMode::RANK_LOCAL_REDUCE);
  ASSERT_TRUE(rejectsEpRequest([&] { latency->prepare(prepareRequest); }));
  latency.reset();

  request.prepareHandle = replacement;
  runtime->dispatch(DispatchRequest{request});
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  request.prepareHandle = empty;
  runtime->dispatch(DispatchRequest{request});
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  ASSERT_TRUE(rejectsEpRequest([&] { replacement.numRecvTokensDevice(); }));

  const auto zero = runtime->prepare({nullptr, 0, CorrectnessTokens, dispatchBlocks_, stream});
  request.input = nullptr;
  request.output = nullptr;
  request.topkIdx = nullptr;
  request.numTokens = 0;
  request.prepareHandle = zero;
  const auto zeroDispatch = runtime->dispatch(DispatchRequest{request});
  runtime->combine(CombineRequest{ThroughputCombineRequest{
      .output = nullptr,
      .outputTopkWeights = nullptr,
      .input = nullptr,
      .handle = zeroDispatch,
      .numBlocks = combineBlocks_,
      .stream = stream,
  }});
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  int received = -1;
  MSCCLPP_CUDATHROW(cudaMemcpy(&received, zero.numRecvTokensDevice(), sizeof(int), cudaMemcpyDeviceToHost));
  ASSERT_EQ(received, 0);

  runtime.reset();
  communicator->bootstrap()->barrier();
}

PERF_TEST(MoERuntimeTest, GraphDispatchCombine32Tokens) {
  runGraphPerformance(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                      mscclpp::ep::DispatchLayout::EXPERT_MAJOR, "expert-major 32 tokens/rank graph D+C");
}

PERF_TEST(MoERuntimeTest, GraphRankMajorDispatchCombine32Tokens) {
  runGraphPerformance(*communicator, gEnv->rank, dispatchBlocks_, combineBlocks_,
                      mscclpp::ep::DispatchLayout::RANK_MAJOR, "rank-major 32 tokens/rank graph D+weighted staging+C");
}
