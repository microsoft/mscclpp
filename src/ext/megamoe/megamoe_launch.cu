// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>
#include <string>
#include <variant>

#include "megamoe_device.cuh"
#include "megamoe_kernel.hpp"
#include "megamoe_quantization.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE {
namespace detail {

constexpr bool isLocalExpert(const NativeConfig& c) {
  return !c.combineRouteWeights && c.weightDtype == WeightDtype::Mxfp8 && c.worldSize == 1 && c.numExperts == 1 &&
         c.topK == 1;
}

size_t aligned(size_t n, size_t alignment = 256) { return (n + alignment - 1) / alignment * alignment; }

size_t appendRegion(size_t& bytes, size_t regionBytes) {
  size_t offset = aligned(bytes);
  if (regionBytes > std::numeric_limits<size_t>::max() - offset)
    throw std::overflow_error("MegaMoE workspace size overflows size_t");
  bytes = offset + regionBytes;
  return offset;
}

size_t blockScaleBytes(size_t rows, int width) {
  const size_t paddedRows = aligned(rows, 128);
  const size_t scalesPerRow = size_t(width) / 32;
  if (scalesPerRow && paddedRows > std::numeric_limits<size_t>::max() / scalesPerRow)
    throw std::overflow_error("MegaMoE block-scale workspace size overflows size_t");
  return paddedRows * scalesPerRow;
}

Workspace localWorkspaceLayout(const NativeConfig& c, void* base, size_t& bytes) {
  Workspace w{};
  bytes = 0;
  const size_t rows = aligned(c.maxTokens, W4TileN);
  if (rows > size_t(std::numeric_limits<int>::max()) ||
      (rows + W4TileN - 1) / W4TileN *
              ((2 * size_t(c.intermediate) + W4TileM - 1) / W4TileM + (size_t(c.hidden) + W4TileM - 1) / W4TileM) >
          size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("MegaMoE local workspace exceeds 32-bit tile indexing");
  w.hiddenReady = at<int>(base, appendRegion(bytes, rows / W4TileN * W4ReadyCounterStride * sizeof(int)));
  w.quantizedHidden = at<uint8_t>(base, appendRegion(bytes, rows * c.intermediate * sizeof(__bfloat16)));
  w.poolRows = int(rows);
  bytes = aligned(bytes);
  return w;
}

Workspace routedWorkspaceLayout(const NativeConfig& c, void* base, size_t& bytes) {
  Workspace w{};
  bytes = 0;
  const bool isW4 = c.weightDtype == WeightDtype::Mxfp4;
  const size_t elemBytes = isW4 ? sizeof(uint8_t) : sizeof(__bfloat16);
  const size_t experts = c.numExperts / c.worldSize;
  const size_t routes = size_t(c.worldSize) * c.maxTokens * c.topK;
  const int inputChunks =
      isW4 ? w4InputChunks<W4A8CollectiveTypes>(c.hidden) : w4InputChunks<W8A16CollectiveTypesT<false>>(c.hidden);
  const size_t rows = aligned(routes + experts * (W4TileN - 1), W4TileN);
  const size_t activationRows = rows / W4TileN * W4TokenStride;
  if (activationRows > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("MegaMoE activation workspace exceeds 32-bit row indexing");
  if (rows > size_t(std::numeric_limits<int>::max()) ||
      (rows + W4TileN - 1) / W4TileN *
              ((2 * size_t(c.intermediate) + W4TileM - 1) / W4TileM + (size_t(c.hidden) + W4TileM - 1) / W4TileM) >
          size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("MegaMoE routing workspace exceeds 32-bit tile indexing");

  w.control = at<Control>(base, appendRegion(bytes, sizeof(Control)));
  w.counts = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
  w.starts = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
  w.cursors = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
  w.inputReady = at<int>(base, appendRegion(bytes, rows / W4TileN * W4ReadyCounterStride * sizeof(int)));
  w.hiddenReady = at<int>(base, appendRegion(bytes, rows / W4TileN * W4ReadyCounterStride * sizeof(int)));
  if (inputChunks > 1)
    w.inputChunkReady =
        at<int>(base, appendRegion(bytes, rows / W4TileN * (inputChunks - 1) * W4ReadyCounterStride * sizeof(int)));
  w.routes = at<Route>(base, appendRegion(bytes, rows * sizeof(Route)));
  w.blocks = at<TokenBlock>(base, appendRegion(bytes, rows / W4TileN * sizeof(TokenBlock)));
  w.quantizedInput = at<uint8_t>(base, appendRegion(bytes, activationRows * c.hidden * elemBytes));
  if (isW4) w.inputScale = at<uint8_t>(base, appendRegion(bytes, blockScaleBytes(activationRows, c.hidden)));
  w.quantizedHidden = at<uint8_t>(base, appendRegion(bytes, activationRows * c.intermediate * elemBytes));
  if (isW4) w.hiddenScale = at<uint8_t>(base, appendRegion(bytes, blockScaleBytes(activationRows, c.intermediate)));
  w.poolRows = int(rows);
  bytes = aligned(bytes);
  return w;
}

Workspace workspaceLayout(const NativeConfig& c, void* base, size_t& bytes) {
  return isLocalExpert(c) ? localWorkspaceLayout(c, base, bytes) : routedWorkspaceLayout(c, base, bytes);
}

__device__ __forceinline__ int canonicalFc1Row(int row, int intermediate) {
  return row / 32 * 16 + row % 16 + (row % 32 >= 16 ? intermediate : 0);
}

template <int ValuesPerByte>
__device__ __forceinline__ void packWeightValues(const NativeConfig& c, const PackedWeights& src,
                                                 const PackedWeights& dst, size_t idx, size_t stride) {
  int experts = c.numExperts / c.worldSize;
  int fc1RowBytes = c.hidden / ValuesPerByte;
  size_t fc1Size = size_t(experts) * 2 * c.intermediate * fc1RowBytes;
  size_t fc2Size = size_t(experts) * c.hidden * (c.intermediate / ValuesPerByte);
  for (size_t i = idx; i < fc1Size; i += stride) {
    int k = i % fc1RowBytes;
    size_t row = i / fc1RowBytes;
    int m = row % (2 * c.intermediate);
    int canonical = canonicalFc1Row(m, c.intermediate);
    dst.fc1[i] = src.fc1[(row / (2 * c.intermediate) * 2 * c.intermediate + canonical) * fc1RowBytes + k];
  }
  for (size_t i = idx; i < fc2Size; i += stride) dst.fc2[i] = src.fc2[i];
}

__global__ void packWeightsKernel(NativeConfig c, PackedWeights src, PackedWeights dst) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  int experts = c.numExperts / c.worldSize;
  size_t fc1Size = size_t(experts) * 2 * c.intermediate * c.hidden;
  size_t fc2Size = size_t(experts) * c.hidden * c.intermediate;
  packWeightValues<1>(c, src, dst, idx, stride);
  for (size_t i = idx; i < fc1Size / 32; i += stride) {
    int m = i % (2 * c.intermediate);
    int k = i / (2 * c.intermediate) % (c.hidden / 32);
    size_t expert = i / (size_t(2) * c.intermediate * (c.hidden / 32));
    int canonical = canonicalFc1Row(m, c.intermediate);
    dst.fc1Scale[i] = src.fc1Scale[(expert * 2 * c.intermediate + canonical) * (c.hidden / 32) + k];
  }
  for (size_t i = idx; i < fc2Size / 32; i += stride) {
    int m = i % c.hidden;
    int k = i / c.hidden % (c.intermediate / 32);
    size_t expert = i / (size_t(c.hidden) * (c.intermediate / 32));
    dst.fc2Scale[i] = src.fc2Scale[(expert * c.hidden + m) * (c.intermediate / 32) + k];
  }
}

template <class Types>
using W4ScaleLayout = typename Types::Mainloop::LayoutSFA;

template <class Types>
__global__ void packW4A8WeightsKernel(NativeConfig c, PackedWeights src, PackedWeights dst,
                                      W4ScaleLayout<Types> fc1ScaleLayout, W4ScaleLayout<Types> fc2ScaleLayout) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  int experts = c.numExperts / c.worldSize;
  size_t fc1Rows = size_t(experts) * 2 * c.intermediate;
  size_t fc2Rows = size_t(experts) * c.hidden;
  packWeightValues<2>(c, src, dst, idx, stride);
  size_t fc1Scales = fc1Rows * (c.hidden / 32);
  for (size_t i = idx; i < fc1Scales; i += stride) {
    int k = i % (c.hidden / 32);
    size_t row = i / (c.hidden / 32);
    int m = row % (2 * c.intermediate);
    int expert = row / (2 * c.intermediate);
    int canonical = canonicalFc1Row(m, c.intermediate);
    size_t destination = fc1ScaleLayout(cute::make_coord(m, k * 32, expert));
    dst.fc1Scale[destination] = src.fc1Scale[(size_t(expert) * 2 * c.intermediate + canonical) * (c.hidden / 32) + k];
  }
  size_t fc2Scales = fc2Rows * (c.intermediate / 32);
  for (size_t i = idx; i < fc2Scales; i += stride) {
    int k = i % (c.intermediate / 32);
    size_t row = i / (c.intermediate / 32);
    int m = row % c.hidden;
    int expert = row / c.hidden;
    size_t destination = fc2ScaleLayout(cute::make_coord(m, k * 32, expert));
    dst.fc2Scale[destination] = src.fc2Scale[i];
  }
}

__global__ void quantizeInputKernel(const __bfloat16* input, uint8_t* quantizedInput, uint8_t* quantizedInputScale,
                                    int tokens, int hidden) {
  constexpr int ValuesPerThread = 8;
  constexpr int ThreadsPerBlockScale = 4;
  constexpr int BlockScalesPerCta = 128 / ThreadsPerBlockScale;
  int group = threadIdx.x / ThreadsPerBlockScale;
  int lane = threadIdx.x % ThreadsPerBlockScale;
  int block = blockIdx.x * BlockScalesPerCta + group;
  int blocks = hidden / 32;
  if (block >= blocks) return;
  const size_t scaleStride = w4SourceScaleStride(hidden);
  for (int token = blockIdx.y; token < tokens; token += gridDim.y) {
    auto values = input + size_t(token) * hidden + block * 32 + lane * ValuesPerThread;
    int4 packed = *reinterpret_cast<const int4*>(values);
    mscclpp::bf16x8 packedValues = mscclpp::bit_cast<mscclpp::bf16x8>(packed);
    mscclpp::f32x8 decoded = mscclpp::to<mscclpp::f32x8>(packedValues);

    float maximum = 0.0f;
#pragma unroll
    for (int i = 0; i < ValuesPerThread; ++i) maximum = fmaxf(maximum, fabsf(decoded[i]));
#pragma unroll
    for (int offset = ThreadsPerBlockScale / 2; offset > 0; offset /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, ThreadsPerBlockScale));
    uint8_t scale = quantizeE8M0Scale(maximum);
    auto quantized = quantizeE4M3(decoded, inverseE8M0Scale(scale));
    auto output = quantizedInput + size_t(token) * hidden + block * 32 + lane * ValuesPerThread;
    *reinterpret_cast<uint2*>(output) = quantized.storage;
    if (lane == 0) quantizedInputScale[size_t(token) * scaleStride + block] = scale;
  }
}

__global__ void quantizeInputStaticKernel(const __bfloat16* input, uint8_t* quantizedInput, float staticScale,
                                          int tokens, int hidden) {
  constexpr int ValuesPerThread = 8;
  int col = (blockIdx.x * blockDim.x + threadIdx.x) * ValuesPerThread;
  if (col >= hidden) return;
  float invScale = staticScale != 0.0f ? (1.0f / staticScale) : 0.0f;
  for (int token = blockIdx.y; token < tokens; token += gridDim.y) {
    auto values = input + size_t(token) * hidden + col;
    int4 packed = *reinterpret_cast<const int4*>(values);
    mscclpp::bf16x8 packedValues = mscclpp::bit_cast<mscclpp::bf16x8>(packed);
    mscclpp::f32x8 decoded = mscclpp::to<mscclpp::f32x8>(packedValues);
    auto quantized = quantizeE4M3(decoded, invScale);
    auto output = quantizedInput + size_t(token) * hidden + col;
    *reinterpret_cast<uint2*>(output) = quantized.storage;
  }
}

template <class Types>
W4ScaleLayout<Types> makeW4ScaleLayout(int m, int n, int k, int experts, bool activation) {
  using namespace cute;
  ProblemShape shape{m, n, k, experts};
  if (!activation) return Types::ScaleConfig::tile_atom_to_shape_SFA(shape);
  auto layout = Types::ScaleConfig::tile_atom_to_shape_SFB(shape);
  return make_layout(cute::shape(layout), make_stride(get<0>(cute::stride(layout)), get<1>(cute::stride(layout)),
                                                      make_stride(_0{}, int32_t(0))));
}

template <class P>
P makeParameters(const NativeConfig& c, void* symmetric, const uint64_t* peers, void* workspace,
                 const PackedWeights& weights) {
  using namespace cute;
  using Mainloop = typename P::Collective::Mainloop;
  P p{};
  p.config = c;
  p.symmetric = getSymmetricLayout(c);
  p.local = symmetric;
  p.peers = peers;
  p.fc1TaskDivisor = cutlass::FastDivmod(2 * c.intermediate / P::Tiles::M);
  p.fc2TaskDivisor = cutlass::FastDivmod((c.hidden + P::Tiles::M - 1) / P::Tiles::M);
  size_t bytes;
  p.workspace = workspaceLayout(c, workspace, bytes);
  int experts = c.numExperts / c.worldSize;
  auto make = [&](int m, int k, int rows, uint8_t* weight, uint8_t* weightScale, void* input,
                  uint8_t* inputScale = nullptr) {
    ProblemShape shape{m, rows, k, experts};
    typename Mainloop::Arguments args{};
    args.ptr_A = reinterpret_cast<decltype(args.ptr_A)>(weight);
    args.dA = make_stride(int64_t(k), _1{}, int64_t(m) * k);
    args.ptr_B = reinterpret_cast<decltype(args.ptr_B)>(input);
    args.dB = make_stride(int64_t(k), _1{}, int64_t(0));
    if constexpr (P::WeightMxfp4) {
      args.ptr_SFA = reinterpret_cast<decltype(args.ptr_SFA)>(weightScale);
      args.layout_SFA = makeW4ScaleLayout<typename P::Collective>(m, rows, k, experts, false);
      args.ptr_SFB = reinterpret_cast<decltype(args.ptr_SFB)>(inputScale);
      args.layout_SFB = makeW4ScaleLayout<typename P::Collective>(m, rows, k, experts, true);
      const size_t actualSfaBytes = size_t(cosize(args.layout_SFA));
      const size_t actualSfbBytes = size_t(cosize(args.layout_SFB));
      const size_t expectedSfaBytes = size_t(m) * (k / 32) * experts;
      const size_t expectedSfbBytes = blockScaleBytes(rows, k);
      if (actualSfaBytes != expectedSfaBytes || actualSfbBytes != expectedSfbBytes)
        throw std::invalid_argument("MegaMoE W4A8 block-scale layout sizes are unsupported: SFA " +
                                    std::to_string(actualSfaBytes) + "/" + std::to_string(expectedSfaBytes) + ", SFB " +
                                    std::to_string(actualSfbBytes) + "/" + std::to_string(expectedSfbBytes));
    } else {
      args.ptr_S = reinterpret_cast<decltype(args.ptr_S)>(weightScale);
      args.layout_S = W8ScaleConfig::tile_atom_to_shape_scale(make_shape(m, k, experts));
    }
    if (!Mainloop::can_implement(shape, args))
      throw std::invalid_argument(P::WeightMxfp4 ? "MegaMoE W4A8 TMA input layout is unsupported"
                                                 : "MegaMoE TMA input layout is unsupported");
    return Mainloop::to_underlying_arguments(shape, args, nullptr);
  };
  if constexpr (P::WeightMxfp4) {
    const int activationRows = w4StorageRow<typename P::Tiles>(p.workspace.poolRows);
    p.fc1 = make(2 * c.intermediate, c.hidden, activationRows, weights.fc1, weights.fc1Scale,
                 p.workspace.quantizedInput, p.workspace.inputScale);
    p.fc2 = make(c.hidden, c.intermediate, activationRows, weights.fc2, weights.fc2Scale, p.workspace.quantizedHidden,
                 p.workspace.hiddenScale);
  } else {
    void* fc1Input = P::LocalExpert ? at<__bfloat16>(symmetric, p.symmetric.input)
                                    : reinterpret_cast<void*>(p.workspace.quantizedInput);
    p.fc1 = make(2 * c.intermediate, c.hidden, p.workspace.poolRows, weights.fc1, weights.fc1Scale, fc1Input);
    p.fc2 = make(c.hidden, c.intermediate, p.workspace.poolRows, weights.fc2, weights.fc2Scale,
                 p.workspace.quantizedHidden);
  }
  return p;
}

cudaLaunchConfig_t makeLaunchConfig(int ctas, int threads, size_t sharedBytes, cudaStream_t stream,
                                    std::array<cudaLaunchAttribute, 2>& attributes, int clusterSize) {
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim = {static_cast<unsigned int>(clusterSize), 1, 1};
  // Routing and the output join synchronize the complete resident grid.
  attributes[1].id = cudaLaunchAttributeCooperative;
  attributes[1].val.cooperative = 1;
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(ctas);
  launch.blockDim = dim3(threads);
  launch.dynamicSmemBytes = sharedBytes;
  launch.stream = stream;
  launch.attrs = attributes.data();
  launch.numAttrs = attributes.size();
  return launch;
}

}  // namespace detail

struct KernelPlan {
  std::variant<detail::W4A8Parameters, detail::W8A16Parameters, detail::W8A16LocalParameters> params;
  int ctas;
  int device;
  size_t sharedBytes = 0;
  bool localExpert = false;
};

void validateNativeConfig(const NativeConfig& c) {
  if (c.worldSize < 1 || c.worldSize > 72 || c.rank < 0 || c.rank >= c.worldSize)
    throw std::invalid_argument("MegaMoE requires 1 <= worldSize <= 72 and rank in [0, worldSize)");
  if (c.weightDtype != WeightDtype::Mxfp4 && c.weightDtype != WeightDtype::Mxfp8)
    throw std::invalid_argument("MegaMoE weightDtype must be Mxfp4 or Mxfp8");
  const int minHidden = c.weightDtype == WeightDtype::Mxfp4 ? 4096 : 128;
  if (c.maxTokens < 1 || c.hidden < minHidden || c.intermediate < 128 || c.hidden % 128 || c.intermediate % 128)
    throw std::invalid_argument("MegaMoE requires positive capacity, hidden >= " + std::to_string(minHidden) +
                                ", and H/I divisible by 128");
  if (c.hidden > std::numeric_limits<int>::max() / 2 || c.intermediate > std::numeric_limits<int>::max() / 2)
    throw std::invalid_argument("MegaMoE H/I exceed 32-bit indexing");
  if (c.numExperts < 1 || c.numExperts % c.worldSize || c.topK < 1 || c.topK > 32 || c.topK > c.numExperts)
    throw std::invalid_argument("MegaMoE requires evenly partitioned experts and 1 <= topK <= min(32, experts)");
  if (c.smMargin < 0 || !std::isfinite(c.gateUpClamp))
    throw std::invalid_argument("MegaMoE smMargin must be nonnegative and gateUpClamp must be finite");
  if (c.weightDtype == WeightDtype::Mxfp8) {
    if (c.activationDtype != ActivationDtype::Bf16)
      throw std::invalid_argument("MegaMoE W8A16 (weightDtype=Mxfp8) requires activationDtype=Bf16");
  } else {
    if (c.activationDtype != ActivationDtype::Mxfp8 && c.activationDtype != ActivationDtype::Fp8Static)
      throw std::invalid_argument("MegaMoE W4A8 requires activationDtype=Mxfp8 or Fp8Static");
    if (c.activationDtype == ActivationDtype::Fp8Static &&
        (!std::isfinite(c.activationScale) || c.activationScale <= 0.0f))
      throw std::invalid_argument("MegaMoE static FP8 activationScale must be a finite positive float");
    if (c.worldSize == 1 && c.numExperts == 1 && c.topK == 1)
      throw std::invalid_argument("MegaMoE W4A8 supports routed experts only");
  }
  if (size_t(c.worldSize) * c.maxTokens * c.topK > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("MegaMoE routing capacity exceeds 32-bit indexing");
  if (!detail::isLocalExpert(c)) {
    const size_t maxScratchInts = c.weightDtype == WeightDtype::Mxfp4
                                      ? sizeof(detail::W4A8SharedStorage::routing) / sizeof(int)
                                      : sizeof(detail::W8A16Parameters::SharedStorage::routing) / sizeof(int);
    if (detail::routingScratchHeaderInts(c.numExperts / c.worldSize, c.worldSize) > maxScratchInts)
      throw std::invalid_argument("MegaMoE local experts exceed the specialization's routing scratch");
  }
  if (c.weightDtype == WeightDtype::Mxfp4) (void)detail::w4a8KernelEntry(c);
}

SymmetricLayout getSymmetricLayout(const NativeConfig& c) {
  validateNativeConfig(c);
  SymmetricLayout layout{};
  const size_t inputTokens =
      detail::isLocalExpert(c) ? detail::aligned(c.maxTokens, detail::LocalTokenAlignment) : size_t(c.maxTokens);
  layout.input = detail::appendRegion(layout.bytes, inputTokens * c.hidden * sizeof(__bfloat16));
  layout.topkIds = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * sizeof(int));
  layout.topkWeights = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * sizeof(float));
  const size_t partialElemBytes =
      c.activationDtype == ActivationDtype::Fp8Static ? sizeof(uint8_t) : sizeof(__bfloat16);
  layout.partialOutput = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * c.hidden * partialElemBytes);
  layout.epoch = detail::appendRegion(layout.bytes, sizeof(uint64_t));
  layout.peerSignals = detail::appendRegion(layout.bytes, size_t(c.worldSize) * sizeof(uint64_t));
  layout.expectedPeerSignals = detail::appendRegion(layout.bytes, size_t(c.worldSize) * sizeof(uint64_t));
  layout.tokenCount = detail::appendRegion(layout.bytes, size_t(c.worldSize) * sizeof(mscclpp::LL8Packet));
  if (!detail::isLocalExpert(c)) {
    layout.routingHeaders = detail::appendRegion(
        layout.bytes, size_t(c.worldSize) * detail::routeChunks(c.maxTokens * c.topK) * sizeof(mscclpp::LL8Packet));
    layout.routingPackets =
        detail::appendRegion(layout.bytes, size_t(c.worldSize) * c.maxTokens * c.topK * sizeof(mscclpp::LL8Packet));
  }
  if (c.weightDtype == WeightDtype::Mxfp4) {
    layout.quantizedInput = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.hidden);
    layout.quantizedInputScale =
        detail::appendRegion(layout.bytes, size_t(c.maxTokens) * detail::w4SourceScaleStride(c.hidden));
  }
  layout.bytes = detail::aligned(layout.bytes);
  return layout;
}

size_t getPrivateWorkspaceBytes(const NativeConfig& c) {
  validateNativeConfig(c);
  size_t bytes;
  detail::workspaceLayout(c, nullptr, bytes);
  return bytes;
}

void packNativeWeights(const NativeConfig& c, const PackedWeights& source, const PackedWeights& destination,
                       cudaStream_t stream) {
  validateNativeConfig(c);
  if (!source.fc1 || !source.fc1Scale || !source.fc2 || !source.fc2Scale || !destination.fc1 || !destination.fc1Scale ||
      !destination.fc2 || !destination.fc2Scale)
    throw std::invalid_argument("MegaMoE weight buffers must be non-null");
  if (c.weightDtype == WeightDtype::Mxfp4) {
    int experts = c.numExperts / c.worldSize;
    auto fc1ScaleLayout =
        detail::makeW4ScaleLayout<detail::W4A8CollectiveTypes>(2 * c.intermediate, 1, c.hidden, experts, false);
    auto fc2ScaleLayout =
        detail::makeW4ScaleLayout<detail::W4A8CollectiveTypes>(c.hidden, 1, c.intermediate, experts, false);
    if (size_t(cute::cosize(fc1ScaleLayout)) != size_t(experts) * 2 * c.intermediate * (c.hidden / 32) ||
        size_t(cute::cosize(fc2ScaleLayout)) != size_t(experts) * c.hidden * (c.intermediate / 32))
      throw std::invalid_argument("MegaMoE W4A8 weight-scale layout size is unsupported");
    detail::packW4A8WeightsKernel<detail::W4A8CollectiveTypes>
        <<<256, 256, 0, stream>>>(c, source, destination, fc1ScaleLayout, fc2ScaleLayout);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    return;
  }
  detail::packWeightsKernel<<<256, 256, 0, stream>>>(c, source, destination);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

KernelResources preflightKernel(const NativeConfig& c) {
  validateNativeConfig(c);
  KernelResources resources{};
  MSCCLPP_CUDATHROW(cudaGetDevice(&resources.device));
  cudaDeviceProp properties{};
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&properties, resources.device));
  if (properties.major != 10 || (properties.minor != 0 && properties.minor != 3 && properties.minor != 7))
    throw std::invalid_argument("Native MegaMoE currently requires an SM100-family GPU");
  const int clusterSize = c.weightDtype == WeightDtype::Mxfp4 ? detail::W4ClusterSize : 2;
  if (c.smMargin > properties.multiProcessorCount - clusterSize)
    throw std::invalid_argument("MegaMoE smMargin must leave enough SMs for the selected cluster");
  resources.ctas = (properties.multiProcessorCount - c.smMargin) / clusterSize * clusterSize;
  const std::string name = c.weightDtype == WeightDtype::Mxfp4 ? "MegaMoE W4A8" : "MegaMoE W8A16";
  auto configure = [&](auto kernel, int threads, int entryRegisters, size_t sharedBytes) {
    resources.sharedBytes = std::max(resources.sharedBytes, sharedBytes);
    cudaFuncAttributes attributes{};
    MSCCLPP_CUDATHROW(cudaFuncGetAttributes(&attributes, kernel));
    if (attributes.numRegs < entryRegisters)
      throw std::runtime_error(name + " entry register allocation is too small for warpgroup reconfiguration");
    if (resources.sharedBytes + attributes.sharedSizeBytes > properties.sharedMemPerBlockOptin)
      throw std::invalid_argument(name + " specialization exceeds the device's shared memory limit");
    MSCCLPP_CUDATHROW(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(resources.sharedBytes)));
    std::array<cudaLaunchAttribute, 2> launchAttributes{};
    auto launch = detail::makeLaunchConfig(resources.ctas, threads, resources.sharedBytes, nullptr, launchAttributes,
                                           clusterSize);
    int clusters = 0;
    MSCCLPP_CUDATHROW(cudaOccupancyMaxActiveClusters(&clusters, kernel, &launch));
    resources.ctas = std::min(resources.ctas, clusters * clusterSize);
    if (resources.ctas < clusterSize) throw std::runtime_error(name + " cannot keep the selected cluster resident");
  };
  if (c.weightDtype == WeightDtype::Mxfp4) {
    configure(detail::w4a8KernelEntry(c), detail::W4A8Parameters::ThreadCount, detail::EntryRegisters,
              sizeof(detail::W4A8SharedStorage));
    configure(detail::w4a8KernelEntry(c, true), detail::W4A8Parameters::ThreadCount, detail::EntryRegisters,
              sizeof(detail::W4A8SharedStorage));
    return resources;
  }
#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
  if (detail::isLocalExpert(c)) {
    configure(detail::w8a16KernelEntry<1>(), detail::W8A16LocalParameters::ThreadCount, detail::LocalEntryRegisters,
              sizeof(detail::W8A16LocalParameters::SharedStorage));
    configure(detail::w8a16KernelEntry<2>(), detail::W8A16LocalParameters::ThreadCount, detail::LocalEntryRegisters,
              sizeof(detail::W8A16LocalParameters::SharedStorage));
  } else {
    configure(detail::w8a16KernelEntry<0>(), detail::W8A16Parameters::ThreadCount, detail::EntryRegisters,
              sizeof(detail::W8A16Parameters::SharedStorage));
  }
#endif
  return resources;
}

std::shared_ptr<KernelPlan> createKernelPlan(const NativeConfig& c, void* symmetric, const uint64_t* peers,
                                             void* workspace, const PackedWeights& weights) {
  if (!symmetric || !peers || !workspace || !weights.fc1 || !weights.fc1Scale || !weights.fc2 || !weights.fc2Scale)
    throw std::invalid_argument("MegaMoE plan requires live workspaces, peer addresses, and packed weights");
  const auto resources = preflightKernel(c);
  auto plan = std::make_shared<KernelPlan>();
  plan->device = resources.device;
  plan->ctas = resources.ctas;
  plan->sharedBytes = resources.sharedBytes;
  plan->localExpert = detail::isLocalExpert(c);
  if (c.weightDtype == WeightDtype::Mxfp4) {
    plan->params = detail::makeParameters<detail::W4A8Parameters>(c, symmetric, peers, workspace, weights);
  } else if (plan->localExpert) {
    plan->params = detail::makeParameters<detail::W8A16LocalParameters>(c, symmetric, peers, workspace, weights);
  } else {
    plan->params = detail::makeParameters<detail::W8A16Parameters>(c, symmetric, peers, workspace, weights);
  }
  return plan;
}

int kernelPlanCtaCount(const KernelPlan& plan) { return plan.ctas; }
size_t kernelPlanSharedBytes(const KernelPlan& plan) { return plan.sharedBytes; }

void launchNativeW4A8(const std::shared_ptr<KernelPlan>& plan, const void* input, const int32_t* ids,
                      const float* scores, int tokens, void* output, cudaStream_t stream, uint32_t* kernelEntrySignal) {
  if (!plan) throw std::invalid_argument("MegaMoE kernel plan is null");
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != plan->device) throw std::invalid_argument("MegaMoE must launch on the device owning its workspace");
  auto& params = std::get<detail::W4A8Parameters>(plan->params);
  if (tokens < 0 || tokens > params.config.maxTokens || (tokens && !output))
    throw std::invalid_argument("MegaMoE token count or output pointer is invalid");
  cudaStreamCaptureStatus capture;
  MSCCLPP_CUDATHROW(cudaStreamIsCapturing(stream, &capture));
  if (tokens) {
    dim3 threads(128);
    dim3 blocks((params.config.hidden / 32 + 31) / 32, std::min(tokens, 65535));
    const auto* source =
        input ? static_cast<const __bfloat16*>(input) : detail::at<__bfloat16>(params.local, params.symmetric.input);
    if (params.config.activationDtype == ActivationDtype::Fp8Static) {
      detail::quantizeInputStaticKernel<<<blocks, threads, 0, stream>>>(
          source, detail::at<uint8_t>(params.local, params.symmetric.quantizedInput), params.config.activationScale,
          tokens, params.config.hidden);
    } else {
      detail::quantizeInputKernel<<<blocks, threads, 0, stream>>>(
          source, detail::at<uint8_t>(params.local, params.symmetric.quantizedInput),
          detail::at<uint8_t>(params.local, params.symmetric.quantizedInputScale), tokens, params.config.hidden);
    }
    MSCCLPP_CUDATHROW(cudaGetLastError());
  }
  std::array<cudaLaunchAttribute, 2> launchAttributes{};
  auto launch = detail::makeLaunchConfig(plan->ctas, detail::W4A8Parameters::ThreadCount, plan->sharedBytes, stream,
                                         launchAttributes, detail::W4ClusterSize);
  MSCCLPP_CUDATHROW(
      cudaLaunchKernelEx(&launch, detail::w4a8KernelEntry(params.config, capture == cudaStreamCaptureStatusActive),
                         params, tokens, static_cast<__bfloat16*>(output), kernelEntrySignal, ids, scores));
}

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
template <int LocalMode>
void launchW8A16Mode(const std::shared_ptr<KernelPlan>& plan, const int32_t* ids, const float* scores, int tokens,
                     void* output, cudaStream_t stream, uint32_t* kernelEntrySignal = nullptr) {
  if (!plan) throw std::invalid_argument("MegaMoE kernel plan is null");
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != plan->device) throw std::invalid_argument("MegaMoE must launch on the device owning its workspace");
  using Parameters = std::conditional_t<LocalMode != 0, detail::W8A16LocalParameters, detail::W8A16Parameters>;
  auto& params = std::get<Parameters>(plan->params);
  if (tokens < 0 || tokens > params.config.maxTokens || (tokens && !output))
    throw std::invalid_argument("MegaMoE token count or output pointer is invalid");
  if constexpr (LocalMode != 0) {
    if (tokens) {
      const size_t tokenBlocks = (tokens + detail::W4TileN - 1) / detail::W4TileN;
      MSCCLPP_CUDATHROW(cudaMemsetAsync(params.workspace.hiddenReady, 0,
                                        tokenBlocks * detail::W4ReadyCounterStride * sizeof(int), stream));
    }
  }
  std::array<cudaLaunchAttribute, 2> launchAttributes{};
  auto launch = detail::makeLaunchConfig(plan->ctas, Parameters::ThreadCount, plan->sharedBytes, stream,
                                         launchAttributes, Parameters::Tiles::ClusterSize);
  MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&launch, detail::w8a16KernelEntry<LocalMode>(), params, tokens,
                                       static_cast<__bfloat16*>(output), kernelEntrySignal, ids, scores));
}

void launchNativeW8A16(const std::shared_ptr<KernelPlan>& plan, const int32_t* ids, const float* scores, int tokens,
                       void* output, cudaStream_t stream, uint32_t* kernelEntrySignal) {
  if (plan && plan->localExpert) {
    launchW8A16Mode<1>(plan, ids, scores, tokens, output, stream, kernelEntrySignal);
    return;
  }
  launchW8A16Mode<0>(plan, ids, scores, tokens, output, stream, kernelEntrySignal);
}

void launchNativeSharedExpert(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream) {
  if (!plan || !plan->localExpert) throw std::invalid_argument("Shared forward requires a single local W8A16 expert");
  launchW8A16Mode<2>(plan, nullptr, nullptr, tokens, output, stream);
}
#endif

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE
