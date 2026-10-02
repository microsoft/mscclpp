// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#include "megamoe_device.cuh"
#include "megamoe_kernel.hpp"

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
#include "megamoe_quantization.cuh"
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE {
namespace detail {

size_t aligned(size_t n, size_t alignment = 256) { return (n + alignment - 1) / alignment * alignment; }

bool isLocalExpert(const NativeConfig& c) { return c.worldSize == 1 && c.numExperts == 1 && c.topK == 1; }

size_t appendRegion(size_t& bytes, size_t regionBytes) {
  size_t offset = aligned(bytes);
  if (regionBytes > std::numeric_limits<size_t>::max() - offset) {
    throw std::overflow_error("MegaMoE workspace size overflows size_t");
  }
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

Workspace workspaceLayout(const NativeConfig& c, void* base, size_t& bytes) {
  const size_t experts = c.numExperts / c.worldSize;
  const size_t routes = size_t(c.worldSize) * c.maxTokens * c.topK;
  const bool local = isLocalExpert(c);
  const int tileM = local ? LocalTileM : TileM;
  const int tileN = local ? LocalTileN : (c.weightMxfp4 ? W4TileN : TileN);
  const size_t rows =
      local ? aligned(c.maxTokens, LocalTokenAlignment) : aligned(routes + experts * (tileN - 1), tileN);
  const size_t activationRows = c.weightMxfp4 ? rows / W4TileN * W4TokenStride : rows;
  if (activationRows > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("MegaMoE activation workspace exceeds 32-bit row indexing");
  if (rows > size_t(std::numeric_limits<int>::max()) ||
      (rows + tileN - 1) / tileN *
              ((2 * size_t(c.intermediate) + tileM - 1) / tileM + (size_t(c.hidden) + tileM - 1) / tileM) >
          size_t(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("MegaMoE routing workspace exceeds 32-bit tile indexing");
  }
  Workspace w{};
  bytes = 0;
  w.control = at<Control>(base, appendRegion(bytes, sizeof(Control)));
  if (local) {
    w.hiddenReady = at<int>(base, appendRegion(bytes, (rows + tileN - 1) / tileN * sizeof(int)));
  } else {
    w.counts = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
    w.starts = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
    w.cursors = at<int>(base, appendRegion(bytes, experts * sizeof(int)));
    w.inputReady = at<int>(base, appendRegion(bytes, rows / tileN * sizeof(int)));
    w.hiddenReady = at<int>(base, appendRegion(bytes, rows / tileN * sizeof(int)));
    w.peerTokenCounts = at<int>(base, appendRegion(bytes, size_t(c.worldSize) * sizeof(int)));
    w.peerTokenOffsets = at<int>(base, appendRegion(bytes, size_t(c.worldSize + 1) * sizeof(int)));
    if (c.weightMxfp4 && w4InputChunks(c.hidden) > 1)
      w.inputChunkReady =
          at<int>(base, appendRegion(bytes, rows / tileN * (w4InputChunks(c.hidden) - 1) * sizeof(int)));
    w.routes = at<Route>(base, appendRegion(bytes, rows * sizeof(Route)));
    w.blocks = at<TokenBlock>(base, appendRegion(bytes, rows / tileN * sizeof(TokenBlock)));
    if (c.weightMxfp4) {
      w.quantizedInput = at<uint8_t>(base, appendRegion(bytes, activationRows * c.hidden));
      w.inputScale = at<uint8_t>(base, appendRegion(bytes, blockScaleBytes(activationRows, c.hidden)));
    } else {
      w.input = at<__bfloat16>(base, appendRegion(bytes, rows * c.hidden * sizeof(__bfloat16)));
    }
  }
  if (c.weightMxfp4) {
    w.quantizedHidden = at<uint8_t>(base, appendRegion(bytes, activationRows * c.intermediate));
    w.hiddenScale = at<uint8_t>(base, appendRegion(bytes, blockScaleBytes(activationRows, c.intermediate)));
  } else {
    w.hidden = at<__bfloat16>(base, appendRegion(bytes, rows * c.intermediate * sizeof(__bfloat16)));
  }
  w.poolRows = int(rows);
  bytes = aligned(bytes);
  return w;
}

__device__ __forceinline__ int canonicalFc1Row(int row, int intermediate) {
  return row / 32 * 16 + row % 16 + (row % 32 >= 16 ? intermediate : 0);
}

template <int ValuesPerByte>
__device__ __forceinline__ void packWeightValues(const NativeConfig& c, const PackedWeights& src,
                                                 const PackedWeights& dst, size_t idx, size_t stride) {
  int experts = c.numExperts / c.worldSize;
  int rowBytes = c.hidden / ValuesPerByte;
  size_t fc1Size = size_t(experts) * 2 * c.intermediate * rowBytes;
  size_t fc2Size = size_t(experts) * c.hidden * (c.intermediate / ValuesPerByte);
  for (size_t i = idx; i < fc1Size; i += stride) {
    int k = i % rowBytes;
    size_t row = i / rowBytes;
    int m = row % (2 * c.intermediate);
    int canonical = canonicalFc1Row(m, c.intermediate);
    dst.fc1[i] = src.fc1[(row / (2 * c.intermediate) * 2 * c.intermediate + canonical) * rowBytes + k];
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

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
using W4ScaleLayout = typename W4A8CollectiveTypes::Mainloop::LayoutSFA;

__global__ void packW4A8WeightsKernel(NativeConfig c, PackedWeights src, PackedWeights dst,
                                      W4ScaleLayout fc1ScaleLayout, W4ScaleLayout fc2ScaleLayout) {
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

__global__ void quantizeInputKernel(W4A8Parameters p, int tokens, const __bfloat16* source) {
  constexpr int ValuesPerThread = 8;
  constexpr int ThreadsPerBlockScale = 4;
  constexpr int BlockScalesPerCta = 128 / ThreadsPerBlockScale;
  int group = threadIdx.x / ThreadsPerBlockScale;
  int lane = threadIdx.x % ThreadsPerBlockScale;
  int block = blockIdx.x * BlockScalesPerCta + group;
  int blocks = p.config.hidden / 32;
  bool active = block < blocks;
  for (int token = blockIdx.y; token < tokens; token += gridDim.y) {
    if (active) {
      auto staged = at<__bfloat16>(p.local, p.symmetric.input) + size_t(token) * p.config.hidden + block * 32;
      auto input = source ? source + size_t(token) * p.config.hidden + block * 32 : staged;
      auto values = input + lane * ValuesPerThread;
      auto stagedValues = staged + lane * ValuesPerThread;
      int4 packed = *reinterpret_cast<const int4*>(values);
      if (values != stagedValues) *reinterpret_cast<int4*>(stagedValues) = packed;
      mscclpp::bf16x8 packedValues = mscclpp::bit_cast<mscclpp::bf16x8>(packed);
      mscclpp::f32x8 decoded = mscclpp::to<mscclpp::f32x8>(packedValues);

      float maximum = 0.0f;
      CUTE_UNROLL
      for (int i = 0; i < ValuesPerThread; ++i) maximum = fmaxf(maximum, fabsf(decoded[i]));
      unsigned int activeMask = __activemask();
      CUTE_UNROLL
      for (int offset = ThreadsPerBlockScale / 2; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(activeMask, maximum, offset, ThreadsPerBlockScale));
      uint8_t scale = quantizeE8M0Scale(maximum);
      float inverse = inverseE8M0Scale(scale);
      mscclpp::f32x8 normalized;
      CUTE_UNROLL
      for (int i = 0; i < ValuesPerThread; ++i) normalized[i] = decoded[i] * inverse;
      mscclpp::f8_e4m3x8 quantized = mscclpp::to<mscclpp::f8_e4m3x8>(normalized);
      auto output = at<uint8_t>(p.local, p.symmetric.quantizedInput) + size_t(token) * p.config.hidden + block * 32 +
                    lane * ValuesPerThread;
      *reinterpret_cast<uint2*>(output) = quantized.storage;
      if (lane == 0)
        at<uint8_t>(p.local,
                    p.symmetric.quantizedInputScale)[size_t(token) * w4SourceScaleStride(p.config.hidden) + block] =
            scale;
    }
  }
}

W4ScaleLayout makeW4ScaleLayout(int m, int n, int k, int experts, bool activation) {
  using namespace cute;
  ProblemShape shape{m, n, k, experts};
  if (!activation) return W4A8CollectiveTypes::ScaleConfig::tile_atom_to_shape_SFA(shape);
  auto layout = W4A8CollectiveTypes::ScaleConfig::tile_atom_to_shape_SFB(shape);
  return make_layout(cute::shape(layout), make_stride(get<0>(cute::stride(layout)), get<1>(cute::stride(layout)),
                                                      make_stride(_0{}, int32_t(0))));
}

#endif

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
  if constexpr (P::WeightMxfp4) {
    p.fc1TaskDivisor = cutlass::FastDivmod(2 * c.intermediate / P::Tiles::M);
    p.fc2TaskDivisor = cutlass::FastDivmod((c.hidden + P::Tiles::M - 1) / P::Tiles::M);
  }
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
#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
    if constexpr (P::WeightMxfp4) {
      args.ptr_SFA = reinterpret_cast<decltype(args.ptr_SFA)>(weightScale);
      args.layout_SFA = makeW4ScaleLayout(m, rows, k, experts, false);
      args.ptr_SFB = reinterpret_cast<decltype(args.ptr_SFB)>(inputScale);
      args.layout_SFB = makeW4ScaleLayout(m, rows, k, experts, true);
      const size_t actualSfaBytes = size_t(cosize(args.layout_SFA));
      const size_t actualSfbBytes = size_t(cosize(args.layout_SFB));
      const size_t expectedSfaBytes = size_t(m) * (k / 32) * experts;
      const size_t expectedSfbBytes = blockScaleBytes(rows, k);
      if (actualSfaBytes != expectedSfaBytes || actualSfbBytes != expectedSfbBytes)
        throw std::invalid_argument("MegaMoE W4A8 block-scale layout sizes are unsupported: SFA " +
                                    std::to_string(actualSfaBytes) + "/" + std::to_string(expectedSfaBytes) + ", SFB " +
                                    std::to_string(actualSfbBytes) + "/" + std::to_string(expectedSfbBytes));
    } else
#endif
    {
      args.ptr_S = reinterpret_cast<decltype(args.ptr_S)>(weightScale);
      args.layout_S = ScaleConfig::tile_atom_to_shape_scale(make_shape(m, k, experts));
    }
    if (!Mainloop::can_implement(shape, args))
      throw std::invalid_argument(P::WeightMxfp4 ? "MegaMoE W4A8 TMA input layout is unsupported"
                                                 : "MegaMoE TMA input layout is unsupported");
    return Mainloop::to_underlying_arguments(shape, args, nullptr);
  };
  if constexpr (P::WeightMxfp4) {
    const int activationRows = w4StorageRow(p.workspace.poolRows);
    p.fc1 = make(2 * c.intermediate, c.hidden, activationRows, weights.fc1, weights.fc1Scale,
                 p.workspace.quantizedInput, p.workspace.inputScale);
    p.fc2 = make(c.hidden, c.intermediate, activationRows, weights.fc2, weights.fc2Scale, p.workspace.quantizedHidden,
                 p.workspace.hiddenScale);
  } else {
    p.fc1 = make(2 * c.intermediate, c.hidden, p.workspace.poolRows, weights.fc1, weights.fc1Scale,
                 isLocalExpert(c) ? at<__bfloat16>(symmetric, p.symmetric.input) : p.workspace.input);
    p.fc2 = make(c.hidden, c.intermediate, p.workspace.poolRows, weights.fc2, weights.fc2Scale, p.workspace.hidden);
  }
  return p;
}

cudaLaunchConfig_t makeLaunchConfig(int ctas, int threads, size_t sharedBytes, cudaStream_t stream,
                                    cudaLaunchAttribute& attribute) {
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {ClusterM, 1, 1};
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(ctas);
  launch.blockDim = dim3(threads);
  launch.dynamicSmemBytes = sharedBytes;
  launch.stream = stream;
  launch.attrs = &attribute;
  launch.numAttrs = 1;
  return launch;
}

}  // namespace detail

struct KernelPlan {
#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
  std::variant<detail::Parameters<false>, detail::Parameters<true>, detail::Parameters<false, true>,
               detail::Parameters<true, true>, detail::W4A8Parameters>
      params;
#else
  std::variant<detail::Parameters<false>, detail::Parameters<true>, detail::Parameters<false, true>,
               detail::Parameters<true, true>>
      params;
#endif
  int ctas;
  int device;
  size_t sharedBytes = 0;
  bool localExpert = false;
};

void validateNativeConfig(const NativeConfig& c) {
  if (c.worldSize < 1 || c.worldSize > 72 || c.rank < 0 || c.rank >= c.worldSize)
    throw std::invalid_argument("MegaMoE requires 1 <= worldSize <= 72 and rank in [0, worldSize)");
  if (c.maxTokens < 1 || c.hidden < 128 || c.intermediate < 128 || c.hidden % 128 || c.intermediate % 128)
    throw std::invalid_argument("MegaMoE requires positive capacity and H/I divisible by 128");
  if (c.hidden > std::numeric_limits<int>::max() / 2 || c.intermediate > std::numeric_limits<int>::max() / 2)
    throw std::invalid_argument("MegaMoE H/I exceed 32-bit indexing");
  if (c.numExperts < 1 || c.numExperts % c.worldSize || c.topK < 1 || c.topK > 32 || c.topK > c.numExperts)
    throw std::invalid_argument("MegaMoE requires evenly partitioned experts and 1 <= topK <= min(32, experts)");
  if (c.smMargin < 0 || !std::isfinite(c.gateUpClamp))
    throw std::invalid_argument("MegaMoE smMargin must be nonnegative and gateUpClamp must be finite");
  if (c.weightE5M2 && c.weightMxfp4)
    throw std::invalid_argument("MegaMoE weightE5M2 and weightMxfp4 are mutually exclusive");
  if (c.weightMxfp4 && detail::isLocalExpert(c))
    throw std::invalid_argument("MegaMoE W4A8 currently supports routed experts only");
  if (size_t(c.worldSize) * c.maxTokens * c.topK > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("MegaMoE routing capacity exceeds 32-bit indexing");
}

SymmetricLayout getSymmetricLayout(const NativeConfig& c) {
  validateNativeConfig(c);
  SymmetricLayout layout{};
  const size_t inputTokens =
      detail::isLocalExpert(c) ? detail::aligned(c.maxTokens, detail::LocalTokenAlignment) : size_t(c.maxTokens);
  layout.input = detail::appendRegion(layout.bytes, inputTokens * c.hidden * sizeof(__bfloat16));
  layout.topkIds = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * sizeof(int));
  layout.topkWeights = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * sizeof(float));
  layout.partialOutput =
      detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * c.hidden * sizeof(__bfloat16));
  layout.epoch = detail::appendRegion(layout.bytes, sizeof(uint64_t));
  layout.peerSignals = detail::appendRegion(layout.bytes, size_t(c.worldSize) * sizeof(uint64_t));
  layout.expectedPeerSignals = detail::appendRegion(layout.bytes, size_t(c.worldSize) * sizeof(uint64_t));
  layout.tokenCount = detail::appendRegion(layout.bytes, sizeof(int));
  if (!detail::isLocalExpert(c)) {
    layout.routingHeader = detail::appendRegion(layout.bytes, sizeof(mscclpp::LLPacket));
    layout.routingPackets =
        detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.topK * sizeof(mscclpp::LLPacket));
  }
#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
  if (c.weightMxfp4) {
    layout.quantizedInput = detail::appendRegion(layout.bytes, size_t(c.maxTokens) * c.hidden);
    layout.quantizedInputScale =
        detail::appendRegion(layout.bytes, size_t(c.maxTokens) * detail::w4SourceScaleStride(c.hidden));
  }
#endif
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
#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
  if (c.weightMxfp4) {
    int experts = c.numExperts / c.worldSize;
    auto fc1ScaleLayout = detail::makeW4ScaleLayout(2 * c.intermediate, 1, c.hidden, experts, false);
    auto fc2ScaleLayout = detail::makeW4ScaleLayout(c.hidden, 1, c.intermediate, experts, false);
    if (size_t(cute::cosize(fc1ScaleLayout)) != size_t(experts) * 2 * c.intermediate * (c.hidden / 32) ||
        size_t(cute::cosize(fc2ScaleLayout)) != size_t(experts) * c.hidden * (c.intermediate / 32))
      throw std::invalid_argument("MegaMoE W4A8 weight-scale layout size is unsupported");
    detail::packW4A8WeightsKernel<<<256, 256, 0, stream>>>(c, source, destination, fc1ScaleLayout, fc2ScaleLayout);
    MSCCLPP_CUDATHROW(cudaGetLastError());
    return;
  }
#endif
  detail::packWeightsKernel<<<256, 256, 0, stream>>>(c, source, destination);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

KernelResources preflightKernel(const NativeConfig& c) {
  validateNativeConfig(c);
  KernelResources resources{};
  MSCCLPP_CUDATHROW(cudaGetDevice(&resources.device));
  cudaDeviceProp properties{};
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&properties, resources.device));
  if (properties.major != 10 || properties.minor != 0)
    throw std::invalid_argument("Native MegaMoE currently requires an SM100 GPU");
  if (c.smMargin > properties.multiProcessorCount - 2)
    throw std::invalid_argument("MegaMoE smMargin must leave at least two SMs");
  resources.ctas = (properties.multiProcessorCount - c.smMargin) / detail::ClusterM * detail::ClusterM;
  const std::string name = c.weightMxfp4 ? "MegaMoE W4A8" : "MegaMoE";
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
    cudaLaunchAttribute attribute{};
    auto launch = detail::makeLaunchConfig(resources.ctas, threads, resources.sharedBytes, nullptr, attribute);
    int clusters = 0;
    MSCCLPP_CUDATHROW(cudaOccupancyMaxActiveClusters(&clusters, kernel, &launch));
    resources.ctas = std::min(resources.ctas, clusters * detail::ClusterM);
    if (resources.ctas < detail::ClusterM) throw std::runtime_error(name + " cannot keep a two-CTA cluster resident");
  };
#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
  if (c.weightMxfp4) {
    configure(detail::w4a8KernelEntry(c), detail::W4Threads, detail::EntryRegisters, sizeof(detail::W4A8SharedStorage));
    return resources;
  }
#endif
  auto configureW8A16 = [&]<bool E5M2, int LocalMode>() {
    constexpr bool Local = LocalMode != 0;
    configure(detail::kernelEntry<E5M2, LocalMode>(), Local ? detail::LocalThreads : detail::Threads,
              Local ? detail::LocalEntryRegisters : detail::EntryRegisters, sizeof(detail::SharedStorage<E5M2, Local>));
  };
  if (detail::isLocalExpert(c)) {
    if (c.weightE5M2) {
      configureW8A16.template operator()<true, 1>();
      configureW8A16.template operator()<true, 2>();
    } else {
      configureW8A16.template operator()<false, 1>();
      configureW8A16.template operator()<false, 2>();
    }
  } else if (c.weightE5M2) {
    configureW8A16.template operator()<true, 0>();
  } else {
    configureW8A16.template operator()<false, 0>();
  }
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
#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
  if (c.weightMxfp4) {
    plan->params = detail::makeParameters<detail::W4A8Parameters>(c, symmetric, peers, workspace, weights);
    return plan;
  }
#endif
  if (plan->localExpert && c.weightE5M2) {
    plan->params = detail::makeParameters<detail::Parameters<true, true>>(c, symmetric, peers, workspace, weights);
  } else if (plan->localExpert) {
    plan->params = detail::makeParameters<detail::Parameters<false, true>>(c, symmetric, peers, workspace, weights);
  } else if (c.weightE5M2) {
    plan->params = detail::makeParameters<detail::Parameters<true>>(c, symmetric, peers, workspace, weights);
  } else {
    plan->params = detail::makeParameters<detail::Parameters<false>>(c, symmetric, peers, workspace, weights);
  }
  return plan;
}

int kernelPlanCtaCount(const KernelPlan& plan) { return plan.ctas; }
size_t kernelPlanSharedBytes(const KernelPlan& plan) { return plan.sharedBytes; }

namespace {
void launchPlan(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream,
                uint32_t* startSignal, bool unweightedShared, const void* input = nullptr, const int32_t* ids = nullptr,
                const float* scores = nullptr) {
  if (!plan) throw std::invalid_argument("MegaMoE kernel plan is null");
  if (unweightedShared && !plan->localExpert)
    throw std::invalid_argument("Shared forward requires a single local expert");
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != plan->device) throw std::invalid_argument("MegaMoE must launch on the device owning its workspace");
  std::visit(
      [&](const auto& params) {
        if (tokens < 0 || tokens > params.config.maxTokens || (tokens && !output))
          throw std::invalid_argument("MegaMoE token count or output pointer is invalid");
        constexpr bool E5M2 = std::decay_t<decltype(params)>::WeightE5M2;
        constexpr bool W4A8 = std::decay_t<decltype(params)>::WeightMxfp4;
        if (plan->localExpert && tokens) {
          const size_t tokenBlocks = (tokens + detail::LocalTileN - 1) / detail::LocalTileN;
          MSCCLPP_CUDATHROW(cudaMemsetAsync(params.workspace.hiddenReady, 0, tokenBlocks * sizeof(int), stream));
        }
        cudaLaunchAttribute attribute{};
        auto launch = detail::makeLaunchConfig(plan->ctas, std::decay_t<decltype(params)>::ThreadCount,
                                               plan->sharedBytes, stream, attribute);
#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
        if constexpr (W4A8) {
          if (tokens) {
            dim3 threads(128);
            dim3 blocks((params.config.hidden / 32 + 31) / 32, std::min(tokens, 65535));
            detail::quantizeInputKernel<<<blocks, threads, 0, stream>>>(params, tokens,
                                                                        static_cast<const __bfloat16*>(input));
            MSCCLPP_CUDATHROW(cudaGetLastError());
          }
          MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&launch, detail::w4a8KernelEntry(params.config), params, tokens,
                                               static_cast<__bfloat16*>(output), startSignal, ids, scores));
          return;
        }
#endif
        if constexpr (!W4A8) {
          auto run = [&]<int LocalMode>(std::integral_constant<int, LocalMode>) {
            MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&launch, detail::kernelEntry<E5M2, LocalMode>(), params, tokens,
                                                 static_cast<__bfloat16*>(output), startSignal));
          };
          if constexpr (std::decay_t<decltype(params)>::LocalExpert) {
            if (unweightedShared) {
              run(std::integral_constant<int, 2>{});
            } else
              run(std::integral_constant<int, 1>{});
          } else {
            run(std::integral_constant<int, 0>{});
          }
        }
      },
      plan->params);
}
}  // namespace

void launchNativeMegaMoe(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream,
                         uint32_t* startSignal) {
  launchPlan(plan, tokens, output, stream, startSignal, false);
}

void launchNativeW4A8(const std::shared_ptr<KernelPlan>& plan, const void* input, const int32_t* ids,
                      const float* scores, int tokens, void* output, cudaStream_t stream, uint32_t* startSignal) {
  launchPlan(plan, tokens, output, stream, startSignal, false, input, ids, scores);
}

void launchNativeSharedExpert(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream) {
  launchPlan(plan, tokens, output, stream, nullptr, true);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE
