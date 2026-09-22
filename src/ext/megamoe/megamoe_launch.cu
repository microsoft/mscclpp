// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>
#include <type_traits>
#include <variant>

#include "megamoe_device.cuh"
#include "megamoe_kernel.hpp"

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

Workspace workspaceLayout(const NativeConfig& c, void* base, size_t& bytes) {
  const size_t experts = c.numExperts / c.worldSize;
  const size_t routes = size_t(c.worldSize) * c.maxTokens * c.topK;
  const bool local = isLocalExpert(c);
  const int tileM = local ? LocalTileM : TileM;
  const int tileN = local ? LocalTileN : TileN;
  const size_t rows =
      local ? aligned(c.maxTokens, LocalTokenAlignment) : aligned(routes + experts * (tileN - 1), tileN);
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
    w.inputReady = at<int>(base, appendRegion(bytes, rows / TileN * sizeof(int)));
    w.hiddenReady = at<int>(base, appendRegion(bytes, rows / TileN * sizeof(int)));
    w.routes = at<Route>(base, appendRegion(bytes, rows * sizeof(Route)));
    w.blocks = at<TokenBlock>(base, appendRegion(bytes, rows / TileN * sizeof(TokenBlock)));
    w.input = at<__bfloat16>(base, appendRegion(bytes, rows * c.hidden * sizeof(__bfloat16)));
  }
  w.hidden = at<__bfloat16>(base, appendRegion(bytes, rows * c.intermediate * sizeof(__bfloat16)));
  w.poolRows = int(rows);
  bytes = aligned(bytes);
  return w;
}

__global__ void packWeightsKernel(NativeConfig c, PackedWeights src, PackedWeights dst) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  int experts = c.numExperts / c.worldSize;
  size_t fc1Size = size_t(experts) * 2 * c.intermediate * c.hidden;
  size_t fc2Size = size_t(experts) * c.hidden * c.intermediate;
  for (size_t i = idx; i < fc1Size; i += stride) {
    int k = i % c.hidden;
    size_t row = i / c.hidden;
    int m = row % (2 * c.intermediate);
    int canonical = (m / 32 * 16 + m % 16) + (m % 32 >= 16 ? c.intermediate : 0);
    dst.fc1[i] = src.fc1[(row / (2 * c.intermediate) * 2 * c.intermediate + canonical) * c.hidden + k];
  }
  for (size_t i = idx; i < fc2Size; i += stride) dst.fc2[i] = src.fc2[i];
  for (size_t i = idx; i < fc1Size / 32; i += stride) {
    int m = i % (2 * c.intermediate);
    int k = i / (2 * c.intermediate) % (c.hidden / 32);
    size_t expert = i / (size_t(2) * c.intermediate * (c.hidden / 32));
    int canonical = (m / 32 * 16 + m % 16) + (m % 32 >= 16 ? c.intermediate : 0);
    dst.fc1Scale[i] = src.fc1Scale[(expert * 2 * c.intermediate + canonical) * (c.hidden / 32) + k];
  }
  for (size_t i = idx; i < fc2Size / 32; i += stride) {
    int m = i % c.hidden;
    int k = i / c.hidden % (c.intermediate / 32);
    size_t expert = i / (size_t(c.hidden) * (c.intermediate / 32));
    dst.fc2Scale[i] = src.fc2Scale[(expert * c.hidden + m) * (c.intermediate / 32) + k];
  }
}

template <bool E5M2, bool Local = false>
Parameters<E5M2, Local> makeParameters(const NativeConfig& c, void* symmetric, const uint64_t* peers, void* workspace,
                                       const PackedWeights& weights) {
  using namespace cute;
  using Mainloop = typename CollectiveTypes<E5M2, Local>::Mainloop;
  using Weight = typename CollectiveTypes<E5M2, Local>::Weight;
  using Scale = typename CollectiveTypes<E5M2, Local>::Scale;
  using Activation = typename CollectiveTypes<E5M2, Local>::Activation;
  Parameters<E5M2, Local> p{};
  p.config = c;
  p.symmetric = getSymmetricLayout(c);
  p.local = symmetric;
  p.peers = peers;
  size_t bytes;
  p.workspace = workspaceLayout(c, workspace, bytes);
  int experts = c.numExperts / c.worldSize;
  auto make = [&](int m, int k, int rows, uint8_t* weight, uint8_t* scale, __bfloat16* input) {
    ProblemShape shape{m, rows, k, experts};
    typename Mainloop::Arguments args{};
    args.ptr_A = reinterpret_cast<const Weight*>(weight);
    args.dA = make_stride(int64_t(k), _1{}, int64_t(m) * k);
    args.ptr_B = reinterpret_cast<const Activation*>(input);
    args.dB = make_stride(int64_t(k), _1{}, int64_t(0));
    args.ptr_S = reinterpret_cast<const Scale*>(scale);
    args.layout_S = ScaleConfig::tile_atom_to_shape_scale(make_shape(m, k, experts));
    if (!Mainloop::can_implement(shape, args)) throw std::invalid_argument("MegaMoE TMA input layout is unsupported");
    return Mainloop::to_underlying_arguments(shape, args, nullptr);
  };
  p.fc1 = make(2 * c.intermediate, c.hidden, p.workspace.poolRows, weights.fc1, weights.fc1Scale,
               isLocalExpert(c) ? at<__bfloat16>(symmetric, p.symmetric.input) : p.workspace.input);
  p.fc2 = make(c.hidden, c.intermediate, p.workspace.poolRows, weights.fc2, weights.fc2Scale, p.workspace.hidden);
  return p;
}

}  // namespace detail

struct KernelPlan {
  std::variant<detail::Parameters<false>, detail::Parameters<true>, detail::Parameters<false, true>,
               detail::Parameters<true, true>>
      params;
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
  auto configure = [&]<bool E5M2, int LocalMode>() {
    constexpr bool Local = LocalMode != 0;
    constexpr int Entry = Local ? detail::LocalEntryRegisters : detail::EntryRegisters;
    auto kernel = detail::kernelEntry<E5M2, LocalMode>();
    resources.sharedBytes = std::max(resources.sharedBytes, sizeof(detail::SharedStorage<E5M2, Local>));
    cudaFuncAttributes attributes{};
    MSCCLPP_CUDATHROW(cudaFuncGetAttributes(&attributes, kernel));
    if (attributes.numRegs < Entry)
      throw std::runtime_error("MegaMoE entry register allocation is too small for warpgroup reconfiguration");
    if (resources.sharedBytes + attributes.sharedSizeBytes > properties.sharedMemPerBlockOptin)
      throw std::invalid_argument("MegaMoE specialization exceeds the device's shared memory limit");
    MSCCLPP_CUDATHROW(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(resources.sharedBytes)));
    cudaLaunchAttribute attribute{};
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {detail::ClusterM, 1, 1};
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(resources.ctas);
    launch.blockDim = dim3(Local ? detail::LocalThreads : detail::Threads);
    launch.dynamicSmemBytes = resources.sharedBytes;
    launch.attrs = &attribute;
    launch.numAttrs = 1;
    int clusters = 0;
    MSCCLPP_CUDATHROW(cudaOccupancyMaxActiveClusters(&clusters, kernel, &launch));
    resources.ctas = std::min(resources.ctas, clusters * detail::ClusterM);
    if (resources.ctas < detail::ClusterM) throw std::runtime_error("MegaMoE cannot keep a two-CTA cluster resident");
  };
  if (detail::isLocalExpert(c)) {
    if (c.weightE5M2) {
      configure.template operator()<true, 1>();
      configure.template operator()<true, 2>();
    } else {
      configure.template operator()<false, 1>();
      configure.template operator()<false, 2>();
    }
  } else if (c.weightE5M2) {
    configure.template operator()<true, 0>();
  } else {
    configure.template operator()<false, 0>();
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
  if (plan->localExpert && c.weightE5M2) {
    plan->params = detail::makeParameters<true, true>(c, symmetric, peers, workspace, weights);
  } else if (plan->localExpert) {
    plan->params = detail::makeParameters<false, true>(c, symmetric, peers, workspace, weights);
  } else if (c.weightE5M2) {
    plan->params = detail::makeParameters<true>(c, symmetric, peers, workspace, weights);
  } else {
    plan->params = detail::makeParameters<false>(c, symmetric, peers, workspace, weights);
  }
  return plan;
}

int kernelPlanCtaCount(const KernelPlan& plan) { return plan.ctas; }
size_t kernelPlanSharedBytes(const KernelPlan& plan) { return plan.sharedBytes; }

namespace {
void launchPlan(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream,
                uint32_t* startSignal, bool unweightedShared) {
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
        if (plan->localExpert && tokens) {
          const size_t tokenBlocks = (tokens + detail::LocalTileN - 1) / detail::LocalTileN;
          MSCCLPP_CUDATHROW(cudaMemsetAsync(params.workspace.hiddenReady, 0, tokenBlocks * sizeof(int), stream));
        }
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim = {detail::ClusterM, 1, 1};
        cudaLaunchConfig_t launch{};
        launch.gridDim = dim3(plan->ctas);
        launch.blockDim = dim3(plan->localExpert ? detail::LocalThreads : detail::Threads);
        launch.dynamicSmemBytes = plan->sharedBytes;
        launch.stream = stream;
        launch.attrs = &attribute;
        launch.numAttrs = 1;
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
      },
      plan->params);
}
}  // namespace

void launchNativeMegaMoe(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream,
                         uint32_t* startSignal) {
  launchPlan(plan, tokens, output, stream, startSignal, false);
}

void launchNativeSharedExpert(const std::shared_ptr<KernelPlan>& plan, int tokens, void* output, cudaStream_t stream) {
  launchPlan(plan, tokens, output, stream, nullptr, true);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE
