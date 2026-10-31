// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpu_net_io.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <mscclpp/env.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/semaphore.hpp>
#include <mscclpp/utils.hpp>
#include <vector>

namespace mscclpp::ep {
namespace {

constexpr Transport kIbTransports[] = {Transport::IB0, Transport::IB1, Transport::IB2, Transport::IB3,
                                       Transport::IB4, Transport::IB5, Transport::IB6, Transport::IB7};

Transport selectIbTransport(int cudaDevice) {
  if (cudaDevice < 0 || cudaDevice >= static_cast<int>(sizeof(kIbTransports) / sizeof(kIbTransports[0]))) {
    throw Error("CUDA device index out of IB transport range: " + std::to_string(cudaDevice), ErrorCode::InvalidUsage);
  }
  const std::string& hcaDevices = env()->hcaDevices;
  const Transport transport =
      (!hcaDevices.empty() && hcaDevices.find(',') == std::string::npos) ? Transport::IB0 : kIbTransports[cudaDevice];
  (void)getIBDeviceName(transport);
  return transport;
}

}  // namespace

struct EpGpuNetIoService::Impl {
  std::shared_ptr<Bootstrap> bootstrap;
  int device;
  bool initialized = false;
  std::shared_ptr<Communicator> communicator;
  std::vector<PortChannel> channels;
  std::shared_ptr<EpPortChannelDeviceHandle> handles;
  std::shared_ptr<EpGpuNetIoSignalState> signalStates;
  std::shared_ptr<EpGpuNetIoDeviceContext> context;

  ~Impl() {
    CudaDeviceGuard guard(device);
    context.reset();
    signalStates.reset();
    handles.reset();
    channels.clear();
    communicator.reset();
  }
};

EpGpuNetIoService::EpGpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, int cudaDevice)
    : impl_(std::make_unique<Impl>()) {
  impl_->bootstrap = std::move(bootstrap);
  impl_->device = cudaDevice;
}

EpGpuNetIoService::~EpGpuNetIoService() = default;

void EpGpuNetIoService::setup(void* buffer, size_t bytes) {
  auto& state = *impl_;
  CudaDeviceGuard guard(state.device);
  if (state.initialized) throw Error("EP GPUNetIO setup called twice", ErrorCode::InvalidUsage);
  state.initialized = true;
  const int rank = state.bootstrap->getRank();
  const int ranks = state.bootstrap->getNranks();
  struct Configuration {
    uint64_t bytes;
    int queues;
    int valid;
  };
  Transport transport = Transport::Unknown;
  std::string localError;
  int queues = 1;
  try {
    transport = selectIbTransport(state.device);
    if (const char* value = std::getenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER")) {
      const std::string text(value);
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), queues);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) queues = 0;
    }
  } catch (const std::exception& error) {
    localError = error.what();
  }
  const bool valid = localError.empty() && buffer != nullptr && bytes > 0 && queues >= 1 && queues <= 64;
  std::vector<Configuration> configurations(ranks);
  configurations[rank] = {bytes, queues, valid ? 1 : 0};
  state.bootstrap->allGather(configurations.data(), sizeof(Configuration));
  for (const auto& configuration : configurations) {
    if (!configuration.valid || configuration.bytes != bytes || configuration.queues != queues)
      throw Error("EP GPUNetIO ranks must agree on buffer size and QPs (1-64). " + localError, ErrorCode::InvalidUsage);
  }
  // One Device-mode connection (QP) per (peer, queue). Memories are registered once and exchanged once per peer,
  // then bound directly into each PortChannel.
  constexpr int ConnectTagBase = 22000;
  constexpr int MemoryTag = 21999;

  state.communicator = std::make_shared<Communicator>(state.bootstrap);
  auto& communicator = *state.communicator;
  RegisteredMemory localMemory;
  try {
    localMemory = communicator.registerMemory(buffer, bytes, transport);
  } catch (const std::exception& error) {
    localError = error.what();
  }
  std::vector<int> registered(ranks);
  registered[rank] = localError.empty() ? 1 : 0;
  state.bootstrap->allGather(registered.data(), sizeof(int));
  if (std::find(registered.begin(), registered.end(), 0) != registered.end())
    throw Error("EP GPUNetIO registration failed on a rank. " + localError, ErrorCode::SystemError);
  std::vector<std::vector<std::shared_future<Connection>>> connections(ranks);
  std::vector<std::shared_future<RegisteredMemory>> remoteMemories(ranks);
  for (int peer = 0; peer < ranks; ++peer) {
    if (peer == rank) continue;
    for (int queue = 0; queue < queues; ++queue) {
      EndpointConfig config;
      config.transport = transport;
      config.device = {DeviceType::GPU, state.device};
      config.ib.mode = EndpointConfig::Ib::Mode::Device;
      connections[peer].push_back(communicator.connect(config, peer, ConnectTagBase + queue));
    }
    communicator.sendMemory(localMemory, peer, MemoryTag);
    remoteMemories[peer] = communicator.recvMemory(peer, MemoryTag);
  }
  std::vector<EpPortChannelDeviceHandle> handles(static_cast<size_t>(ranks) * queues);
  for (int peer = 0; peer < ranks; ++peer) {
    if (peer == rank) continue;
    RegisteredMemory remoteMemory = remoteMemories[peer].get();
    for (int queue = 0; queue < queues; ++queue) {
      auto semaphore = std::make_shared<MemoryDevice2DeviceSemaphore>(communicator, connections[peer][queue].get());
      state.channels.emplace_back(semaphore, remoteMemory, localMemory);
      handles[static_cast<size_t>(peer) * queues + queue] =
          state.channels.back().deviceHandle<PortChannelBackend::Device>();
    }
  }
  state.handles = GpuBuffer<EpPortChannelDeviceHandle>(handles.size()).memory();
  MSCCLPP_CUDATHROW(
      cudaMemcpy(state.handles.get(), handles.data(), handles.size() * sizeof(handles[0]), cudaMemcpyHostToDevice));
  state.signalStates = detail::gpuCallocShared<EpGpuNetIoSignalState>(handles.size());
  const EpGpuNetIoDeviceContext context{state.handles.get(), state.signalStates.get(), ranks, queues};
  state.context = GpuBuffer<EpGpuNetIoDeviceContext>(1).memory();
  MSCCLPP_CUDATHROW(cudaMemcpy(state.context.get(), &context, sizeof(context), cudaMemcpyHostToDevice));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  state.bootstrap->barrier();
}

EpGpuNetIoDeviceContext* EpGpuNetIoService::deviceContext() const { return impl_->context.get(); }

}  // namespace mscclpp::ep
#endif