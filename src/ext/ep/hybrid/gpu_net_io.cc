// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hybrid/include/gpu_net_io.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <dirent.h>
#include <limits.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/semaphore.hpp>
#include <mscclpp/utils.hpp>
#include <vector>

#include "hybrid/include/gpu_net_io_topology.hpp"

namespace mscclpp::ep {
namespace {

int readNumber(const std::string& path, int fallback) {
  std::ifstream file(path);
  int value = fallback;
  if (file.is_open()) file >> value;
  return value;
}

std::string canonicalPath(const std::string& path) {
  char resolved[PATH_MAX] = {};
  return realpath(path.c_str(), resolved) == nullptr ? std::string() : std::string(resolved);
}

std::vector<std::string> selectDevices(const std::string& specification, const std::string& gpuPciBusId,
                                       int gpuNumaNode) {
  if (specification.empty()) {
    const std::string root = "/sys/class/infiniband";
    const auto closeDirectory = [](DIR* directory) { closedir(directory); };
    std::unique_ptr<DIR, decltype(closeDirectory)> directory(opendir(root.c_str()), closeDirectory);
    if (!directory) throw Error("Cannot discover EP GPUNetIO HCAs", ErrorCode::SystemError);
    std::vector<detail::gpunetio::HcaTopology> hcas;
    while (auto* entry = readdir(directory.get())) {
      if (entry->d_name[0] == '.') continue;
      const std::string name = entry->d_name;
      const std::string path = root + "/" + name;
      if (canonicalPath(path + "/device/driver").find("/mlx5_core") == std::string::npos ||
          readNumber(path + "/ports/1/state", 0) != 4)
        continue;
      hcas.push_back({name, canonicalPath(path + "/device"), readNumber(path + "/device/numa_node", -1)});
    }
    auto devices =
        detail::gpunetio::selectClosestHcas(canonicalPath("/sys/bus/pci/devices/" + gpuPciBusId), gpuNumaNode, hcas);
    if (devices.empty()) throw Error("No active EP GPUNetIO HCAs", ErrorCode::InvalidUsage);
    if (devices.size() > 1) {
      int deviceCount = 0;
      if (cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0) {
        int sameNumaIndex = 0;
        int sameNumaCount = 0;
        for (int dev = 0; dev < deviceCount; ++dev) {
          char busId[32] = {};
          if (cudaDeviceGetPCIBusId(busId, sizeof(busId), dev) != cudaSuccess) continue;
          for (char& c : busId) {
            if (c >= 'A' && c <= 'F') c = static_cast<char>(c - 'A' + 'a');
          }
          const int devNuma = readNumber("/sys/bus/pci/devices/" + std::string(busId) + "/numa_node", -1);
          if (devNuma == gpuNumaNode) {
            if (gpuPciBusId == std::string(busId)) sameNumaIndex = sameNumaCount;
            ++sameNumaCount;
          }
        }
        if (sameNumaCount > 0 && devices.size() >= static_cast<size_t>(sameNumaCount) &&
            devices.size() % static_cast<size_t>(sameNumaCount) == 0) {
          const size_t perGpu = devices.size() / static_cast<size_t>(sameNumaCount);
          devices = std::vector<std::string>(devices.begin() + sameNumaIndex * perGpu,
                                             devices.begin() + (sameNumaIndex + 1) * perGpu);
        }
      }
    }
    return devices;
  }
  std::vector<std::string> devices;
  size_t begin = 0;
  while (begin <= specification.size()) {
    const size_t end = specification.find(',', begin);
    const std::string part = specification.substr(begin, end == std::string::npos ? end : end - begin);
    const size_t first = part.find_first_not_of(" \t");
    if (first != std::string::npos) {
      const auto name = part.substr(first, part.find_last_not_of(" \t") - first + 1);
      if (std::find(devices.begin(), devices.end(), name) != devices.end())
        throw Error("Duplicate EP GPUNetIO HCA", ErrorCode::InvalidUsage);
      devices.push_back(name);
    }
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  if (devices.empty()) throw Error("Empty EP GPUNetIO HCA list", ErrorCode::InvalidUsage);
  return devices;
}

// Resolve an HCA name to the IB transport that Context maps to it (honors MSCCLPP_HCA_DEVICES).
Transport ibTransportOf(const std::string& name) {
  constexpr Transport ibs[] = {Transport::IB0, Transport::IB1, Transport::IB2, Transport::IB3,
                               Transport::IB4, Transport::IB5, Transport::IB6, Transport::IB7};
  for (auto transport : ibs) {
    if (getIBDeviceName(transport) == name) return transport;
  }
  throw Error("EP GPUNetIO HCA " + name + " is not an available IB transport", ErrorCode::InvalidUsage);
}

}  // namespace

struct EpGpuNetIoService::Impl {
  std::shared_ptr<Bootstrap> bootstrap;
  std::string devices;
  int device;
  bool initialized = false;
  std::shared_ptr<Communicator> communicator;
  std::shared_ptr<ProxyService> proxyService;
  std::vector<PortChannel> channels;
  std::shared_ptr<PortChannelDeviceHandle> handles;
  std::shared_ptr<EpGpuNetIoSignalState> signalStates;
  std::shared_ptr<EpGpuNetIoDeviceContext> context;

  ~Impl() {
    CudaDeviceGuard guard(device);
    context.reset();
    signalStates.reset();
    handles.reset();
    channels.clear();
    proxyService.reset();
    communicator.reset();
  }
};

EpGpuNetIoService::EpGpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& devices, int cudaDevice)
    : impl_(std::make_unique<Impl>()) {
  impl_->bootstrap = std::move(bootstrap);
  impl_->devices = devices;
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
    int hcas;
    int queues;
    int valid;
  };
  std::vector<std::string> devices;
  std::string localError;
  char pciBusId[32] = {};
  int numaNode = -1;
  int queues = 0;
  try {
    MSCCLPP_CUDATHROW(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), state.device));
    for (char& digit : pciBusId) {
      if (digit >= 'A' && digit <= 'F') digit = static_cast<char>(digit - 'A' + 'a');
    }
    numaNode = readNumber("/sys/bus/pci/devices/" + std::string(pciBusId) + "/numa_node", -1);
    devices = selectDevices(state.devices, pciBusId, numaNode);
    queues = static_cast<int>(devices.size());
    if (const char* value = std::getenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER")) {
      const std::string text(value);
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), queues);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) queues = 0;
    }
  } catch (const std::exception& error) {
    localError = error.what();
  }
  const int hcas = static_cast<int>(devices.size());
  const bool valid = localError.empty() && buffer != nullptr && bytes > 0 && hcas > 0 && queues >= hcas &&
                     queues <= 64 && queues % hcas == 0;
  std::vector<Configuration> configurations(ranks);
  configurations[rank] = {bytes, hcas, queues, valid ? 1 : 0};
  state.bootstrap->allGather(configurations.data(), sizeof(Configuration));
  for (const auto& configuration : configurations) {
    if (!configuration.valid || configuration.bytes != bytes || configuration.hcas != hcas ||
        configuration.queues != queues)
      throw Error(
          "EP GPUNetIO ranks must agree on buffer size, HCA count and QPs (1-64, a multiple of HCAs). " + localError,
          ErrorCode::InvalidUsage);
  }
  if (std::getenv("MSCCLPP_EP_DEBUG_TOPO")) {
    std::string names;
    for (const auto& name : devices) names += (names.empty() ? "" : ",") + name;
    std::fprintf(stderr,
                 "[EPGPUNETIO] rank=%d hcaSelection=%s gpu=%s numa=%d hcas=%s numHcas=%d qpsPerPeer=%d qpsPerHca=%d\n",
                 rank, state.devices.empty() ? "auto" : "explicit", pciBusId, numaNode, names.c_str(), hcas, queues,
                 queues / hcas);
  }
  // One Device-mode connection (QP) per (peer, queue); queue q uses HCA q % hcas. Memories are registered on every
  // selected HCA once and exchanged once per peer, then bound into each channel through ProxyService::portChannel().
  constexpr int ConnectTagBase = 22000;
  constexpr int MemoryTag = 21999;
  std::vector<Transport> transports(hcas);
  TransportFlags flags;
  localError.clear();
  try {
    for (int hca = 0; hca < hcas; ++hca) {
      transports[hca] = ibTransportOf(devices[hca]);
      flags |= transports[hca];
    }
  } catch (const std::exception& error) {
    localError = error.what();
  }
  std::vector<int> resolved(ranks);
  resolved[rank] = localError.empty() ? 1 : 0;
  state.bootstrap->allGather(resolved.data(), sizeof(int));
  if (std::find(resolved.begin(), resolved.end(), 0) != resolved.end())
    throw Error("EP GPUNetIO HCA resolution failed on a rank. " + localError, ErrorCode::InvalidUsage);

  state.communicator = std::make_shared<Communicator>(state.bootstrap);
  state.proxyService = std::make_shared<ProxyService>();
  auto& communicator = *state.communicator;
  RegisteredMemory localMemory;
  try {
    localMemory = communicator.registerMemory(buffer, bytes, flags);
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
      config.transport = transports[queue % hcas];
      config.device = {DeviceType::GPU, state.device};
      config.ib.mode = EndpointConfig::Ib::Mode::Device;
      connections[peer].push_back(communicator.connect(config, peer, ConnectTagBase + queue));
    }
    communicator.sendMemory(localMemory, peer, MemoryTag);
    remoteMemories[peer] = communicator.recvMemory(peer, MemoryTag);
  }
  auto& proxyService = *state.proxyService;
  const MemoryId localMemoryId = proxyService.addMemory(localMemory);
  std::vector<MemoryId> remoteMemoryIds(ranks, localMemoryId);
  std::vector<std::vector<SemaphoreId>> semaphores(ranks);
  for (int peer = 0; peer < ranks; ++peer) {
    if (peer == rank) continue;
    remoteMemoryIds[peer] = proxyService.addMemory(remoteMemories[peer].get());
    for (int queue = 0; queue < queues; ++queue) {
      semaphores[peer].push_back(proxyService.buildAndAddSemaphore(communicator, connections[peer][queue].get()));
    }
  }
  std::vector<PortChannelDeviceHandle> handles(static_cast<size_t>(ranks) * queues);
  for (int peer = 0; peer < ranks; ++peer) {
    if (peer == rank) continue;
    for (int queue = 0; queue < queues; ++queue) {
      state.channels.push_back(proxyService.portChannel(semaphores[peer][queue], remoteMemoryIds[peer], localMemoryId));
      handles[static_cast<size_t>(peer) * queues + queue] = state.channels.back().deviceHandle();
    }
  }
  state.handles = GpuBuffer<PortChannelDeviceHandle>(handles.size()).memory();
  MSCCLPP_CUDATHROW(
      cudaMemcpy(state.handles.get(), handles.data(), handles.size() * sizeof(handles[0]), cudaMemcpyHostToDevice));
  state.signalStates = detail::gpuCallocShared<EpGpuNetIoSignalState>(handles.size());
  const EpGpuNetIoDeviceContext context{state.handles.get(), state.signalStates.get(), ranks, queues, hcas};
  state.context = GpuBuffer<EpGpuNetIoDeviceContext>(1).memory();
  MSCCLPP_CUDATHROW(cudaMemcpy(state.context.get(), &context, sizeof(context), cudaMemcpyHostToDevice));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  state.bootstrap->barrier();
}

EpGpuNetIoDeviceContext* EpGpuNetIoService::deviceContext() const { return impl_->context.get(); }

}  // namespace mscclpp::ep
#endif