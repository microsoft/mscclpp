// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpu_net_io.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <dirent.h>
#include <limits.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mscclpp/gpu_net_io_service.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <vector>

#include "gpu_net_io_topology.hpp"

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

}  // namespace

struct EpGpuNetIoService::Impl {
  std::shared_ptr<Bootstrap> bootstrap;
  std::string devices;
  int device;
  bool initialized = false;
  std::vector<std::shared_ptr<GpuNetIoService>> services;
  std::vector<PortChannel> channels;
  std::shared_ptr<PortChannelDeviceHandle> handles;
  std::shared_ptr<EpGpuNetIoDeviceContext> context;

  ~Impl() {
    CudaDeviceGuard guard(device);
    context.reset();
    handles.reset();
    channels.clear();
    services.clear();
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
  std::vector<PortChannelDeviceHandle> handles(static_cast<size_t>(ranks) * queues);
  for (int hca = 0; hca < hcas; ++hca) {
    auto service = std::make_shared<GpuNetIoService>(state.bootstrap, devices[hca], state.device, queues / hcas);
    service->setup();
    state.services.push_back(service);
    GpuNetIoMemory memory;
    localError.clear();
    try {
      memory = service->registerMemory(buffer, bytes);
    } catch (const std::exception& error) {
      localError = error.what();
    }
    std::vector<int> registered(ranks);
    registered[rank] = localError.empty() ? 1 : 0;
    state.bootstrap->allGather(registered.data(), sizeof(int));
    if (std::find(registered.begin(), registered.end(), 0) != registered.end())
      throw Error("EP GPUNetIO registration failed on a rank. " + localError, ErrorCode::SystemError);
    for (int first = 0; first < ranks; ++first) {
      for (int second = first + 1; second < ranks; ++second) {
        if (rank != first && rank != second) continue;
        const int peer = rank == first ? second : first;
        for (int queue = hca; queue < queues; queue += hcas) {
          const auto connection = service->connect(peer, queue / hcas);
          const auto remote = service->exchangeMemory(connection, memory, 22000 + queue * 2);
          const auto semaphore = service->buildSemaphore(connection, 22001 + queue * 2);
          state.channels.emplace_back(semaphore, remote, memory);
          handles[static_cast<size_t>(peer) * queues + queue] = state.channels.back().deviceHandle();
        }
      }
    }
  }
  state.handles = GpuBuffer<PortChannelDeviceHandle>(handles.size()).memory();
  MSCCLPP_CUDATHROW(
      cudaMemcpy(state.handles.get(), handles.data(), handles.size() * sizeof(handles[0]), cudaMemcpyHostToDevice));
  const EpGpuNetIoDeviceContext context{state.handles.get(), ranks, queues, hcas};
  state.context = GpuBuffer<EpGpuNetIoDeviceContext>(1).memory();
  MSCCLPP_CUDATHROW(cudaMemcpy(state.context.get(), &context, sizeof(context), cudaMemcpyHostToDevice));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  state.bootstrap->barrier();
}

EpGpuNetIoDeviceContext* EpGpuNetIoService::deviceContext() const { return impl_->context.get(); }

}  // namespace mscclpp::ep
#endif