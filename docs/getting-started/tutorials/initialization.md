# Commnunication initialize with mscclpp API

In this tutorial, you will write a simple program to initialize communication between eight GPUs using MSCCL++ C++ API. You will also learn how to use the Python API to initialize communication.

## Prerequisites
A system with eight GPUs is required to run this tutorial.

Also make sure that you have installed MSCCL++ on your system. If not, please follow the [quick start](../quickstart.md).

## Initialize Communication with C++ API
We will setup a mesh topology with eight GPUs. Each GPU will be connected to its neighbors. The following code shows how to initialize communication with MSCCL++ C++ API.

```cpp
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/proxy_channel.hpp>

#include <memory>
#include <string>
#include <vector>

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::ProxyChannel> constProxyChans[8];

void setupMeshTopology(int rank, int worldsize, void* data, size_t dataSize) {
  std::string ip_port = "10.0.0.4:50000";
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, worldsize);
  bootstrap->initialize(ip_port);
  mscclpp::Communicator comm(bootstrap);
  mscclpp::ProxyService proxyService;

  std::vector<mscclpp::SemaphoreId> semaphoreIds;
  std::vector<mscclpp::RegisteredMemory> localMemories;
  std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connections(world_size);
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteMemories;

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
    // Connect with all other ranks
    connections[r] = comm.connectOnSetup(r, 0, transport);
    auto memory = comm.registerMemory(data, dataSize, mscclpp::Transport::CudaIpc | ibTransport);
    localMemories.push_back(memory);
    comm.sendMemoryOnSetup(memory, r, 0);
    remoteMemories.push_back(comm.recvMemoryOnSetup(r, 0));
  }

  comm.setup();

  for (int r = 0; r < world_size; ++r) {
    if (r == rank) continue;
    semaphoreIds.push_back(proxyService.buildAndAddSemaphore(comm, connections[r].get()));
  }

  comm.setup();

  std::vector<DeviceHandle<mscclpp::ProxyChannel>> proxyChannels;
  for (size_t i = 0; i < semaphoreIds.size(); ++i) {
    proxyChannels.push_back(mscclpp::deviceHandle(mscclpp::ProxyChannel(
        proxyService.proxyChannel(semaphoreIds[i]), proxyService.addMemory(remoteMemories[i].get()),
        proxyService.addMemory(localMemories[i]))));
  }

  if (proxyChannels.size() > sizeof(constProxyChans) / sizeof(DeviceHandle<mscclpp::ProxyChannel>)) {
    std::runtime_error("unexpected error");
  }
  CUDACHECK(cudaMemcpyToSymbol(constProxyChans, proxyChannels.data(),
                              sizeof(DeviceHandle<mscclpp::ProxyChannel>) * proxyChannels.size()));
}
```
