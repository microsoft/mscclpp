# MSCCL++

GPU-driven computation & communication stack.

See [Quick Start](docs/quickstart.md) to quickly get started.

See the latest performance evaluation on Azure [NDmv4](docs/performance-ndmv4.md).

Build our Doxygen document by running `doxygen` in [`docs/`](docs/) directory.

## Overview

MSCCL++ is a development kit for implementing highly optimized distributed GPU applications, in terms of both inter-GPU communication and GPU computation. MSCCL++ is specially designed for developers who want to fine-tune inter-GPU communication of their applications at the GPU kernel level, without awareness of detailed communication mechanisms. The key underlying concept of MSCCL++ is GPU-driven execution, where both communication and computation tasks are initiated by GPU not by CPU. That is, the communication and computation interfaces of MSCCL++ are provided as device-side APIs (called inside a GPU kernel), while the host-side APIs of MSCCL++ are for bootstrapping, initial connection setups, or background host threads for inter-GPU DMA and RDMA (called proxies). By using MSCCL++, we expect:

* **Holistic Optimization for High GPU Utilization.** As both communication and computation are scheduled inside a GPU kernel at the same time, we can optimize end-to-end performance of distributed GPU applications from a global view. For example, we can minimize the GPU resource contention between communication and computation, which is known to often substantially degrade throughput of distributed deep learning applications.

* **Fully Pipelined System to Reduce Overhead from the Control Plane.** We can eliminate control overhead from CPU by allowing GPU to autonomously schedule both communication and computation. This significantly reduces GPU scheduling overhead and CPU-GPU synchronization overhead. For example, this allows us to implement a highly fine-grained system pipelining (i.e., hiding communication delays by overlapping with computation), which has been difficult for CPU-controlled applications due to the large control/scheduling overhead.

* **Runtime Performance Optimization for Dynamic Workload.** As we can easily implement flexible communication logics, we can optimize communication performance even during runtime. For example, we can implement the system to automatically choose different communication paths or different collective communication algorithms depending on the dynamic workload at runtime.

## Key Features (v0.2)

MSCCL++ v0.2 supports the following features.

### In-Kernel Communication Interfaces

MSCCL++ provides inter-GPU communication interfaces to be called by a GPU thread. For example, the `put()` method in the following example copies 1KB data from the local GPU to a remote GPU. `channel` is a peer-to-peer communication channel between two GPUs, which consists of information on send/receive buffers. `channel` is initialized from the host side before the kernel execution.

```cpp
__device__ mscclpp::SimpleProxyChannel channel;
__global__ void gpuKernel() {
  ...
  // Only one thread is needed for this method.
  channel.put(/*dstOffset=*/ 0, /*srcOffset=*/ 0, /*size=*/ 1024);
  ...
}
```

MSCCL++ also provides efficient synchronization methods, `signal()`, `flush()`, and `wait()`. For example, we can implement a simple barrier between two ranks (peer-to-peer connected through `channel`) as follows. Explanation of each method is inlined.

```cpp
// Only one thread is needed for this function.
__device__ void barrier() {
  // Inform the peer GPU that I have arrived at this point.
  channel.signal();
  // Flush the previous signal() call, which will wait for completion of signaling.
  channel.flush();
  // Wait for the peer GPU to call signal().
  channel.wait();
  // Now this thread is synchronized with the remote GPU’s thread.
  // Users may call a local synchronize functions (e.g., __syncthreads())
  // to synchronize other local threads as well with the remote side.
}
```

MSCCL++ provides consistent in-kernel interfaces, i.e., the above interfaces are used regardless of the location of the remote GPU (either on the local node or on a remote node) or the underlying link (either NVLink or InfiniBand).

### Host-Side Communication Proxy

Some in-kernel communication interfaces of MSCCL++ send requests (called triggers) to a GPU-external helper that conducts key functionalities such as DMA or RDMA. This helper is called a proxy service or a proxy in short. MSCCL++ provides a default implementation of a proxy, which is a background host thread that busy polls triggers from GPUs and conducts functionalities accordingly. For example, the following is a typical host-side code for MSCCL++.

```cpp
// Bootstrap: initialize control-plane connections between all ranks
auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
// Create a communicator for connection setup
mscclpp::Communicator comm(bootstrap);
// Setup connections here using `comm`
...
// Construct the default proxy
mscclpp::ProxyService proxyService(comm);
// Start the proxy
proxyService.startProxy();
// Run the user application, i.e., launch GPU kernels here
...
// Stop the proxy after the application is finished
proxyService.stopProxy();
```

While the default implementation already enables any kinds of communication, MSCCL++ also supports users to easily implement their own customized proxies for further optimization. For example, the following example re-defines how to interpret triggers from GPUs.

```cpp
// Proxy FIFO is obtained from mscclpp::Proxy on the host and copied to the device.
__device__ mscclpp::DeviceProxyFifo fifo;
__global__ void gpuKernel() {
  ...
  // Only one thread is needed for the followings
  mscclpp::ProxyTrigger trigger;
  // Send a custom request: "1"
  trigger.fst = 1;
  fifo.push(trigger);
  // Send a custom request: "2"
  trigger.fst = 2;
  fifo.push(trigger);
  // Send a custom request: "0xdeadbeef"
  trigger.fst = 0xdeadbeef;
  fifo.push(trigger);
  ...
}

// Host-side custom channel service
class CustomChannelService {
private:
  mscclpp::Proxy proxy_;
public:
  CustomChannelService() : proxy_([&](mscclpp::ProxyTrigger trigger) {
                                    // Custom trigger handler
                                    if (trigger.fst == 1) {
                                      // Handle request "1"
                                    } else if (trigger.fst == 2) {
                                      // Handle request "2"
                                    } else if (trigger.fst == 0xdeadbeef) {
                                      // Handle request "0xdeadbeef"
                                    }
                                  },
                                  [&]() { /* Empty proxy initializer */ }) {}
  void startProxy() { proxy_.start(); }
  void stopProxy()  { proxy_.stop(); }
};
```

Customized proxies can be used for conducting a series of pre-defined data transfers within only a single trigger from GPU at runtime. This would be more efficient than sending a trigger for each data transfer one by one.

### Flexible Customization

Most of key components of MSCCL++ are designed to be easily customized. This enables MSCCL++ to easily adopt a new software / hardware technology and lets users implement algorithms optimized for their own use cases.

## Status & Roadmap

MSCCL++ is under active development and a part of its features will be added in a future release. The following describes key features of each version.

### MSCCL++ v0.4 (TBU)
* Automatic task scheduler
* Dynamic performance tuning

### MSCCL++ v0.3 (TBU)
* Tile-based communication: efficient transport of 2D data patches (tiles)
* GPU computation interfaces

### MSCCL++ v0.2 (Latest Release)
* Basic communication functionalities and new interfaces
    - GPU-side communication interfaces
    - Host-side helpers: bootstrap, communicator, and channel service (proxy)
    - Supports both NVLink and InfiniBand
    - Supports both in-SM copy and DMA/RDMA
* Communication performance optimization
    - Example code outperforms NCCL/MSCCL AllGather/AllReduce/AllToAll
* Development pipeline
* Documentation

### MSCCL++ v0.1
* Proof-of-concept, preliminary interfaces

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
