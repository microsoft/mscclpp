# MSCCL++

[![Latest Release](https://img.shields.io/github/release/microsoft/mscclpp.svg)](https://github.com/microsoft/mscclpp/releases/latest)
[![License](https://img.shields.io/github/license/microsoft/mscclpp.svg)](LICENSE)
[![CodeQL](https://github.com/microsoft/mscclpp/actions/workflows/codeql-analysis.yml/badge.svg?branch=main)](https://github.com/microsoft/mscclpp/actions/workflows/codeql-analysis.yml)

| Pipelines                | Build Status      |
|--------------------------|-------------------|
| Unit Tests (CUDA)        | [![Build Status](https://dev.azure.com/binyli/HPC/_apis/build/status%2Fmscclpp-ut?branchName=main)](https://dev.azure.com/binyli/HPC/_build/latest?definitionId=4&branchName=main) |
| Integration Tests (CUDA) | [![Build Status](https://dev.azure.com/binyli/HPC/_apis/build/status%2Fmscclpp-test?branchName=main)](https://dev.azure.com/binyli/HPC/_build/latest?definitionId=3&branchName=main) |

A GPU-driven communication stack for scalable AI applications.

See [Quick Start](docs/quickstart.md) to quickly get started.

## Overview

MSCCL++ redefines the interface for inter-GPU communication, thereby delivering a highly efficient and customizable communication stack tailored for distributed GPU applications. The followings describe the key features of MSCCL++.

* **On-GPU Interfaces.** MSCCL++ provides communication interfaces to be called by a **GPU thread**. Users can easily implement highly optimized communication logics inside a GPU kernel, without awareness of detailed communication mechanisms. This enables users to implement highly fine-grained system pipelining (i.e., hiding communication delays by overlapping with computation), which has been difficult for CPU-based interfaces.

* **Fine-grained Abstracts.** MSCCL++ provides fine-grained abstracts for communication primitives, such as `put()`, `get()`, `signal()`, `flush()`, and `wait()`. This enables users to easily implement flexible communication logics, such as overlapping communication with computation, or implementing customized collective communication algorithms.

* **Converged Interfaces.** MSCCL++ provides consistent interfaces regardless of the location of the remote GPU (either on the local node or on a remote node) or the underlying link (either NVLink/xGMI or InfiniBand). This simplifies the code for inter-GPU communication, which is often complex and error-prone.

## Feature Examples

The following illustrates key features of MSCCL++.

### On-GPU Communication Interfaces

MSCCL++ provides inter-GPU communication interfaces to be called by a GPU thread. For example, the `put()` method in the following example copies 1KB data from the local GPU to a remote GPU. `channel` is a peer-to-peer communication channel between two GPUs, which consists of information on send/receive buffers. `channel` is initialized from the host side before the kernel execution.

```cpp
__device__ mscclpp::DeviceHandle<mscclpp::SimpleProxyChannel> channel;
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

MSCCL++ provides consistent interfaces, i.e., the above interfaces are used regardless of the location of the remote GPU (either on the local node or on a remote node) or the underlying link (either NVLink or InfiniBand).

### Host-Side Communication Proxy

MSCCL++ interfaces may send a request (called triggers) to a GPU-external helper that conducts key functionalities such as DMA or RDMA. This helper is called a *proxy*. MSCCL++ provides a default implementation of a proxy, which is a background host thread that busy polls triggers from GPUs and conducts functionalities accordingly. For example, the following is a typical host-side code for MSCCL++.

```cpp
// Bootstrap: initialize control-plane connections between all ranks
auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, world_size);
// Create a communicator for connection setup
mscclpp::Communicator comm(bootstrap);
// Setup connections here using `comm`
...
// Construct the default proxy
mscclpp::ProxyService proxyService();
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
__device__ mscclpp::FifoDeviceHandle fifo;
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

// Host-side custom proxy service
class CustomProxyService {
private:
  mscclpp::Proxy proxy_;
public:
  CustomProxyService() : proxy_([&](mscclpp::ProxyTrigger trigger) {
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

### Python Interfaces

MSCCL++ provides Python bindings and interfaces, which simplifies integration with Python applications.

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
