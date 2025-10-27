# Port Channel

```{note}
This tutorial follows the [Memory Channel](./03-memory-channel.md) tutorial.
```

## Build and Run the Example

The code of this tutorial is under [examples/tutorials/04-port-channel](https://github.com/microsoft/mscclpp/blob/main/examples/tutorials/04-port-channel).

Build the example with `make`:

```bash
$ cd examples/tutorials/04-port-channel
$ make
```

Run the example with `./bidir_port_channel`. If you are in a container, you may need to run with root privileges. You should see output similar to the following:

```
# ./bidir_port_channel
GPU 0: Preparing for tests ...
GPU 1: Preparing for tests ...
GPU 0: [Bidir PutWithSignal] bytes 1024, elapsed 0.0204875 ms/iter, BW 0.0499818 GB/s
GPU 0: [Bidir PutWithSignal] bytes 1048576, elapsed 0.0250319 ms/iter, BW 41.8896 GB/s
GPU 0: [Bidir PutWithSignal] bytes 134217728, elapsed 0.365497 ms/iter, BW 367.219 GB/s
Succeed!
```

The example code uses localhost port `50505` by default. If the port is already in use, you can change it by modifying the `PORT_NUMBER` macro in the code.

```{caution}
Note that this example is **NOT** a performance benchmark. The performance numbers are provided to give you an idea of the performance characteristics of `PortChannel`. For optimal performance, synchronization can be further optimized depending on the application scenario and implementation.
```

## Code Overview

The example code implements a bidirectional data transfer using a `PortChannel` between two GPUs on the same machine. The code is similar to the [Memory Channel](./03-memory-channel.md) tutorial, with the main difference being that the construction of a `PortChannel` is done by a `ProxyService` instance. We need to "add" the pre-built `Semaphore` and `RegisteredMemory` objects to the `ProxyService`, which return `SemaphoreId` and `MemoryId`s, respectively:

```cpp
mscclpp::ProxyService proxyService;
mscclpp::SemaphoreId semaId = proxyService.addSemaphore(sema);
mscclpp::MemoryId localMemId = proxyService.addMemory(localRegMem);
mscclpp::MemoryId remoteMemId = proxyService.addMemory(remoteRegMem);
```

Using the IDs, we can create a `PortChannel` associated with the `ProxyService`:

```cpp
mscclpp::PortChannel portChan = proxyService.portChannel(semaId, remoteMemId, localMemId);
```

The procedures for building `Semaphore` and `RegisteredMemory` are explained in the [Basic Concepts](./01-basic-concepts.md) and the [Memory Channel](./03-memory-channel.md) tutorials, respectively.

We need to call `proxyService.startProxy()` before running GPU kernels that use the `PortChannel`. The `ProxyService` runs a background host thread that listens for incoming requests from the `PortChannel` and handles them accordingly. We can call `proxyService.stopProxy()` to stop the background thread after all GPU operations are done.

## PortChannel

**PortChannel** is a communication channel that enables data transfer between GPUs using I/O ports, such as the Copy Engine (CE) of a GPU (e.g., `cudaMemcpyAsync`), InfiniBand queue pairs, or TCP sockets. Compared to `MemoryChannel`, which copies data using GPU threads, `PortChannel` offloads data transfer to dedicated hardware or software components. This reduces interference with other parallel GPU operations, and potentially allows for higher throughput. However, `PortChannel` may introduce additional latency due to the overhead of initiating data transfers.

The device handle of a `PortChannel` provides the following methods. Since the data transfer is offloaded, each method is supposed to be called by a single GPU thread.
- `put()`: Initiates an asynchronous one-way data transfer from the local memory to the remote memory.
- `signal()`: Asynchronously signals the completion of all previous `put()`s to the remote side.
- `wait()`: Blocks the calling GPU thread until the corresponding `signal()` is received from the remote side.
- `poll()`: Non-blocking version of `wait()`. Returns immediately with a boolean indicating whether the signal has been received.
- `flush()`: Synchronizes the local GPU with the `PortChannel`, ensuring that all previous operations are completed.
- Fused methods (e.g., `putWithSignal()`): combines multiple sequential operations into a single call for efficiency.

The following diagram illustrates how the `bidirPutKernel()` function in the example code would work when GPU0 is faster than GPU1. The execution order may vary depending on the relative speeds of the GPUs. 

```{mermaid}
sequenceDiagram
    participant GPU0
    participant GPU1

    GPU0->>GPU1: signal()
    GPU1->>GPU0: signal()

    Note over GPU0: wait() returns by signal()

    GPU0->>GPU1: putWithSignal(): copy local data range<br>[0:copyBytes) to remote range [0:copyBytes)

    Note over GPU1: wait() returns by signal()

    GPU1->>GPU0: putWithSignal(): copy local data range<br>[copyBytes:2*copyBytes) to remote range [copyBytes:2*copyBytes)

    Note over GPU0: wait() returns by putWithSignal()
    Note over GPU1: wait() returns by putWithSignal()
```

## ProxyService

**ProxyService** is a host-side service that assists operation of one or more `PortChannel`s. When a `PortChannel` calls `put()`, `signal()`, or `flush()` methods (or their fused versions) on GPU, it constructs a corresponding request and pushes it into a FIFO queue managed by the `ProxyService` on the host side. The `ProxyService` runs a background thread that processes these requests and performs the actual data transfers or signaling operations using the appropriate implementation, which depends on the transport type of the `Connection` associated with the channel.

In most cases, users only need to use a `ProxyService` instance to create `PortChannel`s and start/stop the proxy thread.

```{caution}
The device handle methods of `PortChannel` are thread-safe except when the number of concurrent threads exceeds the FIFO queue size of the `ProxyService`. The default FIFO queue size is 512, which can be changed by passing a different value to the `ProxyService` constructor.
```

```{note}
Advanced users may want to customize the behavior of `ProxyService` to support custom request types or transport mechanisms, which can be done by subclassing `BaseProxyService`. See an example in [class AllGatherProxyService](https://github.com/microsoft/mscclpp/blob/main/test/mscclpp-test/allgather_test.cu#L503).
```

## Summary and Next Steps

In this tutorial, we learned how to use `PortChannel` for bidirectional data transfer between two GPUs using a `ProxyService`.
