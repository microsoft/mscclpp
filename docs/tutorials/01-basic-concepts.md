# Basic Concepts

In this tutorial, we explain a few basic concepts of the MSCCL++ Primitive API using a simple ping-pong example between two GPUs. The example demonstrates how to set up communication between GPUs.

## Build and Run the Example

The code of this tutorial is under [examples/tutorials/01-basic-concepts](https://github.com/microsoft/mscclpp/blob/main/examples/tutorials/01-basic-concepts).

Build the example with `make`:

```bash
$ cd examples/tutorials/01-basic-concepts
$ make
```

Run the example with `./gpu_ping_pong`. If you are in a container, you may need to run with the root privileges. You should see output similar to the following:

```
# ./gpu_ping_pong
Creating endpoints ...
GPU 0: Creating a connection and a semaphore stub ...
GPU 1: Creating a connection and a semaphore stub ...
GPU 0: Creating a semaphore and a memory channel ...
GPU 1: Creating a semaphore and a memory channel ...
GPU 0: Launching gpuKernel0 ...
GPU 1: Launching gpuKernel1 ...
Elapsed 4.77814 ms per iteration (100)
Succeed!
```

If you see error messages like "At least two GPUs are required" or "GPU 0 cannot access GPU 1", it means that your system does not meet the requirements for running the example. Make sure you have at least two GPUs installed and that they are connected peer-to-peer (through NVLink or under the same PCIe switch). See the {ref}`prerequisites` for more details.

## Code Overview

The example code constructs three key components for communication: **Connection**, **Semaphore**, and **Channel**. The following diagram illustrates the flow of how these components are created and used.

```{mermaid}
sequenceDiagram
    participant ProcessA
    participant ProcessB

    rect rgb(240, 240, 240)
        Note over ProcessA, ProcessB: Create an Endpoint

        ProcessA<<->>ProcessB: Exchange the Endpoints

        Note over ProcessA, ProcessB: Create a Connection using the two Endpoints
    end

    rect rgb(240, 240, 240)
        Note over ProcessA, ProcessB: Construct a SemaphoreStub using the Connection

        ProcessA<<->>ProcessB: Exchange the SemaphoreStubs

        Note over ProcessA, ProcessB: Create a Semaphore using the two SemaphoreStubs
    end

    Note over ProcessA, ProcessB: Create a Channel using the Semaphore and run applications
```

```{note}
Note that ProcessA and ProcessB are not necessarily different processes; they can be the same process running on the same host (like in the example code).
The endpoints constructed by ProcessA and ProcessB are also not necessarily using different GPUs; they can be the same GPU, allowing for intra-GPU communication.
```

## Endpoint and Connection

An **Endpoint** represents an entity that can communicate with another entity, such as a GPU. In this example, we create two endpoints, one for each GPU. A **Connection** is established between these endpoints, allowing them to communicate with each other. Construction of endpoints and connections is done by a **Context** object, which is responsible for managing communication resources.

The example code creates endpoints as follows:

```cpp
// From gpu_ping_pong.cu, lines 70-71
mscclpp::Endpoint ep0 = ctx->createEndpoint({transport, {mscclpp::DeviceType::GPU, 0}});
mscclpp::Endpoint ep1 = ctx->createEndpoint({transport, {mscclpp::DeviceType::GPU, 1}});
```

Both endpoints are created to use the same transport `mscclpp::Transport::CudaIpc`, which uses direct communication supported by CUDA/HIP IPC. The two endpoints must use the same transport type to establish a connection between them. We will introduce other transport types in later tutorials.

`mscclpp::DeviceType::GPU` indicates that these endpoints are for GPUs, and the numbers `0` and `1` specify the GPU IDs.

The connection is created by calling `connect` on the context object:

```cpp
// From gpu_ping_pong.cu, lines 76 and 82
mscclpp::Connection conn0 = ctx->connect(/*localEndpoint*/ ep0, /*remoteEndpoint*/ ep1);
mscclpp::Connection conn1 = ctx->connect(/*localEndpoint*/ ep1, /*remoteEndpoint*/ ep0);
```

The `localEndpoint` and `remoteEndpoint` parameters specify which endpoints are used for the connection. A connection is asymmetric by nature, meaning that we need to create one connection for each endpoint. In this case, `conn0` is created for `ep0` to communicate with `ep1`, and `conn1` is created for `ep1` to communicate with `ep0`.

This example creates both endpoints in a single process for simplicity, so the connections can be established directly using the two endpoints. However, in most real-world applications, the endpoints would be created in different processes. In that case, you can **serialize the endpoints** and send them over a network or through shared memory. For example:

```cpp
// Process A
mscclpp::Endpoint ep0 = ctx->createEndpoint({transport, {mscclpp::DeviceType::GPU, 0}});
std::vector<char> serializedEp0 = ep0.serialize();
sendToProcessB(serializedEp0);  // send serializedEp0 to Process B using any IPC mechanism

// Process B
mscclpp::Endpoint ep1 = ctx->createEndpoint({transport, {mscclpp::DeviceType::GPU, 1}});
std::vector<char> serializedEp0 = recvFromProcessA();  // receive serializedEp0 from Process A
mscclpp::Endpoint ep0 = mscclpp::Endpoint::deserialize(serializedEp0);
mscclpp::Connection conn1 = ctx->connect(/*localEndpoint*/ ep1, /*remoteEndpoint*/ ep0);
```

## SemaphoreStub and Semaphore

After a connection is established, both endpoints know how to communicate with each other. Now, we need a way to synchronize the communication. This is where **Semaphore** comes into play. A Semaphore provides a synchronization mechanism that allows one endpoint to signal the other or wait for a signal from the other endpoint.

To construct a Semaphore, we first need to create a **SemaphoreStub** using the connection from each endpoint. A SemaphoreStub holds one endpoint's resource for a Semaphore, and a Semaphore is constructed using two SemaphoreStubs, one from each endpoint.

```cpp
// From gpu_ping_pong.cu, lines 77 and 83
mscclpp::SemaphoreStub semaStub0(conn0);
mscclpp::SemaphoreStub semaStub1(conn1);
```

The SemaphoreStubs are created using the connections established earlier. They are then exchanged between the two endpoints to create a Semaphore:

```cpp
// From gpu_ping_pong.cu, lines 88 and 98
mscclpp::Semaphore sema0(/*localSemaphoreStub*/ semaStub0, /*remoteSemaphoreStub*/ semaStub1);
mscclpp::Semaphore sema1(/*localSemaphoreStub*/ semaStub1, /*remoteSemaphoreStub*/ semaStub0);
```

Like the connections, the Semaphore is also asymmetric. Each endpoint has its own Semaphore, which is constructed using its own SemaphoreStub and the other endpoint's SemaphoreStub. SemaphoreStubs can be serialized and sent to other processes in the same way as endpoints.

## Channel

Semaphores can be used to synchronize operations between endpoints, but they do not provide a way to transfer data. To facilitate data transfer, we introduce the concept of **Channel**. A Channel is built on top of a semaphore and allows for the transfer of data between endpoints.

However, since this ping-pong example doesn't need to transfer any data, we construct a `BaseMemoryChannel` that is a shallow wrapper around a Semaphore but does not associate with any memory region for data transfer. We will introduce more advanced channels in later tutorials.

```cpp
// From gpu_ping_pong.cu, lines 89 and 99
mscclpp::BaseMemoryChannel memChan0(sema0);
mscclpp::BaseMemoryChannel memChan1(sema1);
```

To let the application (GPU kernels) use the channels, we need to obtain the **device handles** for the channels. The device handle is a lightweight object that can be passed to GPU kernels to perform operations on the channel. Different types of channels have different device handle types.

```cpp
// From gpu_ping_pong.cu, lines 90 and 100
mscclpp::BaseMemoryChannelDeviceHandle memChanHandle0 = memChan0.deviceHandle();
mscclpp::BaseMemoryChannelDeviceHandle memChanHandle1 = memChan1.deviceHandle();
```

The device handles are then copied to the GPU memory and passed to the GPU kernels for execution. For example:

```cpp
// From gpu_ping_pong.cu, lines 91-93
void *devHandle0;
MSCCLPP_CUDATHROW(cudaMalloc(&devHandle0, sizeof(mscclpp::BaseMemoryChannelDeviceHandle)));
MSCCLPP_CUDATHROW(cudaMemcpy(devHandle0, &memChanHandle0, sizeof(memChanHandle0), cudaMemcpyHostToDevice));

// From gpu_ping_pong.cu, line 108
gpuKernel0<<<1, 1>>>(reinterpret_cast<mscclpp::BaseMemoryChannelDeviceHandle *>(devHandle0), iter);

// From gpu_ping_pong.cu, lines 26-35
__global__ void gpuKernel0(mscclpp::BaseMemoryChannelDeviceHandle *devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * gridDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedWait();
      // spin for a few ms
      spin_cycles(1e7);
      devHandle->relaxedSignal();
    }
  }
}
```

The kernel code will be explained in the next section.

```{tip}
If your application (GPU kernel) needs to access the device handle very frequently with low latency requirements, you can consider either of the following approaches:

* Use constant memory to store the device handle, which allows faster access by storing the handle in a read-only memory region on the GPU. However, since constant memory is limited in size, this approach may not be suitable if other parts of the application also require constant memory.
* Copy the device handle into the GPU's shared memory, which incurs a one-time cost of copying the handle from global memory to shared memory, but allows faster access thereafter.
```

(channel-interfaces)=
## Channel Interfaces in GPU Kernels

In the GPU kernels of this example, we use the `relaxedSignal()` and `relaxedWait()` methods of the `BaseMemoryChannelDeviceHandle` to synchronize operations between the two GPUs. The `relaxedWait()` method blocks the calling thread until it receives a signal from the other GPU, while `relaxedSignal()` sends a signal to the other GPU. To demonstrate the synchronization, we put a spin loop of 10 million clock cycles (which takes a few milliseconds) on one side of the ping-pong (GPU 0) and check if the elapsed time is greater than 1 millisecond on the other side (GPU 1).

```cpp
// From gpu_ping_pong.cu, lines 26-44
__global__ void gpuKernel0(mscclpp::BaseMemoryChannelDeviceHandle *devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * gridDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedWait();
      // spin for a few ms
      spin_cycles(1e7);
      devHandle->relaxedSignal();
    }
  }
}

__global__ void gpuKernel1(mscclpp::BaseMemoryChannelDeviceHandle *devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * gridDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedSignal();
      devHandle->relaxedWait();
    }
  }
}
```

`relaxedSignal()` and `relaxedWait()` are used to synchronize the execution flow, but they do not synchronize memory operations. This means that when `relaxedWait()` returns, it guarantees that the other GPU has executed `relaxedSignal()`, but it does not guarantee that the memory operations before `relaxedSignal()` have completed. This can happen because GPUs follow [weakly-ordered memory models](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions). If synchronization of memory operations is needed, you can use `signal()` and `wait()` instead, which will ensure that all memory operations before the signal are visible to the other GPU when its `wait()` returns. In this example, we do not need to synchronize memory operations, so we use `relaxedSignal()` and `relaxedWait()` which are faster.


## Summary and Next Steps

In this tutorial, you have learned the basic concepts of connections, semaphores, and channels in MSCCL++. In the next tutorial, we will introduce `Bootstrap` and `Communicator` interfaces, which provide a convenient way to set up connections across multiple processes.
