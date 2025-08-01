# Memory Channel

```{note}
This tutorial follows the [Bootstrap and Communicator](./02-bootstrap-and-communicator.md) tutorial.
```

In this tutorial, we will introduce comprehensive usage of `MemoryChannel`, which provides direct access to remote GPU memory for communication. We will cover how to create communication buffers, how to use them with `MemoryChannel`, and how to perform efficient data transfer between GPUs using `MemoryChannel`.

## Build and Run the Example

The code of this tutorial is under [examples/tutorials/03-memory-channel](https://github.com/microsoft/mscclpp/blob/main/examples/tutorials/03-memory-channel).

Build the example with `make`:

```bash
$ cd examples/tutorials/03-memory-channel
$ make
```

Run the example with `./perf_memory_channel`. If you are in a container, you may need to run with root privileges. You should see output similar to the following:

```
# ./perf_memory_channel
GPU 1: Preparing for bidirectional copy tests ...
GPU 0: Preparing for bidirectional copy tests ...
GPU 0: bytes 1024, elapsed 0.00654806 ms/iter, BW 0.156382 GB/s
GPU 0: bytes 1048576, elapsed 0.0092823 ms/iter, BW 112.965 GB/s
GPU 0: bytes 134217728, elapsed 0.388549 ms/iter, BW 345.433 GB/s
Succeed!
```

The example code uses localhost port `50505` by default. If the port is already in use, you can change it by modifying the `PORT_NUMER` macro in the code.

## Code Overview

The example code establishes a channel in a similar way to that in the [Bootstrap and Communicator](./02-bootstrap-and-communicator.md) tutorial, but creates a `MemoryChannel` instead of a `BaseMemoryChannel`. To create a `MemoryChannel`, we need to specify the local and remote `RegisteredMemory` objects, which represent the memory regions that will be used for data transfer. The following diagram illustrates how the `RegisteredMemory` objects are created and used to establish a `MemoryChannel`:

```{mermaid}
sequenceDiagram
    participant ProcessA
    participant ProcessB

    Note over ProcessA: Create RegisteredMemory A

    Note over ProcessB: Create RegisteredMemory B

    rect rgb(240, 240, 240)
        ProcessA->>ProcessB: Send and receive RegisteredMemory A
        Note over ProcessB: Create a MemoryChannel using a pre-built Semaphore<br>and RegisteredMemory B and A
    end
    
    rect rgb(240, 240, 240)
        ProcessB->>ProcessA: Send and receive RegisteredMemory B
        Note over ProcessA: Create a MemoryChannel using a pre-built Semaphore<br>and RegisteredMemory A and B
    end
```

The procedure of building a `Semaphore` is explained in the [Basic Concepts](./01-basic-concepts.md) tutorial.

## RegisteredMemory and GpuBuffer

**RegisteredMemory** represents a memory region that can be accessed by local or remote processes. It provides a way to register a memory region for communication, allowing access to the remote memory. In the example code, each process creates a local `RegisteredMemory` object as follows:

```cpp
// From perf_memory_channel.cu, lines 85-86
mscclpp::GpuBuffer buffer(bufferBytes);
mscclpp::RegisteredMemory localRegMem = comm.registerMemory(buffer.data(), buffer.bytes(), transport);
```

Here, we first allocate GPU device memory using `mscclpp::GpuBuffer` (will be explained) and then register its memory region with the `registerMemory()` method of the `Communicator`. If you are using the `Context` interface as shown in the [Basic Concepts](./01-basic-concepts.md) tutorial, you can use `context.registerMemory()` instead. The `transport` parameter specifies the transport types that this memory region can be accessed with. In this example, we use only `mscclpp::Transport::CudaIpc`, which allows the memory to be accessed by other processes using CUDA/HIP IPC. `CudaIpc` transport type is usually used for intra-node communication, but with certain hardware configurations, it can also be used for inter-node communication (such as [NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72) on NVIDIA Grace Blackwell platforms). We will introduce other transport types in later tutorials.

**GpuBuffer** is NOT a must for creating a `RegisteredMemory`; you can register any pre-allocated GPU memory region with `registerMemory()`. However, it is user's responsibility to ensure that the memory region is feasible for their communication operations. Depending on the hardware platform, some communication methods may require a specific way of memory allocation to ensure data consistency and correctness. `GpuBuffer` is a convenient way to allocate GPU memory that is compatible with the communication methods that MSCCL++ supports. It provides a simple interface for allocating GPU memory, and it automatically handles the memory deallocation when it goes out of scope.

```{note}
If you are an optimization expert, we recommend you learn about the details of `GpuBuffer`. It is a thin wrapper around the CUDA/HIP memory allocation APIs with the following features:
* If the GPU device is an NVIDIA GPU that supports [NVLink SHARP](https://docs.nvidia.com/networking/display/sharpv300), it automatically allocates [multimem-addressable](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multimem-addresses) memory. The allocated memory can be still directly accessed by other peer GPUs' threads, and users can run computation kernels on the memory region directly without performance degradation.
* If the GPU device is an AMD GPU that supports [Uncached Memory](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/definitions.html#memory-type), it automatically allocates uncached memory. For such GPU devices, uncached memory must be used if (1) a remote device (CPU, GPU, or NIC) may directly access and update the memory region, and (2) the local GPU device may wait for the update without synchronizing the whole device (e.g., via `hipDeviceSynchronize()` or `hipStreamSynchronize()`). Therefore, you do NOT need to use uncached memory unless you will use the memory region for synchronization flags or counters (such as MSCCL++ Packets). However, in general, we recommend to use uncached memory by default unless you understand the implications. As the name implies, uncached memory is not cached by the GPU, so it can be accessed without polluting the GPU cache (which is good for other parallel computation kernels). For the same reason, running complex computation kernels (such as matrix multiplication) on uncached memory may lead to performance degradation, so it is recommended to use uncached memory only for communication purposes.
```


