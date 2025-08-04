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

Run the example with `./bidir_memory_channel`. If you are in a container, you may need to run with root privileges. You should see output similar to the following:

```
# ./bidir_memory_channel
GPU 1: Preparing for tests ...
GPU 0: Preparing for tests ...
GPU 0: [Bidir Put] bytes 1024, elapsed 0.0065079 ms/iter, BW 0.157347 GB/s
GPU 0: [Bidir Put] bytes 1048576, elapsed 0.00926096 ms/iter, BW 113.225 GB/s
GPU 0: [Bidir Put] bytes 134217728, elapsed 0.389238 ms/iter, BW 344.822 GB/s
GPU 0: [Bidir Get] bytes 1024, elapsed 0.00437581 ms/iter, BW 0.234014 GB/s
GPU 0: [Bidir Get] bytes 1048576, elapsed 0.00768634 ms/iter, BW 136.421 GB/s
GPU 0: [Bidir Get] bytes 134217728, elapsed 0.417454 ms/iter, BW 321.515 GB/s
GPU 0: [Bidir Put Packets] bytes 1024, elapsed 0.00407117 ms/iter, BW 0.251525 GB/s
GPU 0: [Bidir Put Packets] bytes 1048576, elapsed 0.0104925 ms/iter, BW 99.936 GB/s
GPU 0: [Bidir Put Packets] bytes 134217728, elapsed 1.0188 ms/iter, BW 131.741 GB/s
Succeed!
```

The example code uses localhost port `50505` by default. If the port is already in use, you can change it by modifying the `PORT_NUMER` macro in the code.

```{caution}
Note that this example is **NOT** a performance benchmark. The performance numbers are provided to give you an idea of the performance characteristics of `MemoryChannel`. For the best performance, we need to tune the number of thread blocks and the number of threads per block according to the copy size and the hardware specifications. In addition, the synchronization can be more optimized depending on the application scenario and implementation.
```

## Code Overview

The example code establishes a channel in a similar way to that in the [Bootstrap and Communicator](./02-bootstrap-and-communicator.md) tutorial, but creates a `MemoryChannel` instead of a `BaseMemoryChannel`. To create a `MemoryChannel`, we need to specify the local and remote `RegisteredMemory` objects, which represent the memory regions that the channel can transfer data to/from. The following diagram illustrates how the `RegisteredMemory` objects are created and used to establish a `MemoryChannel`:

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

The example code demonstrates three different data transfer methods, `put()`, `get()`, and `putPackets()`. The code examines bidirectional data transfer performance of the three methods.

## RegisteredMemory and GpuBuffer

**RegisteredMemory** represents a memory region that can be accessed by local or remote processes. It provides a way to register a memory region for communication, allowing access to the remote memory. In the example code, each process creates a local `RegisteredMemory` object as follows:

```cpp
mscclpp::GpuBuffer buffer(bufferBytes);
mscclpp::RegisteredMemory localRegMem = comm.registerMemory(buffer.data(), buffer.bytes(), transport);
```

Here, we first allocate GPU device memory using `mscclpp::GpuBuffer` (will be explained) and then register its memory region with the `registerMemory()` method of the `Communicator`. If you are using the `Context` interface as shown in the [Basic Concepts](./01-basic-concepts.md) tutorial, you can use `context.registerMemory()` instead. The `transport` parameter specifies the transport types that this memory region can be accessed with. In this example, we use only `mscclpp::Transport::CudaIpc`, which allows the memory to be accessed by other processes using CUDA/HIP IPC. `CudaIpc` transport type is usually used for intra-node communication, but with certain hardware configurations, it can also be used for inter-node communication (such as [NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72) on NVIDIA Grace Blackwell platforms). We will introduce other transport types in later tutorials.

**GpuBuffer** is NOT a must for creating a `RegisteredMemory`; you can register any pre-allocated GPU memory region with `registerMemory()`. However, it is user's responsibility to ensure that the memory region is feasible for their communication operations. Depending on the hardware platform, some communication methods may require a specific way of memory allocation to ensure data consistency and correctness. `GpuBuffer` is a convenient way to allocate GPU memory that is compatible with the communication methods that MSCCL++ supports. It provides a simple interface for allocating GPU memory, and it automatically handles the memory deallocation when it goes out of scope.

```{note}
If you are an optimization expert, we recommend you learn about the details of `GpuBuffer`. It is a thin wrapper around the CUDA/HIP memory allocation APIs with the following features:
* If the GPU device is an NVIDIA GPU that supports [NVLink SHARP](https://docs.nvidia.com/networking/display/sharpv300), it automatically allocates [multimem-addressable](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multimem-addresses) memory. The allocated memory can be still directly accessed by other peer GPUs' threads, and users can run computation kernels on the memory region directly without performance degradation.
* If the GPU device is an AMD GPU that supports [Uncached Memory](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/definitions.html#memory-type), it automatically allocates uncached memory. For such GPU devices, uncached memory must be used if (1) a remote device (CPU, GPU, or NIC) may directly access and update the memory region, and (2) the local GPU device may wait for the update without synchronizing the whole device (e.g., via `hipDeviceSynchronize()` or `hipStreamSynchronize()`). Therefore, you do NOT need to use uncached memory unless you will use the memory region for synchronization flags or counters (such as MSCCL++ [Packets](#packets)). However, in general, we recommend to use uncached memory by default unless you understand the implications. As the name implies, uncached memory is not cached by the GPU, so it can be accessed without polluting the GPU cache (which is good for other parallel computation kernels). For the same reason, running complex computation kernels (such as matrix multiplication) on uncached memory may lead to performance degradation, so it is recommended to use uncached memory only for communication purposes.
```

## MemoryChannel

**MemoryChannel** is a specialized channel that allows direct access to remote GPU memory. In addition to the synchronization methods that are also provided by `BaseMemoryChannel`, `MemoryChannel` provides methods for data access and transfer between the local and remote memory regions. To construct a `MemoryChannel`, we need to specify the local and remote `RegisteredMemory` objects. `RegisteredMemory` provides `serialize()` and `deserialize()` methods to convert metadata of the memory region into a serialized format that can be sent over the network. While any IPC mechanism can be used to send the serialized data, MSCCL++ `Communicator` provides `sendMemory()` and `recvMemory()` methods to send and receive `RegisteredMemory` objects between processes. The following code shows an example:

```cpp
comm.sendMemory(localRegMem, remoteRank);
auto remoteRegMemFuture = comm.recvMemory(remoteRank);
mscclpp::RegisteredMemory remoteRegMem = remoteRegMemFuture.get();
```

After the `RegisteredMemory` objects are exchanged, we can create a `MemoryChannel` as follows:

```cpp
mscclpp::MemoryChannel memChan(sema, /*dst*/ remoteRegMem, /*src*/ localRegMem.data());
```

Here, `sema` is a pre-built semaphore that is used for synchronization methods, which is introduced in the [Basic Concepts](./01-basic-concepts.md) tutorial. The `remoteRegMem` and `localRegMem` are the destination and source memory regions, respectively. The following diagram illustrates how `memChan` channel uses these memory regions (A and B representing the two GPUs):

```{mermaid}
flowchart TD
    RegMemA -->|"put() from A"| RegMemB
    RegMemB -->|"put() from B"| RegMemA
    RegMemB -->|"get() from A"| RegMemA
    RegMemA -->|"get() from B"| RegMemB
```

### Copy with `put()`

The example code demonstrates a bidirectional data copy between two GPUs using `MemoryChannel` interfaces. Below is the GPU kernel code that performs the data copy using the `put()` method:

```cpp
__global__ void bidirPutKernel(mscclpp::MemoryChannelDeviceHandle *devHandle, size_t copyBytes, int myRank) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    devHandle->relaxedSignal();
    devHandle->relaxedWait();
  }
  devSyncer.sync(gridDim.x);

  const uint64_t srcOffset = myRank * copyBytes;
  const uint64_t dstOffset = srcOffset;
  devHandle->put(dstOffset, srcOffset, copyBytes, /*threadId*/ tid, /*numThreads*/ blockDim.x * gridDim.x);
  devSyncer.sync(gridDim.x);
  if (tid == 0) {
    devHandle->signal();
    devHandle->wait();
  }
}
```

Both GPUs run this kernel concurrently to copy data from their own memory regions to the other GPU's memory region. This code assumes that there is no preceding synchronization between the two GPUs. Therefore, to make sure the other side is ready to receive the data, each kernel needs to check if the other has started execution before proceeding with the data copy. This is done by a single thread (`tid == 0`) in each GPU signaling the other GPU (`relaxedSignal()`), and then waiting for the other GPU to signal that it is ready (`relaxedWait()`). We use the relaxed versions of signal and wait, because the purpose here is execution control, not data synchronization (see [Channel Interfaces in GPU Kernels](./01-basic-concepts.md#channel-interfaces) to recap). After one thread synchronizes the other GPU, all threads in the GPU kernel synchronize with `devSyncer.sync(gridDim.x)`, which ensures that all threads in the GPU kernel starts executing the data copy operation after the other GPU is ready.

The `put()` method is used to copy data from the source offset in the local memory region to the destination offset in the remote memory region. The `threadId` and `numThreads` parameters are used to map the data copy operation to the participating threads in the GPU kernel. Since the example code uses all threads in the GPU kernel to perform the data copy, we pass `tid` as the `threadId` and `blockDim.x * gridDim.x` as the `numThreads`. Users can also use a subset of threads to perform the data copy, in which case they can pass the appropriate values for `threadId` and `numThreads`. This can be useful for optimizing the data copy, especially when there are multiple destinations or sources, or when there is a following computation after `put()` to pipeline the data transfer with computation.

The example code assumes that there can be a following computation that consumes the received data, so it performs another synchronization after the data copy. It first synchronizes all threads in the GPU kernel (`devSyncer.sync(gridDim.x)`) to ensure that all threads have completed the data copy operation, and then the first thread (`tid == 0`) signals the other GPU that the data copy is complete (`devHandle->signal()`) and waits for the other GPU to acknowledge it (`devHandle->wait()`). This ensures that the other GPU can safely access the copied data after the data copy operation is complete.

### Copy with `get()`

While `put()` writes to the remote memory, `get()` reads from the remote memory. The example code demonstrates a bidirectional data copy using `get()` method, which is similar to `put()`, but it reads data from the remote memory region and writes it to the local memory region. The following code shows how to use `get()` in a GPU kernel:

```cpp
__global__ void bidirGetKernel(mscclpp::MemoryChannelDeviceHandle *devHandle, size_t copyBytes, int myRank) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    devHandle->relaxedSignal();
    devHandle->relaxedWait();
  }
  devSyncer.sync(gridDim.x);

  const int remoteRank = myRank ^ 1;
  const uint64_t srcOffset = remoteRank * copyBytes;
  const uint64_t dstOffset = srcOffset;
  devHandle->get(srcOffset, dstOffset, copyBytes, /*threadId*/ tid, /*numThreads*/ blockDim.x * gridDim.x);
}
```

Note that the `get()` method doesn't need explicit data synchronization after the data copy, because it is a read operation. This makes `get()` more efficient than `put()` especially for small data transfers. However, `get()` may not be suitable for all scenarios, especially when the data can be modified by the remote GPU while it is being read. For large data transfers, `put()` is usually considered more efficient, but it highly depends on the hardware implementation and we recommend you to benchmark the performance of both methods for your specific use case.

(packets)=
## Packets

The example code creates one more `MemoryChannel` to demonstrate the use of `putPackets()`, which enables a finer-grained synchronization mechanism for data transfer. The channel is created as follows:

```cpp
mscclpp::MemoryChannel memPktChan(sema, /*dst*/ remotePktRegMem, /*src*/ localRegMem.data(),
                                  /*packetBuffer*/ localPktRegMem.data());
```

Compared to the previous `memChan` channel, this `memPktChan` channel uses the same source (`localRegMem.data()`) but a different destination (`remotePktRegMem`) and an additional packet buffer (`localPktRegMem.data()`). The following diagram illustrates how the `memPktChan` channel uses these memory regions (A and B representing the two GPUs):

```{mermaid}
block-beta
    columns 6
    space:1
    RegMemA space:2 RegMemB
    space:8
    PktRegMemA space:2 PktRegMemB

    RegMemA --"putPackets()"--> PktRegMemB
    RegMemB --"putPackets()"--> PktRegMemA

    PktRegMemA --"unpackPackets()"--> RegMemA
    PktRegMemB --"unpackPackets()"--> RegMemB
```

The `putPackets()` method reads the data from the source memory region, converts it into packets, and writes the packets to the destination memory region. `memPktChan` channel sets the destination memory region to the packet buffer of the remote GPU, so that the remote GPU can use the `unpackPackets()` method, which reads the packets from the packet buffer and writes the data to the source memory region.

`mscclpp::LLPacket` (or `mscclpp::LL16Packet`) is the default packet type used by `putPackets()` and `unpackPackets()`.
