# Memory Channel

```{note}
This tutorial follows the [Bootstrap and Communicator](./02-bootstrap-and-communicator.md) tutorial.
```

In this tutorial, we will introduce the comprehensive usage of `MemoryChannel`, which provides direct access to remote GPU memory for communication. We will cover how to create communication buffers, use them with `MemoryChannel`, and perform efficient data transfer between GPUs.

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

The example code uses localhost port `50505` by default. If the port is already in use, you can change it by modifying the `PORT_NUMBER` macro in the code.

```{caution}
Note that this example is **NOT** a performance benchmark. The performance numbers are provided to give you an idea of the performance characteristics of `MemoryChannel`. For optimal performance, we need to tune the number of thread blocks and threads per block according to the copy size and hardware specifications. Additionally, synchronization can be further optimized depending on the application scenario and implementation.
```

## Code Overview

The example code establishes a channel similarly to the [Bootstrap and Communicator](./02-bootstrap-and-communicator.md) tutorial, but creates a `MemoryChannel` instead of a `BaseMemoryChannel`. To create a `MemoryChannel`, we need to specify the local and remote `RegisteredMemory` objects, which represent the memory regions that the channel can transfer data to/from. The following diagram illustrates how `RegisteredMemory` objects are created and used to establish a `MemoryChannel`:

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

The procedure for building a `Semaphore` is explained in the [Basic Concepts](./01-basic-concepts.md) tutorial.

The example code implements three GPU kernels that perform the same bidirectional data transfer operation using different methods: `put()`, `get()`, and `putPackets()`. The code examines the performance of these three methods.

## RegisteredMemory and GpuBuffer

**RegisteredMemory** represents a memory region that can be accessed by local or remote processes. It provides a way to register a memory region for communication, allowing remote memory access. In the example code, each process creates a local `RegisteredMemory` object as follows:

```cpp
mscclpp::GpuBuffer buffer(bufferBytes);
mscclpp::RegisteredMemory localRegMem = comm.registerMemory(buffer.data(), buffer.bytes(), transport);
```

Here, we first allocate GPU device memory using `mscclpp::GpuBuffer` and then register its memory region with the `registerMemory()` method of the `Communicator`. If you are using the `Context` interface as shown in the [Basic Concepts](./01-basic-concepts.md) tutorial, you can use `context.registerMemory()` instead. The `transport` parameter specifies the transport types that this memory region can be accessed with. In this example, we use only `mscclpp::Transport::CudaIpc`, which allows the memory to be accessed by other processes using CUDA/HIP IPC. The `CudaIpc` transport type is typically used for intra-node communication, but with certain hardware configurations, it can also be used for inter-node communication (will be explained in a later section: {ref}`mc-cross-node`). We will introduce other transport types in later tutorials.

**GpuBuffer** is NOT required for creating a `RegisteredMemory`; you can register any pre-allocated GPU memory region with `registerMemory()`. However, it is the user's responsibility to ensure that the memory region is suitable for their communication operations. Depending on the hardware platform, some communication methods may require specific memory allocation to ensure data consistency and correctness. `GpuBuffer` is a convenient way to allocate GPU memory that is compatible with the communication methods that MSCCL++ supports. It provides a simple interface for allocating GPU memory and automatically handles memory deallocation when it goes out of scope.

```{note}
If you are an optimization expert, we recommend learning about the details of `GpuBuffer`. It is a thin wrapper around the CUDA/HIP memory allocation APIs with the following features:
* If the GPU device is an NVIDIA GPU that supports [NVLink SHARP](https://docs.nvidia.com/networking/display/sharpv300), it automatically allocates [multimem-addressable](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multimem-addresses) memory. The allocated memory can still be directly accessed by other peer GPUs' threads, and users can run computation kernels on the memory region directly without performance degradation.
* If the GPU device is an AMD GPU that supports [Uncached Memory](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/definitions.html#memory-type), it automatically allocates uncached memory. For such GPU devices, uncached memory must be used if (1) a remote device (CPU, GPU, or NIC) may directly access and update the memory region, and (2) the local GPU device may wait for the update without synchronizing the whole device (e.g., via `hipDeviceSynchronize()` or `hipStreamSynchronize()`). Therefore, you do NOT need to use uncached memory unless you will use the memory region for synchronization flags or counters (such as MSCCL++ [Packets](#packets)). However, in general, we recommend using uncached memory by default unless you understand the implications. As the name implies, uncached memory is not cached by the GPU, so it can be accessed without polluting the GPU cache (which is beneficial for other parallel computation kernels). For the same reason, running complex computation kernels (such as matrix multiplication) on uncached memory may lead to performance degradation, so it is recommended to use uncached memory only for communication purposes.
```

## MemoryChannel

**MemoryChannel** is a specialized channel that allows direct access to remote GPU memory. In addition to the synchronization methods provided by `BaseMemoryChannel`, `MemoryChannel` provides methods for data access and transfer between local and remote memory regions. To construct a `MemoryChannel`, we need to specify the local and remote `RegisteredMemory` objects. `RegisteredMemory` provides `serialize()` and `deserialize()` methods to convert memory region metadata into a serialized format that can be sent over the network. While any IPC mechanism can be used to send the serialized data, MSCCL++ `Communicator` provides `sendMemory()` and `recvMemory()` methods to send and receive `RegisteredMemory` objects between processes. The following code shows an example:

```cpp
comm.sendMemory(localRegMem, remoteRank);
auto remoteRegMemFuture = comm.recvMemory(remoteRank);
mscclpp::RegisteredMemory remoteRegMem = remoteRegMemFuture.get();
```

After exchanging the `RegisteredMemory` objects, we can create a `MemoryChannel` as follows:

```cpp
mscclpp::MemoryChannel memChan(sema, /*dst*/ remoteRegMem, /*src*/ localRegMem);
```

Here, `sema` is a pre-built semaphore used for synchronization methods, which is introduced in the [Basic Concepts](./01-basic-concepts.md) tutorial. The `remoteRegMem` and `localRegMem` are the destination and source memory regions, respectively. The following diagram illustrates how the `memChan` channel uses these memory regions (A and B representing the two GPUs):

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

Both GPUs run this kernel concurrently to copy data from their own memory regions to the other GPU's memory region. This code assumes no preceding synchronization between the two GPUs. Therefore, to ensure the other side is ready to receive data, each kernel needs to check if the other has started execution before proceeding with the data copy. This is done by a single thread (`tid == 0`) in each GPU signaling the other GPU (`relaxedSignal()`), and then waiting for the other GPU to signal that it is ready (`relaxedWait()`). We use the relaxed versions of signal and wait because the purpose here is execution control, not data synchronization (see {ref}`channel-interfaces` to recap). After one thread synchronizes with the other GPU, all threads in the GPU kernel synchronize with `devSyncer.sync(gridDim.x)`, which ensures that all threads in the GPU kernel start executing the data copy operation after the other GPU is ready.

The `put()` method copies data from the source offset in the local memory region to the destination offset in the remote memory region. The `threadId` and `numThreads` parameters map the data copy operation to the participating threads in the GPU kernel. Since the example code uses all threads in the GPU kernel to perform the data copy, we pass `tid` as the `threadId` and `blockDim.x * gridDim.x` as the `numThreads`. Users can also use a subset of threads to perform the data copy by passing the appropriate values for `threadId` and `numThreads`. This can be useful for optimizing the data copy, especially when there are multiple destinations or sources, or when following computation after `put()` needs to be pipelined with the data transfer.

The example code assumes there may be following computation that consumes the received data, so it performs another synchronization after the data copy. It first synchronizes all threads in the GPU kernel (`devSyncer.sync(gridDim.x)`) to ensure that all threads have completed the data copy operation, and then the first thread (`tid == 0`) signals the other GPU that the data copy is complete (`devHandle->signal()`) and waits for the other GPU to acknowledge it (`devHandle->wait()`). This ensures that the other GPU can safely access the copied data after the data copy operation is complete.

### Copy with `get()`

While `put()` writes to the remote memory, `get()` reads from the remote memory. The example code demonstrates a bidirectional data copy using the `get()` method, which is similar to `put()`, but reads data from the remote memory region and writes it to the local memory region. The following code shows how to use `get()` in a GPU kernel:

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

Note that the `get()` method doesn't need explicit data synchronization after the data copy because it is a read operation. This makes `get()` more efficient than `put()`, especially for small data transfers. However, `get()` may not be suitable for all scenarios, especially when the data can be modified by the remote GPU while it is being read. For large data transfers, `put()` is usually considered more efficient, but this highly depends on the hardware implementation, and we recommend benchmarking the performance of both methods for your specific use case.

(packets)=
## Packets

In MSCCL++, **Packet** is a data structure that contains user data with metadata (which we call *flags*) that can validate the user data's integrity. This allows the receiver to safely retrieve the user data without explicit synchronization (signal and wait). Using packets is often faster than `put()` for small data transfers and more flexible than `get()` because both the sender and receiver can work at their own pace. However, the goodput of communication using packets is much smaller than that of `put()` or `get()` because packets require additional metadata to be sent along with the user data.

The example code creates one more `MemoryChannel` to demonstrate usage of packets. The channel is created as follows:

```cpp
mscclpp::MemoryChannel memPktChan(sema, /*dst*/ remotePktRegMem, /*src*/ localRegMem,
                                  /*packetBuffer*/ localPktRegMem.data());
```

Compared to the previous `memChan` channel, this `memPktChan` channel uses the same source (`localRegMem`) but a different destination (`remotePktRegMem`) and an additional packet buffer (`localPktRegMem.data()`). The following diagram illustrates how the `memPktChan` channel uses these memory regions (A and B representing the two GPUs):

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

The `putPackets()` method reads data from the source memory region, converts it into packets, and writes the packets to the destination memory region. The `memPktChan` channel sets the destination memory region to the packet buffer of the remote GPU, so that the remote GPU can use the `unpackPackets()` method, which reads packets from the local packet buffer and writes the data to the source memory region locally. The example code demonstrates how to use `putPackets()` and `unpackPackets()` in a GPU kernel:

```cpp
__global__ void bidirPutPacketKernel(mscclpp::MemoryChannelDeviceHandle *devHandle, size_t copyBytes, int myRank,
                                     uint32_t flag) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid == 0) {
    devHandle->relaxedSignal();
    devHandle->relaxedWait();
  }
  devSyncer.sync(gridDim.x);

  const uint64_t srcOffset = myRank * copyBytes;
  const uint64_t dstOffset = srcOffset;
  const uint64_t pktBufOffset = 0;
  devHandle->putPackets(pktBufOffset, srcOffset, copyBytes, tid, blockDim.x * gridDim.x, flag);
  devHandle->unpackPackets(pktBufOffset, dstOffset, copyBytes, tid, blockDim.x * gridDim.x, flag);
}
```

The `flag` parameter is used to construct the packets. It can be any non-zero 4-byte value. If `putPackets()` may directly overwrite previous packets without clearing the packet buffer (as in the example code), the flag value should be different from the previous packets' flags. The figure below illustrates how packets are constructed. `D0-3` are the user data (4 bytes each), and each packet consists of two user data and two flags. We call this packet format `mscclpp::LL16Packet`, which is the default format of `putPackets()` and `unpackPackets()`. The name `LL` stands for "low-latency" (borrowed term from the [LL protocol of NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-proto)) and `16` indicates the packet size.

```{mermaid}
block-beta
columns 2
  block:Data
    Data0["D0"]
    Data1["D1"]
    Data2["D2"]
    Data3["D3"]
  end
  space
  block:PacketA
    PacketA0["D0"]
    PacketA1["flag"]
    PacketA2["D1"]
    PacketA3["flag"]
  end
  block:PacketB
    PacketB0["D2"]
    PacketB1["flag"]
    PacketB2["D3"]
    PacketB3["flag"]
  end
  space
  Data0 --> PacketA0
  Data1 --> PacketA2
  Data2 --> PacketB0
  Data3 --> PacketB2
  style Data fill:#ffffff,stroke:#ffffff
  style PacketA fill:#f0f0f0,stroke:#ffffff
  style PacketB fill:#f0f0f0,stroke:#ffffff
```

Since the flags take 50% of the packet size, the goodput of communication using packets is only 50% compared to transferring raw data. However, this doesn't matter because packets are designed for small data transfers. Packets transfer small data efficiently because the integrity of the user data is guaranteed by only waiting for the correct flags (done by `unpackPackets()`); explicit memory synchronization (signal and wait) is not needed.

(mc-cross-node)=
## Cross-node Execution

For **inter-node** communication, using `PortChannel` (will be explained in the following tutorial) is usually a more accessible option that leverages more widely-used networking interfaces. However, `MemoryChannel` can still be used as long as the underlying hardware allows memory mapping between the two GPUs, such as [Multi-Node NVLink (MNNVL)](https://docs.nvidia.com/multi-node-nvlink-systems/mnnvl-user-guide/overview.html) on NVIDIA Grace Blackwell platforms.

We can use the same example code to test inter-node `MemoryChannel`. The code performs explicit checks to verify MNNVL support and environment readiness. If you need additional details about environment requirements or troubleshooting, please refer to [MNNVL user guide](https://docs.nvidia.com/multi-node-nvlink-systems/mnnvl-user-guide/verifying.html).

Run the program on two nodes with command line arguments:

```
./bidir_memory_channel [<ip_port> <rank> <gpu_id>]
```

For example, assume we use `192.168.0.1:50000` as the bootstrap IP address and port, and both nodes use GPU 0 locally.

On Node 0 (Rank 0):
```bash
$ ./bidir_memory_channel 192.168.0.1:50000 0 0
```

On Node 1 (Rank 1):
```bash
$ ./bidir_memory_channel 192.168.0.1:50000 1 0
```

You should see output indicating successful data transfer.

```{tip}
If your bootstrap IP address is not on the default network interface of your node, you can specify the network interface by passing `interface_name:ip:port` as the first argument (such as `eth1:192.168.0.1:50000`).
```

## Summary and Next Steps

In this tutorial, you have learned how to use `MemoryChannel` for efficient data transfer between GPUs. You have also learned how to create communication buffers using `RegisteredMemory` and `GpuBuffer`, and how to use packets for small data transfers. You can find more complex usage of `MemoryChannel` in the {ref}`mscclpp-test`.

In the next tutorial, we will introduce `PortChannel`, which is another type of channel that provides port-based data transfer methods.
