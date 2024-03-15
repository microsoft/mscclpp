# MSCCL++ Design Document
## Introduction
MSCCL++ redefines inter-GPU communication interfaces, thereby delivering a highly efficient and customizable communication stack for distributed GPU applications. Its design is specifically tailored to accommodate diverse performance optimization scenarios often encountered in state-of-the-art AI applications. The figure below provides a high-level overview of MSCCL++ abstractions in CUDA, C, and Python.

<!-- <center>MSCCL++ Abstractions Overview</center>

![MSCCL++ Abstractions](../figs/abstractions.png) -->


```{figure} ../figs/abstractions.png
:name: MSCCL++ Abstractions
:alt: MSCCL++ Abstractions
:align: center

MSCCL++ Abstractions Overview
```

The followings highlight the key features of MSCCL++.
* **Light-weight and multi-layer abstractions.** MSCCL++ provides communication abstractions at lowest level close to hardware and at the highest level close to application API. The lowest level of abstraction is ultra light weight which enables a user to implement logics of data movement for a collective operation such as AllReduce inside a GPU kernel extremely efficiently without worrying about memory ordering of different ops. The modularity of MSCCL++ enables a user to construct the building blocks of MSCCL++ in a high level abstraction in Python and feed them to a CUDA kernel in order to facilitate the user's productivity.

* **1-sided 0-copy synchronous and asynchronous abstracts.** MSCCL++ provides fine-grained synchronous and asynchronous 0-copy 1-sided abstracts for communication primitives such as `put()`, `get()`, `signal()`, `flush()`, and `wait()`. The 1-sided abstractions allows a user to asynchronously `put()` their data on the remote GPU as soon as it is ready without requiring the remote side to issue any receive instruction. This enables users to easily implement flexible communication logics, such as overlapping communication with computation, or implementing customized collective communication algorithms without worrying about potential deadlocks. Additionally, the 0-copy capability enables MSCCL++ to directly transfer data between user's buffers without using intermediate internal buffers which saves GPU bandwidth and memory capacity.

* **Unified abstractions for different interconnection hardware.** MSCCL++ provides consistent abstractions regardless of the location of the remote GPU (either on the local node or on a remote node) or the underlying link (either NVLink/xGMI or InfiniBand). This simplifies the code for inter-GPU communication, which is often complex due to memory ordering of GPU/CPU read/writes and therefore, is error-prone.

## Concepts

To implement the list of features above, some concepts are introduced.
### Channel
MSCCL++ provides peer-to-peer communication methods between GPUs. A peer-to-peer connection between two GPUs is called a Channel. Channels are constructed by MSCCL++ host-side interfaces and copied to GPUs during initialization. Channels provide GPU-side interfaces, which means that all communication methods are defined as a device function to be called from a GPU kernel code. Following code shows the basic usage for channel, the put() method in the following code copies 1KB data from the local GPU to a remote GPU.
```cpp
__global__ void gpuKernel() {
  ...
  // Only one thread is needed for this method.
  channel.put(/*dstOffset=*/ 0, /*srcOffset=*/ 0, /*size=*/ 1024);
  ...
}
```

#### SmChannel & ProxyChannel
MSCCL++ delivers two types of channels, ProxyChannel and SmChannel. ProxyChannel provides (R)DMA-based data copy and synchronization methods. When called, these methods send/receive a signal to/from a host-side proxy (hence the name ProxyChannel), which will trigger (R)DMA (such as cudaMemcpy* or ibv_post_send) or issue synchronization methods (such as cudaStreamSynchronize or ibv_poll_cq). Since the key functionalities are run by the proxy, ProxyChannel requires only a single GPU thread to call its methods.
On the other hand, SmChannel provides memory-mapping-based copy and synchronization methods. When called, these methods will directly use GPU threads to read/write from/to the remote GPU's memory space. Comparing against ProxyChannel, SmChannel is especially performant for low-latency scenarios, while it may need many GPU threads to call copying methods at the same time to achieve high copying bandwidth. See all SmChannel methods from here.

### Fifo & Trigger
To offload the communication logic from the GPU to the CPU, MSCCL++ introduces the concept of `Fifo` and `Trigger`. A Fifo is a circular buffer that shared between the GPU and the CPU. It is used to store `Trigger`. A Trigger is a signal that is sent from the GPU to the CPU to notify the CPU that there are commands in the Fifo that need to be processed. The CPU will then process the commands in the Fifo and send a signal back to the GPU to notify the GPU that the commands have been processed.

### ProxyService
Proxy service is a persistent service that resides in the CPU side. It functions as a polling service that receives the message trigger from the GPU side and then transfers data according to the command.  When we use ProxyChannel for communication, a trigger is sent from the GPU side to the ProxyService. Then ProxyService will invoke cudaMemcpy* or IB verbs to transfer data to the targe GPU.
