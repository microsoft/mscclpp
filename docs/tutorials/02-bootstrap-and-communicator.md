# Bootstrap and Communicator

```{note}
This tutorial follows the [Basic Concepts](./01-basic-concepts.md) tutorial.
```

In this tutorial, we introduce `Bootstrap` and `Communicator` interfaces, which provide a convenient way to set up connections across multiple processes. The example code implements the same ping-pong example as in the [Basic Concepts](./01-basic-concepts.md) tutorial, but using one process per GPU and the `Bootstrap` and `Communicator` interfaces to establish connections.

## Build and Run the Example

The code of this tutorial is under [examples/tutorials/02-bootstrap-and-communicator](https://github.com/microsoft/mscclpp/blob/main/examples/tutorials/02-bootstrap-and-communicator).

Build the example with `make`:

```bash
$ cd examples/tutorials/02-bootstrap-and-communicator
$ make
```

Run the example with `./gpu_ping_pong_mp`. If you are in a container, you may need to run with root privileges. You should see output similar to the following:

```
# ./gpu_ping_pong_mp
GPU 1: Initializing a bootstrap ...
GPU 0: Initializing a bootstrap ...
GPU 0: Creating a connection ...
GPU 1: Creating a connection ...
GPU 0: Creating a semaphore ...
GPU 1: Creating a semaphore ...
GPU 1: Creating a channel ...
GPU 0: Creating a channel ...
GPU 1: Launching a GPU kernel ...
GPU 0: Launching a GPU kernel ...
Elapsed 4.78082 ms per iteration (100)
Succeed!
```

The example code uses localhost port `50505` by default. If the port is already in use, you can change it by modifying the `PORT_NUMER` macro in the code.

If you see error messages like "At least two GPUs are required" or "GPU 0 cannot access GPU 1", it means that your system does not meet the requirements for running the example. Make sure you have at least two GPUs installed and that they are connected peer-to-peer (through NVLink or under the same PCIe switch). See the {ref}`prerequisites` for more details.

## Code Overview

The example code is similar to that in the [Basic Concepts](./01-basic-concepts.md) tutorial, but uses `Bootstrap` and `Communicator` interfaces to establish connections between GPUs. The code spawns one child process per GPU. The parent process waits for the child processes to finish before exiting.

## Bootstrap

**Bootstrap** is an abstract class that defines common inter-process communication (IPC) interfaces such as `send()`, `recv()`, `allGather()`, and `barrier()`. Bootstrap is used to exchange serialized MSCCL++ objects between host processes, or to synchronize the processes. `TcpBootstrap` is a concrete implementation of the `Bootstrap` interface that uses TCP sockets for communication.

In the example code, two processes create and initialize a `TcpBootstrap` instance as follows:

```cpp
// From gpu_ping_pong_mp.cu, lines 100-101
auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
bootstrap->initialize("lo:127.0.0.1:" PORT_NUMER);
```

`myRank` is the rank of the current process, and `nRanks` is the total number of processes. The `initialize()` method sets up the bootstrap connection between all processes. In this example, we pass an `ifIpPortTrio` string, which has the format `if:ip:port`, where `if` is the network interface (e.g., `lo` for localhost), `ip` is the IP address, and `port` is the port number. The `TcpBootstrap` will listen on the specified port and accept connections from other processes.

```{note}
Alternatively, `TcpBootstrap` can be initialized with a **UniqueId**, which is a unique identifier for the bootstrap connection. This is similar to what NCCL does with its [`ncclGetUniqueId()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclgetuniqueid) and [`ncclCommInitRank()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#ncclcomminitrank) functions. The UniqueId should be shared between processes using an external mechanism, such as using MPI like the following:

```cpp
auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(myRank, nRanks);
mscclpp::UniqueId id;
if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
bootstrap->initialize(id);
```

## Communicator

While `Bootstrap` provides general IPC interfaces, `Communicator` is a wrapper around `Bootstrap` that provides more specific methods for building channels between GPUs.

In the example code, `Communicator` is constructed as follows:

```cpp
// From gpu_ping_pong_mp.cu, line 102
mscclpp::Communicator comm(bootstrap);
```

Then it creates a GPU endpoint that connects to the remote rank:

```cpp
// From gpu_ping_pong_mp.cu, lines 106-107
auto connFuture = comm.connect({transport, {mscclpp::DeviceType::GPU, gpuId}}, remoteRank);
auto conn = connFuture.get();
```

The `connect()` method builds a connection asynchronously; it returns a future of a connection object. The `get()` method is called later on the future to retrieve the connection object. In this example, we call `get()` immediately since we don't have other tasks in between.

After the connection is established, we create a semaphore for synchronization:

```cpp
// From gpu_ping_pong_mp.cu, lines 111-112
auto semaFuture = comm.buildSemaphore(conn, remoteRank);
auto sema = semaFuture.get();
```

Like `connect()`, `buildSemaphore()` is an asynchronous method that returns a future of a semaphore object.

We omit explaining the rest of the code, as it is similar to that in the [Basic Concepts](./01-basic-concepts.md) tutorial.

## Summary and Next Steps

In this tutorial, you have learned how to use `Bootstrap` and `Communicator` interfaces to establish connections between multiple processes. Note that `Bootstrap` and `Communicator` are still optional interfaces for convenience. As noted in the [Basic Concepts](./01-basic-concepts.md) tutorial, you can still use your own IPC mechanisms to build connections and semaphores. For advanced examples that use Redis or `torch.distributed` for IPC, see the [Advanced Connections](../guide/advanced-connections.md) guide.

In the next tutorial, we will introduce more comprehensive usage of `MemoryChannel` including how to use it for efficient data transfer between GPUs. It will also cover how to create communication buffers and how to use them with channels.
