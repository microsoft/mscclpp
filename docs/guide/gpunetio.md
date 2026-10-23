# GPU-Initiated PortChannel with GPUNetIO

By default, `PortChannel` operations over InfiniBand are pushed by GPU threads into a FIFO and posted to the NIC by the
host proxy thread of a `ProxyService`. With the optional GPUNetIO backend, GPU threads build the RDMA work requests and
ring the NIC doorbell themselves through [DOCA GPUNetIO](https://github.com/NVIDIA-DOCA/gpunetio) (GDAKI), removing
the proxy from the data path. The setup flow and the `PortChannel` device API are unchanged; the backend is selected
per endpoint with `EndpointConfig::Ib::Mode::GpuNetIo`.

## Build

```bash
cmake -DMSCCLPP_USE_GPUNETIO=ON ..
```

GPUNetIO requires a CUDA build with InfiniBand support (`MSCCLPP_USE_IB=ON`). At configure time, CMake fetches the
GPUNetIO sources (pinned to v4.0.1, see `cmake/GpuNetIo.cmake`) and compiles them into `libmscclpp`. To build without
network access, point CMake to a local checkout of the same revision:

```bash
cmake -DMSCCLPP_USE_GPUNETIO=ON -DFETCHCONTENT_SOURCE_DIR_GPUNETIO=/path/to/gpunetio ..
```

The build compiles a patched copy of `doca_gpunetio_high_level.cpp` that backports an upstream fix for a resource leak
on a QP creation failure path; the fetched tree itself is not modified. The GPUNetIO license is installed to
`share/licenses/mscclpp/gpunetio`.

## Requirements

- An NVIDIA GPU and an RDMA NIC that support GPU-rung doorbells (`DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB`).
- RDMA atomics on both sides. Signals are NIC atomic adds on the remote semaphore token, so a GPUNetIO endpoint cannot
  connect to an endpoint in `HostNoAtomic` mode.
- The local endpoint must be on a GPU (`EndpointConfig::device.type == DeviceType::GPU`).

## Usage

Create the connection with the GPUNetIO mode and build the channel with `ProxyService` exactly as for other IB modes:

```cpp
mscclpp::EndpointConfig cfg;
cfg.transport = mscclpp::Transport::IB0;
cfg.ib.mode = mscclpp::EndpointConfig::Ib::Mode::GpuNetIo;
auto connection = communicator->connect(cfg, remoteRank).get();

auto localMem = communicator->registerMemory(buff, bytes, mscclpp::Transport::IB0);
communicator->sendMemory(localMem, remoteRank);
auto remoteMem = communicator->recvMemory(remoteRank).get();

auto semaphore = communicator->buildSemaphore(connection, remoteRank).get();
mscclpp::ProxyService proxyService;
auto semaphoreId = proxyService.addSemaphore(semaphore);
auto dst = proxyService.addMemory(remoteMem);
auto src = proxyService.addMemory(localMem);
// Add every memory the channel will address before creating it.
auto channel = proxyService.portChannel(semaphoreId, dst, src);
auto deviceHandle = channel.deviceHandle();
```

In the kernel, `put`, `signal`, `putWithSignal`, `putWithSignalAndFlush`, `flush`, `accumulate`, `wait` and `poll`
behave as for proxy-backed channels. The signal of `putWithSignal` is posted on the same queue pair after the payload, so
the remote side observes the data before the semaphore update. `flush` waits for the local completion of all work
requests posted on the queue pair.

Only `port` and `gidIndex` of `EndpointConfig::Ib` are used in this mode; queue sizes are chosen by the backend.

## Using the device API from another project

Targets that link `mscclpp` inside the MSCCL++ build tree inherit the required compile definition and include paths.
Kernels compiled against an installed MSCCL++ must add:

```bash
-DMSCCLPP_USE_GPUNETIO -I<install-prefix>/include/mscclpp/gpunetio
```

## Limitations

- Host-side `Connection` operations (`write`, `updateAndSync`, `accumulate`, `flush`) and `Host2DeviceSemaphore::signal`
  are not supported on GPUNetIO connections and throw `InvalidUsage`.
- Channels must be created with `ProxyService::portChannel()` / `basePortChannel()`. A `PortChannel` constructed
  directly over a GPUNetIO connection has no memory bindings, so its `deviceHandle()` throws `InvalidUsage`.
- A channel captures the memories registered with its `ProxyService` when it is created; memories added later are not
  addressable by it.
- On RoCE devices only port 1 is supported (the pinned DOCA GPUNetIO resolves the destination MAC on port 1).
- `accumulate` requires an 8-byte-aligned destination offset.
