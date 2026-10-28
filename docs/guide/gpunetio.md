# GPU-Initiated PortChannel with GPUNetIO

By default, `PortChannel` operations over InfiniBand are pushed by GPU threads into a FIFO and posted to the NIC by the
host proxy thread of a `ProxyService`. With the optional GPUNetIO backend, GPU threads build the RDMA work requests and
ring the NIC doorbell themselves through [DOCA GPUNetIO](https://github.com/NVIDIA-DOCA/gpunetio) (GDAKI), removing
the proxy from the data path. The setup flow and the `PortChannel` device API are unchanged; the backend is selected
per endpoint with `EndpointConfig::Ib::Mode::GpuNetIo`.

## Build

The backend is built automatically for CUDA builds with InfiniBand support (`MSCCLPP_USE_IB=ON`, the default); no
extra CMake option is needed. ROCm builds and builds with `MSCCLPP_USE_IB=OFF` skip it. At configure time, CMake
fetches the GPUNetIO sources (pinned to v5.0.0-rc1, see `cmake/GpuNetIo.cmake`) and compiles them into `libmscclpp`.
To build without network access, point CMake to a local checkout of the same revision:

```bash
cmake -DFETCHCONTENT_SOURCE_DIR_GPUNETIO=/path/to/gpunetio ..
```

The build compiles a patched copy of `doca_gpunetio_high_level.cpp` that fixes a double free on a CQ creation failure
path; the fetched tree itself is not modified. The GPUNetIO license is installed to `share/licenses/mscclpp/gpunetio`.

## Requirements

- A Hopper or newer NVIDIA GPU and an RDMA NIC with a BlueFlame UAR
  (`DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF`). Device code must target `sm_90` or newer.
- RDMA atomics on both sides for `accumulate`; a GPUNetIO endpoint cannot connect to an endpoint in `HostNoAtomic`
  mode.
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
Small `put` and `putWithSignal` payloads, as well as semaphore signals, are embedded in the work request rather than
read separately by the NIC.

Callers must make source data ready before `put` or `putWithSignal`, synchronizing with its producers as needed.
These operations rely on DOCA's queue-publication release ordering rather than adding a full system fence for
every submission. Keep non-inline source buffers unchanged until their transfers complete.

### Exclusively accessed queue pairs

GPUNetIO always uses exclusive QPs. Callers must serialize **all** GPU operations on each connection, including
`flush` and channels sharing that connection. A different thread may take over after a barrier or release/acquire
handoff; the caller need not be thread zero. Different QPs can be used concurrently.

All requests, including strong-ordered inline semaphore signals (`DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_STRONG_ORDERING`),
atomics and non-inline transfers, are submitted through BlueFlame.
GPU-resident recovery WQEs are retained. BF submission uses 8 KiB of per-block shared-memory WQE scratch (64 pairs of
WQE slots) and waits for async source consumption before reusing a slot. Thread zero and blocks with up to 64 threads
have dedicated slot pairs; additional threads in larger blocks serialize access to their hashed slot pair using an
internal workspace lock, so different QPs cannot overwrite each other's WQEs.
There is no regular-doorbell-only or CPU-proxy fallback.

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
