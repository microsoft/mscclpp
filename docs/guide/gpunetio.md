# GPU-Initiated PortChannel with GPUNetIO

By default, `PortChannel` operations over InfiniBand are pushed by GPU threads into a FIFO and posted to the NIC by the
host proxy thread of a `ProxyService`. With the optional GPUNetIO backend, GPU threads build the RDMA work requests and
ring the NIC doorbell themselves through [DOCA GPUNetIO](https://github.com/NVIDIA-DOCA/gpunetio) (GDAKI), removing
the proxy from the data path. The host setup flow and `PortChannel` posting methods are unchanged; kernels provide
shared-memory WQE scratch as shown below. The backend is selected per endpoint with `EndpointConfig::Ib::Mode::Device`.

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

Create the connection with `EndpointConfig::Ib::Mode::Device` and construct the `PortChannel` directly from the
semaphore and registered memories (no `ProxyService` is needed):

```cpp
mscclpp::EndpointConfig cfg;
cfg.transport = mscclpp::Transport::IB0;
cfg.ib.mode = mscclpp::EndpointConfig::Ib::Mode::Device;
auto connection = communicator->connect(cfg, remoteRank).get();

auto localMem = communicator->registerMemory(buff, bytes, mscclpp::Transport::IB0);
communicator->sendMemory(localMem, remoteRank);
auto remoteMem = communicator->recvMemory(remoteRank).get();

auto semaphore = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator, connection);
mscclpp::PortChannel channel(semaphore, remoteMem, localMem);
auto deviceHandle = channel.deviceHandle<mscclpp::PortChannelBackend::Device>();
```

Host-created device handles use regular doorbell submission (`submit_db`, preparing WQEs in GPU global memory) by
default without requiring shared memory. To enable BlueFlame submission for transfers up to `device_.blueFlameLimit`
(default: 28 bytes), bind caller-owned shared-memory scratch on a **local copy** of the handle:

```cpp
__global__ void send(mscclpp::PortChannelDeviceHandleT<mscclpp::PortChannelBackend::Device> channel, size_t bytes) {
  __shared__ mscclpp::GpuNetIoWqeScratch scratch;
  if (threadIdx.x != 0) return;
  auto portChan = channel;
  portChan.device_.wqeScratchShm = &scratch;
  portChan.putWithSignal<mscclpp::PortChannelSharingMode::Exclusive>(0, bytes);
  portChan.flush<mscclpp::PortChannelSharingMode::Exclusive>();
}

// One block exclusively owns this QP.
send<<<1, 32>>>(deviceHandle, bytes);
```

Never write a block-shared pointer into a constant-memory or globally shared channel handle. When `wqeScratchShm` is
null or a transfer exceeds `blueFlameLimit`, work requests are submitted via regular doorbell without shared-memory
staging. The proxy backend ignores the `device_` fields, so the same kernel can support either backend.

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

### Queue pair sharing modes and BlueFlame configuration

By default, queue pairs are accessed in `PortChannelSharingMode::Gpu` (safe when threads across the GPU share a QP).
Callers may also pass `PortChannelSharingMode::Cta` (shared within a CTA) or `PortChannelSharingMode::Exclusive`
(operations on a QP are serialized by the caller) as a template parameter to posting and `flush` methods.

`device_.blueFlameLimit` controls the maximum payload size (in bytes) submitted through BlueFlame when `wqeScratchShm`
is bound (default: `kDefaultBlueFlameLimit = 28`, matching inline payloads; set to `0` to disable BlueFlame or increase
to use BlueFlame for larger transfers). Payloads above `blueFlameLimit`—or any request when
`wqeScratchShm == nullptr`—write their WQEs directly to the GPU-resident send queue and ring the regular doorbell,
matching NCCL GIN's `submit_db` path.

### Caller-owned WQE scratch and optional lock

`mscclpp::GpuNetIoWqeScratch` is an opaque 128-byte, 16-byte-aligned shared-memory buffer for two WQEs. When BlueFlame
is enabled (`wqeScratchShm != nullptr` and `size <= blueFlameLimit`), each scratch entry must either:
- Be owned exclusively by one posting thread at a time (`reuseSharedWqe == false`), or
- Be shared across concurrent submitters by setting `device_.reuseSharedWqe = true` and binding `device_.wqeScratchLock`
  to a zero-initialized `uint32_t` spinlock in shared or device memory.

BlueFlame submission waits for async consumption of the scratch source before each posting call returns; one thread can
therefore reuse one entry for sequential operations across different QPs without locking. There is no internal shared
memory allocation or fixed connection-slot cap in MSCCL++.

## Limitations

- Host-side `Connection` operations (`write`, `updateAndSync`, `accumulate`, `flush`) and `Host2DeviceSemaphore::signal`
  are not supported on GPUNetIO connections and throw `InvalidUsage`.
- On RoCE devices only port 1 is supported (the pinned DOCA GPUNetIO resolves the destination MAC on port 1).
- `accumulate` requires an 8-byte-aligned destination offset.
