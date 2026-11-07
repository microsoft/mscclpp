# GPU-Initiated PortChannel with GPUNetIO

By default, `PortChannel` operations over InfiniBand are pushed by GPU threads into a FIFO and posted to the NIC by the
host proxy thread of a `ProxyService`. With the optional GPUNetIO backend, GPU threads build the RDMA work requests and
ring the NIC doorbell themselves through [DOCA GPUNetIO](https://github.com/NVIDIA-DOCA/gpunetio) (GDAKI), removing
the proxy from the data path. The host setup flow and `PortChannel` posting methods are unchanged; kernels provide
shared-memory WQE scratch as shown below. The backend is selected per endpoint with `EndpointConfig::Ib::Mode::Device`.

## Build

The backend is built automatically for CUDA builds with InfiniBand support (`MSCCLPP_USE_IB=ON`, the default); no
extra CMake option is needed. ROCm builds and builds with `MSCCLPP_USE_IB=OFF` skip it. At configure time, CMake
fetches the GPUNetIO sources (see `cmake/GpuNetIo.cmake`) and compiles them into `libmscclpp`.
To build without network access, point CMake to a local checkout of the same revision:

```bash
cmake -DFETCHCONTENT_SOURCE_DIR_GPUNETIO=/path/to/gpunetio ..
```

The GPUNetIO license is installed to `share/licenses/mscclpp/gpunetio`.

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

auto semaphore = std::make_shared<mscclpp::PortDevice2DeviceSemaphore>(*communicator, connection);
mscclpp::PortChannel channel(semaphore, remoteMem, localMem);
auto deviceHandle = channel.deviceHandle<mscclpp::PortChannelBackend::Device>();
```

Exclusive-mode operations automatically use internal shared-memory WQE slots for BlueFlame submission when the
transfer fits `device_.blueFlameLimit` (default: 512 KiB) and the calling thread has a slot:

```cpp
__global__ void send(mscclpp::PortChannelDeviceHandleT<mscclpp::PortChannelBackend::Device> channel, size_t bytes) {
  if (threadIdx.x != 0) return;
  channel.putWithSignal<mscclpp::PortChannelSharingMode::Exclusive>(0, bytes);
  channel.flush<mscclpp::PortChannelSharingMode::Exclusive>();
}

// One block exclusively owns this QP.
send<<<1, 32>>>(deviceHandle, bytes);
```

Handles do not store shared-memory pointers and can be loaded directly from constant or global memory. Operations
without an available slot, or transfers above `blueFlameLimit`, use regular doorbell submission without WQE staging.

In the kernel, `put`, `signal`, `putWithSignal`, `putWithSignalAndFlush`, `flush`, `accumulate`, `wait` and `poll`
behave as for proxy-backed channels. `flush` waits for the local completion of all work requests posted on the queue
pair.

`signal` and `accumulate` use an RDMA atomic fetch-and-add, which allows any number of writers to the same word.
Device-mode channels also provide `putWithAccumulate`, a put followed by an accumulate, posted together;
`putWithSignal` is `putWithAccumulate` on the semaphore. The responder applies an atomic only after placing the data
of earlier RDMA writes on the same queue pair, so observing the updated word guarantees that those writes are visible.
A plain RDMA write gives no such guarantee, even with strong ordering, because packets may be placed out of order at
the responder.

Only `port` and `gidIndex` of `EndpointConfig::Ib` are used in this mode; queue sizes are chosen by the backend.
Small `put` and `putWithSignal` payloads are embedded in the work request rather than read separately by the NIC.

Callers must make source data ready before `put` or `putWithSignal`, synchronizing with its producers as needed.
These operations rely on DOCA's queue-publication release ordering rather than adding a full system fence for
every submission. Keep non-inline source buffers unchanged until their transfers complete.

### Queue pair sharing modes and BlueFlame configuration

By default, queue pairs are accessed in `PortChannelSharingMode::Gpu` (safe when threads across the GPU share a QP).
Callers may also pass `PortChannelSharingMode::Cta` (shared within a CTA) or `PortChannelSharingMode::Exclusive`
(operations on a QP are serialized by the caller) as a template parameter to posting and `flush` methods.

`device_.blueFlameLimit` controls the maximum payload size (in bytes) submitted through BlueFlame (default:
`kDefaultBlueFlameLimit = 512 * 1024`; set to `0` to disable BlueFlame or adjust it to change which transfer sizes use
BlueFlame). Payloads above this limit write their WQEs directly to the GPU-resident send queue and ring the regular
doorbell, matching NCCL GIN's `submit_db` path.
Only `PortChannelSharingMode::Exclusive` operations submit through BlueFlame. The NIC executes a BlueFlame work
request directly only if all earlier ones were already submitted, which threads sharing a queue pair cannot ensure
without serializing each other, so `Cta` and `Gpu` operations always ring the regular doorbell.

### Internal WQE slots

`PortChannelDeviceHandleT<Backend, MaxWqeSlots>` and `BasePortChannelDeviceHandleT<Backend, MaxWqeSlots>` configure the
internal capacity at compile time. The default is 32 slots (2 KiB per CTA); use
`channel.deviceHandle<mscclpp::PortChannelBackend::Device, 8>()` for eight slots, or a capacity of zero to disable
BlueFlame without allocating scratch. The host backend ignores this parameter.

The default slot index is the linear thread index, including all three block dimensions. Posting methods accept an
optional `wqeSlotIdx` (`-1` selects the default); out-of-range indices fall back to regular doorbells. For example:

```cpp
using Mode = mscclpp::PortChannelSharingMode;
channel.put<Mode::Exclusive>(dstOffset, srcOffset, bytes, slotIdx);
channel.signal<Mode::Exclusive>(slotIdx);
```

For `put` and `putWithSignal`, specify both source and destination offsets when passing an explicit index, avoiding
ambiguity with the existing two-argument overloads. For `putWithSignalAndFlush`, the index follows `maxSpinCount`.

Slots are 64 bytes, 16-byte aligned, and shared across all handles with the same capacity in a CTA. Concurrent posting
threads must use distinct indices, even when they own different QPs. BlueFlame waits for async source consumption
before returning, so one thread can reuse a slot sequentially across QPs without an extra barrier or flush.
Prefer one capacity per kernel: different capacities instantiate separate scratch arrays. Scratch is statically
allocated for compiled BlueFlame paths, so runtime doorbell fallback does not reduce its footprint. Large capacities
must respect the kernel's shared-memory limits; `Cta` and `Gpu` operations do not use this scratch.

## Limitations

- CQ initialization has known upstream error-handling issues when its CUDA copy fails; see
  [NVIDIA-DOCA/gpunetio#23](https://github.com/NVIDIA-DOCA/gpunetio/issues/23). No local workaround is applied.
- Host-side `Connection` operations (`write`, `updateAndSync`, `accumulate`, `flush`) and `Host2DeviceSemaphore::signal`
  are not supported on GPUNetIO connections and throw `InvalidUsage`.
- On RoCE devices only port 1 is supported (the pinned DOCA GPUNetIO resolves the destination MAC on port 1).
- `accumulate` requires an 8-byte-aligned destination offset.
