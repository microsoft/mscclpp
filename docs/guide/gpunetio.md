# Device-backed PortChannel

`PortChannel` has two execution modes. Host channels send work through a `ProxyService`; Device channels post
work directly from the GPU and do not use a proxy. Host ownership types remain non-template.
Use `deviceHandle()` for a Host GPU view and `deviceBackendHandle()` for a Device GPU view.
The views have different types, so kernels select the backend at compilation, not during GPU operations.

The current Device implementation uses [DOCA GPUNetIO](https://github.com/NVIDIA-DOCA/gpunetio). Its queue pointers,
registration keys, and descriptors are implementation details, not channel API fields.

## Build

CUDA builds with InfiniBand support (`MSCCLPP_USE_IB=ON`, the default) build the Device implementation automatically.
ROCm and IB-off builds do not. CMake fetches the pinned GPUNetIO sources; for offline configuration, supply a local
checkout of the same revision:

```bash
cmake -DFETCHCONTENT_SOURCE_DIR_GPUNETIO=/path/to/gpunetio ..
```

The installation includes the required inline implementation and dependency headers. The installed
`mscclpp/config.hpp` records library capability, so standalone consuming kernels do not need `USE_IBVERBS` or
GPUNetIO feature definitions. Compile consuming kernels as C++20 with the matching installed headers and library.
Rebuild C++/Python consumers together after changing versions; handle ABI is not frozen.

The GPUNetIO license is installed to `share/licenses/mscclpp/gpunetio`.

## Requirements

- A Hopper or newer NVIDIA GPU and an RDMA NIC with a BlueFlame UAR
  (`DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF`). Device kernels must target `sm_90` or newer.
- RDMA atomics on both sides. A Device endpoint cannot connect to a `HostNoAtomic` endpoint.
- A local GPU endpoint (`EndpointConfig::device.type == DeviceType::GPU`).

Unsupported endpoint creation throws `InvalidUsage`. Device operations on an unavailable build or architecture
and operations on unassigned views fail explicitly with a device error, including in release builds. Host-incompatible
concurrency modes fail at compilation. Device execution never silently falls back to Host execution.

## Usage

Select Device mode per endpoint, exchange registered memory, and construct from a raw semaphore:

```cpp
mscclpp::EndpointConfig cfg;
cfg.transport = mscclpp::Transport::IB0;
cfg.ib.mode = mscclpp::EndpointConfig::Ib::Mode::Device;
auto connection = comm.connect(cfg, remoteRank).get();
auto sema = comm.buildSemaphore(connection, remoteRank).get();

auto localRegMem = comm.registerMemory(buff, bytes, cfg.transport);
comm.sendMemory(localRegMem, remoteRank);
auto remoteRegMem = comm.recvMemory(remoteRank).get();
mscclpp::PortChannel channel(sema, remoteRegMem, localRegMem);
auto handle = channel.deviceBackendHandle();
```

No `ProxyService` is constructed, started, or stopped. Share an algorithm source by templating the kernel:

```cpp
template <class Handle>
__global__ void send(Handle channel, size_t bytes) {
  if (threadIdx.x != 0) return;
  channel.putWithSignal(0, bytes);
  channel.flush();
}
// Launch the Device instantiation. A Host channel instead uses channel.deviceHandle().
send<<<1, 1>>>(handle, bytes);
```

Host channels retain their [existing ProxyService IDs, factories and lifecycle](../tutorials/04-port-channel.md).
`PortChannelDeviceHandle` and `BasePortChannelDeviceHandle` remain the Host aliases. Their Device counterparts are
`PortChannelDeviceHandleT<PortChannelBackend::Device>` and `BasePortChannelDeviceHandleT<PortChannelBackend::Device>`.
Backend is the only handle template argument. The nested `DeviceHandle` aliases and generic
`mscclpp::deviceHandle(channel)` helper remain Host-only. A mismatched extraction throws `InvalidUsage` on the CPU.

Select the extraction and kernel instantiation on the host. Store different backends in separate typed GPU arrays;
do not reinterpret raw handle bytes as another backend. `BasePortChannel` memory-ID posting is Host-only. Device
applications bind `RegisteredMemory` through `PortChannel`, not public descriptors. `putWithAccumulate` is available
only on the bound Device view. These capability differences are checked at compilation.

Python exposes `device_backend_handle()` and distinct `CppDeviceBasePortChannelDeviceHandle` and
`CppDevicePortChannelDeviceHandle` raw views. Existing `device_handle()` and Host properties remain Host-only.
Pass raw bytes only to a kernel compiled for that view type. ProxyService-based Python helpers and the execution-plan
executor continue to use Host views; they do not automatically select Device execution.

Default Host bound views have no initialized memory IDs. Assigning only their base view does not initialize bounds.
Bound constructors and extraction accept registry ID zero. Python `src_` or `dst_` assignment initializes Host bounds,
including zero IDs; reading unset bounds raises `InvalidUsage` without initializing them.

The proxy-free raw constructors reject Host connections. Host constructors reject Device connections or null
proxies, and constructors taking explicit owner pointers reject null owners. Device constructors also reject
uninitialized registered regions. Extraction rejects unassigned channels and invalid connection state. Extracting a Host view from a Device channel or a Device view from a Host channel
throws `InvalidUsage`. Device construction/extraction validates that the remote signal token is nonzero and
8-byte aligned, resolving current registration metadata rather than caching it in a semaphore handle.

Raw construction creates an internal semaphore owner, like `MemoryChannel` raw construction. Copies share its
counter and lifetime; separate constructions from the same raw semaphore create separate logical expected counters.
Extraction does not allocate or reset a counter. Keep a channel copy and the application buffers alive during GPU
use: registered memory retains metadata, not buffer ownership. Advanced callers can supply a shared
`Device2DeviceSemaphore` owner when they intentionally share progress across channels. Its expected counter is
allocated on the endpoint GPU while restoring the caller's current device.

`Device2DeviceSemaphore::deviceHandle()` remains Memory-facing and rejects Device connections. Port signaling is
exposed through the channel, not a second public semaphore/transport descriptor API. The deprecated C++ names
`MemoryDevice2DeviceSemaphore` and `SmDevice2DeviceSemaphore` alias the actual `Device2DeviceSemaphore` class.
Python keeps its existing `MemoryDevice2DeviceSemaphore` name.

## Ordering and completion

`put`, `signal`, `putWithSignal`, `putWithSignalAndFlush`, `flush`, `accumulate`, `wait` and `poll` retain their
channel meanings. Device `flush` waits for local completion of work posted so far on the queue pair.

Device `signal` and `accumulate` use RDMA atomic fetch-and-add. `putWithAccumulate` posts a put followed by an
atomic add; `putWithSignal` uses the signal token as that add's target. The responder applies the atomic after
placing earlier writes on the same queue pair, so observing it publishes those writes. A plain RDMA write alone
provides no such visibility guarantee. Host views do not provide `putWithAccumulate`.

Make source data ready before posting, synchronizing with its producers as needed. Queue publication uses DOCA's
release ordering rather than adding a full system fence per submission. Keep non-inline source buffers unchanged
until transfers complete. Small payloads can be embedded in the work request rather than DMA-read separately.

## Concurrency and scratch capacity

Use `PortChannelDeviceHandle::Concurrency` as the first operation template argument:

- `Device` (default): concurrent callers may span the GPU. Supported by both backends.
- `Block`: Device callers are confined to one CTA.
- `Exclusive`: Device callers serialize access to the queue pair.

Host views reject `Block` and `Exclusive` at compilation. Choose the default for a kernel source shared by both
backends, or select a concurrency template argument in a backend-specific kernel instantiation.

The second template argument selects internal shared-memory slots per CTA, not a different handle type:

```cpp
using C = mscclpp::PortChannelDeviceHandle::Concurrency;
channel.put<C::Exclusive, 8>(dstOffset, srcOffset, bytes, slotIdx);
channel.signal<C::Exclusive, 8>(slotIdx);
channel.flush<C::Exclusive>();
```

The default is 32 slots (2 KiB); zero disables scratch submission without allocating scratch. The maximum is 1024;
larger capacities produce a compile-time error. Host operations do not use these slots. Handles contain no
shared-memory pointers and can be stored in constant/global memory.

The default `slotIdx=-1` is the linear CTA thread index, including all block dimensions. Out-of-range indices use
regular doorbells. Pass both offsets to `put`/`putWithSignal` when specifying a slot; for `putWithSignalAndFlush`,
the slot follows `maxSpinCount`. Slots are 64 bytes, 16-byte aligned, and shared across handles using the same
capacity in a CTA. Concurrent posting threads must use distinct indices, including when using different QPs.
Submission waits for async source consumption, so one thread can reuse a slot sequentially across QPs.

Prefer one capacity per kernel: different capacities instantiate separate arrays. Scratch is statically allocated
for compiled scratch paths; runtime fallback does not remove its footprint. Large capacities must respect kernel
shared-memory limits. `Block` and `Device` operations use regular doorbells, not scratch.

## Endpoint-specific tuning

Only Exclusive operations can submit through BlueFlame. Concurrent callers cannot ensure all earlier WQEs were
submitted before a direct BlueFlame WQE, so Block/Device calls always use regular doorbells.

Configure the payload threshold before connecting:

```cpp
cfg.ib.gpuNetIo.blueFlameLimit = 512 * 1024; // Default: 512 KiB.
// cfg.ib.gpuNetIo.blueFlameLimit = 0;      // Disable BlueFlame for this endpoint.
```

This policy is private in extracted handles and is not mutable at the channel/kernel API. Larger transfers use
regular doorbells. A zero threshold disables BlueFlame but does not remove scratch compiled into an Exclusive
kernel; use a scratch capacity of zero when that footprint must disappear. Python exposes this endpoint setting as
`cfg.ib.gpunetio.blue_flame_limit`.

The implementation also uses `port` and `gidIndex`; queue sizes are chosen internally. Host/HostNoAtomic ignore
`gpuNetIo` settings.

## Limitations

- CQ initialization has known upstream error handling issues on CUDA-copy failure; see
  [NVIDIA-DOCA/gpunetio#23](https://github.com/NVIDIA-DOCA/gpunetio/issues/23). No local workaround is applied.
- Host-side `Connection` operations (`write`, `updateAndSync`, `accumulate`, `flush`) and
  `Host2DeviceSemaphore::signal` are unsupported on Device connections and throw `InvalidUsage`.
- On RoCE devices only port 1 is supported by the pinned implementation's destination-MAC resolution.
- `accumulate` requires an 8-byte-aligned destination offset.
