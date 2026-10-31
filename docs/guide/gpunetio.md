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
(default: 4096 bytes), bind caller-owned shared-memory WQE slot storage on a **local copy** of the handle:

```cpp
__global__ void send(mscclpp::PortChannelDeviceHandleT<mscclpp::PortChannelBackend::Device> channel, size_t bytes) {
  __shared__ mscclpp::GpuNetIoWqeSlot wqeSlot;
  if (threadIdx.x != 0) return;
  auto portChan = channel;
  portChan.setWqeSlot<mscclpp::PortChannelSharingMode::Exclusive>(&wqeSlot);
  portChan.putWithSignal<mscclpp::PortChannelSharingMode::Exclusive>(0, bytes);
  portChan.flush<mscclpp::PortChannelSharingMode::Exclusive>();
}

// One block exclusively owns this QP.
send<<<1, 32>>>(deviceHandle, bytes);
```

Never bind a block-shared pointer on a constant-memory or globally shared channel handle. When no WQE slot is bound
or a transfer exceeds `blueFlameLimit`, work requests are submitted via regular doorbell without shared-memory
staging.

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

### EP NIC selection and queue pairs

Cross-domain EP uses `MSCCLPP_HCA_DEVICES` to map each GPU to its IB transport (`IB0`–`IB7`), or discovers IB
devices via `ibverbs` when `MSCCLPP_HCA_DEVICES` is unset. If a rank's process sets `MSCCLPP_HCA_DEVICES` to a
single HCA name (for example `export MSCCLPP_HCA_DEVICES=mlx5_0`), that rank uses `IB0`.

For latency `RANK_MAJOR` EP, `MSCCLPP_EP_GPUNETIO_QPS_PER_PEER` selects the number of QPs created per peer (1 or 4,
default 1). Hybrid dispatch and aggregate combine distribute chunks across all configured QPs; direct combine uses one
payload QP per peer.

### Hybrid dispatch

Cross-domain EP automatically uses the send-and-distribute hybrid BF16 `RANK_MAJOR` dispatch with
`RANK_LOCAL_REDUCE` when ranks span two CUDA IPC domains. It requires exactly two equal, fully IPC-mapped NVLink
domains and one or four QPs per peer. Initialization collectively checks the actual IPC mapping matrix. Sorted members
of the two domains are paired: each source sends to its corresponding gateway in the other domain, not to one central
gateway. Incompatible domain mappings or QP counts fail initialization. Other layouts and IPC-only paths retain their
existing behavior.

Each input token with any cross-domain destination is sent once, carrying fresh top-k IDs, weights, and the
source-allocated compact destination slots. IB chunks are striped round-robin over the gateway QPs.
`MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS` controls their size (1–128 rows, default eight).
Each QP has one independent posting warp that preserves chunk order. QP 0's posting thread also sends the forwarding
ACK after all chunk posters finish, so the hybrid dispatch requires no QP locks.
A self-validating LL header carries fresh destination counts, which the gateway publishes locally before completion.
The row count and every destination count each carry their own epoch in an `LL8Packet`; only the active
`(1 + world_size) * sizeof(LL8Packet)` bytes are sent by one plain RDMA write. Every consumed word validates its own
epoch, so no separate header-ready signal is needed under the existing LL eight-byte IB/RDMA atomicity contract.
Header arrival does **not** establish payload readiness: every payload chunk still has its strong-ordered signal,
including on relaxed-ordering registered memory.
Dedicated gateway CTAs and completed producer CTAs forward arrived rows over NVLink while later chunks are still
transferring. Reused producer CTAs retain two independent warp/TMA row slots and join the same completion counter
as dedicated forwarders. The specialization retains its shared-memory allocation and full-grid residency check.
Row assignment and completion counts use the active workload, configured chunk/QP counts, and launch grid;
there is no special-case token count, hidden dimension, balanced routing pattern, or fixed number of chunks.
Separate send and gateway landing regions in the runtime's symmetric registered allocation are reserved at
runtime construction. Chunk flags are cumulative; consumers use dedicated per-original-source readiness epochs,
and an explicit gateway acknowledgment protects staging reuse even when no route targets the gateway itself.
After all forwarding writers join, the last forwarder release-publishes local completion and publishes local
readiness independently. QP 0's posting thread acquires forwarding completion, sends the acknowledgment, and only
then waits for the peer's acknowledgment. All outgoing QPs are drained independently before dispatch returns.
Empty rounds participate in the same protocol.

By default, Combine returns each rank's preweighted BF16 partials directly to the source.
Without the aggregation option below, set `MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS=0` (the default).
It requires more than `world_size + 9` dispatch blocks, with the full grid resident as checked by the launcher. Two stream-ordered
workspace resets are part of each dispatch without aggregate Combine: include them in end-to-end CUDA graph timing.
Row writers use device fences, and each QP poster acquires the completed chunk's release sequence before a system
fence and the NIC post. Headers larger than the inline-transfer limit retain system publication;
inline headers are copied into the WQE after their writers join. Forwarding writers retain system fences before their
completion join, covering ordinary NVLink metadata stores as well as completed TMA payloads. The joined writers already establish visibility for
the inline acknowledgment, so its posting thread does not repeat the system fence. Standalone LL count packets
validate their own epochs; nonempty payloads use a separate ready signal.

### Hybrid combine

`MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE=1` additionally enables reverse-gateway aggregation. It defaults off and
requires BF16 `RANK_MAJOR`/`RANK_LOCAL_REDUCE`, `topk <= 8`, and the same validated two-domain,
one-or-four-QP topology. `MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS` selects 1–128 rows per chunk (default eight when
aggregation is enabled). Incompatible rank-major requests fail collective initialization; other layouts and
IPC-only runtimes retain their existing paths.

The paired gateway reads the local-domain **BF16 preweighted rank partials**, sums them in FP32, and returns
compact FP32 rows in independently posted, ordered per-QP IB chunks. Gateway production prioritizes the first
chunk on each QP before later chunks, and completes before independent local-domain reads begin. Those local
FP32 reductions overlap the posted IB transfers. The final result adds the two FP32 domain partials and casts to
BF16 once; there is no intermediate BF16 group rounding. **FP32 addition grouping changes**, so this mode is numerically distinct from direct Combine
and must be explicitly opted into.

Dispatch and Combine assign one submitter to each outgoing QP, so they use the existing PortChannel GPUNetIO
operations without EP locks. The EP device context only indexes channels and notification state.
It does not implement a separate put, signal, batching, or flush API.

Gateway row writers join at a CTA barrier before thread zero updates the device-scoped release counter, so they do
not need an additional per-writer fence. Each QP poster acquires its completed chunk and retains the cumulative
system fence before NIC DMA. DOCA's GPU-scoped queue-publication release alone is not used as a substitute for
publishing source rows written by other CTAs.

Dispatch records a private token-to-compact-row map and caches incoming IDs, destination slots, and row counts on
the gateway. These caches cannot be overwritten by another rank's next dispatch. FP32 send and receive buffers
are separate regions in the runtime's symmetric registered allocation; the local FP32 accumulator is private. All workers finish
their assigned gateway production before waiting for returned rows. Local start/end handshakes and outgoing
QP completion drains are retained. The end handshake starts only after every local worker finishes all NVLink
reads, overlapping the remaining FP32 IB returns; the final grid join still waits for every result consumer,
posting warp, QP drain, and end handshake before resetting workspace. The gateway's next Dispatch must receive the owner's next-Dispatch ACK before
its next Combine can overwrite that owner's FP32 landing, including empty rounds. This relies on the existing
stream-ordered Dispatch/Combine sequence.
There must be **one fresh Dispatch before every aggregate Combine**, including CUDA graph replays.
Repeated eager use of a consumed dispatch handle is already rejected by the host runtime. A device-side epoch
guard rejects aggregate Combine-only graph replays without a new Dispatch; this reports a diagnostic and traps
the CUDA context, rather than allowing unsafe landing-buffer reuse.

One fused `hybridAggregateCombineKernel` performs all reduction, posting, and finalization, using the requested
Combine grid and its full-residency check. After the completion join and local end handshakes, it resets the private
Dispatch/Combine counters and readiness epochs for the next round, eliminating the four separate workspace resets
in a paired graph. Network counters, arrival baselines, and replay validation epochs remain persistent.

Both hybrid algorithms reuse `LatencyRuntimeContext`, including its symmetric allocation, workspace, IPC channels,
and single device-context copy. The context also owns the network service and private pipeline state, and waits
for outstanding device work before releasing the service and registered memory.

### Queue pair sharing modes and BlueFlame configuration

By default, queue pairs are accessed in `PortChannelSharingMode::Gpu` (safe when threads across the GPU share a QP).
Callers may also pass `PortChannelSharingMode::Cta` (shared within a CTA) or `PortChannelSharingMode::Exclusive`
(operations on a QP are serialized by the caller) as a template parameter to posting and `flush` methods.

`device_.blueFlameLimit` controls the maximum payload size (in bytes) submitted through BlueFlame when a WQE slot
is bound via `setWqeSlot` (default: `kDefaultBlueFlameLimit = 4096`; set to `0` to disable BlueFlame or increase
to use BlueFlame for larger transfers). Payloads above `blueFlameLimit`—or any request when no WQE slot is bound—write
their WQEs directly to the GPU-resident send queue and ring the regular doorbell, matching NCCL GIN's `submit_db` path.

### Caller-owned WQE slot and optional lock

`mscclpp::GpuNetIoWqeSlot` is an opaque 64-byte, 16-byte-aligned shared-memory buffer for one WQE. Call
`setWqeSlot<Mode>(&wqeSlot, &wqeSlotLock, reuseSharedWqe)` on a local copy of the handle:
- In `PortChannelSharingMode::Exclusive` with `reuseSharedWqe == false`, each slot is owned exclusively by one posting
  thread at a time without locking (`wqeSlotLock` may be omitted).
- In `Cta`/`Gpu` sharing modes or when `reuseSharedWqe == true`, pass a zero-initialized `uint32_t` spinlock in shared
  or device memory (`&wqeSlotLock`); if `wqeSlotLock` is null in those modes, `setWqeSlot` leaves BlueFlame disabled and
  falls back to regular doorbell submission.

BlueFlame submission waits for async consumption of the scratch source before each posting call returns; one thread can
therefore reuse one entry for sequential operations across different QPs without locking. There is no internal shared
memory allocation or fixed connection-slot cap in MSCCL++.

## Limitations

- Host-side `Connection` operations (`write`, `updateAndSync`, `accumulate`, `flush`) and `Host2DeviceSemaphore::signal`
  are not supported on GPUNetIO connections and throw `InvalidUsage`.
- On RoCE devices only port 1 is supported (the pinned DOCA GPUNetIO resolves the destination MAC on port 1).
- `accumulate` requires an 8-byte-aligned destination offset.
