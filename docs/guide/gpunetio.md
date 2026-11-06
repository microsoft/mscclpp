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

auto semaphore = std::make_shared<mscclpp::PortDevice2DeviceSemaphore>(*communicator, connection);
mscclpp::PortChannel channel(semaphore, remoteMem, localMem);
auto deviceHandle = channel.deviceHandle<mscclpp::PortChannelBackend::Device>();
```

Host-created device handles use regular doorbell submission (`submit_db`, preparing WQEs in GPU global memory) by
default without requiring shared memory. To enable BlueFlame submission for transfers up to `device_.blueFlameLimit`
(default: 512 KiB), bind caller-owned shared-memory WQE slot storage on a **local copy** of the handle:

```cpp
__global__ void send(mscclpp::PortChannelDeviceHandleT<mscclpp::PortChannelBackend::Device> channel, size_t bytes) {
  __shared__ mscclpp::GpuNetIoWqeSlot wqeSlot;
  if (threadIdx.x != 0) return;
  auto portChan = channel;
  portChan.setWqeSlot(&wqeSlot);
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

### EP NIC selection and queue pairs

Cross-domain EP uses `MSCCLPP_HCA_DEVICES` to map each GPU to its IB transport (`IB0`–`IB7`), or discovers IB
devices via `ibverbs` when `MSCCLPP_HCA_DEVICES` is unset. If a rank's process sets `MSCCLPP_HCA_DEVICES` to a
single HCA name (for example `export MSCCLPP_HCA_DEVICES=mlx5_0`), that rank uses `IB0`.

For latency `RANK_MAJOR` EP, `qpsPerPeer` (`qps_per_peer` in `MoECommunicatorConfig`) selects the number of QPs created
per peer (1 or 4, default 4). Hybrid dispatch and combine stripe their chunks round-robin across all configured QPs.

### Hybrid dispatch

Latency `RANK_MAJOR` EP automatically uses the hybrid algorithm when ranks span CUDA IPC domains. It requires exactly
two equal, fully IPC-mapped domains, 1 or 4 QPs per peer, `topk <= 8`, and BF16 or `FP8_E4M3_STATIC` dispatch;
initialization or dispatch fails otherwise. Rank `r` is paired with gateway rank `(r + domainSize) % worldSize` in
the other domain, so every rank both sends to and forwards for exactly one remote peer. `nTokensPerChunk`
(`n_tokens_per_chunk` in `MoECommunicatorConfig`) sets the tokens per IB chunk (1–128, default eight).

The dispatch grid needs at least 10 resident blocks, and the launcher checks full-grid residency:

- **Control block (block 0).** Counts tokens per destination rank and the number of remote tokens, writes a
  self-validating header (`nRemoteTokens` and per-rank counts, each an `LL8Packet` carrying the round epoch), and
  publishes NVLink counts and payload-ready signals for local destinations. One posting warp per QP (lane 0,
  `PortChannelSharingMode::Exclusive`, no locks) sends chunk `c` on QP `c % nQps`. QP 0 first sends the active
  `(1 + worldSize) * sizeof(LL8Packet)` header bytes with one plain RDMA write, then its chunks, then the
  buffer-reuse acknowledgment described below.
- **Producer blocks (blocks 9 and up).** Each of the two TMA warps per block handles one input token. It allocates
  destination slots, stores the token directly to NVLink-local destinations, and, if any top-k destination is in
  the other domain, copies the token once (payload, top-k IDs, weights, and destination slots) into a compact
  remote-token slot of the send buffer. It then increments that chunk's ready counter with a device-scoped release.
  Producer blocks join the forwarders when they finish.
- **Forwarder blocks (blocks 1 and up).** These read the peer's header. Forwarder 0 republishes the per-destination
  counts to the local peers. Each forwarding warp waits until the RDMA atomic for its token's chunk has incremented
  the per-QP `qpSignalCount` past the chunk's position on that QP. It then TMA-loads the token from the landing
  buffer, stores it and its metadata to local destinations over NVLink, and caches the routes for combine. The last
  forwarder resets `qpSignalCount`, sets the local forward-complete flag, and publishes the per-source
  payload-ready epoch to every local peer with a system-scoped release.
- **Receive.** Each block finalizes the counts and padding metadata for a grid-stride subset of source ranks. It
  waits on the NVLink semaphore for local sources and on the payload-ready epoch for remote sources.

Memory ordering: the RDMA atomic for a chunk is applied only after earlier writes on the same QP are placed, so
observing `qpSignalCount` publishes the chunk payload; the header publishes itself through its per-word epochs.
Send-buffer and header writers join through warp or CTA barriers and device-scoped release counters, and the
posting thread acquires those counters. The writers' data stays in GPU memory, so DOCA's GPU-scoped release
before the doorbell publishes it to the NIC; the hybrid path adds no system fences.

Buffer reuse: dispatch does not drain its QPs before returning. After its forwarders finish, each rank sends a
`bufferReuseReady` RDMA atomic to its paired rank. The next Combine control block waits for and clears the peer's
acknowledgment. This guarantees that the peer has consumed this rank's chunks and header, so the next Dispatch may
overwrite this rank's send buffer and the peer's landing buffer. Counters are per round, not cumulative: dispatch
chunk counters are reset by Combine, and Combine's chunk counters and `qpSignalCount` are reset at the start of the
next Dispatch, before its header is sent.

### Hybrid combine

Hybrid runtimes always use gateway aggregation, for both `RANK_LOCAL_REDUCE` and `DIRECT_SEND`. Each gateway
reduces, in FP32, the local-domain expert outputs for the tokens it forwarded. Inputs are the preweighted BF16 rank
partials for `RANK_LOCAL_REDUCE` and the per-route outputs, weighted on the fly, for `DIRECT_SEND`. The gateway
returns compact rows (FP32, or BF16 when the combine input is `FP8_E4M3_STATIC`) in ordered per-QP chunks. Each
owner reduces its own local-domain contributions in FP32, adds the returned row, and casts to BF16 once.
**FP32 addition grouping differs** from single-domain combine.

- **Control block (block 0).** Exchanges start signals with local peers and publishes their ready epochs. One
  posting warp per QP sends chunk `c` once all of its gateway rows are reduced. The block also resets the dispatch
  counters, and its thread 0 waits for and clears `bufferReuseReady`.
- **Worker blocks.** Reduce the gateway rows first. A row in chunk `c` waits until chunk `c - nQps` is complete,
  which prioritizes the first chunk on each QP. Workers then wait for all gateway rows, reduce their local tokens,
  and wait for each token's returned chunk before writing the final BF16 output.

Combine requires at least `worldSize` blocks. There must be **exactly one fresh Dispatch before every Combine**,
including CUDA graph replays. No device-side guard enforces this: replaying Combine without a new Dispatch reuses
stale chunk counters and blocks in the `bufferReuseReady` wait.

The hybrid paths reuse `LatencyRuntimeContext`, including its workspace, IPC channels, and device context. Its
symmetric allocation additionally reserves the send and landing regions and the control blocks targeted by RDMA.

### Queue pair sharing modes and BlueFlame configuration

By default, queue pairs are accessed in `PortChannelSharingMode::Gpu` (safe when threads across the GPU share a QP).
Callers may also pass `PortChannelSharingMode::Cta` (shared within a CTA) or `PortChannelSharingMode::Exclusive`
(operations on a QP are serialized by the caller) as a template parameter to posting and `flush` methods.

`device_.blueFlameLimit` controls the maximum payload size (in bytes) submitted through BlueFlame when a WQE slot
is bound via `setWqeSlot` (default: `kDefaultBlueFlameLimit = 512 * 1024`; set to `0` to disable BlueFlame or adjust
it to change which transfer sizes use BlueFlame). Payloads above `blueFlameLimit`, or any request when no WQE slot is
bound, write their WQEs directly to the GPU-resident send queue and ring the regular doorbell, matching NCCL GIN's
`submit_db` path.
Only `PortChannelSharingMode::Exclusive` operations submit through BlueFlame. The NIC executes a BlueFlame work
request directly only if all earlier ones were already submitted, which threads sharing a queue pair cannot ensure
without serializing each other, so `Cta` and `Gpu` operations always ring the regular doorbell.

### Caller-owned WQE slot

`mscclpp::GpuNetIoWqeSlot` is an opaque 64-byte, 16-byte-aligned shared-memory buffer for one WQE. Bind it with
`setWqeSlot(&wqeSlot)` on a local copy of the handle. Only one thread may post through a slot at a time.
BlueFlame submission waits for async consumption of the slot before each posting call returns; one thread can
therefore reuse one slot for sequential operations across different QPs.

## Limitations

- Host-side `Connection` operations (`write`, `updateAndSync`, `accumulate`, `flush`) and `Host2DeviceSemaphore::signal`
  are not supported on GPUNetIO connections and throw `InvalidUsage`.
- On RoCE devices only port 1 is supported (the pinned DOCA GPUNetIO resolves the destination MAC on port 1).
- `accumulate` requires an 8-byte-aligned destination offset.
