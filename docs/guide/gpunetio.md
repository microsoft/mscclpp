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
GPUNetIO sources (pinned to v5.0.0-rc1, see `cmake/GpuNetIo.cmake`) and compiles them into `libmscclpp`. To build
without network access, point CMake to a local checkout of the same revision:

```bash
cmake -DMSCCLPP_USE_GPUNETIO=ON -DFETCHCONTENT_SOURCE_DIR_GPUNETIO=/path/to/gpunetio ..
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

### Hybrid EP queue pairs

For latency `RANK_MAJOR` EP, `MSCCLPP_EP_GPUNETIO_QPS_PER_PEER` selects the number of QPs created per peer. It defaults
to the number of selected HCAs and must be a multiple of that count. QP `q` uses HCA `q % numHcas`, so multiple QPs
do not change the GPU-to-NIC mapping.

Batched dispatch distributes its chunks over all configured QPs by default.
`MSCCLPP_EP_GPUNETIO_BATCH_ALL_QPS=0` retains dispatch striping by HCA count instead.
Unchunked combine stripes peer payloads by selected HCA count and waits for those stripes before consuming their rows.
With one selected HCA per GPU, it uses one payload QP per peer even when multiple QPs are configured.
Chunked combine distributes chunks over the configured QPs.

`MSCCLPP_EP_GPUNETIO_ASYNC_DISPATCH=1` opts into dedicated receive CTAs and per-peer completion publication instead
of waiting for the global producer join. It requires batched dispatch, a positive dispatch chunk size, and at least
`world_size + 2` dispatch blocks. This path drains its sends before returning, so it does not use deferred send flush.
It also enqueues three workspace resets before each dispatch; include those operations when comparing complete
CUDA graph latency, not just dispatch kernel time. The default remains disabled.

### Optional send-and-distribute dispatch

`MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE=1` selects an experimental BF16 `RANK_MAJOR` dispatch with
`RANK_LOCAL_REDUCE`. It requires exactly two equal, fully IPC-mapped NVLink domains and one or four QPs per peer.
Initialization collectively checks the actual IPC mapping matrix. Sorted members of the two domains are paired:
each source sends to its corresponding gateway in the other domain, not to one central gateway.
Incompatible domain mappings or QP counts fail initialization. Other layouts and IPC-only paths retain their
existing behavior; hybrid initialization logs a fallback for unsupported layouts.

Each input token with any cross-domain destination is sent once, carrying fresh top-k IDs, weights, and the
source-allocated compact destination slots. IB chunks are striped round-robin over the gateway QPs.
`MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS` controls their size (1–128 rows, default eight).
Each QP has an independent posting warp that preserves chunk order while holding one posting-phase queue lease.
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
Separate registered send and gateway landing allocations are reserved at
runtime construction. Chunk flags are cumulative; consumers use dedicated per-original-source readiness epochs,
and an explicit gateway acknowledgment protects staging reuse even when no route targets the gateway itself.
After all forwarding writers join, separate warps publish the acknowledgment and local readiness concurrently.
Every posting lease ends before waiting for the acknowledgment; all outgoing QPs are then drained independently
before dispatch returns. Empty rounds participate in the same protocol.

This specialization overrides the direct dispatch batching/async/flush tuning choices; by default it does not change
Combine. Without the aggregation option below, set `MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS=0` (the default). It requires more than
`world_size + 9` dispatch blocks, with the full grid resident as checked by the launcher. Two stream-ordered
workspace resets are part of each dispatch without aggregate Combine: include them in end-to-end CUDA graph timing.
`MSCCLPP_EP_GPUNETIO_ROW_FENCE` defaults to `gpu+poster`: row writers use device fences, and each QP poster
acquires the completed chunk's release sequence before a system fence and the NIC post. `sys` retains system
fences on every row writer. Unlike the direct path, the poster's system fence is unconditional, including when
`gpu` is selected. Header, forwarding-completion, and acknowledgment publication remains system-scoped.
`MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE_STATS=1` prints actual unique rows, direct-path remote rows, hidden payload
bytes, row wire bytes (including routing metadata and alignment), and header bytes per rank/round for validation.
These byte counts exclude transport framing and completion signals. Disable statistics for performance measurements.
Payload reduction alone does not establish a latency improvement; compare measured complete dispatch/combine
rounds on the same routing workload and topology.

### Optional FP32 gateway aggregation for Combine

`MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE=1` additionally enables reverse-gateway aggregation. It defaults off and
requires send-and-distribute, BF16 `RANK_MAJOR`/`RANK_LOCAL_REDUCE`, `topk <= 8`, and the same validated two-domain,
one-or-four-QP topology. `MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS` selects 1–128 rows per chunk (default eight when
aggregation is enabled). Incompatible rank-major requests fail collective initialization; other layouts and
IPC-only runtimes retain their existing paths.

The paired gateway reads the local-domain **BF16 preweighted rank partials**, sums them in FP32, and returns
compact FP32 rows in independently posted, ordered per-QP IB chunks. Gateway production prioritizes the first
chunk on each QP before later chunks, and completes before independent local-domain reads begin. Those local
FP32 reductions overlap the posted IB transfers. The final result adds the two FP32 domain partials and casts to BF16 once; there is no intermediate
BF16 group rounding. **FP32 addition grouping changes**, so this mode is numerically distinct from direct Combine
and must be explicitly opted into.

Dispatch records a private token-to-compact-row map and caches incoming IDs, destination slots, and row counts on
the gateway. These caches cannot be overwritten by another rank's next dispatch. FP32 send and receive buffers
are separate runtime-owned, registered allocations; the local FP32 accumulator is private. All workers finish
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

One fused `rankMajorAggregateCombineKernel` performs all reduction, posting, and finalization, using the requested
Combine grid and its full-residency check. After the completion join and local end handshakes, it resets the private
Dispatch/Combine counters and readiness epochs for the next round, eliminating the four separate workspace resets
in a paired graph. Network counters, arrival baselines, and replay validation epochs remain persistent.
With send-and-distribute statistics enabled, `[EP-AGGREGATE-COMBINE]` reports actual returned FP32
rows and payload bytes. No latency gain is implied by the reduction in transferred rows.

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
