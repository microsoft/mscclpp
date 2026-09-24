# Dependency: DOCA GPUNetIO (GDAKI) device verbs

The third-party GPUNetIO sources are provided by CMake FetchContent from:
https://github.com/NVIDIA-DOCA/gpunetio.

- **License:** BSD-3-Clause (NVIDIA CORPORATION & AFFILIATES).
- **Revision:** pinned by a full `GIT_TAG` commit SHA in `src/gpunetio/CMakeLists.txt`; update it explicitly and test
  both the CPU-proxy and GDAKI PortChannel backends before advancing the pin.
- **Build:** gated behind the `MSCCLPP_USE_GPUNETIO` CMake option. Host sources
  compile into the `mscclpp_gpunetio_obj` object library with
  `-DDOCA_VERBS_USE_NET_WRAPPER` (selects the dlopen ibverbs/mlx5dv wrappers).
  Device headers (`include/device/*.cuh`) are consumed only by the GPUNetIO
  PortChannel backend implementation
  (`include/mscclpp/internal/port_channel_gpunetio_device_impl.hpp`).

Configuring with `MSCCLPP_USE_GPUNETIO=ON` fetches the dependency automatically.
No Git submodule initialization is required. OFF builds do not declare, fetch,
or add GPUNetIO dependency targets.

## Main-Targeted Scope

The PortChannel implementation is derived from
`qinghuazhou/gpunetio_port_channel_merge_feature_ep` at `3d796fc`, adapted to
main's `accumulate` and proxy trigger APIs. It includes the reviewed explicit
peer/signal binding, bounded flush, atomic-result scratch, CUDA-device lifetime,
and minimum-path-MTU fixes. It does not include EP kernels, layouts, Python EP
interfaces, EP benchmarks, automatic HCA selection, or multi-HCA policy.
Each service uses one explicitly selected local HCA. It defaults to one QP per
remote rank; the explicit four-argument constructor supports 1-64 QPs per peer.

Upstream dependency: GPUNetIO **v4.0.1**, commit
`bfe3e5484f16a01ac91a906b2f0046dbc8bd61a4`. The fetched sources are unmodified;
MSCCL++ does not carry patches to NVIDIA's implementation. Its source list and
host API differ from the old snapshot, so dependency updates require new build
and hardware validation. No installed DOCA SDK is required; upstream can load
DOCA SDK libraries when available according to its own runtime policy.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMSCCLPP_USE_CUDA=ON -DMSCCLPP_USE_IB=ON -DMSCCLPP_USE_GPUNETIO=ON
cmake --build build -j
```

The option defaults to OFF. OFF builds do not require GPUNetIO sources or DOCA
headers. GDAKI host channel construction is rejected by an OFF library.
Compile each CUDA translation unit using GDAKI with `-DMSCCLPP_USE_GPUNETIO`
and `-I<build>/_deps/gpunetio-src/include` for the default FetchContent layout, or with
`-I<prefix>/include/mscclpp/gpunetio` after installation. Do not enable the
device macro globally for unrelated core kernels. Normal public headers remain
usable without any DOCA includes when the macro is absent.

The first ON configure needs Git and network access to the upstream repository.
For offline builds, provide an existing checkout at the pinned SHA using
`-DFETCHCONTENT_SOURCE_DIR_GPUNETIO=/absolute/path/to/gpunetio`.
This standard FetchContent override bypasses fetching and pin enforcement;
the operator must verify its revision and cleanliness. `FETCHCONTENT_BASE_DIR`
may also relocate the download cache. In-tree targets use
`mscclpp_gpunetio_headers` for dependency include paths, including overrides.

For existing submodule checkouts, applying the migration patch removes the
gitlink and `.gitmodules`. Git may leave the old populated directory on disk;
it is no longer used by the build. Preserve any local upstream changes before
removing that obsolete checkout. Reconfigure the build after applying the patch.

## Select a Backend

Existing `ProxyService::portChannel(...)` calls remain CPU-proxy channels.
No environment variable changes their backend. For GDAKI, initialize a service
collectively on every bootstrap rank, select an already-connected QP, build a
semaphore for it, and bind separately chosen registrations to `PortChannel`:

```cpp
#include <mscclpp/gpu_net_io_service.hpp>
#include <mscclpp/port_channel.hpp>

mscclpp::GpuNetIoService service(bootstrap, ibDeviceName, cudaDeviceId);
service.setup();
auto connection = service.connect(peerRank, 0);
auto source = service.registerMemory(sendBuffer, sendBytes, sendOwner);
auto receive = service.registerMemory(recvBuffer, recvBytes, recvOwner);
auto destination = service.exchangeMemory(connection, receive, memoryTag);
auto semaphore = service.buildSemaphore(connection, semaphoreTag);
mscclpp::PortChannel channel(semaphore, destination, source);
auto handle = channel.deviceHandle();
```

The kernel uses the existing `put`, `signal`, `putWithSignal`,
`putWithSignalAndFlush`, `accumulate`, `flush`, `poll`, and `wait` methods.
Offsets are relative to each selected registration, not a service-wide symmetric
payload. `sendOwner` and `recvOwner` are optional shared allocation owners;
without them, the caller must keep buffers alive while any handle uses them.
Peers can register different sizes and addresses. `exchangeMemory` exchanges a
local destination registration with the selected peer/QP; it is not collective.
Both peers call with matching tags and queue indices. Use distinct nonnegative
tags for simultaneous setup operations and other bootstrap traffic, and serialize
setup calls in matching order. Local registration/allocation failures before an
exchange can still leave the peer waiting; use a launcher timeout for recovery.

`buildSemaphore` allocates/zeros/registers private uint64 inbound and expected
counters, completes initialization, and exchanges signal metadata on the selected
connection. Payload and signal registrations are independent. Writes, signals,
accumulates, and flushes all use that connection's QP. `poll` and `wait` use its
semaphore counters. Multiple semaphores may share a QP, but ordering/completion
on that QP is shared; select distinct QPs for independent queues.

For explicit memory IDs, construct `BasePortChannel(semaphore, {remote, local,
otherRemote, otherLocal})`. IDs index that immutable per-channel table; they are
not ignored or interpreted as peer ranks. Every device operation validates the
ID, owner rank, and bounds before posting; atomics also require 8-byte alignment.
Host construction rejects foreign-service/foreign-peer handles and requires a
remote destination/local source for `PortChannel`. Proxy-service memory IDs and
GPUNetIO channel-table IDs are separate namespaces.

Connections validate self/out-of-range peers, queue indices and completed setup
using retained host metadata. Channels retain their semaphore, connection,
registrations and transport even if the service wrapper is destroyed. Remote
registrations also retain the local buffer exported in their exchange. Device
handles alone do not retain host resources: synchronize GPU work before releasing
the final channel and allocation owners. Producers must make payload writes
visible before issuing network operations. The legacy `setup(buffer, bytes)`
and raw context operations remain available with their symmetric-layout contract,
but ordinary channels no longer need that path or raw signal pointers.

The pinned dependency is restricted to direct `GPU_SM_DB` doorbells, valid DBRs,
and non-collapsed GPU-resident CQs. `AUTO`, CPU-proxy/free-flow handlers,
software-emulated DBRs, host CQs, and CPU UMEM are not selected. If direct GPU
doorbells are unavailable, setup fails instead of falling back. QP-creation
status is exchanged before the QP-info collective so every rank rejects an
unsupported peer. The service never starts a DOCA CPU progress thread. The
existing MSCCL++ FIFO/CPU-proxy backend remains available and unchanged.

## Validation

`gpunetio_channel_release_test` and `gpunetio_channel_debug_test` execute the
actual common device-handle routing with CPU transport stubs. They do not
validate NIC ordering or CUDA memory visibility. The two-rank hardware check is
`mpirun -np 2 build/bin/mp_unit_tests --filter=PortChannelOneToOneTest.GpuNetIoP2P`.
Run it on a configured GPU/RDMA pair with a launcher timeout, then run existing
proxy tests with the same build. A skipped setup is not a hardware pass;
asymmetric setup failures during collectives can require launcher termination.

### Multi-QP Bandwidth

`PortChannelOneToOneTest.GpuNetIoMultiQpBandwidth` is ported from
`qinghuazhou/gpunetio_port_channel_merge_feature_ep` at `3d796fc`. Build with
`MSCCLPP_USE_GPUNETIO=ON` and `MSCCLPP_BUILD_TESTS=ON`, then on two configured ranks:

```bash
export MSCCLPP_GPUNETIO_QPS_PER_PEER=4
timeout 300s mpirun -np 2 -x MSCCLPP_GPUNETIO_QPS_PER_PEER \
  build/bin/mp_unit_tests --filter=PortChannelOneToOneTest.GpuNetIoMultiQpBandwidth
```

Set the same count on both ranks. The test accepts 1-64, defaulting to 1, and
retains `MSCCLPP_EP_GPUNETIO_QPS_PER_PEER` as a fallback when the generic variable
is unset. Malformed, out-of-range, or mismatched counts fail collectively before
buffer allocation. These variables configure this test only; production service
construction is explicit and does not read EP environment variables.

The source benchmark's workload is unchanged: one GPU thread per QP sends from
rank 0 to rank 1, with 10 warmup puts and 256 timed puts per queue, at 256 B,
16 KiB, 256 KiB, 1 MiB, 4 MiB, and 8 MiB per QP. It reports aggregate GB/s and
microseconds per iteration (all QPs). Timing includes launch/synchronization and
the final queue drains. Like the source test, it transfers zero-filled buffers
without checking received payload contents; it is not a correctness proof.

The supporting API is `GpuNetIoService(bootstrap, ibDeviceName, cudaDeviceId,
numQpsPerPeer)`. Setup validates matching QP counts and symmetric sizes before
variable-size QP exchange, pairs matching queue indices, and allocates atomic
result scratch per peer/QP. Device-context operations accept a final optional
`qpIndex=0`; flushing one queue does not drain the others. The benchmark now
constructs real PortChannels with separate registrations and semaphores on each
QP; its kernel does not bypass PortChannel. No multi-HCA or EP scheduling policy
is introduced, and the upstream dependency remains unmodified.

`gpunetio_channel_multi_qp_test` runs the real device implementation with stubbed
DOCA calls on CPU at QP counts 1, 2, 4, 8, and 64. It checks per-peer addressing,
per-QP scratch, and queue-local completion, not hardware ordering or bandwidth.

### Two Independent Channels

`PortChannelOneToOneTest.GpuNetIoBoundChannels` creates one transport service,
two connections to the same peer on QPs 0/1, two owned semaphores, and separate
send/receive registrations with differing per-rank sizes. It releases the service
wrapper before using the channels, exercises one channel while checking the other
buffer/counter stays untouched, then verifies both payloads, guard bytes and
33 consumed signals per channel through the PortChannel API. Run on two ranks:

```bash
timeout 300s mpirun -np 2 build/bin/mp_unit_tests \
  --filter=PortChannelOneToOneTest.GpuNetIoBoundChannels
```

The hardware test is compiled but not executed by the CPU validation suite.
CPU tests exercise actual registered device operations through two handles and
check exact addresses, keys, QPs, atomic scratch, signal counters, invalid IDs,
owner/range/alignment rejection, and both peer-order branches of metadata exchange.

## Upstream Limitations

### Sparse QP Setup Failure (2026-09-22)

The user reported CPU tests passing on both nodes but `GpuNetIoP2P` crashing
with SIGSEGV/RC=139 at address `0x58` in `doca_gpu_verbs_qp_flat_list_create_hl`,
including explicit retries with `mlx5_ib0`, `mlx5_ib1`, `mlx5_ib2`, and `mlx5_ib3`.
The service passed a peer-major QP list containing null entries for its own rank
to an upstream helper that unconditionally dereferences every entry. In the
pinned upstream ABI, `offsetof(doca_gpu_verbs_qp_hl, qp_gverbs)` is `0x58`.
This is an integration input-contract error, independent of HCA selection.

The service now builds its own peer-major descriptor table: self slots remain
zero, every remote QP descriptor is checked and copied to its original index,
and a service-owned CUDA allocation receives the complete table. Cleanup uses
`cudaFree`; no NVIDIA source changes or self-QP connections are needed.
`gpunetio_channel_qp_table_test` exercises this production adapter using actual
upstream types on CPU: 28 layouts (1/2/4 ranks, 1/2/4/64 QPs, every self rank)
and 296 invalid inputs. Earlier routing tests did not exercise host flattening.

The reported hardware status remains a reproducible setup failure until both
nodes are rebuilt with this fix and the two-rank P2P test is rerun. Passing the
new CPU regression is not evidence of GPU/NIC runtime correctness.

### Remaining Dependency Issues

The official v4.0.1 pin is not equivalent to the patched DOCA snapshot in the
EP branch. In upstream `src/doca_gpunetio.cpp`, the GPU_CPU fallback without
GDRCopy still uses `calloc(alignment, size)` and keeps the original memory-type
bookkeeping; upstream's CPU doorbell service also shares a plain `bool running`
between its worker and shutdown thread. The EP branch corrected these paths.
Upstream `main` at `586453728bcab2d4c50574924dc6cf43543c9ed4` does not contain
those corrections either. This integration deliberately does not patch the
upstream sources or maintain a local fork.

The service explicitly excludes these affected paths while retaining the
unmodified pin. In v4.0.1, explicit GPU_SM_DB fails UAR export rather than entering
the AUTO-to-CPU fallback. Valid DBRs with a GPU CQ keep QP descriptor allocation
in GPU memory instead of GPU_CPU; CPU UMEM and host CQ options remain off.
Returned QPs are also checked for direct handler/CQ/DBR state before publication,
and all CPU-service creation/progress code has been removed from the integration.

The upstream defects themselves are not fixed. Do not enable any excluded mode
without a corrected upstream revision and renewed validation. Successful builds
and CPU policy tests do not qualify hardware compatibility or wire correctness;
real two-rank GDAKI and existing proxy runtime tests remain required for this
restricted configuration.

## Review Regression Coverage

The linked host API test exercises real service metadata for 1/2/4 ranks and
rejects negative, self, world-size, INT_MAX, and valid-but-not-initialized peers
at connection selection. Empty semaphore bindings are rejected by both channel
types. Compile-time checks prevent raw device-pointer host construction. The QP-table test also
checks the production direct-QP policy using actual upstream types and a stubbed
create call: 13 cases cover exact requested attributes, error propagation,
unsupported returned modes, null descriptors, and absence of fallback retries.
These checks execute on CPU and do not initialize GPU/NIC resources.

Two CUDA compile-only targets, `gpunetio_header_clean_compile` and
`gpunetio_header_existing_macros_compile`, check that the public PortChannel
header neither defines the unused `DO_PRAGMA`/`NVCC_PRAGMA_UNROLL*` macros nor
changes caller-provided definitions. The pinned upstream headers use native
unroll pragmas and require no compatibility shim.

## Integration Verification (2026-09-21)

These historical checks predate the FetchContent migration.

- CUDA 13.0.88: core library, `unit_tests`, and `mp_unit_tests` compile and link
  with GPUNetIO ON and OFF for SM80/90/100/120. Existing test warnings remain.
- Three focused CPU tests pass in each configuration, including linked host
  construction and OFF rejection. No CUDA kernels are executed by these tests.
- An SM90 CUDA consumer compiles and links using installed headers/library only.
- OFF configures without the submodule; ON rejects a missing submodule or IB OFF.
- Repository C++ lint and all 131 Python files pass the formatting check.
- GPU/NIC runtime correctness, performance, and ROCm compilation are unverified.

## FetchContent Verification (2026-09-24)

- Clean source snapshot contains neither `.gitmodules` nor `src/gpunetio/vendor`.
  ON configuration fetched exactly `bfe3e5484f16a01ac91a906b2f0046dbc8bd61a4`.
- CUDA 13.0.88 Release builds of the core, `unit_tests`, and `mp_unit_tests`
  pass with GPUNetIO ON/OFF for SM80/90/100/120. Five ON and three OFF CPU tests
  pass; existing test warnings remain.
- A disconnected OFF configure/build creates no GPUNetIO download or targets.
  A disconnected ON source override compiles the QP-table and both header probes.
- Both macro probes compile across the configured CUDA architectures, and at
  SM90 using installed headers only. The installed upstream license matches.
- C++ lint and all 131 Python formatting checks pass. These changes do not alter
  runtime protocol or dependency revision; no GPU/NIC workloads were executed.

## Channel Binding Verification (2026-09-24)

- This patch replaces the service/peer/raw-counter host constructors with
  `PortChannel(semaphore, remoteDestination, localSource)` and
  `BasePortChannel(semaphore, memoryTable)`. Update callers to the connection,
  registration exchange and semaphore flow above. The CPU-proxy API is unchanged.
- Release CUDA 13.0.88 ON/OFF core, `unit_tests`, and `mp_unit_tests` compile and
  link for SM80/90/100/120; both macro compile probes still pass.
- Five ON and three OFF CPU tests pass. Coverage includes two real device handles
  on different QPs/registrations, 12 valid paired metadata exchanges and 96
  mismatched exchanges, invalid access rejection before DOCA calls, and existing
  peer-validation, sparse QP-table and direct-only policy regressions.
- Installed host binding API syntax checks and SM90 device-header compilation
  pass. Device handles remain trivially copyable/default-constructible.
- `GpuNetIoBoundChannels` and the migrated P2P/bandwidth tests are compiled but
  not run on GPU/NIC hardware. Allocation, RDMA ordering, remote lifetime and
  teardown require cluster verification; CPU tests do not establish those results.
- Transport setup still preconnects a collective, uniform-QP-count mesh on one
  HCA. Per-channel memory sizes/addresses and semaphore creation are independent
  thereafter. On-demand QP creation and concurrent setup calls are not introduced.
