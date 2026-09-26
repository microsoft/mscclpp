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
Both `mscclpp` and `mscclpp_static` carry the GPUNetIO feature definition and
dependency include paths as consumer usage requirements when built ON. Linking
either target enables the real device implementation automatically. OFF targets
export neither the feature macro nor DOCA header paths. Internal object compilation
is configured separately; ordinary CPU-proxy channels remain the default backend.

Installed consumers can use the relocatable CMake package:

```cmake
find_package(mscclpp CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE mscclpp::mscclpp)
# Or mscclpp::mscclpp_static for the static library.
```

Set `CMAKE_PREFIX_PATH` to the install prefix. No manual GPUNetIO compile
definition, vendor include path or FetchContent checkout is needed by the consumer.
For non-CMake/manual compilation against an ON library, still supply
`-DMSCCLPP_USE_GPUNETIO` and `-I<prefix>/include/mscclpp/gpunetio` in addition
to the normal public include path. Do not mix ON and OFF installations.

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
std::vector<int> peerQpCounts(bootstrap->getNranks(), 0);
peerQpCounts[peerRank] = 1;
service.setup(peerQpCounts, setupTag);
auto connection = service.connect(peerRank, 0);
auto source = service.registerMemory(sendBuffer, sendBytes, sendOwner);
auto receive = service.registerMemory(recvBuffer, recvBytes, recvOwner);
auto destination = service.exchangeMemory(connection, receive, memoryTag);
auto semaphore = service.buildSemaphore(connection, semaphoreTag);
mscclpp::PortChannel channel(semaphore, destination, source);
auto handle = channel.deviceHandle();
```

The plan contains one count per bootstrap rank, zero for self and 0-64 for each
peer. Both ends of an edge must request the same count, but different edges may
request different counts. All ranks participate once, including idle ranks with
all-zero plans, using the same nonnegative `setupTag`. Plans and tags are
validated collectively before QP creation. Reserve the tag until setup completes
and serialize setup calls with other bootstrap traffic. The example requests
only one peer; its peer must request the reciprocal edge.

Only requested QPs/CQs and their atomic-result slots are allocated. The QP
descriptor table is compact: `peerQpOffsets[peer] + qpIndex` selects both the QP
and scratch slot. Idle ranks allocate no QPs, CQs or atomic scratch, although
they still initialize a local IB context/protection domain and device metadata.
QP connection metadata is sent directly to requested peers, not all-gathered.
The count-plan validation still all-gathers `worldSize` integers per rank,
requiring O(worldSize squared) control metadata per rank; this is sparse
collective setup, not fully noncollective or dynamically extensible setup.

The existing `setup()` and `setup(buffer, bytes)` overloads retain full-mesh
behavior with the constructor's uniform count (and reserve bootstrap tag zero).
The explicit sparse overload ignores that constructor default and registers
channel payloads separately. `connect(peer, qpIndex)` selects only QPs in the
completed plan; it cannot add new connections. Unrequested peers and excess
queue indices are rejected by host and device checks.

### Port and GID Selection

The service uses port 1 on the explicitly selected HCA and reads the same
`env()->ibGidIndex` configuration as ordinary IB endpoints (`MSCCLPP_IB_GID_INDEX`,
default 0). Set the variable before the process initializes MSCCL++'s cached
environment configuration. Different ranks may use different local indices;
choose the appropriate entry for each node rather than assuming index 0 is routable.

Before creating QPs, each rank checks that its port is active, has an IB or
Ethernet link layer and valid MTU, and that the configured GID index is in the
port's table and fits the 8-bit source-GID field (0-255). GID query failures are
rejected; RoCE or GRH-required ports also require a nonzero GID. The validated
local GID is cached for QP metadata exchange, and the same configured local
index is used for address-handle programming. There is no fallback to another
GID entry. Validating an entry does not prove network reachability or select a
RoCE version automatically.

Port/GID validation status is exchanged across all ranks, including idle ranks,
so a failed preflight prevents QP creation everywhere. The HCA must also report
RDMA atomic support through `IbCtx::supportsRdmaAtomics()`, including on idle ranks;
signals and `accumulate` require fetch-add and have no safe fallback. Errors identify
the failed phase and rank, with the local diagnostic (bounded to 255 bytes).
Port/GID diagnostics retain the HCA, port and configured index. Later phases use
the status protocol below; physical link changes still require runtime handling.
CPU regression checks exercise nonzero indices and invalid/query-failure cases;
actual RoCE connectivity still requires testing on the intended network.

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
status is exchanged before QP metadata exchange so every rank rejects an
unsupported peer. The service never starts a DOCA CPU progress thread. The
existing MSCCL++ FIFO/CPU-proxy backend remains available and unchanged.

### Setup Failure Coordination

Each failure-prone local phase is followed by a status all-gather before any
rank advances: configuration preparation, plan preparation/validation, HCA and
port/GID/atomic admission, registration/device resources, QP creation, metadata
preparation, QP transitions, device QP-table preparation, and final device-context
publication. Host buffers for peer exchanges are allocated in the preceding phase.
All ranks participate, including idle ranks. A reported failure makes every rank
throw with the same first failing rank/phase; no rank publishes a successful
context before final agreement. Partial resources remain owned for RAII cleanup.

The small status buffer is allocated during service construction. Callers must
successfully construct services on all ranks before entering setup and must not
retry a failed setup on the same service. This protocol coordinates exceptions
from local work; it cannot recover a failed process, a hung driver call, or an
exception/hang inside bootstrap send/receive/all-gather itself. Keep an external
launcher timeout for those failures. Post-setup registration/semaphore exchanges
remain paired operations and are not covered by the collective setup protocol.

## Validation

`gpunetio_channel_release_test` and `gpunetio_channel_debug_test` execute the
actual common device-handle routing with CPU transport stubs. They do not
validate NIC ordering or CUDA memory visibility. The two-rank hardware check is
`mpirun -np 2 build/bin/mp_unit_tests --filter=PortChannelOneToOneTest.GpuNetIoP2P`.
Run it on a configured GPU/RDMA pair with a launcher timeout, then run existing
proxy tests with the same build. A skipped setup is not a hardware pass;
asymmetric setup failures during collectives can require launcher termination.

### LL Ping-Pong

`PortChannelOneToOneTest.GpuNetIoLLPingPong` and
`PortChannelOneToOneTest.GpuNetIoLLPingPongPerf` are ported from
`feature/ep-experimental` at `672e30d733bca64b13c2976bc0353f74d3f6dca9`.
Both use separately registered send/receive buffers bound to a PortChannel on
QP 0. LL packet flags provide receive readiness; no separate signal is posted.
Each launch uses one block of 512 threads with alternating send/receive ranks.

The correctness test checks both payload words in every packet for 1,000
iterations at logical payload sizes 8 B, 4 KiB, 4 MiB, and 16 MiB. LL16 packet
flags double the wire bytes. The performance test uses an 8 B payload with
100,000 warmup iterations and 100,000 timed iterations. It reports `us/iter`,
including launch and device synchronization, not isolated one-way wire latency.
Performance mode does not check payload values. Packet flags are cleared and
initialization is synchronized across ranks before every launch, including
between warmup and timing.

Build with GPUNetIO and tests enabled, then on two configured GPU/RDMA ranks:

```bash
timeout 300s mpirun -np 2 build/bin/mp_unit_tests \
  --filter=PortChannelOneToOneTest.GpuNetIoLLPingPong --exclude-perf-tests
timeout 300s mpirun -np 2 build/bin/mp_unit_tests \
  --filter=PortChannelOneToOneTest.GpuNetIoLLPingPongPerf
```

The filter is a substring match: omitting `--exclude-perf-tests` from the first
command runs both tests. Both require exactly two ranks and skip in OFF builds.
Diagnostics from both ranks include payload mismatches (code 1), bounded CQ
completion failures (code 100), and receive flag timeouts (code 200). The tests
use the channel's selected QP for bounded `tryFlush`, since ordinary `flush`
may fall back to a blocking drain. These spin bounds do not bound setup or
upstream posting waits; retain an external launcher timeout. Compilation and
CPU checks do not establish hardware correctness or latency for these tests.

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
numQpsPerPeer)`. Legacy setup validates matching QP counts and symmetric sizes,
exchanges requested peer metadata, pairs matching queue indices, and allocates atomic
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

## Channel Binding Verification (2026-09-24, Historical)

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
- At that revision, transport setup preconnected a collective, uniform-QP-count mesh on one
  HCA. Per-channel memory sizes/addresses and semaphore creation are independent
  thereafter. On-demand QP creation and concurrent setup calls are not introduced.

## Sparse Connection Plans

`setup(peerQpCounts, tag)` removes the mandatory full mesh while retaining the
existing binding API and explicit one-HCA service. CPU coverage includes 45
empty/ring/unequal-star plan cases, asymmetric and invalid count rejection,
compact descriptor copies, blocking eight-rank peer exchanges with an idle rank,
and actual device QP/atomic-scratch lookup with missing-peer and excess-queue
rejection. These checks do not establish GPU/NIC ordering or hardware resource
availability. Setup is still collective. Local setup exceptions are coordinated
at phase boundaries; bootstrap or driver hangs can require launcher termination.

The four-rank hardware test requests one QP between ranks 0/1, three between
ranks 1/2, and none for rank 3. It checks compact offsets, idle-rank resources,
unrequested connection rejection, and payload/guard bytes plus five signals on
each requested channel. Compile with tests and GPUNetIO enabled, then run on
four configured GPU/RDMA ranks:

```bash
timeout 300s mpirun -np 4 build/bin/mp_unit_tests \
  --filter=PortChannelSparseTest.GpuNetIoSparseConnections
```

This test is compiled but has not been executed as part of CPU validation.
Rebuild all consumers when updating the device-context layout; raw contexts
without an offsets table retain the legacy peer-major indexing convention.

## Consumer and Setup Regressions

`gpunetio_setup_test` compiles the actual setup body extracted at CMake configure
time with CPU resource/bootstrap stubs. It injects failures on each of four ranks
at device selection, HCA/atomic/port admission, registration, CUDA operations, QP
creation/metadata/transition, and final synchronization. Tests check common failure
outcomes and no premature successful publication. Its stub cleanup is not proof
of real CUDA/DOCA teardown behavior. The host test also covers allocation and
unknown exceptions and threaded phase agreement. Run these CPU checks with:

```bash
ctest --test-dir build -R '^gpunetio_.*_test$' --output-on-failure
```

The installed-consumer fixture compiles and links shared/static CUDA programs
without manually supplying GPUNetIO flags or include directories:

```bash
cmake -S test/unit/consumer -B consumer-build \
  -DCMAKE_PREFIX_PATH=/path/to/install -DEXPECT_GPUNETIO=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build consumer-build -j2
```

Use `EXPECT_GPUNETIO=OFF` for an OFF installation. Compilation is not GPU/NIC
runtime qualification; this fixture does not launch its device kernel.
