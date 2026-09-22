# Dependency: DOCA GPUNetIO (GDAKI) device verbs

The third-party GPUNetIO sources are provided by the `vendor` Git submodule:
https://github.com/NVIDIA-DOCA/gpunetio.

- **License:** BSD-3-Clause (NVIDIA CORPORATION & AFFILIATES).
- **Revision:** pinned by the submodule gitlink; update it explicitly and test
  both the CPU-proxy and GDAKI PortChannel backends before advancing the pin.
- **Build:** gated behind the `MSCCLPP_USE_GPUNETIO` CMake option. Host sources
  compile into the `mscclpp_gpunetio_obj` object library with
  `-DDOCA_VERBS_USE_NET_WRAPPER` (selects the dlopen ibverbs/mlx5dv wrappers).
  Device headers (`include/device/*.cuh`) are consumed only by the GPUNetIO
  PortChannel backend implementation
  (`include/mscclpp/internal/port_channel_gpunetio_device_impl.hpp`).

Clone with `--recurse-submodules`, or run `git submodule update --init --recursive`
before configuring with `MSCCLPP_USE_GPUNETIO=ON`.

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
`bfe3e5484f16a01ac91a906b2f0046dbc8bd61a4`. The submodule is unmodified;
MSCCL++ does not carry patches to NVIDIA's implementation. Its source list and
host API differ from the old snapshot, so dependency updates require new build
and hardware validation. No installed DOCA SDK is required; upstream can load
DOCA SDK libraries when available according to its own runtime policy.

## Build

```bash
git submodule update --init --recursive src/gpunetio/vendor
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMSCCLPP_USE_CUDA=ON -DMSCCLPP_USE_IB=ON -DMSCCLPP_USE_GPUNETIO=ON
cmake --build build -j
```

The option defaults to OFF. OFF builds do not require the submodule or DOCA
headers. GDAKI host channel construction is rejected by an OFF library.
Compile each CUDA translation unit using GDAKI with `-DMSCCLPP_USE_GPUNETIO`
and `-I<source>/src/gpunetio/vendor/include`, or with
`-I<prefix>/include/mscclpp/gpunetio` after installation. Do not enable the
device macro globally for unrelated core kernels. Normal public headers remain
usable without any DOCA includes when the macro is absent.

## Select a Backend

Existing `ProxyService::portChannel(...)` calls remain CPU-proxy channels.
No environment variable changes their backend. For GDAKI, initialize a service
collectively on every bootstrap rank, then construct the same `PortChannel`
type from its device context:

```cpp
#include <mscclpp/gpu_net_io_service.hpp>
#include <mscclpp/port_channel.hpp>

mscclpp::GpuNetIoService service(bootstrap, ibDeviceName, cudaDeviceId);
service.setup(symmetricBuffer, symmetricBytes);
mscclpp::PortChannel channel(service.deviceContext(), peerRank,
                            remoteSignalOffset, localInboundCounter,
                            localExpectedCounter);
auto handle = channel.deviceHandle();
```

The kernel uses the existing `put`, `signal`, `putWithSignal`,
`putWithSignalAndFlush`, `accumulate`, `flush`, `poll`, and `wait` methods.
Offsets are relative to the registered symmetric buffer; arbitrary proxy
`MemoryId` registrations are not interchangeable with GDAKI registrations.
All ranks must use equal buffer sizes and the same offset layout. Signal
counters are aligned 64-bit values, initially zero, disjoint from payloads;
each peer/channel pair needs its own inbound/expected pair. `peerRank` is a
remote bootstrap rank, not self. Producers must make payload writes visible
before issuing network operations. Stop all GPU use and synchronize streams
before destroying the service, then free the buffers and counters.

The backend preserves DOCA's `AUTO` NIC-handler selection. Kernels post WQEs,
but DOCA can use CPU doorbell assistance when direct GPU doorbells are not
available. This is separate from MSCCL++'s FIFO/CPU-proxy backend; selecting
GPUNetIO alone does not promise a CPU-free doorbell path on every machine.

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
`qpIndex=0`; flushing one queue does not drain the others. The common
`PortChannelDeviceHandle` still uses QP 0. No multi-HCA or EP scheduling policy
is introduced, and the upstream submodule remains unchanged.

`gpunetio_channel_multi_qp_test` runs the real device implementation with stubbed
DOCA calls on CPU at QP counts 1, 2, 4, 8, and 64. It checks per-peer addressing,
per-QP scratch, and queue-local completion, not hardware ordering or bandwidth.

## Known Upstream Merge Blockers

The official v4.0.1 pin is not equivalent to the patched DOCA snapshot in the
EP branch. In upstream `src/doca_gpunetio.cpp`, the GPU_CPU fallback without
GDRCopy still uses `calloc(alignment, size)` and keeps the original memory-type
bookkeeping; upstream's CPU doorbell service also shares a plain `bool running`
between its worker and shutdown thread. The EP branch corrected these paths.
Upstream `main` at `586453728bcab2d4c50574924dc6cf43543c9ed4` does not contain
those corrections either. This integration deliberately does not patch the
submodule or maintain a local fork.

Before merging into main, obtain an upstream revision that resolves these
issues or explicitly review and qualify a supported configuration that excludes
the affected paths. Successful compilation and CPU routing tests do not qualify
GPU/NIC resource allocation, fallback progress, shutdown, or wire correctness.
Real two-rank GDAKI and existing proxy runtime tests remain required.

## Integration Verification (2026-09-21)

- CUDA 13.0.88: core library, `unit_tests`, and `mp_unit_tests` compile and link
  with GPUNetIO ON and OFF for SM80/90/100/120. Existing test warnings remain.
- Three focused CPU tests pass in each configuration, including linked host
  construction and OFF rejection. No CUDA kernels are executed by these tests.
- An SM90 CUDA consumer compiles and links using installed headers/library only.
- OFF configures without the submodule; ON rejects a missing submodule or IB OFF.
- Repository C++ lint and all 131 Python files pass the formatting check.
- GPU/NIC runtime correctness, performance, and ROCm compilation are unverified.
