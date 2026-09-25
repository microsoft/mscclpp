# Dependency: DOCA GPUNetIO (GDAKI)

The bundled DOCA sources and headers have been removed. CMake FetchContent
uses the unmodified official NVIDIA repository, https://github.com/NVIDIA-DOCA/gpunetio,
pinned to v4.0.1 commit `bfe3e5484f16a01ac91a906b2f0046dbc8bd61a4`.
The dependency is BSD-3-Clause licensed. Its headers are installed under
`include/mscclpp/gpunetio` and its license under `share/licenses/mscclpp/gpunetio`.
No Git submodule initialization is needed despite the branch name.

The integration is ported from `qinghuazhou/gpunetio_submodule` at
`b11bccc544f1cbe3d524d8e753922aecab0a8cf7`, on top of `feature/ep-experimental`
at `672e30d733bca64b13c2976bc0353f74d3f6dca9`. EP's existing CPU-proxy
`atomicAdd` API and bitmask trigger encoding are retained; the donor branch's
unrelated `accumulate` and proxy-opcode migration is not included.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DMSCCLPP_USE_CUDA=ON -DMSCCLPP_USE_IB=ON -DMSCCLPP_USE_GPUNETIO=ON \
  -DMSCCLPP_BUILD_PYTHON_BINDINGS=ON -DMSCCLPP_BUILD_TESTS=ON \
  -DMSCCLPP_GPU_ARCHS=100a
cmake --build build -j2
```

Select the architecture for the target hardware. GPUNetIO defaults OFF and
requires CUDA, InfiniBand and no ROCm. OFF builds neither declare nor fetch
the dependency. ON builds need Git/network access unless an existing checkout
is supplied using `-DFETCHCONTENT_SOURCE_DIR_GPUNETIO=/absolute/path`.
That override bypasses pin enforcement: verify the checkout's exact revision
and cleanliness. Reconfigure existing builds after this migration.

## Shared API

Include `<mscclpp/gpu_net_io_service.hpp>`. Each shared `GpuNetIoService` owns
one explicitly selected HCA and 1-64 QPs per peer. `setup()` creates transport;
`connect(peer, queue)` selects a connected QP; `registerMemory` registers a
buffer; `exchangeMemory` exchanges the selected peer's registration; and
`buildSemaphore` creates owned signal counters. Construct
`PortChannel(semaphore, remoteDestination, localSource)` from these handles.
Offsets are relative to the selected registrations. Channels retain transport,
registrations and semaphore state. Synchronize GPU work before releasing the
last channel or its externally owned payload buffer.

## EP Adapter

EP's internal `EpGpuNetIoService` in `src/ext/ep/gpu_net_io.cc` composes one
shared service per HCA. It preserves `MSCCLPP_EP_ENABLE_GPUNETIO`, the plural
`MSCCLPP_EP_GPUNETIO_HCAS` setting (preferred over singular `..._HCA`), and
automatic selection of the full best-affinity active HCA set. Explicit lists
retain their order; automatic lists are sorted by name and may be shared by
nearby GPUs. Existing topology selection tests remain applicable.

`MSCCLPP_EP_GPUNETIO_QPS_PER_PEER` remains the total logical QP count, defaulting
to the HCA count. It must be a complete integer in 1-64 and a positive multiple
of the HCA count; malformed values are now rejected rather than partially parsed.
All bootstrap ranks must agree on buffer size and HCA/QP counts. Logical QP
`q` maps to HCA `q % numHcas` and that service's QP `q / numHcas`.

The adapter registers the existing symmetric EP allocation separately on each
HCA and builds owned PortChannels per peer/logical QP using the shared API.
EP payload layouts, per-QP generation flags, combine stripe markers, batching,
and final acknowledgements are preserved. Symmetric flags stay in the EP
allocation, distinct from each channel's private semaphore. Batched metadata
and sparse warp writes obtain their QP, addresses and keys from the channel
binding, not from a duplicated flattened DOCA QP table. The latency context
synchronizes and destroys the adapter before freeing its payload allocation.

## Safety and Validation

The shared service requires direct `GPU_SM_DB` doorbells, valid DBRs and
noncollapsed GPU-resident CQs. CPU-assisted fallback, AUTO selection, host CQs,
CPU UMEM and a DOCA CPU progress service are not enabled. Unsupported hardware
fails setup. This avoids the pinned upstream's known GPU_CPU allocation fallback
and CPU service shutdown defects; it does not fix those upstream defects or
carry the old bundled DOCA patches. No upstream source is modified.

Use an external launcher timeout for GPU tests: local allocation failures,
bootstrap exchanges and upstream posting waits are not end-to-end bounded.
A skipped setup is not a hardware pass. Build/CPU checks cannot establish GPU
memory visibility, NIC ordering, numerical results or performance.

```bash
ctest --test-dir build -R '^gpunetio_.*_test$' --output-on-failure
PYTHONPATH=test/python/ep python3 -m unittest discover \
  -s test/python/ep -p 'test_gpunetio*.py'
timeout 300s mpirun -np 2 build/bin/mp_unit_tests \
  --filter=PortChannelOneToOneTest.GpuNetIo
```

The mp-unit filter includes P2P, LL correctness/latency, multi-QP bandwidth and
independent channel bindings. The multi-QP mp-unit benchmark now uses the shared
single-HCA service; EP multi-HCA behavior is exercised through the EP runtime,
not that benchmark. Run the existing EP rank-major and top-k-expanded numerical
and graph-replay tests on the intended multi-node/HCA configuration before
claiming runtime parity. These GPU workloads are not executed by the CPU suite.

CPU coverage includes the shared real-header routing, registration exchange,
sparse QP-table adapter and direct-only policy tests, plus EP logical-QP mapping,
collective rejection, retained ownership, partial setup cleanup, batched WQEs
and 32-thread sparse warp tests. Tests of the removed bundled allocator, CPU
worker and modified upstream WQE internals have been retired; they are not
claims about the fetched dependency. Existing EP protocol and layout checks
remain, with fixtures updated for the new binding API.
