# Customized Collective Algorithm with NCCL API

```{note}
This tutorial builds on concepts from the previous tutorials (e.g. Port Channel). Here we demonstrate how to plug a **custom collective algorithm** (an AllGather variant) into the MSCCL++ NCCL interposition / algorithm registration path and invoke it transparently via the standard NCCL API (`ncclAllGather`).
```

## Overview
The example shows how to:

1. Define a device kernel (`allgather`) that uses `PortChannel` device handles to exchange data.
2. Wrap that kernel inside an algorithm class (`AllgatherAlgo`) responsible for:
   - Connection discovery / proxy setup.
   - Context key generation (so contexts can be reused / cached).
   - Launch function binding (kernel wrapper executed when NCCL all-gather is called).
3. Register the algorithm with the global `AlgorithmFactory` and install a **selector** deciding which implementation to return for a given collective request.
4. Run a multi-process (multi-rank) test using standard NCCL calls. The user program remains unchanged apart from initialization / registration code.
5. (Optionally) Capture the sequence of `ncclAllGather` calls into a CUDA Graph for efficient replay.

## Location
Example source directory:
```
examples/tutorials/05-customized-collective-algorithm/
```
Key file: `customized_allgather.cu`.

## Build and Run
From the repository root:
```bash
cd examples/tutorials/05-customized-collective-algorithm
make
```
Run (inside container you may need root privileges depending on GPU access):
```bash
LD_PRELOAD=<MSCCLPP_INSTALL_DIR>/lib/libmscclpp_nccl.so ./customized_allgather
```
Expected (abbreviated) output on success:
```
GPU 0: bytes 268435456, elapsed 6.25382 ms/iter, BW 128.77 GB/s
Succeed!
```
