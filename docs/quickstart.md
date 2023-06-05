# Quick Start

## Prerequisites

* Azure SKUs
    * [ND_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series)
    * [NDm_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/ndm-a100-v4-series)
    * ND_H100_v5
    * [NC_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series) (TBD)
* Non-Azure Systems
    * NVIDIA A100 GPUs + CUDA >= 11.1.1
    * NVIDIA H100 GPUs + CUDA >= 12.0.0
* OS: tested over Ubuntu 18.04 and 20.04
* Libraries: [libnuma](https://github.com/numactl/numactl), [GDRCopy](https://github.com/NVIDIA/gdrcopy) (optional), MPI (optional)

## Build from Source

```
$ git clone https://github.com/microsoft/mscclpp.git
$ mkdir -p mscclpp/build && cd mscclpp/build
$ cmake ..
$ make -j
```

## Install from Source

```
# Install the generated headers and binaries to /usr/local
$ cmake --install . --prefix /usr/local
```

## Install from Package

TBU

## (Optional) Unit Tests

TBU
