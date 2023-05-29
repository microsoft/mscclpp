# Quick Start

## Preliminaries

- OS: tested over Ubuntu 18.04 and 20.04
- Libraries: CUDA >= 11.1.1, [libnuma](https://github.com/numactl/numactl), (optional) [GDRCopy](https://github.com/NVIDIA/gdrcopy), (optional) MPI
- GPUs: A100, H100
- Azure SKUs: [ND_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series), [NDm_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/ndm-a100-v4-series), ND_H100_v5 (TBD: [NC_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series))

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
