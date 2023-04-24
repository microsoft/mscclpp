# MSCCL++

GPU-driven computation & communication stack.

## Quick Start

### Preliminaries

- OS: tested over Ubuntu 18.04 and 20.04
- Libraries: CUDA >= 11.1.1, [libnuma](https://github.com/numactl/numactl)
- GPUs: A100 (TBU: H100)
- Azure SKUs: [ND_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series), [NDm_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/ndm-a100-v4-series) (TBD: [NC_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series))


### Compile Library

Run `make` in the top directory. To use MPI for test code, pass `MPI_HOME` (`/usr/local/mpi` by default). For example:

```
$ MPI_HOME=/usr/local/mpi make -j
```

If you do not want to use MPI, pass `USE_MPI_FOR_TESTS=0`.

```
# Do not use MPI
$ USE_MPI_FOR_TESTS=0 make -j
```

`make` will create a header file `build/include/mscclpp.h` and a shared library `build/lib/libmscclpp.so`.

### (Optional) Tests

For verification, one can try provided sample code `bootstrap_test` or `p2p_test`. First add the MSCCL++ library path to `LD_LIBRARY_PATH`.

```
$ export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH
```

Run tests using MPI:

```
$ mpirun -np 8 ./build/bin/tests/bootstrap_test 127.0.0.1:50000
$ mpirun -np 8 ./build/bin/tests/p2p_test 127.0.0.1:50000
```

If tests are compiled without MPI, pass a rank and the number of ranks as the following example. Usage of `p2p_test` is also the same as `bootstrap_test`.

```
# Terminal 1: Rank 0, #Ranks 2
$ ./build/bin/tests/bootstrap_test 127.0.0.1:50000 0 2
# Terminal 2: Rank 1, #Ranks 2
$ ./build/bin/tests/bootstrap_test 127.0.0.1:50000 1 2
```

## Performance

All results from NDv4. NCCL version 2.17.1+cuda11.8, reported in-place numbers.

nccl-tests command example:
```bash
mpirun --bind-to numa -hostfile /mnt/hostfile --tag-output --allow-run-as-root -map-by ppr:8:node --bind-to numa -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x PATH -x LD_PRELOAD=/mnt/nccl/build/lib/libnccl.so -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_SOCKET_IFNAME=eth0 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/mnt/ndv4-topo.xml -x NCCL_DEBUG=WARN ./build/all_gather_perf -b 1K -e 1K -g 1 -c 1 -w 10 -n 10 -G 1
```

mscclpp-tests command example:
```bash
mpirun -allow-run-as-root -map-by ppr:8:node -hostfile /mnt/hostfile -x LD_LIBRARY_PATH=/mnt/mscclpp/build/lib:$LD_LIBRARY_PATH ./build/bin/tests/allgather_test_perf -b 1K -e 1K -w 10 -n 10 -G 1 -k 0
```

**NOTE:** NCCL AllGather leverages Ring algorithm instead of all-pairs alike algorithm, which greatly reduces inter-node transmission, causing significant higher performance. MSCCL++ should do something similar in the future

### 1 node, 8 gpus/node
**Latency (us)**
| Message Size | NCCL AllGather | NCCL AllReduce | NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:-------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1K           | 12.53          | **16.96**      | 9.34          | **7.76** / 21.06 / 28.50       | 157.91 / 143.21 / 447.0    | 326.4             |

**BusBW (GB/s)**
| Message Size | NCCL AllGather | NCCL AllReduce | NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2   | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:-------------:|:------------------------------:|:----------------------------:|:-----------------:|
| 1G           | 253.59         | **132.31**     | 254.69        | 217.05 / 216.98 / 217.15       | 144.21 / **255.06** / 142.47 | 12.81             |

### 2 nodes, 1 gpu/node
**Latency (us)**
| Message Size | NCCL AllGather | NCCL AllReduce |  NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:--------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1K           | 16.08          | **21.27**      | 29.84          | 14.67 / 29.12 / 35.43          | 15.32 / **13.84** / 26.08  | -                 |

**BusBW (GB/s)**
| Message Size | NCCL AllGather | NCCL AllReduce | NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:-------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1G           | 31.71          | **18.65**      | 30.93         | 13.94 / 13.83 / 14.10          | 23.26 / 23.29 / **43.20**  | -                 |

### 2 nodes, 8 gpus/node
**Latency (us)**
| Message Size | NCCL AllGather | NCCL AllReduce | NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:-------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1K           | 33.74          | **35.85**      | 49.75         | **22.55** / 39.33 / 56.93      | 159.14 / 230.52 / 462.7    | -                 |

**BusBW (GB/s)**
| Message Size | NCCL AllGather | NCCL AllReduce | NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:-------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1G           | 189.73         | **97.72**      | 40.16         | 40.17 / 40.18 / 40.23          | 44.08 / 9.31 / **225.72**  | -                 |
| 4G           | 198.87         | **100.36**     | 40.56         | - / - / -                      | 47.54 / - / **249.01**     | -                 |



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
