# NDmv4 Performance

All results from NDmv4. NCCL version 2.17.1+cuda11.8, reported in-place numbers.

nccl-tests command example:
```bash
mpirun --bind-to numa -hostfile /mnt/hostfile --tag-output --allow-run-as-root -map-by ppr:8:node --bind-to numa -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x PATH -x LD_PRELOAD=/mnt/nccl/build/lib/libnccl.so -x NCCL_IB_PCI_RELAXED_ORDERING=1 -x NCCL_SOCKET_IFNAME=eth0 -x CUDA_DEVICE_ORDER=PCI_BUS_ID -x NCCL_NET_GDR_LEVEL=5 -x NCCL_TOPO_FILE=/mnt/ndv4-topo.xml -x NCCL_DEBUG=WARN ./build/all_gather_perf -b 1K -e 1K -g 1 -c 1 -w 10 -n 10 -G 1
```

mscclpp-tests command example:
```bash
mpirun -allow-run-as-root -map-by ppr:8:node -hostfile /mnt/hostfile ./build/test/mscclpp-test/allgather_test_perf -b 1K -e 1K -w 10 -n 10 -G 10 -k 0
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
| 1G           | 253.59         | **231.45**     | 254.69        | 217.05 / 216.98 / 217.15       | 125.06 / **255.64** / 124.89 | 22.55             |

### 2 nodes, 1 gpu/node
**Latency (us)**
| Message Size | NCCL AllGather | NCCL AllReduce |  NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:--------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1K           | 16.08          | **21.27**      | 29.84          | 14.67 / 29.12 / 35.43          | 15.32 / **13.84** / 26.08  | -                 |

**BusBW (GB/s)**
| Message Size | NCCL AllGather | NCCL AllReduce | NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:-------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1G           | 15.84          | **18.65**      | 15.48         | 13.94 / 13.83 / 14.10          | **23.30** / 23.29 / 21.60  | -                 |

### 2 nodes, 8 gpus/node
**Latency (us)**
| Message Size | NCCL AllGather | NCCL AllReduce | NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:-------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1K           | 33.74          | **35.85**      | 49.75         | **22.55** / 39.33 / 56.93      | 159.14 / 230.52 / 462.7    | -                 |

**BusBW (GB/s)**
| Message Size | NCCL AllGather | NCCL AllReduce | NCCL AllToAll | MSCCL AllToAll LL/LL128/Simple | MSCCL++ AllGather K0/K1/K2 | MSCCL++ AllReduce |
|:------------:|:--------------:|:--------------:|:-------------:|:------------------------------:|:--------------------------:|:-----------------:|
| 1G           | 177.05         | **183.82**     | 37.80         | 40.17 / 40.18 / 40.23          | 44.19 / 9.31 / **209.33**  | -                 |
| 4G           | 186.01         | **188.18**     | 37.81         | - / - / -                      | 44.60 / - / **234.08**     | -                 |

