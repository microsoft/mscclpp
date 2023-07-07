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

CMake 3.26 or later is required.

```bash
$ git clone https://github.com/microsoft/mscclpp.git
$ mkdir -p mscclpp/build && cd mscclpp/build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j
```

## Install from Source

```bash
# Install the generated headers and binaries to /usr/local/mscclpp
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/mscclpp ..
$ make -j mscclpp
$ sudo make install/fast
```

## Docker Images

Our base image installs all prerequisites for MSCCL++.

```bash
$ docker pull ghcr.io/microsoft/mscclpp/mscclpp:base-cuda12.1
```

## Unit Tests

`unit_tests` require one GPU on the system. It only tests operation of basic components.

```bash
$ make -j unit_tests
$ ./test/unit_tests
```

For thorough testing of MSCCL++ features, we need to use `mp_unit_tests` that require at least two GPUs on the system. `mp_unit_tests` also requires MPI to be installed on the system. For example, the following commands run `mp_unit_tests` with two processes (two GPUs). The number of GPUs can be changed by changing the number of processes.

```bash
$ make -j mp_unit_tests
$ mpirun -np 2 ./test/mp_unit_tests
```

To run `mp_unit_tests` with more than two nodes, you need to specify the `-ip_port` argument that is accessible from all nodes. For example:

```bash
$ mpirun -np 16 -npernode 8 -hostfile hostfile ./test/mp_unit_tests -ip_port 10.0.0.5:50000
```

## mscclpp-test

mscclpp-test is a set of performance benchmarks for MSCCL++. It requires MPI to be installed on the system.

```bash
$ make -j sendrecv_test_perf allgather_test_perf allreduce_test_perf alltoall_test_perf
```

For example, the following command runs the AllReduce benchmark with 8 GPUs starting from 3MB to 48MB messages, by doubling the message size in between.

```bash
$ mpirun -np 8 ./test/mscclpp-test/allreduce_test_perf -b 3m -e 48m -G 100 -n 100 -w 20 -f 2 -k 4
```

Check the help message for more details.

```bash
$ ./test/mscclpp-test/allreduce_test_perf --help
USAGE: allreduce_test_perf 
        [-b,--minbytes <min size in bytes>] 
        [-e,--maxbytes <max size in bytes>] 
        [-i,--stepbytes <increment size>] 
        [-f,--stepfactor <increment factor>] 
        [-n,--iters <iteration count>] 
        [-w,--warmup_iters <warmup iteration count>] 
        [-c,--check <0/1>] 
        [-T,--timeout <time in seconds>] 
        [-G,--cudagraph <num graph launches>] 
        [-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] 
        [-k,--kernel_num <kernel number of commnication primitive>] 
        [-o, --output_file <output file name>] 
        [-h,--help]
```
