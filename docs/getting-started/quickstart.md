# Quick Start

## Prerequisites

* Azure SKUs
    * [ND_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series)
    * [NDm_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/ndm-a100-v4-series)
    * [ND_H100_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nd-h100-v5-series)
    * [NC_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nc-a100-v4-series) (TBD)
* Non-Azure Systems
    * NVIDIA A100 GPUs + CUDA >= 11.8
    * NVIDIA H100 GPUs + CUDA >= 12.0
    * AMD MI250X GPUs + ROCm >= 5.7
    * AMD MI300X GPUs + ROCm >= 6.0
* OS: tested over Ubuntu 18.04 and 20.04
* Libraries
    * [libnuma](https://github.com/numactl/numactl)
        ```bash
        sudo apt-get install libnuma-dev
        ```
    * (Optional, for [building the Python module](#install-from-source-python-module)) Python >= 3.8 and Python Development Package
        ```bash
        sudo apt-get satisfy "python3 (>=3.8), python3-dev (>=3.8)"
        ```
        If you don't want to build Python module, you need to set `-DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF` in your `cmake` command (see details in [Install from Source (Libraries and Headers)](#install-from-source-libraries-and-headers)).
    * (Optional, for benchmarks) MPI
* Others
    * For NVIDIA platforms, `nvidia_peermem` driver should be loaded on all nodes. Check it via:
        ```
        lsmod | grep nvidia_peermem
        ```
    * For GPU with nvls support, the IMEX channels should be set up (refer [cuMemCreate](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html#group__CUDA__VA_1g899d69a862bba36449789c64b430dc7c)). You can set up the channels manually via:
        ```
        sudo nvidia-modprobe -s -i <start:number of minors>
        ```

## Build with Docker Images

We provide docker images which package all prerequisites for MSCCL++. You can setup your dev environment with the following command.

```bash
$ docker run -it --privileged --net=host --ipc=host --gpus all ghcr.io/microsoft/mscclpp/mscclpp:base-dev-cuda12.2 mscclpp-dev bash
```

See all available images [here](https://github.com/microsoft/mscclpp/pkgs/container/mscclpp%2Fmscclpp).

(build-from-source)=
## Build from Source

CMake 3.25 or later is required.

```bash
$ git clone https://github.com/microsoft/mscclpp.git
$ mkdir -p mscclpp/build && cd mscclpp/build
```

For NVIDIA platforms, build MSCCL++ as follows.

```bash
# For NVIDIA platforms
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j
```

For AMD platforms, use HIPCC instead of the default C++ compiler. Replace `/path/to/hipcc` from the command below into the your HIPCC path.

```bash
# For AMD platforms
$ CXX=/path/to/hipcc cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j
```

(install-from-source-libraries-and-headers)=
## Install from Source (Libraries and Headers)

```bash
# Install the generated headers and binaries to /usr/local/mscclpp
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/mscclpp -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF ..
$ make -j mscclpp mscclpp_static
$ sudo make install/fast
```

(install-from-source-python-module)=
## Install from Source (Python Module)

Python 3.8 or later is required.

```bash
# For NVIDIA platforms
$ python -m pip install .
# For AMD platforms
$ CXX=/path/to/hipcc python -m pip install .
```

## Docker Images

Our base image installs all prerequisites for MSCCL++.

```bash
$ docker pull ghcr.io/microsoft/mscclpp/mscclpp:base-dev-cuda12.3
```

See all available images [here](https://github.com/microsoft/mscclpp/pkgs/container/mscclpp%2Fmscclpp).

## Unit Tests

`unit_tests` require one GPU on the system. It only tests operation of basic components.

```bash
$ make -j unit_tests
$ ./test/unit_tests
```

For thorough testing of MSCCL++ features, we need to use `mp_unit_tests` that require at least two GPUs on the system. `mp_unit_tests` also requires MPI to be installed on the system. For example, the following commands compile and run `mp_unit_tests` with two processes (two GPUs). The number of GPUs can be changed by changing the number of processes.

```bash
$ make -j mp_unit_tests
$ mpirun -np 2 ./test/mp_unit_tests
```

To run `mp_unit_tests` with more than two nodes, you need to specify the `-ip_port` argument that is accessible from all nodes. For example:

```bash
$ mpirun -np 16 -npernode 8 -hostfile hostfile ./test/mp_unit_tests -ip_port 10.0.0.5:50000
```

## Performance Benchmark

### Python Benchmark

[Install the MSCCL++ Python package](#install-from-source-python-module) and run our Python AllReduce benchmark as follows. It requires MPI on the system.

```bash
# Choose `requirements_*.txt` according to your CUDA/ROCm version.
$ python3 -m pip install -r ./python/requirements_cuda12.txt
$ mpirun -tag-output -np 8 python3 ./python/mscclpp_benchmark/allreduce_bench.py
```

### C++ Benchmark (mscclpp-test)

*NOTE: mscclpp-test will be retired soon and will be maintained only as an example of C++ implementation. If you want to get the latest performance numbers, please use the Python benchmark instead.*

mscclpp-test is a set of C++ performance benchmarks. It requires MPI on the system, and the path should be provided via `MPI_HOME` environment variable to the CMake build system.

```bash
$ MPI_HOME=/path/to/mpi cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j allgather_test_perf allreduce_test_perf
```

For example, the following command runs the `allreduce5` algorithm with 8 GPUs starting from 3MB to 48MB messages, by doubling the message size in between. You can try different algorithms by changing the `-k 5` option to another value (e.g., `-k 3` runs `allreduce3`). Check all algorithms from the code: [allreduce_test.cu](https://github.com/microsoft/mscclpp/blob/main/test/mscclpp-test/allreduce_test.cu) and [allgather_test.cu](https://github.com/microsoft/mscclpp/blob/main/test/mscclpp-test/allgather_test.cu).

```bash
$ mpirun --bind-to numa -np 8 ./test/mscclpp-test/allreduce_test_perf -b 3m -e 48m -G 100 -n 100 -w 20 -f 2 -k 5
```

*NOTE: a few algorithms set a condition on the total data size, such as to be a multiple of 3. If the condition is unmet, the command will throw a regarding error.*

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

## NCCL over MSCCL++

We implement [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html) APIs using MSCCL++. How to use:

1. [Build MSCCL++ from source](#build-from-source).
2. Replace your `libnccl.so` library with `libmscclpp_nccl.so`, which is compiled under `./build/apps/nccl/` directory.

For example, you can run [nccl-tests](https://github.com/NVIDIA/nccl-tests) using `libmscclpp_nccl.so` as follows, where `MSCCLPP_BUILD` is your MSCCL++ build directory.

```bash
mpirun -np 8 --bind-to numa --allow-run-as-root -x LD_PRELOAD=$MSCCLPP_BUILD/apps/nccl/libmscclpp_nccl.so ./build/all_reduce_perf -b 1K -e 256M -f 2 -d half -G 20 -w 10 -n 50
```

If MSCCL++ is built on AMD platforms, `libmscclpp_nccl.so` would replace the [RCCL](https://github.com/ROCm/rccl) library (i.e., `librccl.so`).

See limitations of the current NCCL over MSCCL++ from [here](../design/nccl-over-mscclpp.md#limitations).
