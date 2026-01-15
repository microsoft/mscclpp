# Quick Start

(prerequisites)=
## Prerequisites

* GPUs
    * NVIDIA CUDA architecture 7.0 (Volta) or later, or AMD CDNA 2 architecture (GFX90a) or later are required. Features are more thoroughly tested on CUDA architecture 8.0 (Ampere) or later and AMD CDNA 3 architecture (GFX942) or later.
    * A part of the features require GPUs to be connected peer-to-peer (through NVLink/xGMI or under the same PCIe switch).
        * On NVIDIA platforms, check the connectivity via `nvidia-smi topo -m`. If the output shows `NV#` or `PIX`, it means the GPUs are connected peer-to-peer.
        * On AMD platforms, check the connectivity via `rocm-smi --showtopohops`. If the output shows `1`, it means the GPUs are connected peer-to-peer.
    * Below are example systems that meet the requirements:
        * Azure SKUs
            * [ND_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/nda100-v4-series)
            * [NDm_A100_v4](https://learn.microsoft.com/en-us/azure/virtual-machines/ndm-a100-v4-series)
            * [ND_H100_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nd-h100-v5-series)
        * Non-Azure Systems
            * NVIDIA A100 GPUs + CUDA >= 11.8
            * NVIDIA H100 GPUs + CUDA >= 12.0
            * AMD MI250X GPUs + ROCm >= 5.7
            * AMD MI300X GPUs + ROCm >= 6.0
* OS
    * Tested on Ubuntu 20.04 and later
* Libraries
    * [libnuma](https://github.com/numactl/numactl)
        ```bash
        sudo apt-get install libnuma-dev
        ```
    * (Optional, for [building the Python module](#install-from-source-python-module)) Python >= 3.8 and Python Development Package
        ```bash
        sudo apt-get satisfy "python3 (>=3.8), python3-dev (>=3.8)"
        ```
        If you don't want to build Python module, you need to set `-DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF` in your `cmake` command (see details in [Install from Source](#install-from-source)).
    * (Optional, for benchmarks) MPI
* Others
    * For RDMA (InfiniBand or RoCE) support on NVIDIA platforms, [GPUDirect RDMA](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html#gpudirect-rdma-and-gpudirect-storage) should be supported by the system. See the detailed prerequisites from [this NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-rdma.html#common-prerequisites).
    * For NVLink SHARP (NVLS) support on NVIDIA platforms, the Linux kernel version should be 5.6 or above.

(docker-images)=
## Docker Images

We provide docker images which package all prerequisites for MSCCL++. You can setup your dev environment with the following command. Note that our docker images don't contain MSCCL++ by default, so you need to build it from source inside the container (see [Install from Source](#install-from-source) below).

```bash
# For NVIDIA platforms
$ docker run -it --privileged --net=host --ipc=host --gpus all --name mscclpp-dev ghcr.io/microsoft/mscclpp/mscclpp:base-dev-cuda12.8 bash
# For AMD platforms
$ docker run -it --privileged --net=host --ipc=host --security-opt=seccomp=unconfined --group-add=video --name mscclpp-dev ghcr.io/microsoft/mscclpp/mscclpp:base-dev-rocm6.2 bash
```

See all available images [here](https://github.com/microsoft/mscclpp/pkgs/container/mscclpp%2Fmscclpp).

(install-from-source)=
## Install from Source

If you want to install only the Python module, you can skip this section and go to [Install from Source (Python Module)](#install-from-source-python-module).

CMake 3.25 or later is required.

```bash
$ git clone https://github.com/microsoft/mscclpp.git
$ mkdir -p mscclpp/build && cd mscclpp/build
```

For NVIDIA platforms, build MSCCL++ as follows. Replace `/usr` with your desired installation path.

```bash
# For NVIDIA platforms
$ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
$ make -j$(nproc)
```

For AMD platforms, use HIPCC instead of the default C++ compiler. The HIPCC path is usually `/opt/rocm/bin/hipcc` in official ROCm installations. If the path is different in your environment, please change it accordingly.

```bash
# For AMD platforms
$ CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
$ make -j$(nproc)
```

After build succeeds, install the headers and binaries.

```bash
$ sudo make install
```

```{tip}
There are a few optional CMake options you can set:
- `-DMSCCLPP_GPU_ARCHS=<arch-list>`: Specify the GPU architectures to build for. For example, `-DMSCCLPP_GPU_ARCHS="80,90"` for NVIDIA A100 and H100 GPUs, `-DMSCCLPP_GPU_ARCHS=gfx942` for AMD MI300x GPU.
- `-DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON`: If the build environment doesn't have GPUs and only has CUDA installed, you can set these options to bypass GPU checks and use CUDA APIs. This is useful for building on CI systems or environments without GPUs.
- `-DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_ROCM=ON`: If the build environment doesn't have GPUs and only has ROCm installed, you can set these options to bypass GPU checks and use ROCm APIs.
- `-DMSCCLPP_USE_IB=OFF`: Don't build InfiniBand support.
- `-DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF`: Don't build the Python module.
- `-DMSCCLPP_BUILD_TESTS=OFF`: Don't build the tests.
- `-DMSCCLPP_BUILD_APPS_NCCL=OFF`: Don't build the NCCL API.
```

(install-from-source-python-module)=
## Install from Source (Python Module)

Python 3.8 or later is required.

```bash
# For NVIDIA platforms
$ python -m pip install .
# For AMD platforms, set the C++ compiler to HIPCC
$ CXX=/opt/rocm/bin/hipcc python -m pip install .
```

(vscode-dev-container)=
## VSCode Dev Container

If you are using VSCode, you can use our VSCode Dev Container that automatically launches a development environment and installs MSCCL++ in it. Steps to use our VSCode Dev Container:

1. Open the MSCCL++ repository in VSCode.
2. Make sure your Docker is running.
3. Make sure you have the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed in VSCode.
4. Open the command palette with `Ctrl`+`Shift`+`P` and select
   `Dev Containers: Rebuild and Reopen in Container`.
5. Wait for the container to build and open (may take a few minutes).

```{note}
- Our Dev Container is set up for NVIDIA GPUs by default. If you are using AMD GPUs, you need to copy [`devcontainer_amd.json`](https://github.com/microsoft/mscclpp/blob/main/.devcontainer/devcontainer_amd.json) to [`devcontainer.json`](https://github.com/microsoft/mscclpp/blob/main/.devcontainer/devcontainer.json).
- Our Dev Container runs an SSH server over the host network and the port number is `22345` by default. You can change the port number by modifying the `SSH_PORT` argument in the [`devcontainer.json`](https://github.com/microsoft/mscclpp/blob/main/.devcontainer/devcontainer.json) file.
- Our Dev Container uses a non-root user `devuser` by default, but note that you may need the root privileges to enable all hardware features of the GPUs inside the container. `devuser` is already configured to have `sudo` privileges without a password.
```

For more details on how to use the Dev Container, see the [Dev Containers tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial).

## Unit Tests

`unit_tests` require one GPU on the system. It only tests operation of basic components.

```bash
$ make -j unit_tests
$ ./bin/unit_tests
```

For thorough testing of MSCCL++ features, we need to use `mp_unit_tests` that require at least two GPUs on the system. `mp_unit_tests` also requires MPI to be installed on the system. For example, the following commands compile and run `mp_unit_tests` with two processes (two GPUs). The number of GPUs can be changed by changing the number of processes.

```bash
$ make -j mp_unit_tests
$ mpirun -np 2 ./bin/mp_unit_tests
```

To run `mp_unit_tests` with more than two nodes, you need to specify the `-ip_port` argument that is accessible from all nodes. For example:

```bash
$ mpirun -np 16 -npernode 8 -hostfile hostfile ./bin/mp_unit_tests -ip_port 10.0.0.5:50000
```

## Performance Benchmark

### Python Benchmark

[Install the MSCCL++ Python package](#install-from-source-python-module) and run our Python AllReduce benchmark as follows. It requires MPI on the system.

```bash
# Choose `requirements_*.txt` according to your CUDA/ROCm version.
$ python3 -m pip install -r ./python/requirements_cuda12.txt
$ mpirun -tag-output -np 8 python3 ./python/mscclpp_benchmark/allreduce_bench.py
```

(nccl-benchmark)=
### NCCL/RCCL Benchmark over MSCCL++

We implement [NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html) APIs using MSCCL++. How to use:

1. [Build MSCCL++ from source](#install-from-source).
2. Replace your `libnccl.so` library with `libmscclpp_nccl.so`, which is compiled under `./build/lib/` directory.

For example, you can run [nccl-tests](https://github.com/NVIDIA/nccl-tests) using `libmscclpp_nccl.so` as follows, where `MSCCLPP_BUILD` is your MSCCL++ build directory.

```bash
mpirun -np 8 --bind-to numa --allow-run-as-root -x LD_PRELOAD=$MSCCLPP_BUILD/lib/libmscclpp_nccl.so ./build/all_reduce_perf -b 1K -e 256M -f 2 -d half -G 20 -w 10 -n 50
```

If MSCCL++ is built on AMD platforms, `libmscclpp_nccl.so` would replace the [RCCL](https://github.com/ROCm/rccl) library (i.e., `librccl.so`).

MSCCL++ also supports fallback to NCCL/RCCL collectives by adding following environment variables.
```bash
-x MSCCLPP_ENABLE_NCCL_FALLBACK=TRUE
-x MSCCLPP_NCCL_LIB_PATH=/path_to_nccl_lib/libnccl.so (or /path_to_rccl_lib/librccl.so for AMD platforms)
-x MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION="list of collective name[s]"
```

The value `"list of collective name[s]"` can be a combination of collectives, such as `"allgather"`, `"allreduce"`, `"broadcast"`, and `"reducescatter"`. Alternatively, it can simply be set to `"all"` to enable fallback for all these collectives.
By default, if the parameter `MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION` is not specified, `"all"` will be applied.

Example 1, Allreduce will fallback to NCCL ncclAllReduce since allreduce is in the fallback list.
```bash
mpirun -np 8 --bind-to numa --allow-run-as-root -x LD_PRELOAD=$MSCCLPP_BUILD/lib/libmscclpp_nccl.so -x MSCCLPP_ENABLE_NCCL_FALLBACK=TRUE -x MSCCLPP_NCCL_LIB_PATH=$NCCL_BUILD/lib/libnccl.so -x MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION="allreduce,allgather" ./build/all_reduce_perf -b 1K -e 256M -f 2 -d half -G 20 -w 10 -n 50
```

Example 2, ReduceScatter will still use msccl++ implementation since reducescatter is not in the fallbacklist.
```bash
mpirun -np 8 --bind-to numa --allow-run-as-root -x LD_PRELOAD=$MSCCLPP_BUILD/lib/libmscclpp_nccl.so -x MSCCLPP_ENABLE_NCCL_FALLBACK=TRUE -x MSCCLPP_NCCL_LIB_PATH=$NCCL_BUILD/lib/libnccl.so -x MSCCLPP_FORCE_NCCL_FALLBACK_OPERATION="broadcast" -x MSCCLPP_EXECUTION_PLAN_DIR=/$PATH_TO_EXECUTION_PLANS/execution-files ./build/reduce_scatter_perf -b 1K -e 256M -f 2 -d half -G 20 -w 10 -n 50
```

On AMD platforms, you need to add `RCCL_MSCCL_ENABLE=0` to avoid conflicts with the fallback features.

**NOTE:** We also provide an NCCL audit shim library that can be used as a drop-in replacement for `libnccl.so` without modifying the original application. Set `LD_PRELOAD` as a global environment variable will cause applications to load cuda libraries from the host system, which may lead to errors in some environments (such as building pipeline in the CPU machine). To avoid this, you can use the audit shim library instead of setting `LD_PRELOAD` directly.
```bash
export LD_AUDIT=$MSCCLPP_INSTALL_DIR/libmscclpp_audit_nccl.so
export LD_LIBRARY_PATH=$MSCCLPP_INSTALL_DIR:$LD_LIBRARY_PATH
torchrun --nnodes=1 --nproc_per_node=8 your_script.py
```

## Version Tracking

The MSCCL++ Python package includes comprehensive version tracking that captures git repository information at build time. This feature allows users to identify the exact source code version of their installed package.

### Version Format

The package version includes the git commit hash directly in the version string for development builds:
- **Release version**: `0.7.0`
- **Development version**: `mscclpp-0.8.0.post1.dev0+gc632fee37.d20251007`

### Checking Version Information

After installation, you can check the version information in several ways:

**From Python:**
```python
import mscclpp

# Access individual attributes
print(f"Version: {mscclpp.__version__}")           # Full version with commit
Version: 0.8.0.post1.dev0+gc632fee37.d20251007

# Get as dictionary
mscclpp.version
{'version': '0.8.0.post1.dev0+gc632fee37.d20251007', 'git_commit': 'g50382c567'}
```

