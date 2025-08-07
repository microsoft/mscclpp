# MSCCL++ Coding Agent Instructions

## Repository Overview

MSCCL++ is a GPU-driven communication stack for scalable AI applications, designed to provide efficient and customizable inter-GPU communication for distributed GPU applications. This repository contains:

- **Language**: C++17 with CUDA/HIP extensions, Python bindings
- **Size**: ~150 source files across multiple directories  
- **Target**: NVIDIA CUDA 11.8+ or AMD ROCm 5.7+ platforms
- **Purpose**: High-performance GPU communication primitives and collective operations
- **Architecture**: Multi-layer abstractions (low-level GPU kernels to high-level Python APIs)

## Build Instructions

### Prerequisites
Always install these dependencies first:
```bash
sudo apt-get update && sudo apt-get install -y libnuma-dev python3-dev cmake build-essential
```

### Building the Project

**ALWAYS use these exact command sequences for reliable builds:**

#### For NVIDIA Platforms:
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
make -j$(nproc)
sudo make install
```

#### For AMD Platforms:
```bash
mkdir -p build && cd build  
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
make -j$(nproc)
sudo make install
```

#### For CI/Testing Environments (No GPU):
```bash
mkdir -p build && cd build
# For NVIDIA
cmake -DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON -DCMAKE_BUILD_TYPE=Release ..
# For AMD  
CXX=/opt/rocm/bin/hipcc cmake -DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_ROCM=ON -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Build Time**: Full builds take 5-10 minutes. Always use `make -j$(nproc)` for parallel compilation.

**Common Build Options:**
- `-DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF` - Skip Python bindings (use if Python build fails)
- `-DMSCCLPP_BUILD_TESTS=OFF` - Skip tests (faster builds)
- `-DMSCCLPP_BUILD_APPS_NCCL=OFF` - Skip NCCL interface
- `-DMSCCLPP_GPU_ARCHS="80,90"` - Specify GPU architectures explicitly

### Python Module Installation
**ALWAYS install Python module separately after C++ build:**
```bash
# NVIDIA
python -m pip install .
# AMD
CXX=/opt/rocm/bin/hipcc python -m pip install .
```

## Testing

### Run Tests in This Order:

1. **Unit Tests** (requires 1 GPU):
```bash
make -j unit_tests
./test/unit_tests
```

2. **Multi-Process Tests** (requires 2+ GPUs, MPI):
```bash  
make -j mp_unit_tests
mpirun -np 2 ./test/mp_unit_tests
# For more GPUs: mpirun -np 8 ./test/mp_unit_tests
```

3. **Python Tests** (requires MPI, Python bindings):
```bash
pip install -r python/requirements_cuda12.txt  # or requirements_cuda11.txt, requirements_rocm6.txt
mpirun -np 8 python3 -m pytest python/test/test_mscclpp.py -x
```

**Test Time**: Unit tests ~30 seconds, multi-process tests ~2-3 minutes.

## Linting & Code Formatting

**ALWAYS run linting before committing:**

```bash
# Install tools first
pip install black
sudo apt-get install clang-format

# Check code formatting (dry run)
bash ./tools/lint.sh cpp dry
bash ./tools/lint.sh py dry

# Apply formatting
bash ./tools/lint.sh cpp
bash ./tools/lint.sh py
```

**Note**: Current codebase has some existing linting violations. Only fix violations in files you modify.

## Project Architecture & Key Directories

### Source Code Layout:
- `src/` - Core C++ implementation (communicator, channels, memory management)
- `include/mscclpp/` - Public headers and device interfaces  
- `python/mscclpp/` - Python bindings and high-level APIs
- `apps/nccl/` - NCCL API compatibility layer
- `test/` - Unit tests and integration tests
- `examples/tutorials/` - Tutorial code and examples

### Configuration Files:
- `CMakeLists.txt` - Main build configuration
- `pyproject.toml` - Python packaging configuration  
- `.clang-format` - C++ code formatting rules (Google style, 120 char width)
- `tools/lint.sh` - Unified linting script for C++ and Python

### Key Source Files:
- `src/communicator.cc` - Main communication setup
- `src/proxy.cc` - Host-side communication proxy
- `src/memory_channel.cc` - Direct memory-based channels
- `src/port_channel.cc` - Port-mapping-based channels
- `include/mscclpp/port_channel_device.hpp` - GPU device interfaces

## CI/CD & Validation

### GitHub Actions (run on every PR):
- `codeql-analysis.yml` - Security scanning
- `lint.yml` - Code formatting checks  
- `doc-build.yaml` - Documentation building

### Azure Pipelines (GPU testing):
- `ut.yml` - Unit tests on A100/H100 GPUs
- `integration-test.yml` - Multi-node integration tests
- `integration-test-rocm.yml` - AMD GPU testing

**Validation Commands You Can Run Locally:**
```bash
# Check for build issues
mkdir build && cd build && cmake .. && make -j

# Run formatters
bash tools/lint.sh cpp dry && bash tools/lint.sh py dry

# Quick unit test
make unit_tests && ./test/unit_tests
```

## Common Issues & Solutions

1. **GPU Detection Failure**: Always use `-DMSCCLPP_BYPASS_GPU_CHECK=ON` in CI environments
2. **CUDA/ROCm Not Found**: Install CUDA 11.8+ or ROCm 5.7+, or use bypass flags  
3. **Python Build Fails**: Use `-DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF` and build separately
4. **MPI Tests Fail**: Install OpenMPI: `sudo apt-get install libopenmpi-dev`
5. **Linting Errors**: Run `bash tools/lint.sh cpp` and `bash tools/lint.sh py` to auto-fix
6. **Build Times Out**: Use `make -j$(nproc)` and allow 10 minutes for full builds

## Dependencies Not Obvious from Structure

- **Critical**: libnuma-dev (always required, not optional)
- **MPI**: Required for multi-process tests and benchmarks
- **ninja_peermem**: Required on NVIDIA platforms for peer memory access
- **Docker**: Used extensively in CI, images available at `ghcr.io/microsoft/mscclpp/mscclpp`

## Quick Install Script
For automated installation:
```bash
# NVIDIA
bash tools/install.sh nvidia /usr
# AMD  
bash tools/install.sh amd /usr
```

## Trust These Instructions
These instructions are comprehensive and tested. Only search for additional information if:
- Build commands fail with specific errors not covered above
- New dependencies are introduced  
- CI pipelines are modified
- You need to understand specific algorithm implementations

For any standard development tasks (building, testing, linting, installing), follow these instructions exactly as written.