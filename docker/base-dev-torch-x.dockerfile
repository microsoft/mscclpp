# Benchmark-only image for the unified EP benchmark (python/mscclpp_benchmark/ep).
# Adds PyTorch on top of a base-dev image. Not used by the regular test pipelines.
#
# NCCL-EP and DeepEP are intentionally not included: the benchmark pipeline installs
# their latest versions at the start of every run (test/deploy/install-ep-backends.sh).
#
# When using this image:
# - Install MSCCL++ without the `ep` extra (e.g. `pip install ".[cuda13,benchmark,test]"`).
#   The `ep` extra makes pip re-resolve PyTorch's NCCL pin and can downgrade the NCCL
#   that NCCL-EP and DeepEP need. The EP module is built either way.
# - On GB200 hosts with Data Direct RoCE devices (mlx5_bond_*), NCCL crashes in IB
#   detection with this image's OFED userspace. Single-node runs need NCCL_IB_DISABLE=1
#   (set by test/deploy/run-ep-benchmark.sh).
ARG BASE_IMAGE=ghcr.io/microsoft/mscclpp/mscclpp:base-dev-cuda13.3
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source=https://github.com/microsoft/mscclpp

# PyTorch has no CUDA 13.3 wheels; the cu130 wheels run on any CUDA 13.x toolkit.
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

# NCCL-EP compiles kernels at runtime and DeepEP is built from source in the pipeline;
# both locate the toolkit through CUDA_HOME.
ENV CUDA_HOME=/usr/local/cuda

RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" \
        --index-url ${TORCH_INDEX_URL} --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir ninja

# System headers needed to build DeepEP (its DeepJIT dependency uses elfutils).
RUN apt-get update && \
    apt-get install -y --no-install-recommends libdw-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -c "from importlib.metadata import version; print('torch', version('torch'))" && \
    test -f /usr/include/elfutils/libdwfl.h

WORKDIR /
