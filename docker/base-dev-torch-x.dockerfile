# Image for the unified EP benchmark pipeline (.azure-pipelines/ep-benchmark.yml).
# Adds PyTorch to a base-dev image. NCCL-EP and DeepEP are installed by the pipeline
# on every run so that it always benchmarks their latest releases.
ARG BASE_IMAGE=ghcr.io/microsoft/mscclpp/mscclpp:base-dev-cuda13.3
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source=https://github.com/microsoft/mscclpp

# PyTorch does not publish CUDA 13.3 wheels; cu130 wheels work with CUDA 13.x.
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV CUDA_HOME=/usr/local/cuda

RUN pip install --no-cache-dir "torch==${TORCH_VERSION}" \
        --index-url ${TORCH_INDEX_URL} --extra-index-url https://pypi.org/simple && \
    pip install --no-cache-dir ninja

# Required to build DeepEP.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libdw-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -c "from importlib.metadata import version; print('torch', version('torch'))" && \
    test -f /usr/include/elfutils/libdwfl.h

WORKDIR /
