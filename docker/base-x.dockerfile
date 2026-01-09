ARG BASE_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source=https://github.com/microsoft/mscclpp

ENV DEBIAN_FRONTEND=noninteractive
USER root

RUN rm -rf /opt/nvidia

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libcap2 \
        libnuma-dev \
        lsb-release \
        openssh-client \
        openssh-server \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        python3-venv \
        sudo \
        wget

# Install OFED
ARG OFED_VERSION=24.10-3.2.5.0
RUN cd /tmp && \
    OS_ARCH=$(uname -m) && \
    OS_VERSION=$(lsb_release -rs) && \
    OS_VERSION=ubuntu${OS_VERSION} && \
    wget -q https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-${OS_VERSION}-${OS_ARCH}.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-${OS_VERSION}-${OS_ARCH}.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-${OS_VERSION}-${OS_ARCH}/mlnxofedinstall --user-space-only --without-fw-update --without-ucx-cuda --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*

# Install OpenMPI (should be done after the OFED installation) & clean apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenmpi-dev \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# OpenMPI short link (for compatibility with old images)
RUN ln -s /usr/lib/x86_64-linux-gnu/openmpi /usr/local/mpi

ARG EXTRA_LD_PATH=
ENV LD_LIBRARY_PATH="${EXTRA_LD_PATH}:${LD_LIBRARY_PATH}"
RUN echo LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" >> /etc/environment

ENTRYPOINT []
WORKDIR /
