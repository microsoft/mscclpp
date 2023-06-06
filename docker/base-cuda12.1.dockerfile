FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source https://github.com/microsoft/mscclpp

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libcap2 \
        libnuma-dev \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        wget \
        && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Install OFED
ENV OFED_VERSION=5.2-2.2.3.0
RUN cd /tmp && \
    wget -q https://content.mellanox.com/ofed/MLNX_OFED-${OFED_VERSION}/MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    tar xzf MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64.tgz && \
    MLNX_OFED_LINUX-${OFED_VERSION}-ubuntu20.04-x86_64/mlnxofedinstall --user-space-only --without-fw-update --force --all && \
    rm -rf /tmp/MLNX_OFED_LINUX-${OFED_VERSION}*

# Install OpenMPI
ENV OPENMPI_VERSION=4.1.5
RUN cd /tmp && \
    export ompi_v_parsed="$(echo ${OPENMPI_VERSION} | sed -E 's/^([0-9]+)\.([0-9]+)\..*/\1.\2/')" && \
    wget -q https://download.open-mpi.org/release/open-mpi/v${ompi_v_parsed}/openmpi-${OPENMPI_VERSION}.tar.gz && \
    tar xzf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/mpi && \
    make -j && \
    make install && \
    cd .. && \
    rm -rf /tmp/openmpi-${OPENMPI_VERSION}

ENV PATH="${PATH}" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/mpi/lib:/usr/local/cuda-12.1/compat:/usr/local/cuda-12.1/lib64"

RUN echo PATH="${PATH}" > /etc/environment && \
    echo LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" >> /etc/environment
