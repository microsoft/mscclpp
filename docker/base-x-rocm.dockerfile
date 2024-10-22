ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source=https://github.com/microsoft/mscclpp

ENV DEBIAN_FRONTEND=noninteractive

ENV RCCL_VERSION=rocm-6.2.0
ARG ARCH=gfx942
ENV ARCH_TARGET=${ARCH}
RUN cd /tmp && \
    git clone --branch ${RCCL_VERSION} --depth 1  https://github.com/ROCm/rccl.git && \
    cd rccl && \
    ./install.sh --prefix=/opt/rocm --amdgpu_targets ${ARCH_TARGET} && \
    cd .. && \
    rm -rf /tmp/rccl

WORKDIR /
