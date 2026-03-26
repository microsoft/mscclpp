ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source=https://github.com/microsoft/mscclpp

# Install cmake (not in base image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        htop \
        lcov \
        vim \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Install CMake 3.26.4
RUN OS_ARCH=$(uname -m) && \
    CMAKE_VERSION="3.26.4" && \
    CMAKE_HOME="/tmp/cmake-${CMAKE_VERSION}-linux-${OS_ARCH}" && \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${OS_ARCH}.tar.gz" && \
    curl -L ${CMAKE_URL} -o ${CMAKE_HOME}.tar.gz && \
    tar xzf ${CMAKE_HOME}.tar.gz -C /usr/local && \
    rm -rf ${CMAKE_HOME}.tar.gz && \
    ln -s /usr/local/cmake-${CMAKE_VERSION}-linux-${OS_ARCH}/bin/* /usr/bin/

# Create Python venv
RUN python3 -m venv /root/venv && \
    echo 'source /root/venv/bin/activate' >> /root/.bashrc
ENV PATH="/root/venv/bin:${PATH}"

# Install SGLang
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install sglang

WORKDIR /
