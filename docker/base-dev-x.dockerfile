ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source=https://github.com/microsoft/mscclpp

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
RUN ARCH=$(uname -m) && \
    CMAKE_VERSION="3.26.4" && \
    CMAKE_HOME="/tmp/cmake-${CMAKE_VERSION}-linux-${ARCH}" && \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${ARCH}.tar.gz" && \
    curl -L ${CMAKE_URL} -o ${CMAKE_HOME}.tar.gz && \
    tar xzf ${CMAKE_HOME}.tar.gz -C /usr/local && \
    rm -rf ${CMAKE_HOME}.tar.gz && \
    ln -s /usr/local/cmake-${CMAKE_VERSION}-linux-${ARCH}/bin/* /usr/bin/

# Install Python dependencies
ADD . /tmp/mscclpp
WORKDIR /tmp/mscclpp
ARG TARGET="cuda12.1"
RUN target_type=$(echo $TARGET | sed 's/\.[0-9]*$//') && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r python/requirements_${target_type}.txt

# Cleanup
RUN rm -rf /tmp/mscclpp
WORKDIR /
