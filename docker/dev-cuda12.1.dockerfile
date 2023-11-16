FROM ghcr.io/microsoft/mscclpp/mscclpp:base-cuda12.1

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source https://github.com/microsoft/mscclpp

ENV MSCCLPP_SRC_DIR="/tmp/mscclpp" \
    CMAKE_VERSION="3.26.4"

ADD . ${MSCCLPP_SRC_DIR}
WORKDIR ${MSCCLPP_SRC_DIR}

# Install cmake 3.26.4
ENV CMAKE_HOME="/tmp/cmake-${CMAKE_VERSION}-linux-x86_64" \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
RUN curl -L ${CMAKE_URL} -o ${CMAKE_HOME}.tar.gz && \
    tar xzf ${CMAKE_HOME}.tar.gz -C /usr/local
ENV PATH="/usr/local/cmake-${CMAKE_VERSION}-linux-x86_64/bin:${PATH}"

# Install pytest & dependencies
RUN python3 -m pip install --no-cache-dir -r python/test/requirements_cu12.txt

# Set PATH
RUN echo PATH="${PATH}" > /etc/environment

# Cleanup
WORKDIR /
RUN rm -rf ${MSCCLPP_SRC_DIR}
