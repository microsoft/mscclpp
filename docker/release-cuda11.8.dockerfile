FROM ghcr.io/microsoft/mscclpp/mscclpp:base-cuda11.8

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source https://github.com/microsoft/mscclpp

ENV MSCCLPP_HOME="/usr/local/mscclpp" \
    MSCCLPP_SRC_DIR="/tmp/mscclpp" \
    CMAKE_VERSION="3.26.4"

# Download cmake 3.26.4
ENV CMAKE_HOME="/tmp/cmake-${CMAKE_VERSION}-linux-x86_64" \
    CMAKE_URL="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz"
RUN curl -L ${CMAKE_URL} -o ${CMAKE_HOME}.tar.gz && \
    tar xzf ${CMAKE_HOME}.tar.gz -C /tmp

# Install MSCCL++
ADD . ${MSCCLPP_SRC_DIR}
WORKDIR ${MSCCLPP_SRC_DIR}
RUN rm -rf build && \
    mkdir build && \
    cd build && \
    ${CMAKE_HOME}/bin/cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${MSCCLPP_HOME} .. && \
    make -j mscclpp && \
    make install/fast && \
    strip ${MSCCLPP_HOME}/lib/libmscclpp.so.[0-9]*.[0-9]*.[0-9]*

ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MSCCLPP_HOME}/lib"
RUN echo LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" >> /etc/environment

# Cleanup
WORKDIR /
RUN rm -rf ${CMAKE_HOME}* ${MSCCLPP_SRC_DIR}
