ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source=https://github.com/microsoft/mscclpp

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        htop \
        vim \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Install lcov 2.2
RUN LCOV_VERSION="2.2" && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cpanminus \
        gcc \
        make \
        perl \
        && \
    cpanm --notest \
        Capture::Tiny \
        DateTime \
        JSON::XS \
        Memory::Process \
        TimeDate \
        && \
    cd /tmp && \
    curl -L https://github.com/linux-test-project/lcov/releases/download/v${LCOV_VERSION}/lcov-${LCOV_VERSION}.tar.gz -o lcov.tar.gz && \
    tar xzf lcov.tar.gz && \
    cd lcov-${LCOV_VERSION} && \
    make install && \
    cd / && rm -rf /tmp/lcov* && \
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

# Install GDRCopy userspace library for CUDA targets
ARG TARGET="cuda13.0"
RUN if echo "$TARGET" | grep -q "^cuda"; then \
        GDRCOPY_VERSION="2.5.2" && \
        apt-get update -y && \
        apt-get install -y --no-install-recommends devscripts debhelper fakeroot pkg-config dkms && \
        cd /tmp && \
        curl -L https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz -o gdrcopy.tar.gz && \
        tar xzf gdrcopy.tar.gz && \
        cd gdrcopy-${GDRCOPY_VERSION}/packages && \
        ./build-deb-packages.sh -k -t && \
        dpkg -i libgdrapi_*.deb && \
        cd / && rm -rf /tmp/gdrcopy* && \
        apt-get autoremove -y && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* /tmp/*; \
    fi

# Install ROCm-specific packages if building for ROCm
RUN if echo "$TARGET" | grep -q "^rocm"; then \
        apt-get update -y && \
        apt-get install -y hipblas hipsparse rocsparse rocrand hiprand rocthrust rocsolver rocfft hipfft hipcub rocprim rccl roctracer-dev && \
        apt-get autoremove -y && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/* /tmp/*; \
    fi

# Create Python venv
RUN python3 -m venv /root/venv && \
    echo 'source /root/venv/bin/activate' >> /root/.bashrc
ENV PATH="/root/venv/bin:${PATH}"

# Install Python dependencies
ADD . /tmp/mscclpp
WORKDIR /tmp/mscclpp
RUN target_type=$(echo $TARGET | sed 's/\.[0-9]*$//') && \
    if echo "$TARGET" | grep -q "^rocm"; then \
        export CUPY_INSTALL_USE_HIP=1 && export ROCM_HOME=/opt/rocm; \
    fi && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r python/requirements_${target_type}.txt && \
    pip install --no-cache-dir coverage xlsxwriter

# Cleanup
RUN rm -rf /tmp/mscclpp
WORKDIR /
