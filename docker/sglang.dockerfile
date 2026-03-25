ARG BASE_IMAGE
FROM ${BASE_IMAGE}

LABEL maintainer="MSCCL++"
LABEL org.opencontainers.image.source=https://github.com/microsoft/mscclpp

# Install cmake (not in base image)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        cmake \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

# Create Python venv
RUN python3 -m venv /root/venv && \
    echo 'source /root/venv/bin/activate' >> /root/.bashrc
ENV PATH="/root/venv/bin:${PATH}"

# Install SGLang
RUN pip install --upgrade pip && \
    pip install uv && \
    uv pip install sglang

WORKDIR /
