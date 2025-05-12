ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG USERNAME=mscclpp
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Dev tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    clang-format \
    htop \
    openssh-server \
    gdb \
    python3-dbg \
    && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*
RUN python3 -m pip install --no-cache-dir \
    black

USER $USERNAME
