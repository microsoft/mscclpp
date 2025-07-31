ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG SSH_PORT=22345

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

RUN rm -r /etc/apt/sources.list.d/cuda-* && \
    apt-get update && \
    apt install -y --no-install-recommends \
        clang-format \
        openssh-server \
        gdb \
        doxygen \
        graphviz \
        && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

RUN python3 -m pip install --no-cache-dir \
    black \
    pytest \
    breathe \
    sphinx_rtd_theme \
    myst_parser \
    sphinxcontrib.mermaid

RUN sed -i "s/^Port 22/Port ${SSH_PORT}/" /etc/ssh/sshd_config && \
    mkdir -p /home/$USERNAME/.ssh && \
    ssh-keygen -t rsa -f /home/$USERNAME/.ssh/id_rsa -N "" -q && \
    cat /home/$USERNAME/.ssh/id_rsa.pub >> /home/$USERNAME/.ssh/authorized_keys && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

USER $USERNAME
