ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG USERNAME=mscclpp
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG SSH_PORT=22345

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

RUN apt-get update && \
    apt install -y --no-install-recommends \
        openssh-server \
        gdb \
        && \
    apt-get autoremove && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

RUN sed -i "s/^Port 22/Port ${SSH_PORT}/" /etc/ssh/sshd_config && \
    mkdir -p /home/$USERNAME/.ssh && \
    ssh-keygen -t rsa -f /home/$USERNAME/.ssh/id_rsa -N "" -q && \
    cat /home/$USERNAME/.ssh/id_rsa.pub >> /home/$USERNAME/.ssh/authorized_keys && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

USER $USERNAME
