ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ARG SSH_PORT=22345

# Create or modify the user
RUN if getent group $USER_GID > /dev/null; then \
        EXISTING_GROUP=$(getent group $USER_GID | cut -d: -f1); \
        if [ "$EXISTING_GROUP" != "$USERNAME" ]; then \
            groupmod -n $USERNAME $EXISTING_GROUP; \
        fi; \
    else \
        groupadd --gid $USER_GID $USERNAME; \
    fi && \
    if id -u $USER_UID > /dev/null 2>&1; then \
        EXISTING_USER=$(getent passwd $USER_UID | cut -d: -f1); \
        if [ "$EXISTING_USER" != "$USERNAME" ]; then \
            usermod -l $USERNAME -d /home/$USERNAME -m $EXISTING_USER; \
        fi; \
    else \
        useradd --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi && \
    usermod -g $USERNAME $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN rm -rf /etc/apt/sources.list.d/cuda-* && \
    apt-get update && \
    apt install -y --no-install-recommends \
        clang-format \
        openssh-server \
        gdb \
        doxygen \
        graphviz \
        && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/*

RUN python3 -m pip install --no-cache-dir \
    black \
    pytest \
    breathe \
    sphinx_rtd_theme \
    myst_parser \
    sphinxcontrib.mermaid

RUN sed -i '/^[#]*Port[[:space:]]\+[0-9]\+/d' /etc/ssh/sshd_config && \
    echo "Port ${SSH_PORT}" >> /etc/ssh/sshd_config && \
    mkdir -p /home/$USERNAME/.ssh && \
    ssh-keygen -t rsa -f /home/$USERNAME/.ssh/id_rsa -N "" -q && \
    cat /home/$USERNAME/.ssh/id_rsa.pub >> /home/$USERNAME/.ssh/authorized_keys && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

USER $USERNAME
