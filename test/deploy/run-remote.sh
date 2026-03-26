#!/bin/bash
# Run a command on remote CI VMs via parallel-ssh.
# By default, runs inside the mscclpp-test docker container.
#
# Usage:
#   run-remote.sh [OPTIONS] < <command_script>
#
# Options:
#   --no-docker   Run command directly on the host, not inside docker
#   --no-log      Don't tail the log file in the background
#   --hostfile    Override hostfile path (default: test/deploy/hostfile_ci)
#   --host        Run command on a single host (uses parallel-ssh -H)
#   --user        SSH user when using --host or custom hostfile
#   --container   Docker container name to exec into (default: mscclpp-test)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${SCRIPT_DIR}/hostfile_ci"
SSH_OPTION="StrictHostKeyChecking=no"
KeyFilePath="${SSHKEYFILE_SECUREFILEPATH}"

USE_DOCKER=true
USE_LOG=true
TARGET_HOST=""
REMOTE_USER=""
CONTAINER_NAME="mscclpp-test"

usage() {
    echo "Usage: $0 [--no-docker] [--no-log] [--hostfile <path>] [--host <name>] [--user <name>] [--container <name>] < <command_script>" >&2
}

require_value() {
    local opt="$1"
    local val="$2"
    if [ -z "$val" ]; then
        echo "Missing value for ${opt}" >&2
        exit 1
    fi
}

while [[ "$1" == --* ]]; do
    case "$1" in
        --no-docker) USE_DOCKER=false; shift ;;
        --no-log)    USE_LOG=false; shift ;;
        --hostfile)
            require_value "--hostfile" "${2-}"
            HOSTFILE="$2"
            shift 2
            ;;
        --host)
            require_value "--host" "${2-}"
            TARGET_HOST="$2"
            shift 2
            ;;
        --user)
            require_value "--user" "${2-}"
            REMOTE_USER="$2"
            shift 2
            ;;
        --container)
            require_value "--container" "${2-}"
            CONTAINER_NAME="$2"
            shift 2
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ $# -ne 0 ] || [ -t 0 ]; then
    usage
    exit 1
fi

CMD=$(cat)
if [ -z "$CMD" ]; then
    usage
    exit 1
fi
CMD_B64=$(printf '%s' "$CMD" | base64 | tr -d '\n')

PSSH_TARGET_ARGS=()
if [ -n "$TARGET_HOST" ]; then
    PSSH_TARGET_ARGS=(-H "$TARGET_HOST")
else
    PSSH_TARGET_ARGS=(-h "$HOSTFILE")
fi

PSSH_USER_ARGS=()
if [ -n "$REMOTE_USER" ]; then
    PSSH_USER_ARGS=(-l "$REMOTE_USER")
fi

PSSH_COMMON=(
    -t 0
    "${PSSH_TARGET_ARGS[@]}"
    "${PSSH_USER_ARGS[@]}"
    -x "-i ${KeyFilePath}"
    -O "$SSH_OPTION"
)

if $USE_DOCKER; then
    # If using the sglang container, launch it first
    if [ "${CONTAINER_NAME}" = "mscclpp-sglang-test" ]; then
        parallel-ssh -i "${PSSH_COMMON[@]}" \
            "sudo docker rm -f ${CONTAINER_NAME} 2>/dev/null; \
             sudo docker run -itd --name=${CONTAINER_NAME} --privileged --net=host --ipc=host --gpus=all -w /root -v /mnt:/mnt lmsysorg/sglang:latest bash"
    fi

    INNER="set -euxo pipefail;"
    INNER+=" cd /root/mscclpp;"
    INNER+=" export LD_LIBRARY_PATH=/root/mscclpp/build/lib:\\\$LD_LIBRARY_PATH;"
    INNER+=" CMD_B64='${CMD_B64}';"
    INNER+=" printf '%s' \\\"\\\$CMD_B64\\\" | base64 -d | bash -euxo pipefail"

    parallel-ssh -i "${PSSH_COMMON[@]}" \
        "sudo docker exec ${CONTAINER_NAME} bash -c \"${INNER}\""
else
    parallel-ssh -i "${PSSH_COMMON[@]}" \
        "set -euxo pipefail; CMD_B64='${CMD_B64}'; printf '%s' \"\$CMD_B64\" | base64 -d | bash -euxo pipefail"
fi
