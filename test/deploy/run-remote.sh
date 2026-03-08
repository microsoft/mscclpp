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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTFILE="${SCRIPT_DIR}/hostfile_ci"
SSH_OPTION="StrictHostKeyChecking=no"
KeyFilePath="${SSHKEYFILE_SECUREFILEPATH}"

USE_DOCKER=true
USE_LOG=true
TARGET_HOST=""
REMOTE_USER=""

usage() {
    echo "Usage: $0 [--no-docker] [--no-log] [--hostfile <path>] [--host <name>] [--user <name>] < <command_script>" >&2
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

if $USE_LOG; then
    HOST="${TARGET_HOST:-$(head -1 "${HOSTFILE}")}"
    HOST="${HOST##*@}"
    : > "${HOST}"
    tail -f "${HOST}" &
    CHILD_PID=$!
    trap "kill $CHILD_PID 2>/dev/null" EXIT
fi

if $USE_DOCKER; then
    parallel-ssh -t 0 "${PSSH_TARGET_ARGS[@]}" "${PSSH_USER_ARGS[@]}" -x "-i ${KeyFilePath}" -o . \
    -O "$SSH_OPTION" "sudo docker exec -t mscclpp-test bash -c \"set -euxo pipefail; pushd /root/mscclpp >/dev/null; trap 'popd >/dev/null' EXIT; CMD_B64='${CMD_B64}'; printf '%s' \\\"\\\$CMD_B64\\\" | base64 -d | bash -euxo pipefail\""
else
    parallel-ssh -i -t 0 "${PSSH_TARGET_ARGS[@]}" "${PSSH_USER_ARGS[@]}" -x "-i ${KeyFilePath}" \
        -O "$SSH_OPTION" "set -euxo pipefail; CMD_B64='${CMD_B64}'; printf '%s' \"\$CMD_B64\" | base64 -d | bash -euxo pipefail"
fi
