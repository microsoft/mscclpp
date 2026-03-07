#!/bin/bash
# Run a command on remote CI VMs via parallel-ssh.
# By default, runs inside the mscclpp-test docker container.
#
# Usage:
#   run-remote.sh [OPTIONS] <command>
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

while [[ "$1" == --* ]]; do
    case "$1" in
        --no-docker) USE_DOCKER=false; shift ;;
        --no-log)    USE_LOG=false; shift ;;
        --hostfile)
            if [ -z "$2" ]; then
                echo "Missing value for --hostfile" >&2
                exit 1
            fi
            HOSTFILE="$2"
            shift 2
            ;;
        --host)
            if [ -z "$2" ]; then
                echo "Missing value for --host" >&2
                exit 1
            fi
            TARGET_HOST="$2"
            shift 2
            ;;
        --user)
            if [ -z "$2" ]; then
                echo "Missing value for --user" >&2
                exit 1
            fi
            REMOTE_USER="$2"
            shift 2
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--no-docker] [--no-log] <command>" >&2
    exit 1
fi
CMD="$*"

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
    if [ -n "$TARGET_HOST" ]; then
        HOST="$TARGET_HOST"
    else
        HOST=$(head -1 "${HOSTFILE}")
        HOST="${HOST##*@}"
    fi
    : > "${HOST}"
    tail -f "${HOST}" &
    CHILD_PID=$!
    trap "kill $CHILD_PID 2>/dev/null" EXIT
fi

if $USE_DOCKER; then
    parallel-ssh -t 0 "${PSSH_TARGET_ARGS[@]}" "${PSSH_USER_ARGS[@]}" -x "-i ${KeyFilePath}" -o . \
        -O "$SSH_OPTION" "sudo docker exec -t mscclpp-test bash -c \"set -ex; pushd /root/mscclpp >/dev/null; trap 'popd >/dev/null' EXIT; ${CMD}\""
else
    parallel-ssh -i -t 0 "${PSSH_TARGET_ARGS[@]}" "${PSSH_USER_ARGS[@]}" -x "-i ${KeyFilePath}" \
        -O "$SSH_OPTION" "set -ex; ${CMD}"
fi
