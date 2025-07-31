#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=$(dirname "$(realpath "$0")")/..
TMP_BUILD_DIR=$(mktemp -d)
INSTALL_DIR=/usr
NVIDIA=false
AMD=false

usage() {
    echo "Usage: $0 <nvidia|amd> [install_dir]"
    echo "  nvidia       Install for NVIDIA platforms"
    echo "  amd          Install for AMD platforms"
    echo "  install_dir  Directory to install to (default: /usr)"
}

if [ ! -d "$TMP_BUILD_DIR" ]; then
    echo "Error: Failed to create temporary build directory."
    exit 1
fi

# Parse arguments
if [ $# -lt 1 ]; then
    usage
    exit 1
fi
case "$1" in
    nvidia)
        NVIDIA=true
        ;;
    amd)
        AMD=true
        ;;
    *)
        echo "Error: Unknown argument '$1'"
        usage
        exit 1
        ;;
esac
if [ $# -ge 2 ]; then
    INSTALL_DIR="$2"
fi
if [ ! -d "$INSTALL_DIR" ]; then
    echo "Error: Install directory '$INSTALL_DIR' does not exist."
    exit 1
fi

trap 'rm -rf "$TMP_BUILD_DIR"' EXIT

pushd "$TMP_BUILD_DIR" || exit 1

if $AMD; then
    export CXX=/opt/rocm/bin/hipcc
    CMAKE="cmake -DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_ROCM=ON"
elif $NVIDIA; then
    CMAKE="cmake -DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_CUDA=ON"
else
    echo "Error: No valid platform specified."
    exit 1
fi

$CMAKE \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF \
    -DMSCCLPP_BUILD_TESTS=OFF \
    "$PROJECT_ROOT"

make -j$(nproc)

sudo make install/fast

popd || exit 1

echo "Installation completed successfully."
