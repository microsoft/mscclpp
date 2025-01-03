#!/usr/bin/env bash

set -e

declare -A baseImageTable
baseImageTable=(
    ["cuda11.8"]="nvidia/cuda:11.8.0-devel-ubuntu20.04"
    ["cuda12.1"]="nvidia/cuda:12.1.1-devel-ubuntu20.04"
    ["cuda12.2"]="nvidia/cuda:12.2.2-devel-ubuntu20.04"
    ["cuda12.3"]="nvidia/cuda:12.3.2-devel-ubuntu20.04"
    ["cuda12.4"]="nvidia/cuda:12.4.1-devel-ubuntu22.04"
    ["rocm6.2"]="rocm/rocm-terminal:6.2.1"
)

declare -A extraLdPathTable
extraLdPathTable=(
    ["cuda11.8"]="/usr/local/cuda-11.8/lib64"
    ["cuda12.1"]="/usr/local/cuda-12.1/compat:/usr/local/cuda-12.1/lib64"
    ["cuda12.2"]="/usr/local/cuda-12.2/compat:/usr/local/cuda-12.2/lib64"
    ["cuda12.3"]="/usr/local/cuda-12.3/compat:/usr/local/cuda-12.3/lib64"
    ["rocm6.2"]="/opt/rocm/lib"
)

declare -A ofedVersionTable
ofedVersionTable=(
    ["cuda12.4"]="23.07-0.5.1.2"
)

GHCR="ghcr.io/microsoft/mscclpp/mscclpp"
TARGET=${1}

print_usage() {
    echo "Usage: $0 [cuda11.8|cuda12.1|cuda12.2|cuda12.3|cuda12.4|rocm6.2]"
}

if [[ ! -v "baseImageTable[${TARGET}]" ]]; then
    echo "Invalid target: ${TARGET}"
    print_usage
    exit 1
fi
echo "Target: ${TARGET}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

cd ${SCRIPT_DIR}/..

DEFAULT_OFED_VERSION="5.2-2.2.3.0"
OFED_VERSION=${ofedVersionTable[${TARGET}]}
if [[ -z ${OFED_VERSION} ]]; then
    OFED_VERSION=${DEFAULT_OFED_VERSION}
fi

docker build -t ${GHCR}-common:base-${TARGET} \
    -f docker/base-x.dockerfile \
    --build-arg BASE_IMAGE=${baseImageTable[${TARGET}]} \
    --build-arg EXTRA_LD_PATH=${extraLdPathTable[${TARGET}]} \
    --build-arg TARGET=${TARGET} \
    --build-arg OFED_VERSION=${OFED_VERSION} .

if [[ ${TARGET} == rocm* ]]; then
    echo "Building ROCm base image..."
    docker build -t ${GHCR}:base-${TARGET} \
        -f docker/base-x-rocm.dockerfile \
        --build-arg BASE_IMAGE=${GHCR}-common:base-${TARGET} \
        --build-arg EXTRA_LD_PATH=${extraLdPathTable[${TARGET}]} \
        --build-arg TARGET=${TARGET} \
        --build-arg ARCH="gfx942" .
else
    echo "Building CUDA base image..."
    docker tag ${GHCR}-common:base-${TARGET} ${GHCR}:base-${TARGET}
fi

docker build -t ${GHCR}:base-dev-${TARGET} \
    -f docker/base-dev-x.dockerfile \
    --build-arg BASE_IMAGE=${GHCR}:base-${TARGET} \
    --build-arg TARGET=${TARGET} .
