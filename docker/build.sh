#!/usr/bin/env bash

set -e

declare -A baseImageTable
baseImageTable=(
    ["cuda11.8"]="nvidia/cuda:11.8.0-devel-ubuntu20.04"
    ["cuda12.1"]="nvidia/cuda:12.1.1-devel-ubuntu20.04"
    ["cuda12.2"]="nvidia/cuda:12.2.2-devel-ubuntu20.04"
    ["cuda12.3"]="nvidia/cuda:12.3.2-devel-ubuntu20.04"
    ["cuda12.4"]="nvidia/cuda:12.4.1-devel-ubuntu22.04"
    ["cuda12.8"]="nvidia/cuda:12.8.1-devel-ubuntu22.04"
    ["cuda12.9"]="nvidia/cuda:12.9.1-devel-ubuntu22.04"
    ["cuda13.0"]="nvidia/cuda:13.0.2-devel-ubuntu24.04"
    ["rocm6.2"]="rocm/dev-ubuntu-22.04:6.2.2"
)

declare -A extraLdPathTable
extraLdPathTable=(
    ["cuda12.1"]="/usr/local/cuda-12.1/compat:/usr/local/cuda-12.1/lib64"
    ["cuda12.2"]="/usr/local/cuda-12.2/compat:/usr/local/cuda-12.2/lib64"
    ["cuda12.3"]="/usr/local/cuda-12.3/compat:/usr/local/cuda-12.3/lib64"
    ["rocm6.2"]="/opt/rocm/lib"
)

declare -A ofedVersionTable
ofedVersionTable=(
    ["cuda12.4"]="23.07-0.5.1.2"
    ["cuda12.8"]="24.10-1.1.4.0"
    ["cuda12.9"]="24.10-1.1.4.0"
    ["cuda13.0"]="24.10-3.2.5.0"
    ["rocm6.2"]="24.10-1.1.4.0"
)

TARGET=${1}
OS_ARCH=$(uname -m)

print_usage() {
    echo "Usage: $0 [cuda11.8|cuda12.1|cuda12.2|cuda12.3|cuda12.4|cuda12.8|cuda12.9|cuda13.0|rocm6.2]"
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

TAG_TMP="tmp-${TARGET}-${OS_ARCH}"
TAG_BASE="base-${TARGET}-${OS_ARCH}"
TAG_BASE_DEV="base-dev-${TARGET}-${OS_ARCH}"

docker build -t ${TAG_TMP} \
    -f docker/base-x.dockerfile \
    --build-arg BASE_IMAGE=${baseImageTable[${TARGET}]} \
    --build-arg EXTRA_LD_PATH=${extraLdPathTable[${TARGET}]} \
    --build-arg TARGET=${TARGET} \
    --build-arg OFED_VERSION=${OFED_VERSION} .

if [[ ${TARGET} == rocm* ]]; then
    echo "Building ROCm base image..."
else
    echo "Building CUDA base image..."
fi
docker tag ${TAG_TMP} ${TAG_BASE}
docker rmi --no-prune ${TAG_TMP}

docker build -t ${TAG_BASE_DEV} \
    -f docker/base-dev-x.dockerfile \
    --build-arg BASE_IMAGE=${TAG_BASE} \
    --build-arg TARGET=${TARGET} .

GHCR="ghcr.io/microsoft/mscclpp/mscclpp"
GHCR_TAG_BASE_DEV=${GHCR}:base-dev-${TARGET}
GHCR_TAG_BASE_DEV_ARCH=${GHCR}:base-dev-${TARGET}-${OS_ARCH}

echo "Successfully built images:"
echo "  - ${TAG_BASE}"
echo "  - ${TAG_BASE_DEV}"
echo ""
echo "To push the base-dev image to ghcr.io,"
echo ""
echo "0. Login to ghcr.io:"
echo ""
echo "    docker login ghcr.io"
echo ""
echo "1. Tag and push the arch-specific image:"
echo ""
echo "    docker tag ${TAG_BASE_DEV} ${GHCR_TAG_BASE_DEV_ARCH} && \\"
echo "    docker push ${GHCR_TAG_BASE_DEV_ARCH}"
echo ""
echo "2. Create or update the multi-arch manifest:"
echo ""
echo "   If \`${GHCR_TAG_BASE_DEV}\` already exists (adding another arch):"
echo ""
echo "    docker buildx imagetools create \\"
echo "        --tag ${GHCR_TAG_BASE_DEV} \\"
echo "        --append ${GHCR_TAG_BASE_DEV_ARCH}"
echo ""
echo "   If \`${GHCR_TAG_BASE_DEV}\` does not exist yet:"
echo ""
echo "    docker buildx imagetools create \\"
echo "        --tag ${GHCR_TAG_BASE_DEV} \\"
echo "        ${GHCR_TAG_BASE_DEV_ARCH}"
echo ""
