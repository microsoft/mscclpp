#!/usr/bin/env bash

set -euo pipefail

declare -A baseImageTable
baseImageTable=(
    ["cuda12.4"]="nvidia/cuda:12.4.1-devel-ubuntu22.04"
    ["cuda12.8"]="nvidia/cuda:12.8.1-devel-ubuntu22.04"
    ["cuda12.9"]="nvidia/cuda:12.9.1-devel-ubuntu24.04"
    ["cuda13.0"]="nvidia/cuda:13.0.2-devel-ubuntu24.04"
    ["cuda13.3"]="nvidia/cuda:13.3.1-devel-ubuntu24.04"
    ["rocm6.2"]="rocm/dev-ubuntu-22.04:6.2.2"
    ["rocm7.2"]="rocm/dev-ubuntu-24.04:7.2.4"
)

declare -A extraLdPathTable
extraLdPathTable=(
    ["rocm6.2"]="/opt/rocm/lib"
    ["rocm7.2"]="/opt/rocm/lib"
)

declare -A ofedVersionTable
ofedVersionTable=(
    ["cuda12.4"]="23.07-0.5.1.2"
    ["cuda12.8"]="24.10-1.1.4.0"
    ["cuda12.9"]="24.10-1.1.4.0"
    ["cuda13.0"]="24.10-3.2.5.0"
    ["cuda13.3"]="24.10-3.2.5.0"
    ["rocm6.2"]="24.10-1.1.4.0"
    ["rocm7.2"]="24.10-3.2.5.0"
)

TARGET=""
DOCKER_PLATFORM=""
PRINT_CONFIG=false

print_usage() {
    echo "Usage: $0 [--platform linux/amd64|linux/arm64] [--print-config] [cuda12.4|cuda12.8|cuda12.9|cuda13.0|cuda13.3|rocm6.2|rocm7.2]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --platform" >&2
                print_usage >&2
                exit 1
            fi
            DOCKER_PLATFORM="$2"
            shift 2
            ;;
        --platform=*)
            DOCKER_PLATFORM="${1#*=}"
            shift
            ;;
        --print-config)
            PRINT_CONFIG=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            print_usage >&2
            exit 1
            ;;
        *)
            if [[ -n "$TARGET" ]]; then
                echo "Unexpected argument: $1" >&2
                print_usage >&2
                exit 1
            fi
            TARGET="$1"
            shift
            ;;
    esac
done

if [[ -z "$TARGET" || -z "${baseImageTable[$TARGET]-}" ]]; then
    echo "Invalid target: ${TARGET:-<none>}" >&2
    print_usage >&2
    exit 1
fi

if [[ -z "$DOCKER_PLATFORM" ]]; then
    case "$(uname -m)" in
        x86_64|amd64) DOCKER_PLATFORM="linux/amd64" ;;
        aarch64|arm64) DOCKER_PLATFORM="linux/arm64" ;;
        *)
            echo "Unsupported host architecture: $(uname -m)" >&2
            exit 1
            ;;
    esac
fi

case "$DOCKER_PLATFORM" in
    linux/amd64) OS_ARCH="x86_64" ;;
    linux/arm64) OS_ARCH="aarch64" ;;
    *)
        echo "Unsupported platform: ${DOCKER_PLATFORM}; expected linux/amd64 or linux/arm64" >&2
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "${SCRIPT_DIR}/.."

DEFAULT_OFED_VERSION="5.2-2.2.3.0"
OFED_VERSION="${ofedVersionTable[$TARGET]:-$DEFAULT_OFED_VERSION}"
BASE_IMAGE="${baseImageTable[$TARGET]}"
EXTRA_LD_PATH="${extraLdPathTable[$TARGET]-}"

echo "Target: ${TARGET}"
echo "Base image: ${BASE_IMAGE}"
echo "Platform: ${DOCKER_PLATFORM}"
echo "Image architecture: ${OS_ARCH}"
echo "OFED version: ${OFED_VERSION}"

if $PRINT_CONFIG; then
    exit 0
fi

TAG_TMP="tmp-${TARGET}-${OS_ARCH}"
TAG_BASE="base-${TARGET}-${OS_ARCH}"
TAG_BASE_DEV="base-dev-${TARGET}-${OS_ARCH}"

docker build --platform "${DOCKER_PLATFORM}" -t "${TAG_TMP}" \
    -f docker/base-x.dockerfile \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    --build-arg EXTRA_LD_PATH="${EXTRA_LD_PATH}" \
    --build-arg TARGET="${TARGET}" \
    --build-arg OFED_VERSION="${OFED_VERSION}" .

if [[ ${TARGET} == rocm* ]]; then
    echo "Building ROCm base image..."
else
    echo "Building CUDA base image..."
fi
docker tag "${TAG_TMP}" "${TAG_BASE}"
docker rmi --no-prune "${TAG_TMP}"

docker build --platform "${DOCKER_PLATFORM}" -t "${TAG_BASE_DEV}" \
    -f docker/base-dev-x.dockerfile \
    --build-arg BASE_IMAGE="${TAG_BASE}" \
    --build-arg TARGET="${TARGET}" .


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
echo "2. (Re)create the multi-arch manifest — ALWAYS list EVERY arch tag you have"
echo "   published for this target. \`imagetools create\` REPLACES the manifest,"
echo "   so omitting an arch silently drops it:"
echo ""
echo "    docker buildx imagetools create \\"
echo "        --tag ${GHCR_TAG_BASE_DEV} \\"
echo "        ${GHCR}:base-dev-${TARGET}-x86_64 \\"
echo "        ${GHCR}:base-dev-${TARGET}-aarch64"
echo ""
echo "   (Include only the arch tags that actually exist for this target — e.g."
echo "    ROCm targets are x86_64-only. Verify afterward:)"
echo ""
echo "    docker buildx imagetools inspect ${GHCR_TAG_BASE_DEV}"
echo ""