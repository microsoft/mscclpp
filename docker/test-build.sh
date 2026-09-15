#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
BUILD_SCRIPT="${SCRIPT_DIR}/build.sh"

assert_contains() {
    local output="$1"
    local expected="$2"
    if [[ "$output" != *"$expected"* ]]; then
        echo "Expected output to contain: ${expected}" >&2
        echo "$output" >&2
        exit 1
    fi
}

bash -n "$BUILD_SCRIPT" "$0"

arm_config=$("$BUILD_SCRIPT" --print-config --platform linux/arm64 cuda13.3)
assert_contains "$arm_config" "Target: cuda13.3"
assert_contains "$arm_config" "Base image: nvidia/cuda:13.3.1-devel-ubuntu24.04"
assert_contains "$arm_config" "Platform: linux/arm64"
assert_contains "$arm_config" "Image architecture: aarch64"
assert_contains "$arm_config" "OFED version: 24.10-3.2.5.0"

amd_config=$("$BUILD_SCRIPT" --platform=linux/amd64 --print-config cuda12.4)
assert_contains "$amd_config" "Base image: nvidia/cuda:12.4.1-devel-ubuntu22.04"
assert_contains "$amd_config" "Platform: linux/amd64"
assert_contains "$amd_config" "Image architecture: x86_64"

if "$BUILD_SCRIPT" --print-config --platform linux/s390x cuda13.3 >/dev/null 2>&1; then
    echo "Unsupported platform was accepted" >&2
    exit 1
fi

if "$BUILD_SCRIPT" --print-config cuda13.3.1 >/dev/null 2>&1; then
    echo "Unsupported target was accepted" >&2
    exit 1
fi

if "$BUILD_SCRIPT" --print-config >/dev/null 2>&1; then
    echo "Missing target was accepted" >&2
    exit 1
fi

for dockerfile in "${SCRIPT_DIR}/base-x.dockerfile" "${SCRIPT_DIR}/base-dev-x.dockerfile"; do
    grep -q '^ARG BASE_IMAGE$' "$dockerfile"
    grep -q '^FROM ${BASE_IMAGE}$' "$dockerfile"
done

echo "Docker build configuration tests passed"
