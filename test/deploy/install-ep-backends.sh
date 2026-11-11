#!/bin/bash
# Install the latest NCCL-EP and DeepEP for the unified EP benchmark. Run after MSCCL++ is installed.
#
# NCCL-EP requires the NCCL version it is validated with, while DeepEP requires a newer one.
# NCCL-EP's NCCL stays in the Python environment. DeepEP is built against the latest NCCL,
# installed into DEEPEP_NCCL_DIR, which run-ep-benchmark.sh adds to PYTHONPATH for DeepEP only.
#
# Environment overrides:
#   NCCL_EP_SPEC          pip requirement for NCCL-EP  (default: nccl-extensions[cu13])
#   CUDA_CORE_SPEC        pip requirement for cuda-core (default: cuda-core[cu13])
#   DEEPEP_NCCL_SPEC      pip requirement for DeepEP's NCCL (default: nvidia-nccl-cu13, i.e. latest)
#   DEEPEP_NCCL_DIR       install directory for DeepEP's NCCL (default: /opt/deepep-nccl)
#   DEEPEP_REPO           DeepEP git URL               (default: deepseek-ai/DeepEP on GitHub)
#   DEEPEP_REF            DeepEP branch, tag or commit (default: the repository's default branch)
#   TORCH_CUDA_ARCH_LIST  GPU architecture for DeepEP  (default: detected with nvidia-smi)
#   EP_BACKEND_ENV        file recording DeepEP's NCCL location for run-ep-benchmark.sh
#                         (default: /tmp/ep-backend-env)
#   EP_VERSIONS_FILE      JSON file recording installed versions (default: /tmp/ep-backend-versions.json)

set -euo pipefail

NCCL_EP_SPEC="${NCCL_EP_SPEC:-nccl-extensions[cu13]}"
CUDA_CORE_SPEC="${CUDA_CORE_SPEC:-cuda-core[cu13]}"
DEEPEP_NCCL_SPEC="${DEEPEP_NCCL_SPEC:-nvidia-nccl-cu13}"
DEEPEP_NCCL_DIR="${DEEPEP_NCCL_DIR:-/opt/deepep-nccl}"
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/deepseek-ai/DeepEP.git}"
DEEPEP_REF="${DEEPEP_REF:-}"
EP_BACKEND_ENV="${EP_BACKEND_ENV:-/tmp/ep-backend-env}"
EP_VERSIONS_FILE="${EP_VERSIONS_FILE:-/tmp/ep-backend-versions.json}"
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
fi
export TORCH_CUDA_ARCH_LIST
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "=== Installing latest NCCL-EP (${NCCL_EP_SPEC}) and its validated NCCL"
# NCCL-EP depends on cuda-core without its CUDA extra, so pip does not enforce cuda-core's
# cuda-bindings floor (e.g. cuda-core 1.3.0 needs cuda-bindings>=13.5). Request the extra here.
pip install --no-cache-dir --upgrade "${NCCL_EP_SPEC}" "${CUDA_CORE_SPEC}"

echo "=== Installing latest NCCL for DeepEP (${DEEPEP_NCCL_SPEC}) into ${DEEPEP_NCCL_DIR}"
rm -rf "${DEEPEP_NCCL_DIR}"
pip install --no-cache-dir --no-deps --target "${DEEPEP_NCCL_DIR}" "${DEEPEP_NCCL_SPEC}"
DEEPEP_NCCL_ROOT="${DEEPEP_NCCL_DIR}/nvidia/nccl"
DEEPEP_NCCL_VERSION="$(PYTHONPATH="${DEEPEP_NCCL_DIR}" python -c \
    "from importlib.metadata import version; print(version('${DEEPEP_NCCL_SPEC%%[<>=!~ ]*}'))")"

echo "=== Building latest DeepEP for TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
DEEPEP_SRC="$(mktemp -d)"
trap 'rm -rf "${DEEPEP_SRC}"' EXIT
if [ -n "${DEEPEP_REF}" ]; then
    git clone --quiet "${DEEPEP_REPO}" "${DEEPEP_SRC}"
    git -C "${DEEPEP_SRC}" checkout --quiet "${DEEPEP_REF}"
else
    git clone --quiet --depth 1 "${DEEPEP_REPO}" "${DEEPEP_SRC}"
fi
git -C "${DEEPEP_SRC}" submodule update --quiet --init --recursive --depth 1
DEEPEP_COMMIT="$(git -C "${DEEPEP_SRC}" log -1 --format='%h %cs %s')"
echo "DeepEP commit: ${DEEPEP_COMMIT}"
(
    cd "${DEEPEP_SRC}"
    EP_NCCL_ROOT_DIR="${DEEPEP_NCCL_ROOT}" MAX_JOBS="$(nproc)" python setup.py bdist_wheel
    pip install --no-cache-dir --no-deps --force-reinstall dist/*.whl
)

printf 'DEEPEP_PYTHONPATH=%s\n' "${DEEPEP_NCCL_DIR}" > "${EP_BACKEND_ENV}"

echo "=== Verifying EP backends"
# Run outside the repository so Python imports the installed packages.
cd /tmp
python -c "import torch, nccl.core, nccl.ep, mscclpp.ep"
PYTHONPATH="${DEEPEP_NCCL_DIR}" python -c "
import torch, deep_ep
loaded = {l.split()[-1] for l in open('/proc/self/maps') if 'libnccl' in l}
assert loaded == {'${DEEPEP_NCCL_ROOT}/lib/libnccl.so.2'}, f'DeepEP run loads unexpected NCCL: {loaded}'
assert hasattr(deep_ep, 'EPBuffer'), 'Installed DeepEP does not provide deep_ep.EPBuffer, which the benchmark requires'
"

DEEPEP_COMMIT="${DEEPEP_COMMIT}" DEEPEP_NCCL_VERSION="${DEEPEP_NCCL_VERSION}" \
EP_VERSIONS_FILE="${EP_VERSIONS_FILE}" python - <<'PY'
import json
import os
from importlib.metadata import version

import torch

versions = {
    "torch": version("torch"),
    "torch_cuda": torch.version.cuda,
    "mscclpp": version("mscclpp"),
    "nccl_ep": version("nccl-extensions"),
    "nccl4py": version("nccl4py"),
    "nccl_for_nccl_ep_and_mscclpp": version("nvidia-nccl-cu13"),
    "deep_ep": version("deep_ep"),
    "deep_ep_commit": os.environ["DEEPEP_COMMIT"],
    "nccl_for_deep_ep": os.environ["DEEPEP_NCCL_VERSION"],
}
with open(os.environ["EP_VERSIONS_FILE"], "w") as f:
    json.dump(versions, f, indent=2)
    f.write("\n")
for key, value in versions.items():
    print(f"{key}: {value}")
PY
