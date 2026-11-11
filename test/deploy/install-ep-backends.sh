#!/bin/bash
# Install PyTorch and the latest NCCL-EP and DeepEP for the unified EP benchmark. Run after
# MSCCL++ is installed, in the base-dev-cuda13.x image. Installing PyTorch here rather than
# using a separate torch image avoids pulling a second multi-GB image onto the GB200 nodes.
#
# The GB200 nodes' root disk has no room for these packages (~7 GB), so everything is
# installed under EP_ENV_DIR, on the host's RAM-backed /dev/shm (shared with the container by
# --ipc=host): a venv layered on the image's Python environment, DeepEP's NCCL and nvcc, and
# pip's temporary files. Remove EP_ENV_DIR when done; it persists on the host until reboot.
#
# NCCL-EP requires the NCCL version it is validated with, while DeepEP requires a newer one.
# NCCL-EP's NCCL stays in the Python environment. DeepEP is built against the latest NCCL,
# installed into DEEPEP_NCCL_DIR, which run-ep-benchmark.sh adds to PYTHONPATH for DeepEP only.
#
# DeepEP JIT-compiles kernels at run time with ${CUDA_HOME}/bin/nvcc, and CUDA 13.0's ptxas
# rejects its PTX. A newer nvcc is installed into DEEPEP_CUDA_DIR, which run-ep-benchmark.sh
# uses as CUDA_HOME for DeepEP only.
#
# Environment overrides:
#   EP_ENV_DIR            directory for everything installed here (default: /dev/shm/mscclpp-ep)
#   TORCH_SPEC            pip requirement for PyTorch (default: torch==2.11.0)
#   TORCH_INDEX_URL       PyTorch wheel index (default: https://download.pytorch.org/whl/cu130)
#   NCCL_EP_SPEC          pip requirement for NCCL-EP  (default: nccl-extensions[cu13])
#   CUDA_CORE_SPEC        pip requirement for cuda-core (default: cuda-core[cu13])
#   DEEPEP_NCCL_SPEC      pip requirement for DeepEP's NCCL (default: nvidia-nccl-cu13, i.e. latest)
#   DEEPEP_NCCL_DIR       install directory for DeepEP's NCCL (default: EP_ENV_DIR/deepep-nccl)
#   DEEPEP_CUDA_VERSION   CUDA version of DeepEP's JIT compiler (default: 13.3.*)
#   DEEPEP_CUDA_DIR       install directory for DeepEP's JIT compiler (default: EP_ENV_DIR/deepep-cuda)
#   DEEPEP_REPO           DeepEP git URL               (default: deepseek-ai/DeepEP on GitHub)
#   DEEPEP_REF            DeepEP branch, tag or commit (default: the repository's default branch)
#   TORCH_CUDA_ARCH_LIST  GPU architecture for DeepEP  (default: detected with nvidia-smi)
#   EP_BACKEND_ENV        file recording the venv and DeepEP's NCCL and CUDA for run-ep-benchmark.sh
#                         (default: /tmp/ep-backend-env)
#   EP_VERSIONS_FILE      JSON file recording installed versions (default: /tmp/ep-backend-versions.json)

set -euo pipefail

EP_ENV_DIR="${EP_ENV_DIR:-/dev/shm/mscclpp-ep}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.11.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
NCCL_EP_SPEC="${NCCL_EP_SPEC:-nccl-extensions[cu13]}"
CUDA_CORE_SPEC="${CUDA_CORE_SPEC:-cuda-core[cu13]}"
DEEPEP_NCCL_SPEC="${DEEPEP_NCCL_SPEC:-nvidia-nccl-cu13}"
DEEPEP_NCCL_DIR="${DEEPEP_NCCL_DIR:-${EP_ENV_DIR}/deepep-nccl}"
DEEPEP_CUDA_VERSION="${DEEPEP_CUDA_VERSION:-13.3.*}"
DEEPEP_CUDA_DIR="${DEEPEP_CUDA_DIR:-${EP_ENV_DIR}/deepep-cuda}"
DEEPEP_REPO="${DEEPEP_REPO:-https://github.com/deepseek-ai/DeepEP.git}"
DEEPEP_REF="${DEEPEP_REF:-}"
EP_BACKEND_ENV="${EP_BACKEND_ENV:-/tmp/ep-backend-env}"
EP_VERSIONS_FILE="${EP_VERSIONS_FILE:-/tmp/ep-backend-versions.json}"
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
fi
export TORCH_CUDA_ARCH_LIST
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

echo "=== Creating the EP venv in ${EP_ENV_DIR}"
rm -rf "${EP_ENV_DIR}"
mkdir -p "${EP_ENV_DIR}/tmp"
export TMPDIR="${EP_ENV_DIR}/tmp"
EP_VENV="${EP_ENV_DIR}/venv"
BASE_SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
python -m venv "${EP_VENV}"
# Make the image's packages (MSCCL++, cuda-bindings, ...) importable after the venv's own,
# so packages installed here take precedence.
echo "import site; site.addsitedir('${BASE_SITE_PACKAGES}')" \
    > "$("${EP_VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/base-env.pth"
export PATH="${EP_VENV}/bin:${PATH}"
python -c "import mscclpp"

echo "=== Installing PyTorch (${TORCH_SPEC}) and DeepEP build dependencies"
pip install --no-cache-dir "${TORCH_SPEC}" ninja \
    --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple
if [ ! -f /usr/include/elfutils/libdwfl.h ]; then
    apt-get update
    apt-get install -y --no-install-recommends libdw-dev
    rm -rf /var/lib/apt/lists/*
fi

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

echo "=== Installing CUDA ${DEEPEP_CUDA_VERSION} nvcc for DeepEP's JIT into ${DEEPEP_CUDA_DIR}"
rm -rf "${DEEPEP_CUDA_DIR}"
pip install --no-cache-dir --no-deps --target "${DEEPEP_CUDA_DIR}" \
    "nvidia-cuda-nvcc==${DEEPEP_CUDA_VERSION}" "nvidia-cuda-crt==${DEEPEP_CUDA_VERSION}" \
    "nvidia-nvvm==${DEEPEP_CUDA_VERSION}" "nvidia-cuda-runtime==${DEEPEP_CUDA_VERSION}" \
    "nvidia-cuda-cccl==${DEEPEP_CUDA_VERSION}"
DEEPEP_CUDA_HOME="${DEEPEP_CUDA_DIR}/nvidia/cu13"
DEEPEP_NVCC_VERSION="$("${DEEPEP_CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
echo "DeepEP JIT nvcc: ${DEEPEP_NVCC_VERSION}"

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

printf 'EP_VENV=%s\nDEEPEP_PYTHONPATH=%s\nDEEPEP_CUDA_HOME=%s\n' \
    "${EP_VENV}" "${DEEPEP_NCCL_DIR}" "${DEEPEP_CUDA_HOME}" > "${EP_BACKEND_ENV}"

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
DEEPEP_NVCC_VERSION="${DEEPEP_NVCC_VERSION}" \
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
    "nvcc_for_deep_ep_jit": os.environ["DEEPEP_NVCC_VERSION"],
}
with open(os.environ["EP_VERSIONS_FILE"], "w") as f:
    json.dump(versions, f, indent=2)
    f.write("\n")
for key, value in versions.items():
    print(f"{key}: {value}")
PY
