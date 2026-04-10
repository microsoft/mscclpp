set -e

PLATFORM="${1:-cuda}"

mkdir -p /root/.ssh
mv /root/mscclpp/sshkey.pub /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
chmod 400 /root/mscclpp/sshkey
chown root:root /root/mscclpp/sshkey

# Generate SSH config from hostfile_mpi
HOSTFILE_MPI=/root/mscclpp/test/deploy/hostfile_mpi
if [ -f "${HOSTFILE_MPI}" ]; then
    > /root/.ssh/config
    while IFS= read -r host; do
        echo "Host ${host}" >> /root/.ssh/config
        echo "  Port 22345" >> /root/.ssh/config
        echo "  IdentityFile /root/mscclpp/sshkey" >> /root/.ssh/config
        echo "  StrictHostKeyChecking no" >> /root/.ssh/config
    done < "${HOSTFILE_MPI}"
    chown root:root /root/.ssh/config
fi

if [ "${PLATFORM}" == "cuda" ]; then
    nvidia-smi -pm 1
    for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
        nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i $i | sed 's/\ //') -i $i
    done
fi

make -C /root/mscclpp/tools/peer-access-test
set +e
/root/mscclpp/tools/peer-access-test/peer_access_test
PEER_ACCESS_EXIT_CODE=$?
set -e
if [ ${PEER_ACCESS_EXIT_CODE} -eq 2 ]; then
    # Exit code 2 = CUDA init failure (e.g., driver/toolkit version mismatch).
    # Add CUDA compat libs for forward compatibility and retry.
    CUDA_COMPAT_PATH="/usr/local/cuda/compat"
    if [ -d "${CUDA_COMPAT_PATH}" ]; then
        echo "Adding ${CUDA_COMPAT_PATH} to LD_LIBRARY_PATH for forward compatibility"
        export LD_LIBRARY_PATH="${CUDA_COMPAT_PATH}:${LD_LIBRARY_PATH}"
        /root/mscclpp/tools/peer-access-test/peer_access_test
    else
        echo "CUDA compat libs not found at ${CUDA_COMPAT_PATH}"
        exit 1
    fi
elif [ ${PEER_ACCESS_EXIT_CODE} -ne 0 ]; then
    exit ${PEER_ACCESS_EXIT_CODE}
fi
make -C /root/mscclpp/tools/peer-access-test clean

if [[ "${CUDA_VERSION}" == *"11."* ]]; then
    pip3 install -r /root/mscclpp/python/requirements_cuda11.txt
elif [[ "${CUDA_VERSION}" == *"12."* ]]; then
    pip3 install -r /root/mscclpp/python/requirements_cuda12.txt
fi

if [ "${PLATFORM}" == "rocm" ]; then
    export CXX=/opt/rocm/bin/hipcc
fi

PIP_CMAKE_ARGS_FILE="/root/mscclpp/pip_cmake_args.txt"
if [ -f "${PIP_CMAKE_ARGS_FILE}" ]; then
    export CMAKE_ARGS="$(cat ${PIP_CMAKE_ARGS_FILE})"
    echo "Using CMAKE_ARGS: ${CMAKE_ARGS}"
fi
cd /root/mscclpp && pip3 install .
pip3 install setuptools_scm
python3 -m setuptools_scm --force-write-version-files

mkdir -p /var/run/sshd
/usr/sbin/sshd -p 22345
