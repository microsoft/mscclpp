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
# Newer host drivers break with the container's host-injected CUDA compat libs (803);
# probe without them first, fall back to compat only if the native driver fails.
NATIVE_LD_PATH=$(echo "${LD_LIBRARY_PATH}" | tr ':' '\n' | grep -v '/compat' | paste -sd ':' -)
LD_LIBRARY_PATH="${NATIVE_LD_PATH}" /root/mscclpp/tools/peer-access-test/peer_access_test
PEER_ACCESS_EXIT_CODE=$?
set -e
RESOLVED_LD_PATH="${NATIVE_LD_PATH}"
if [ ${PEER_ACCESS_EXIT_CODE} -ne 0 ] && [ "${PLATFORM}" == "cuda" ] && [ -d "/usr/local/cuda/compat" ]; then
    echo "Native driver failed (exit ${PEER_ACCESS_EXIT_CODE}); retrying with CUDA compat libs"
    RESOLVED_LD_PATH="/usr/local/cuda/compat:${NATIVE_LD_PATH}"
    LD_LIBRARY_PATH="${RESOLVED_LD_PATH}" /root/mscclpp/tools/peer-access-test/peer_access_test
elif [ ${PEER_ACCESS_EXIT_CODE} -ne 0 ]; then
    exit ${PEER_ACCESS_EXIT_CODE}
fi
# Persist the resolved LD_LIBRARY_PATH so later docker exec sessions (run-remote.sh)
# reuse it and are not affected by the host runtime's compat-first injection.
echo "export LD_LIBRARY_PATH=\"${RESOLVED_LD_PATH}\"" > /root/mscclpp/.ldpath
make -C /root/mscclpp/tools/peer-access-test clean

if [ "${PLATFORM}" == "rocm" ]; then
    export CXX=/opt/rocm/bin/hipcc
fi

PIP_CMAKE_ARGS_FILE="/root/mscclpp/pip_cmake_args.txt"
if [ -f "${PIP_CMAKE_ARGS_FILE}" ]; then
    export CMAKE_ARGS="$(cat ${PIP_CMAKE_ARGS_FILE})"
    echo "Using CMAKE_ARGS: ${CMAKE_ARGS}"
fi

cd /root/mscclpp
if [[ "${CUDA_VERSION}" == *"12."* ]]; then
    pip3 install ".[cuda12,benchmark,test]"
elif [[ "${CUDA_VERSION}" == *"13."* ]]; then
    pip3 install ".[cuda13,benchmark,test]"
elif [ "${PLATFORM}" == "rocm" ]; then
    ROCM_VERSION=$(cat /opt/rocm/.info/version)
    ROCM_MAJOR="${ROCM_VERSION%%.*}"
    pip3 install ".[rocm${ROCM_MAJOR},benchmark,test]"
else
    pip3 install ".[benchmark,test]"
fi
pip3 install setuptools_scm
python3 -m setuptools_scm --force-write-version-files

mkdir -p /var/run/sshd
/usr/sbin/sshd -p 22345
