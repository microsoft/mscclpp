set -e

PLATFORM="${1:-cuda}"

mkdir -p /root/.ssh
mv /root/mscclpp/sshkey.pub /root/.ssh/authorized_keys
chown root:root /root/.ssh/authorized_keys
mv /root/mscclpp/test/deploy/config /root/.ssh/config
chown root:root /root/.ssh/config
chmod 400 /root/mscclpp/sshkey
chown root:root /root/mscclpp/sshkey

if [ "${PLATFORM}" == "cuda" ]; then
    nvidia-smi -pm 1
    for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
        nvidia-smi -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i $i | sed 's/\ //') -i $i
    done
fi

make -C /root/mscclpp/tools/peer-access-test
/root/mscclpp/tools/peer-access-test/peer_access_test
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
if [[ "${CUDA_VERSION}" == *"11."* ]]; then
    pip3 install ".[cuda11,benchmark,test]"
elif [[ "${CUDA_VERSION}" == *"12."* ]]; then
    pip3 install ".[cuda12,benchmark,test]"
elif [ "${PLATFORM}" == "rocm" ]; then
    pip3 install ".[rocm6,benchmark,test]"
else
    pip3 install ".[benchmark,test]"
fi
pip3 install setuptools_scm
python3 -m setuptools_scm --force-write-version-files

mkdir -p /var/run/sshd
/usr/sbin/sshd -p 22345
