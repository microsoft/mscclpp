#!/bin/bash
# deploy.sh — Provisions remote hosts, copies sources, and launches Docker containers
# for mscclpp CI/CD test environments.
#
# Usage: deploy.sh <test_name> [ib_environment] [platform] [container_name]
#   test_name       : Test suite to deploy (e.g. single-node-test, nccltest-single-node)
#   ib_environment  : Enable InfiniBand networking (default: true)
#   platform        : Target GPU platform — "cuda" or "rocm" (default: cuda)
#   container_name  : Docker container name (default: mscclpp-test)

set -ex

###############################################################################
# 1. Parse arguments
###############################################################################
TEST_NAME=$1
IB_ENVIRONMENT="${2:-true}"
PLATFORM="${3:-cuda}"
CONTAINER_NAME="${4}"

###############################################################################
# 2. Resolve paths and host file
###############################################################################
KeyFilePath=${SSHKEYFILE_SECUREFILEPATH}
ROOT_DIR="${SYSTEM_DEFAULTWORKINGDIRECTORY}/"

if [ "${TEST_NAME}" == "nccltest-single-node" ]; then
  ROOT_DIR="${ROOT_DIR}/mscclpp"
  SYSTEM_DEFAULTWORKINGDIRECTORY="${SYSTEM_DEFAULTWORKINGDIRECTORY}/mscclpp"
fi

DST_DIR="/tmp/mscclpp"

if [ "${TEST_NAME}" == "nccltest-single-node" ] || [ "${TEST_NAME}" == "single-node-test" ]; then
  HOSTFILE="${SYSTEM_DEFAULTWORKINGDIRECTORY}/test/deploy/hostfile_ci"
else
  HOSTFILE="${SYSTEM_DEFAULTWORKINGDIRECTORY}/test/deploy/hostfile"
fi

SSH_OPTION="StrictHostKeyChecking=no"

###############################################################################
# 3. Prepare SSH keys
###############################################################################
chmod 400 ${KeyFilePath}
ssh-keygen -t rsa -f sshkey -P ""

###############################################################################
# 4. Wait for remote hosts to be reachable
###############################################################################
while true; do
  set +e
  parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION "hostname"
  if [ $? -eq 0 ]; then
    break
  fi
  echo "Waiting for sshd to start..."
  sleep 5
done
set -e

###############################################################################
# 5. Copy source tree to remote hosts
###############################################################################
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION "sudo rm -rf ${DST_DIR}"
parallel-scp -t 0 -r -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION ${ROOT_DIR} ${DST_DIR}

###############################################################################
# 6. Platform-specific setup (ROCm kernel module)
###############################################################################
if [ "${PLATFORM}" == "rocm" ]; then
  parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION "sudo modprobe amdgpu"
fi

###############################################################################
# 7. Pull the latest container image
###############################################################################
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker pull ${CONTAINERIMAGE}"

###############################################################################
# 8. Remove any existing container with the same name
###############################################################################
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker rm -f ${CONTAINER_NAME} 2>/dev/null || true"

###############################################################################
# 9. Launch Docker container
###############################################################################

if [ "${CONTAINER_NAME}" == "sglang-mscclpp-test" ]; then
  parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
    "sudo docker run -itd --name=sglang-mscclpp-test --privileged --net=host --ipc=host --gpus=all -w /root -v ${DST_DIR}:/root/mscclpp --entrypoint /bin/bash lmsysorg/sglang:latest"
else
  # Set GPU passthrough flags based on platform
  LAUNCH_OPTION="--gpus=all"
  if [ "${PLATFORM}" == "rocm" ]; then
    LAUNCH_OPTION="--device=/dev/kfd --device=/dev/dri --group-add=video"
  fi

  if [ "${IB_ENVIRONMENT}" == "true" ]; then
    # InfiniBand: use --privileged for RDMA device access
    parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
      "sudo docker run --rm -itd --privileged --net=host --ipc=host ${LAUNCH_OPTION} \
      -w /root -v ${DST_DIR}:/root/mscclpp -v /opt/microsoft:/opt/microsoft --ulimit memlock=-1:-1 --name=mscclpp-test \
      --entrypoint /bin/bash ${CONTAINERIMAGE}"
  else
    # Non-IB: grant SYS_ADMIN and disable seccomp instead of full --privileged
    parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
      "sudo docker run --rm -itd --net=host --ipc=host ${LAUNCH_OPTION} --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
      -w /root -v ${DST_DIR}:/root/mscclpp -v /opt/microsoft:/opt/microsoft --ulimit memlock=-1:-1 --name=mscclpp-test \
      --entrypoint /bin/bash ${CONTAINERIMAGE}"
  fi
fi

###############################################################################
# 9. Run setup script inside the container
###############################################################################
if [ "${CONTAINER_NAME}" = "sglang-mscclpp-test" ]; then
  parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
    "sudo docker exec -t --user root sglang-mscclpp-test bash '/root/mscclpp/test/deploy/setup.sh' ${PLATFORM}"
else
  parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
    "sudo docker exec -t --user root mscclpp-test bash '/root/mscclpp/test/deploy/setup.sh' ${PLATFORM}"
fi
