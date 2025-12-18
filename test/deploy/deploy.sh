set -e

# get parameter from $1 and $2
TEST_NAME=$1
IB_ENVIRONMENT="${2:-true}"

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

chmod 400 ${KeyFilePath}
ssh-keygen -t rsa -f sshkey -P ""

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
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION "sudo rm -rf ${DST_DIR}"
parallel-scp -t 0 -r -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION ${ROOT_DIR} ${DST_DIR}

# force to pull the latest image
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker pull ${CONTAINERIMAGE}"
if [ "${IB_ENVIRONMENT}" == "true" ]; then
  parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
    "sudo docker run --rm -itd --privileged --net=host --ipc=host --gpus=all \
    -w /root -v ${DST_DIR}:/root/mscclpp -v /opt/microsoft:/opt/microsoft --ulimit memlock=-1:-1 --name=mscclpp-test \
    --entrypoint /bin/bash ${CONTAINERIMAGE}"
else
  parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
    "sudo docker run --rm -itd --net=host --ipc=host --gpus=all --cap-add=SYS_ADMIN --security-opt seccomp=unconfined \
    -w /root -v ${DST_DIR}:/root/mscclpp -v /opt/microsoft:/opt/microsoft --ulimit memlock=-1:-1 --name=mscclpp-test \
    --entrypoint /bin/bash ${CONTAINERIMAGE}"
fi
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker exec -t --user root mscclpp-test bash '/root/mscclpp/test/deploy/setup.sh'"

