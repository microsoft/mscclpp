set -e

KeyFilePath=${SSHKEYFILE_SECUREFILEPATH}
SRC_DIR="${SYSTEM_DEFAULTWORKINGDIRECTORY}/build"
DST_DIR="/tmp/mscclpp"
HOSTFILE="${SYSTEM_DEFAULTWORKINGDIRECTORY}/test/deploy/hostfile"
DEPLOY_DIR="${SYSTEM_DEFAULTWORKINGDIRECTORY}/test/deploy"
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
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION "rm -rf ${DST_DIR}"
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION "mkdir -p ${DST_DIR}"
parallel-scp -t 0 -r -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION ${SRC_DIR} ${DST_DIR}

parallel-scp -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION sshkey ${DST_DIR}
parallel-scp -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION sshkey.pub ${DST_DIR}
parallel-scp -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION ${DEPLOY_DIR}/* ${DST_DIR}

# force to pull the latest image
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker pull ghcr.io/microsoft/mscclpp/mscclpp:base-cuda12.1"
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker run --rm -itd --privileged --net=host --ipc=host --gpus=all \
  -w /root -v ${DST_DIR}:/root/mscclpp --name=mscclpp-test \
  --entrypoint /bin/bash ghcr.io/microsoft/mscclpp/mscclpp:base-cuda12.1"
parallel-ssh -i -t 0 -h ${HOSTFILE} -x "-i ${KeyFilePath}" -O $SSH_OPTION \
  "sudo docker exec -t --user root mscclpp-test bash '/root/mscclpp/setup.sh'"

