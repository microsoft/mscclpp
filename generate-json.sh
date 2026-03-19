#!/bin/bash
set -ex

# Check if the number of arguments is exactly 3
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <hostfile> <nnodes> <ppn>"
    exit 1
fi

HOSTFILE=$1
NNODES=$2
PPN=$3

parallel-scp -h "$HOSTFILE" -p128 -t1800 -r python/test/executor_test.py /home/azhpcuser/mahdieh/mscclpp/python/test/

parallel-scp -h "$HOSTFILE" -p128 -t1800 -r python/mscclpp/default_algos/mscclpp_send_recv.py /home/azhpcuser/mahdieh/mscclpp/python/mscclpp/default_algos/ 

parallel-ssh -h "$HOSTFILE" -p128 -i -t1800 "cd /home/azhpcuser/mahdieh/mscclpp && source mscclpp/bin/activate && python3 python/mscclpp/default_algos/mscclpp_send_recv.py --name send_recv_test --nnodes $NNODES --gpus_per_node $PPN --split_mask 0x3 > test.json "
