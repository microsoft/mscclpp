#!/bin/bash
set -ex

# Check if the number of arguments is exactly 1
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi
export MSCCLPPHOME=/home/azhpcuser/mahdieh/mscclpp-unittest/mscclpp/

HOSTFILE=$1

parallel-scp -h "$HOSTFILE" -p128 -t1800 -r  ./*.json $MSCCLPPHOME

parallel-scp -h "$HOSTFILE" -p128 -t1800 -r ./python/test/executor_test.py $MSCCLPPHOME/python/test/

parallel-scp -h "$HOSTFILE" -p128 -t1800 -r ./python/test/executor_test_verifier.cu $MSCCLPPHOME/python/test/
