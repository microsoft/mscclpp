#!/bin/bash
set -ex

# Check if the number of arguments is exactly 1
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <hostfile>"
    exit 1
fi

HOSTFILE=$1

parallel-ssh -h "$HOSTFILE" -p32 -i -t1800 "mkdir -p /home/azhpcuser/mahdieh/mscclpp/build/lib/"

parallel-scp -h "$HOSTFILE" -p32 -t1800 -r /home/azhpcuser/mahdieh/mscclpp/build/lib/libmscclpp_nccl.so /home/azhpcuser/mahdieh/mscclpp/build/lib/

