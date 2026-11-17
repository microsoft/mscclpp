#!/bin/bash
set -e

if [[ $OMPI_COMM_WORLD_RANK == 0 ]]
then
  python3 -m pytest /root/mscclpp/python/test -x -v
else
  python3 -m pytest /root/mscclpp/python/test -x 2>&1 >/dev/null
fi
