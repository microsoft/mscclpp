#!/usr/bin/env bash

LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./build/lib
mpirun -allow-run-as-root \
       -tag-output \
       -map-by ppr:8:node \
       -bind-to numa \
       -x MSCCLPP_DEBUG_SUBSYS=ALL \
       -x MSCCLPP_SOCKET_IFNAME=eth0 \
       ./build/bin/tests/p2p_test_mpi 172.17.0.4:50000
