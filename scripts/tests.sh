#!/usr/bin/env bash

mpirun -allow-run-as-root \
       -tag-output \
       -map-by ppr:8:node \
       -bind-to numa \
       -x MSCCLPP_DEBUG_SUBSYS=ALL \
       -x MSCCLPP_DEBUG=TRACE \
       ./build/src/bootstrap/bootstrap_test
# MSCCLPP_DEBUG_SUBSYS=ALL MSCCLPP_DEBUG=TRACE ./build/src/bootstrap/init_test
