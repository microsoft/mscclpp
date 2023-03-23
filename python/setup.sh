#!/bin/bash

set -ex
cmake -S . -B build
cmake --build build --clean-first -v
ldd build/py_mscclpp.cpython-39-x86_64-linux-gnu.so

