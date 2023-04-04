#!/bin/bash

set -ex

if ! [ -d build ] ; then
  ./setup.sh
fi

cmake --build build

cd build
pytest -s mscclpp
