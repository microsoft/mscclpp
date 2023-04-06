#!/bin/bash

set -ex

isort src
black src

clang-format -i $(find src -name '*.cpp' -or -name '*.h')

