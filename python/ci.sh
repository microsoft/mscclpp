#!/bin/bash
# CI hook script.

set -ex

# CD to this directory.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# clean env
rm -rf .venv build

# setup a python virtual env
python -m venv .venv

# activate the virtual env
source .venv/bin/activate

# install venv deps.
pip install -r dev-requirements.txt

# run the build and test.
./test.sh

