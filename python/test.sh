#!/bin/bash

set -ex

pip install -e .

cd src
pytest -vs mscclpp
