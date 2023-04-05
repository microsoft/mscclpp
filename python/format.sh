#!/bin/bash

set -ex

isort src
black src

clang-format -style='{
    "BasedOnStyle": "google",
    "BinPackParameters": false,
    "BinPackArguments": false,
    "AlignAfterOpenBracket": "AlwaysBreak"
}' -i src/*.cpp

